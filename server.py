#!/usr/bin/env python3
"""
Local Speech-to-Text (STT) server using Vosk with streaming WebSocket.

Endpoints:
 - WS /ws    : binary stream of PCM S16LE mono @16kHz; emits JSON partial/final transcripts
 - GET /health

Setup:
 - pip install -r stt/requirements.txt
 - On first run, downloads Vosk small English model to stt/models/

Run:
 - python stt/server.py  (defaults to 0.0.0.0:8001)

Notes:
 - Only the STT lives here; no changes to os/ automation code.
 - Frontend will send 16kHz 16-bit mono PCM frames over WebSocket.
"""

import asyncio
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional, List
import io

import uvicorn
import numpy as np
import soundfile as sf
import shutil
import subprocess
import tarfile
import stat
import tempfile
import platform
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
BIN_DIR = BASE_DIR / "bin"
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_DIR = MODELS_DIR / MODEL_NAME
MODEL_ZIP = MODELS_DIR / f"{MODEL_NAME}.zip"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"


def ensure_model() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_DIR.exists():
        return
    # Download model zip if not present
    try:
        import requests  # type: ignore
    except Exception:
        print("Installing requests for model download...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests  # type: ignore

    print(f"Downloading Vosk model from {MODEL_URL} ...")
    with requests.get(MODEL_URL, stream=True, timeout=300) as r:  # type: ignore
        r.raise_for_status()
        with open(MODEL_ZIP, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("Download complete. Extracting...")
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall(MODELS_DIR)
    print("Model extracted.")
    try:
        MODEL_ZIP.unlink(missing_ok=True)  # type: ignore[attr-defined]
    except Exception:
        pass


app = FastAPI(title="Local STT Server (Vosk)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "engine": "vosk", "model": MODEL_NAME}

@app.get("/")
def root():
    return JSONResponse({
        "service": "Local STT (Vosk)",
        "endpoints": [
            "/health",
            "/ws",
            "/v1/listen",
            "/v1/transcribe",
            "/v1/models",
        ],
        "docs": "/docs"
    })

@app.get("/favicon.ico")
def favicon():
    # No favicon, return empty 204
    return PlainTextResponse("", status_code=204)


def which_ffmpeg() -> Optional[str]:
    # 1) PATH
    path = shutil.which("ffmpeg")
    if path:
        return path
    # 2) bundled
    bundled = BIN_DIR / ("ffmpeg.exe" if platform.system().lower().startswith("win") else "ffmpeg")
    if bundled.exists():
        return str(bundled)
    return None


def ensure_ffmpeg() -> Optional[str]:
    path = which_ffmpeg()
    if path:
        return path
    # Attempt to download static build for linux amd64 if allowed
    if platform.system().lower() == 'linux':
        try:
            BIN_DIR.mkdir(parents=True, exist_ok=True)
            url = os.environ.get(
                "FFMPEG_STATIC_URL",
                "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            )
            print(f"Downloading static ffmpeg from {url} ...")
            import requests  # type: ignore
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".tar.xz", delete=False) as tf:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            tf.write(chunk)
                    tmp_path = tf.name
            # Extract ffmpeg binary
            with tarfile.open(tmp_path, "r:xz") as tar:
                member = next((m for m in tar.getmembers() if m.name.endswith("/ffmpeg")), None)
                if not member:
                    raise RuntimeError("ffmpeg binary not found in archive")
                tar.extract(member, path=BIN_DIR)
                extracted = BIN_DIR / member.name
                # Move to BIN_DIR/ffmpeg
                target = BIN_DIR / "ffmpeg"
                if target.exists():
                    target.unlink()
                extracted.rename(target)
                # Make executable
                target.chmod(target.stat().st_mode | stat.S_IEXEC)
            os.unlink(tmp_path)
            return str(BIN_DIR / "ffmpeg")
        except Exception as e:
            print(f"Failed to download ffmpeg: {e}")
            return None
    return None


def decode_with_ffmpeg(content: bytes, target_sr: int = 16000) -> Optional[bytes]:
    ffmpeg_path = ensure_ffmpeg()
    if not ffmpeg_path:
        return None
    try:
        # Decode any input to 16kHz mono s16le on stdout
        proc = subprocess.Popen(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel","error",
                "-i","pipe:0",
                "-ac","1",
                "-ar", str(target_sr),
                "-f","s16le",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = proc.communicate(input=content, timeout=60)
        if proc.returncode == 0 and out:
            return out
        else:
            print(f"ffmpeg decode error: rc={proc.returncode} err={err.decode(errors='ignore')}")
            return None
    except Exception as e:
        print(f"ffmpeg exception: {e}")
        return None


class VoskSession:
    def __init__(self, sample_rate: int = 16000):
        from vosk import Model, KaldiRecognizer  # type: ignore

        self.model = Model(str(MODEL_DIR))
        self.sample_rate = sample_rate
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.recognizer.SetWords(True)

    def accept_waveform(self, data: bytes) -> Optional[dict]:
        if self.recognizer.AcceptWaveform(data):
            return json.loads(self.recognizer.Result())
        else:
            # Partial result
            return None

    def partial(self) -> dict:
        return json.loads(self.recognizer.PartialResult())

    def final(self) -> dict:
        return json.loads(self.recognizer.FinalResult())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    # Optional: client may send a JSON init message with sampleRate
    sample_rate = 16000
    vosk_session: Optional[VoskSession] = None
    try:
        while True:
            message = await ws.receive()
            if "text" in message and message["text"]:
                # Expect optional init message
                try:
                    obj = json.loads(message["text"])  # type: ignore[arg-type]
                    if isinstance(obj, dict) and obj.get("type") == "init":
                        sample_rate = int(obj.get("sampleRate", 16000))
                        # Initialize recognizer on init
                        vosk_session = VoskSession(sample_rate=sample_rate)
                        await ws.send_text(json.dumps({"type": "ready"}))
                        continue
                except Exception:
                    # Non-JSON text messages are ignored
                    pass
                continue

            data: bytes = message.get("bytes")  # type: ignore[assignment]
            if not data:
                continue

            if vosk_session is None:
                vosk_session = VoskSession(sample_rate=sample_rate)

            # Feed PCM chunk
            result = vosk_session.accept_waveform(data)
            if result and result.get("text"):
                await ws.send_text(json.dumps({
                    "type": "final",
                    "text": result.get("text", "")
                }))
            else:
                partial = vosk_session.partial()
                if partial and partial.get("partial"):
                    await ws.send_text(json.dumps({
                        "type": "partial",
                        "text": partial.get("partial", "")
                    }))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        # Send final result if any
        if vosk_session is not None:
            final = vosk_session.final()
            if final and final.get("text"):
                try:
                    await ws.send_text(json.dumps({
                        "type": "final",
                        "text": final.get("text", "")
                    }))
                except Exception:
                    pass
        await ws.close()


# --- Deepgram/ElevenLabs-like HTTP/WS API surface ---

@app.get("/v1/models")
def list_models():
    return {
        "models": [
            {
                "name": "vosk-small-en",
                "language": "en-US",
                "sample_rate": 16000,
                "features": ["streaming", "partial_results"],
            }
        ]
    }


@app.post("/v1/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = "en-US",
    punctuate: bool = True,
    words: bool = False,
):
    try:
        # Read entire file into memory and decode
        content = await file.read()

        pcm_bytes: Optional[bytes] = None
        target_sr = 16000
        # Try ffmpeg first for broad codec support
        pcm_bytes = decode_with_ffmpeg(content, target_sr=target_sr)
        # Fallback to soundfile (wav/flac/ogg; mp3 if supported by libsndfile)
        if pcm_bytes is None:
            try:
                data, sr = sf.read(io.BytesIO(content), dtype='float32', always_2d=True)
                # Mixdown to mono
                mono = data.mean(axis=1)
                # Resample to 16kHz if needed (linear interpolation)
                if sr != target_sr and len(mono) > 0:
                    x_old = np.linspace(0, 1, num=len(mono), endpoint=False)
                    x_new = np.linspace(0, 1, num=int(len(mono) * (target_sr / sr)), endpoint=False)
                    mono = np.interp(x_new, x_old, mono).astype(np.float32)
                # Convert to int16 PCM
                pcm_i16 = np.clip(mono, -1.0, 1.0)
                pcm_i16 = (pcm_i16 * 32767).astype(np.int16)
                pcm_bytes = pcm_i16.tobytes()
            except Exception:
                pass
        # Fallback: naive WAV header strip or raw
        if pcm_bytes is None:
            if content[:4] == b"RIFF" and b"WAVEfmt" in content[:64]:
                idx = content.find(b"data")
                if idx != -1 and idx + 8 < len(content):
                    pcm_bytes = content[idx + 8 :]
            if pcm_bytes is None:
                pcm_bytes = content

        # Recognize
        session = VoskSession(sample_rate=target_sr)
        text_parts: List[str] = []
        words_list: List[dict] = []
        if pcm_bytes:
            # 100ms chunks at 16kHz mono int16 = 16000 samples * 2 bytes * 0.1 = 3200 bytes
            chunk_size = 3200
            for i in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[i : i + chunk_size]
                res = session.accept_waveform(chunk)
                if res and res.get("text"):
                    text_parts.append(res.get("text"))
                    if words and res.get("result"):
                        words_list.extend(res.get("result"))
        final = session.final()
        if final.get("text"):
            text_parts.append(final.get("text"))
            if words and final.get("result"):
                words_list.extend(final.get("result"))

        transcript = " ".join([t for t in text_parts if t])
        resp = {
            "request_id": None,
            "results": [
                {
                    "alternatives": [
                        {"transcript": transcript, "confidence": None, "words": (words_list if words else None)}
                    ],
                    "language": language,
                }
            ],
            "metadata": {"model": "vosk-small-en", "punctuate": punctuate},
        }
        return resp
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/v1/listen")
async def websocket_listen(ws: WebSocket):
    await ws.accept()
    sample_rate = 16000
    interim = True
    session: Optional[VoskSession] = None
    try:
        while True:
            message = await ws.receive()
            if "text" in message and message["text"]:
                # Expect Deepgram-like start message
                try:
                    obj = json.loads(message["text"])  # type: ignore[arg-type]
                    if obj.get("type") in ("start", "config"):
                        if "sample_rate" in obj:
                            sample_rate = int(obj.get("sample_rate"))
                        if "interim_results" in obj:
                            interim = bool(obj.get("interim_results"))
                        session = VoskSession(sample_rate=sample_rate)
                        await ws.send_text(json.dumps({"type": "open"}))
                        continue
                except Exception:
                    pass
                continue

            data: bytes = message.get("bytes")  # type: ignore[assignment]
            if not data:
                continue
            if session is None:
                session = VoskSession(sample_rate=sample_rate)

            if session.accept_waveform(data):
                res = session.final()
                alt = {"transcript": res.get("text", "")}
                await ws.send_text(json.dumps({
                    "type": "transcript",
                    "is_final": True,
                    "channel": {"alternatives": [alt]},
                }))
            elif interim:
                part = session.partial()
                alt = {"transcript": part.get("partial", "")}
                if alt["transcript"]:
                    await ws.send_text(json.dumps({
                        "type": "transcript",
                        "is_final": False,
                        "channel": {"alternatives": [alt]},
                    }))
    except WebSocketDisconnect:
        pass
    finally:
        try:
            if session is not None:
                res = session.final()
                if res.get("text"):
                    await ws.send_text(json.dumps({
                        "type": "transcript",
                        "is_final": True,
                        "channel": {"alternatives": [{"transcript": res.get("text", "")} ]},
                    }))
        except Exception:
            pass
        await ws.close()


def main():
    # Ensure Vosk model exists
    ensure_model()
    host = os.environ.get("STT_HOST", "0.0.0.0")
    # Prefer Render's PORT if present
    port = int(os.environ.get("PORT", os.environ.get("STT_PORT", "8001")))
    # Avoid import-by-string so this runs from any working directory (Render-friendly)
    uvicorn.run(app, host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()


