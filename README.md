# Local STT (Vosk) for Electron/Web Fallback

This adds a self-contained Speech-to-Text server that the app can use when Web Speech API is unavailable (e.g., in Electron).

- Server: FastAPI + Vosk with a streaming WebSocket at ws://localhost:8001/ws
- Audio input: PCM 16-bit, mono, 16kHz frames
- Messages:
  - Client sends optional text init: { "type": "init", "sampleRate": 16000 }
  - Then client streams binary PCM frames
  - Server sends JSON text messages: { type: 'partial'|'final', text: string }

## Install

```bash
python -m venv venv
venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
python server.py
```

On first run, it will download the small English model (~50–60MB) into stt/models/.

## Configure Frontend

In `useSpeechRecognition.tsx`, when Web Speech API is unavailable or in Electron, connect to ws://localhost:8001/ws, send an init message, and stream mic audio as 16kHz PCM mono chunks. Handle `partial` and `final` messages to update transcript and fire result callbacks. Alternatively, a Deepgram-like WS is available at `ws://localhost:8001/v1/listen`, and HTTP file transcription at `POST /v1/transcribe`.

This folder is isolated. It does not modify `os/` or automation code.


