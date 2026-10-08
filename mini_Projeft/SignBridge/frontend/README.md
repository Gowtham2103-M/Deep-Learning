# SignBridge frontend (React + TypeScript + Vite)

```powershell
cd frontend
npm install        # first time only
npm run dev        # http://localhost:5173  (backend must run on port 8000)
npm run build      # type-check + production build into dist/
```

The dev server forwards `/api/*` to `http://127.0.0.1:8000` (see `vite.config.ts`).
To use another backend address create `.env` with `VITE_API_BASE=http://host:port`.

Files:
* `src/api.ts` - every backend call and response type (URLs match `backend/app/routes`).
* `src/hooks/useRecognition.ts` - camera, sliding frame buffer, prediction loop, stability rule.
* `src/hooks/useSpeak.ts` - text-to-speech with cooldown (no repeated speaking).
* `src/hooks/useSpeechRecognition.ts` - microphone speech-to-text (Web Speech API, Chrome/Edge).
* `src/components/` - `RecognitionPanel`, `SpeechPanel`, `MetricsPanel`, `StatusBanner`.
