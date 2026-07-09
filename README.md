# Korean Audio Dataset Stats API

Endpoint that receives Korean TTS audio describing a dataset, transcribes it,
parses the spoken table into a DataFrame, and returns the required statistics JSON.

## What you submit
Only the **API endpoint URL** (the `POST /` route). The grader sends:
```json
{"audio_id": "q0", "audio_base64": "..."}
```
and strict-matches the returned JSON:
```
rows, columns, mean, std, variance, min, max, median, mode,
range, allowed_values, value_range, correlation
```

## Key design fact
Earlier attempts treated the audio as raw PCM — that failed. Grader feedback
proved the **columns come from the spoken content** (e.g. `온도` = temperature).
So this server transcribes Korean speech (Whisper) and parses the described table.

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Deploy for a public URL

### Render (recommended)
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- First request is slow (Whisper model download + load). Consider `WHISPER_MODEL=base` for speed.

### Any host + tunnel (quick test)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
# in another shell:
cloudflared tunnel --url http://localhost:8000   # or: ngrok http 8000
```
Submit the public https URL.

## Env vars
- `WHISPER_MODEL` (default `small`) — `base` is faster, `medium` more accurate.
- `WHISPER_LANG` (default `ko`).
- `ALLOWED_VALUES_MAX_UNIQUE` (default `20`) — columns with more distinct values get no `allowed_values` entry.
- `DEBUG_TOKEN` — protects the debug endpoints.

## Tuning to an exact match (important)
The one unknown is exactly HOW the dataset is spoken. Capture a real sample:
1. Deploy, submit the URL so the grader hits it once.
2. `GET /debug/list` → see captured `audio_id`s.
3. `GET /debug/transcript/{audio_id}` → see the Whisper transcript + how numbers/columns come through.
4. `GET /debug/audio/{audio_id}` → download the actual audio to listen.
5. Adjust `parse_transcript()` in `main.py` to match the real spoken format, redeploy.

Add `?token=YOUR_DEBUG_TOKEN` to debug URLs if `DEBUG_TOKEN` is set.
```
