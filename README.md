# VTracer Backend — Railway Deploy

## Files
- `vtracer_server.py` — Flask app (main entrypoint, has `/vectorize`, `/export_eps`, `/segment`, `/health`)
- `eps_export.py`, `color_segmentation.py`, `path_smoothing.py`, `path_cleanup.py`, `background_removal.py` — helper modules imported by the server
- `requirements.txt` — Python deps (includes `gunicorn`)
- `Procfile` — tells Railway how to start the app
- `Nur Meta AI - VTracer.html` — the frontend tool (open this in a browser, or host it separately)

## Deploy on Railway

1. Push this folder to a GitHub repo (all files in the repo root — don't nest them in a subfolder, or update the Procfile path accordingly).
2. On Railway: **New Project → Deploy from GitHub repo** → pick the repo.
3. Railway auto-detects Python and reads `Procfile` + `requirements.txt`. No extra config needed — `$PORT` is provided automatically by Railway and the app already reads it.
4. Wait for the build to finish, then open **Settings → Networking → Generate Domain** to get a public URL, e.g. `https://your-app.up.railway.app`.
5. Test it: visit `https://your-app.up.railway.app/health` — it should return `ok`.

## Point the HTML tool at the deployed backend

1. Open `Nur Meta AI - VTracer.html` (locally, or host it wherever you like — it's a static file).
2. In the **AI Vectorizer → VTracer Backend Server** card, set **Backend URL** to:
   `https://your-app.up.railway.app/vectorize`
3. Click **Save URL**, then **Test Connection** — should say "Connected".

## Notes
- `vtracer` and `opencv-python-headless` can take a few minutes to build on first deploy — this is normal.
- The server runs with `--workers 1` in the Procfile since VTracer tracing is CPU-heavy; bump this only if you upgrade to a bigger Railway instance.
- CORS is open (`CORS(app)`), so any frontend origin (including a file opened straight from disk) can call this backend.
