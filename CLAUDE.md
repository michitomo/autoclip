# CLAUDE.md

autoclip: generate vertical social clips of a specific Diet member's questioning from
衆議院TV (shugiintv.go.jp) committee video. Flow: date→committee→member→(topic) →
burned-in subtitles + JetCut (filler/dead-air removal) + Q&A-hierarchy editing.
Human-review required (don't cut political speech out of context).
Legal basis: 著作権法 40-1 + 48 (source attribution required).

## Layout
- `backend/` — Python 3.12 / FastAPI. Copy of an upstream Diet-transcription
  pipeline (scrape→Whisper→LLM correction) + video-clip layer. **Imports use
  `from src.xxx`; run with cwd=`backend/`.**
- `frontend/` — Vite + React + TS. Wizard + Q&A tree editor. Vite dev server proxies
  `/api` → backend :8000.
- `media/clips/<session_id>/` — outputs (mp4/ass/project.json). **gitignored.**
- `kokkaidb/` — reference symlink to the upstream pipeline, gitignored (local only).

## Run
```bash
cd backend
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
# keys in backend/.env (DEEPINFRA_API_KEY, OPENROUTER_API_KEY); or reuse an existing
# pipeline .env -> `set -a; source <path-to>/.env; set +a`
.venv/bin/uvicorn src.app:app --port 8000 --reload
# frontend (separate term): cd frontend && npm install && npm run dev   # :5173
# one clip via CLI: .venv/bin/python -m src.clip_service --session-id 56345 \
#   --member 辰巳孝太郎 --out-dir media/clips/56345
```
Tests: `cd backend && .venv/bin/python -m pytest -m "not integration" -q` (pure logic,
no net). Frontend typecheck: `cd frontend && npm run build`.

## Gotchas (load-bearing)
1. **ffmpeg needs libass** for `subtitles=` filter. Plain `brew install ffmpeg` lacks it
   on macOS; use `brew install ffmpeg-full`. `src/video/ffmpeg.py` auto-detects it.
2. **Renderer uses trim+concat**, not one big `select='between()+between()...'` expr.
   The single-expr form blows up ffmpeg's parser at ~100+ ranges (exit 244 / OOM).
   See `src/video/renderer.py`. Perf notes: render floor for a ~16-min source is
   ~70s — bottleneck is decode + 96-range trim/concat + libass burn, NOT encode (HW
   videotoolbox tested = only ~5% faster, dropped). Export = x264 `veryfast` full-res;
   preview = `ultrafast` + `scale_factor=1/3` (360p) + `fps=15` (ASS stays full
   PlayRes so libass auto-scales). Whisper window concurrency = env
   `WHISPER_WINDOW_CONCURRENCY` (default 24, was 8); Q&A annotate chunks small (20) ×
   parallel for lower latency.
3. **Two timelines, never mix:**
   - member-WAV time (member audio starts at 0): `WhisperWord`, `KeepRange`, `EditRange`,
     `QASegment.start/end`, `QASentence.start/end`, `KeptWord.old_*`.
   - post-cut time (concatenated enabled keep_ranges = the rendered `_clip.mp4`):
     `EditCaption.start/end`, `KeptWord.new_*`.
   - Convert: member→post-cut = `_member_to_post_cut` (and TS `Editor.memberToClip`,
     keep them identical). post-cut→member→filtered = `_remap_caption_time`.
4. **Q&A tree = selection layer; flat EditRange = render contract.** `ClipProject.qa_tree`
   (topic>speaker>sentence) only drives the UI. Rendering always consumes flat
   `ClipProject.ranges` (`rerender_project`/`render_topic_clips`/`render_clip`).
   - Only leaf `QASentence.enabled` is truth; parent state derived (tri-state).
   - `apply_tree_to_ranges` recomputes `ranges[i].enabled` by "range midpoint inside an
     enabled sentence span" (called by `/edit`, `/export`).
   - `qa_tree=None` (old projects) → flat range-toggle UI fallback.
5. **Editor never re-renders.** Toggles are instant; preview `<video>` plays the long
   `_clip.mp4` and skips off-sentence spans via `onTimeUpdate`. Real output only via
   "export topics/full" (`/export`). Long `_clip.mp4` is preview-only; deliverables are
   `_topicN_clip.mp4`. React hooks must precede early returns.
6. **Subtitle styling:** questioner=mint `#5FE0B7`, answerer/参考人=orange
   (`_role_spans_post_cut` → `build_ass_*(role_spans=...)`). Persistent top banner after
   the title panel (`_banner_event`). LLM annotation (`qa_annotate.py`) sets importance
   高/中/低 but all sentences start `enabled=True` (low is initially ON; human removes).
7. **ASR/LLM:** ASR = DeepInfra `whisper-large-v3-turbo` (word timestamps confirmed;
   OpenRouter ASR returns no timestamps). `ASR_PROVIDER=groq` switches. LLM = OpenRouter,
   default `google/gemma-4-26b-a4b-it` (env `AUTOCLIP_LLM_MODEL`). Member WAV extracted
   WITHOUT silenceremove (else word times desync from video).

## Key files
- `src/clip_service.py` — orchestration + CLI: `generate_clip`, `rerender_project`,
  `render_topic_clips`, `apply_tree_to_ranges`, `_render_span_clip`,
  `_remap_caption_time`, `_member_to_post_cut`, `_role_spans_post_cut`.
- `src/video/subtitles.py` — ASS: `build_ass_karaoke/rolling/from_captions`, title panel
  (`_title_event`/`_build_title_layout`), banner (`_banner_event`), color (`_role_colour_at`).
- `src/video/renderer.py` — `render_clip` (trim+concat).
- `src/video/jetcut.py` — `build_edl` (→ keep_ranges + kept_words).
- `src/video/qaseg.py` — `segment_qa`, `build_qa_tree`, `extract_answerers`.
- `src/video/qa_annotate.py` — LLM summary + importance + topic labels.
- `src/models.py` — `ClipProject`, `QATree/QATopic/QATurn/QASentence`,
  `EDL/KeepRange/KeptWord`, `EditRange/EditCaption`.
- `src/app.py` — `GET /api/sessions`, `/sessions/{id}/members`, `POST /api/clips`,
  `GET .../project`, `POST .../edit`, `POST .../export?mode=topics|full|both`,
  `GET /api/clips/file/...`.
- `frontend/src/{App,Editor,QATreeEditor}.tsx`, `api.ts`, `App.css`.

## Conventions
- Commit/push only when asked; avoid direct push to `main`, branch if needed.
- Never print API keys. `media/`, `*.mp4`, `.venv`, `node_modules`, `.env` are gitignored.
- External access via Tailscale `http://<tailscale-ip>:5173` (Vite `host:true` +
  `allowedHosts`). ngrok configured but unneeded with Tailscale.
