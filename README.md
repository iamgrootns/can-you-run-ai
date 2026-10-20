# Can You Run AI

Detect this PC → browse Hugging Face → see what **fits** → download → **run it here** if we have an adapter.

No hardcoded model list. Hub API when online; cache + downloads work offline.

## Jobs that actually run in-app

| Job | What | Needs | Runner |
|-----|------|--------|--------|
| **Talk** | Chat (optional JSON schema in Advanced) | GGUF LLM | native (`node-llama-cpp`) |
| **See** | Text-to-image, img2img, inpaint | Single-file SD ggml/safetensors | native (sd.cpp, Vulkan → CPU fallback) |
| **See** | Text-to-image (multi-file diffusers pipelines — most of the Hub) | Any diffusers repo (unet/vae/text_encoder folders) | Python (auto-installed, see below) |
| **Hear** | Transcribe / Speak | whisper ggml / piper onnx | native |
| **See it move** | Text-to-video | Any diffusers video pipeline | Python |
| **Hear it play** | Text-to-music | MusicGen/AudioGen-style transformers checkpoint | Python |
| **Embed** | Vectors + cosine | GGUF embed model (not a chat GGUF) | native |
| **Rank** | Query vs documents | GGUF reranker | native |

Everything else on the Hub: **download only** (honest). No fake Generate.

### Zero-setup Python, for the jobs that genuinely need it

Text-to-image's multi-file case, video, and music have no native/ggml runtime
anywhere — the entire ecosystem for those (diffusers, transformers) is
Python. Rather than skip them, the app manages Python itself: it downloads a
small, pinned, sha256-verified `uv` binary (astral-sh/uv) per platform, and
`uv` does the rest — including downloading the exact Python build it needs if
none exists on the system. The user never installs Python, opens a terminal,
or runs pip. Each model family gets its own isolated virtual environment
(under `runtimes/py/<family>/`) so an image-gen stack's torch/diffusers pins
can never conflict with a video-gen stack's. See `electron/engine/pyhost.js`
and `electron/adapters/py{diffusion,video,music}/`.

This was built and live-verified (real `uv install → venv → pip install
torch+diffusers → download a model → generate → write a file` run, not just
unit tests) during development — see that work's own notes in
`electron/adapters/pydiffusion/index.js`. Video generation follows the exact
same code path but has not itself been live-tested end-to-end (real
text-to-video checkpoints are multi-GB with no small CI-style test fixture
available) — flagged there rather than implied.

## Platforms

Windows, Linux (x64/arm64), and macOS (Apple Silicon) are all real targets,
not just Windows with others aspirational. The native binary adapters
(sd.cpp, whisper.cpp, piper) each download a platform-specific, sha256-pinned
build — verified by actually downloading, hashing, extracting, and running
every pinned asset per platform during development, not assumed from release
filenames. One real gap: whisper.cpp has no standalone macOS CLI build
upstream (only an xcframework meant for embedding in Apple apps) — transcribe
is Windows/Linux only until someone builds that from source. Everything else
(chat, image, video, music) is fully cross-platform, since `node-llama-cpp`
and the Python stacks are already platform-agnostic.

## Screens

- **Home** — your specs + one recommended model per job
- **Explore** — Hub search; Run only if `canRunHere === yes`
- **Library** — downloads; Talk models chat here, others jump to Explore

Known gap: the Explore/Library UI (`src/js/app.js`) has Studio panels for
Talk/See/Hear but not yet for the new video/music jobs — the engine layer
(fit-checking, download-free generate via the Python adapters, IPC handlers:
`video:generate`/`music:generate`) is wired and tested; the UI panels to
drive them from Explore are the next piece of work, not yet built.

## Run

```powershell
# installer (Windows)
release\Can You Run AI Setup 1.0.0.exe
# or portable
release\Can You Run AI 1.0.0.exe
# or from source
npm start
```

Selftest: `npm run selftest`
Smoke: `node scripts/smoke.js`

## Data

`%APPDATA%\can-you-run-ai\` (Windows) / `~/.config/can-you-run-ai/` (Linux) /
`~/Library/Application Support/can-you-run-ai/` (macOS) —
`store/` (cache), `models/` (weights; `models/hf-cache/` is where the Python
adapters' own Hugging Face downloads land, kept inside the app's storage
rather than the user's home directory), `runtimes/` (sd/whisper/piper/uv
binaries + the per-family Python venvs).

## Not in v1

Vision/caption, Flux/SD3, ControlNet/upscale (needs a 2nd model file), NER,
Comfy/Ollama. (Python sidecar is no longer excluded — see above — but it's
scoped to image/video/music specifically, not a general plugin system.)
