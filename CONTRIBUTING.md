# Contributing to Face-Tracking-System

Contributions are welcome — whether you are improving detector accuracy, adding a new
acceleration backend, fixing a bug, or writing a test. The project aims to stay
production-grade, so every change needs tests and must pass the full suite before review.
CPU-only contributors are fully welcome; CUDA and OpenCL are optional throughout.

---

## Ways to contribute

- **Bug reports** — camera recovery failures, wrong detections, crash on a specific
  OpenCV build
- **New detector backends** — replacing or extending the ResNet-SSD / Haar hybrid
- **Acceleration backends** — e.g. CoreML/ANE on Apple Silicon, DirectML on Windows
- **Performance improvements** — NMS speed, temporal filter tuning, frame-skipping logic
- **Fault-recovery improvements** — circuit-breaker thresholds, retry strategies
- **REST API** — new endpoints, response fields, auth
- **Tests** — increasing the 96% line / 93% branch coverage
- **Documentation** — architecture, deployment, responsible-use guidance

---

## Getting started

Python 3.10, 3.11, or 3.12 is required. **opencv must stay below 5** — opencv 5.x
breaks initialization of the bundled ResNet-SSD and Haar models.

### Path A — CPU-only (CI, development, no camera needed)

```bash
git clone https://github.com/Romil2112/Face-Tracking-System.git
cd Face-Tracking-System
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

`requirements-dev.txt` pulls `requirements-api.txt` (headless OpenCV, FastAPI,
uvicorn) plus imutils, pytest, pytest-cov, and httpx2. No GUI libs are installed,
which is intentional — all tests mock cv2's capture and display calls.

### Path B — Full runtime with GUI (live webcam tracker)

```bash
pip install -r requirements.txt   # adds opencv-python (GUI build) + imutils
python src/main.py --width 1280 --height 720
# press Q to quit
```

The tracker prints which acceleration backend it selected at startup
(`CUDA`, `OpenCL`, or `CPU`).

### Path C — REST API locally

```bash
pip install -r requirements-api.txt
uvicorn src.api:app --port 8000
# POST an image: curl -X POST http://localhost:8000/detect -F "file=@face.jpg"
```

### Path D — Docker (API only)

```bash
docker build -t face-detection-api .
docker run -p 8000:8000 face-detection-api
curl http://localhost:8000/health
```

The Dockerfile uses `python:3.12-slim` with `opencv-python-headless` and installs
`libgl1` + `libglib2.0-0` to satisfy OpenCV's runtime linkage.

---

## Project structure

```
src/
  acceleration.py       — hardware backend selection (CUDA → OpenCL → CPU)
  api.py                — FastAPI service: POST /detect, GET /health
  cli_detect.py         — headless CLI for image / video files
  config.py             — all tuneable constants (thresholds, paths, priorities)
  error_handling.py     — ErrorHandler, @retry decorator, circuit-breaker logic
  face_detector.py      — FaceDetector: DNN + Haar hybrid, NMS merge, fallback
  geometry.py           — rect/center helpers
  main.py               — live webcam tracking loop with pacing and recovery
  models/               — bundled ResNet-SSD weights (.caffemodel + .prototxt)
                          and haarcascade_frontalface_default.xml
  motion_utils.py       — optical-flow motion gate (optional, toggled via config)
  nms_utils.py          — non-maximum suppression
  temporal_filter.py    — 5-frame consistency window; filters one-frame FPs
  tracking_visualizer.py — draws bounding boxes, scores, and status text
  video_capture.py      — VideoCapture wrapper with reconnect and pacing

tests/
  test_acceleration.py, test_api.py, test_cli_detect.py, test_config.py,
  test_error_handling.py, test_error_handling_more.py, test_face_detector.py,
  test_geometry.py, test_main.py, test_motion_utils.py, test_nms_utils.py,
  test_package.py, test_temporal_filter.py, test_tracking_visualizer.py,
  test_video_capture.py

conftest.py             — shared fixtures; mocks cv2.VideoCapture and cv2.imshow
```

---

## How to add a new detector backend

The cleanest contribution path is adding a detector alongside the existing DNN/Haar
hybrid.

1. **Add your detector class** in `src/face_detector.py` (or a new `src/` module).
   It must expose a method with this signature:

   ```python
   def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
       """Return a list of (x, y, w, h) face rects for the given BGR frame."""
   ```

2. **Wire it into `FaceDetector`** — initialize it in `__post_init__` with a
   try/except that falls back gracefully if the backend is unavailable, and add
   it to the NMS merge in `detect_faces`.

3. **Add acceleration support** if applicable — see `acceleration.py`. The
   `select_acceleration` function is a pure function of `(priority, capabilities)`
   so it can be unit-tested without real hardware.

4. **Update `config.py`** for any new thresholds or paths your detector needs.

5. **Write tests** — mock the detector's underlying library calls; do not require
   a camera or real model files. See `test_face_detector.py` for the fixture pattern
   (`scope="module"` for the real detector, `np.zeros` frames for unit cases).

---

## Code style

The project uses **ruff** (not black) with a 100-character line length and these
rule sets: `E`, `W`, `F` (pyflakes), `I` (isort), `N` (pep8-naming), `UP`
(pyupgrade), `B` (bugbear). Config is in `pyproject.toml`.

```bash
pip install ruff
ruff check src/ tests/          # lint
ruff check --fix src/ tests/    # auto-fix safe issues
```

Type annotations are checked with mypy:

```bash
pip install mypy
mypy src/
```

`ignore_missing_imports = true` is set for cv2, numpy, imutils, and FastAPI internals
(they lack complete stubs).

---

## Running tests

```bash
python -m pytest tests/ -v                          # full suite
python -m pytest tests/ --cov=src --cov-report=term-missing  # with coverage
python -m pytest tests/test_face_detector.py -v    # single file
```

All 204 tests are hermetic — no camera, display, or GPU is needed. cv2's GUI and
capture calls are mocked in `conftest.py`. There are no `@pytest.mark.gpu` or
similar markers; CUDA paths are exercised by injecting mock capabilities into
`select_acceleration`.

---

## PR guidelines

- Keep changes focused — one logical change per PR.
- **Tests are required.** A PR that reduces coverage will not be merged.
- For detector or NMS changes, include a brief benchmark note in the PR description
  (e.g. average inference time on your hardware, CPU-only baseline).
- opencv must remain pinned below 5. Do not bump the pin without a documented fix
  for the bundled-model initialization breakage.
- CI runs the full test suite on Python 3.10, 3.11, and 3.12. CUDA/OpenCL are not
  available in CI; GPU paths must be testable via mocked capabilities.
- Run ruff and mypy locally before pushing — CI will catch failures.

---

## Reporting bugs

Open an issue and include:

- **OpenCV version** — `python -c "import cv2; print(cv2.__version__)"`
- **Python version** — `python --version`
- **CUDA version** (if using GPU) — `nvcc --version`
- **Hardware** — GPU model, OS
- **How you installed** — requirements.txt / requirements-api.txt / Docker
- **Minimal reproduction** — the command or code that triggers the bug
- **Full traceback** if there is one

---

## Responsible use

This project processes biometric-related data (images of human faces). Use it only
with images or video you own or are authorized to process. Obtaining any legally
required notice and consent is the operator's responsibility. See the README's
Privacy & Responsible Use section and [SECURITY.md](SECURITY.md) for details.
