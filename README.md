# Real-time face tracking

A face detector that runs on a webcam, image files, or as an HTTP service, and keeps working when a camera or GPU backend fails. I built it with Parshav.

## Detection

It runs two detectors on each frame. A ResNet-SSD network is accurate on frontal faces but misses some at sharp angles; a Haar cascade catches a different set. Running both and merging the boxes with non-maximum suppression (NMS) finds more faces than either one alone, and NMS throws out the duplicate overlapping boxes so a single face isn't reported three times.

## Fallback chain

It tries CUDA first for the DNN. If there's no NVIDIA GPU or the CUDA backend fails to initialize, it drops to OpenCL through OpenCV's T-API, which runs on Intel, AMD, or Apple GPUs. If that isn't available either, it runs on CPU. CPU is always there, so detection never stops for lack of a GPU.

## Temporal filtering

A single frame can throw a false detection, a flicker that isn't a face. The temporal filter holds a candidate across a 5-frame window and only reports it once it shows up consistently, which clears out most one-frame false positives before they reach the output.

## Why opencv is pinned below 5

`requirements` pins opencv below version 5. OpenCV 5.x breaks initialization of the bundled ResNet-SSD and Haar detectors, so on 5.x they fail to load and detection never starts. Pinning `<5` keeps the bundled detectors working until they're ported.

## A bug the tests found

Writing coverage for the camera-failure path found a real bug. The recovery code called `ErrorHandler.handle_camera_error` as if it were a standalone function, with no `ErrorHandler` instance bound to it. So on any camera failure it raised an `AttributeError` instead of resetting the camera and continuing, which is the opposite of what recovery code is supposed to do. The fix gave the capture loop its own `ErrorHandler` instance. The test that caught it forces a camera failure and checks that recovery actually runs.

## API

Run it as a headless service and it exposes two endpoints. `GET /health` returns `{"status": "ok"}`. `POST /detect` takes a multipart image upload and returns the faces it found:

```bash
curl -X POST http://localhost:8000/detect -F "file=@face.jpg"
# {"count": 1, "faces": [{"rect": [120, 80, 90, 90], "center": [165, 125], "confidence": 0.99}]}
```

## Tests

204 pytest tests at 96% line and 93% branch coverage, run on Python 3.10, 3.11, and 3.12 through GitHub Actions.

## Running it

```bash
docker build -t face-detection-api .
docker run -p 8000:8000 face-detection-api
```

Or locally: `pip install -r requirements-api.txt`, then `uvicorn src.api:app --port 8000` for the service, or `python src/main.py` for the webcam tracker.
