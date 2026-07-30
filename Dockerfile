# Headless face-detection REST API service.
FROM python:3.12-slim

# libGL/libglib are pulled in by some OpenCV builds even in headless mode.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Pre-download YOLO nano face weights (~6 MB) at build time so the container
# starts without requiring any network I/O at inference time.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-face.pt')"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
