# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

# Use an accessible Jetson PyTorch base image.
# The original nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3 tag is not available.
FROM dustynv/pytorch:2.7-r36.4.0

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3-serial \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_CONFIG_FILE=/dev/null

# Install project runtime dependencies from public PyPI only.
RUN python3 -m pip install --isolated \
    --index-url https://pypi.org/simple \
    --prefer-binary \
    --timeout 120 \
    --retries 10 \
    "numpy<2" \
    "ultralytics>=8.3" \
    "paho-mqtt>=2.0" \
    "Jetson.GPIO>=2.1" \
    "fastapi>=0.115" \
    "uvicorn>=0.30" \
    "aiofiles>=24.1" \
    "python-multipart>=0.0.20" \
    "Adafruit-PCA9685>=1.0.1" \
    "smbus2>=0.4.3"

RUN python3 -c "import numpy as np, cv2, ultralytics; print(np.__version__, cv2.__version__, ultralytics.__version__)"

# Copy project
COPY pyproject.toml pdm.lock* ./
COPY src/ ./src/
COPY models/ ./models/

ENV MODEL_PATH=/app/models/best_fp16.engine
ENV PYTHONPATH=/app

# Dùng CMD thay vì ENTRYPOINT để có thể override khi test
CMD ["python3", "-m", "src.main"]
