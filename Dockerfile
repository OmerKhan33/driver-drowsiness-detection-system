# ──────────────────────────────────────────────────────────────────────────────
# Driver Drowsiness Detection System — Docker Image
#
# Multi-stage build for a lightweight production container.
# Uses PyTorch CPU wheels to keep the image small.
#
# Build:  docker build -t drowsiness-detector .
# Run:    docker run -p 8501:8501 drowsiness-detector
# ──────────────────────────────────────────────────────────────────────────────

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system deps for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU first (smaller), then rest of requirements
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt


# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="OmerKhan33"
LABEL description="Driver Drowsiness Detection System"
LABEL version="1.0.0"

WORKDIR /app

# Install runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files
COPY src/ ./src/
COPY app/ ./app/
COPY data/scripts/ ./data/scripts/
COPY models/ ./models/
COPY requirements.txt .
COPY README.md .

# Create necessary directories
RUN mkdir -p models/weights models/results data/processed

# Set Python path so 'from src.xxx' imports work
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Default command: run Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
