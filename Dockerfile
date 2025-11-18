# SwimVision Pro - Multi-stage Docker Build
# Supports both CPU and GPU environments

# ========================================
# Base Stage - Common dependencies
# ========================================
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# ========================================
# Dependencies Stage
# ========================================
FROM base AS dependencies

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ========================================
# Development Stage
# ========================================
FROM dependencies AS development

# Install development dependencies
RUN pip install -r requirements-dev.txt

# Copy project files
COPY . .

# Install package in editable mode
RUN pip install -e .

# Expose Streamlit port
EXPOSE 8501

# Set default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ========================================
# Production Stage
# ========================================
FROM dependencies AS production

# Copy only necessary files
COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml ./
COPY README.md ./

# Install package
RUN pip install .

# Create non-root user
RUN useradd -m -u 1000 swimvision && \
    chown -R swimvision:swimvision /app

USER swimvision

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Set default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ========================================
# GPU Stage (CUDA support)
# ========================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS gpu-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy project
COPY . .
RUN pip install .

# Create non-root user
RUN useradd -m -u 1000 swimvision && \
    chown -R swimvision:swimvision /app

USER swimvision

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
