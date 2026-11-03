# syntax=docker/dockerfile:1
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system meteorological and geospatial dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    cdo \
    libeccodes-dev \
    libeccodes-tools \
    python3-eccodes \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libhdf5-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Copy dependency definition first for caching
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Create venv and install package in editable mode
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN uv pip install -e ".[dev]"

CMD ["/bin/bash"]
