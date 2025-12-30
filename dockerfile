FROM nvcr.io/nvidia/pytorch:25.04-py3

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1


RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Install system dependencies and ffmpeg in one layer
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o ffmpeg.tar.xz \
    && tar -xf ffmpeg.tar.xz \
    && cd ffmpeg-*-static \
    && install -m 0755 ffmpeg /usr/local/bin/ffmpeg \
    && install -m 0755 ffprobe /usr/local/bin/ffprobe \
    && cd / \
    && rm -rf /tmp/ffmpeg* \
    && ffprobe -version

# Copy requirements file first (for better caching)
COPY requirements.txt /tmp/requirements.txt

# Install Python packages in optimized order
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda110 && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Set default command
CMD ["/bin/bash"]
