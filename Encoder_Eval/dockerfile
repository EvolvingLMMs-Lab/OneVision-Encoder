FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

# Set up environment variables (optional, but often recommended)
ENV DEBIAN_FRONTEND=noninteractive

# Install Python packages
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda110 \
    && pip install --no-cache-dir decord==0.6.0 \
    && pip install --no-cache-dir timm \
    && pip install --no-cache-dir transformers==4.53.1

# (Optional) Set working directory
WORKDIR /workspace

# (Optional) Copy your code into the container
# COPY . /workspace

# (Optional) Set entrypoint or CMD
# CMD ["python", "your_script.py"]