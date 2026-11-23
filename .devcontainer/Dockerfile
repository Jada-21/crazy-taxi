# Official Ultralytics image for AMD64/x86 (CPU version; smaller & simpler)
FROM ultralytics/ultralytics:latest-cpu

ENV DEBIAN_FRONTEND=noninteractive

# Add only the extra packages not already included
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    imagemagick \
    libgl1 \
    libglx-mesa0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

run pip install --no-cache-dir --upgrade pip

CMD ["bash"]