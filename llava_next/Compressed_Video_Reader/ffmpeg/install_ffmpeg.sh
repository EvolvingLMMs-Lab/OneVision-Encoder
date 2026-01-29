#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname -- "$0")")
cd "${SCRIPT_DIR}" || exit

# Install FFMPEG dependencies
echo "Installing FFMPEG dependencies"
apt update -qq --fix-missing && \
apt upgrade -y && \
apt -y install \
    libass-dev \
    libfreetype6-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    texinfo \
    zlib1g-dev \
    nasm \
    yasm \
    libx264-dev \
    libx265-dev \
    libnuma-dev \
    libvpx-dev \
    libmp3lame-dev \
    libopus-dev

# Download FFMPEG source
FFMPEG_VERSION="5.1"
echo "Downloading FFMPEG source"
mkdir -p "$SCRIPT_DIR"/ffmpeg_source
wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-"$FFMPEG_VERSION".tar.bz2
tar xjvf ffmpeg-snapshot.tar.bz2 -C "$SCRIPT_DIR"/ffmpeg_source --strip-components=1
rm -rf "$SCRIPT_DIR"/ffmpeg-snapshot.tar.bz2

# Install patch for FFMPEG
echo "Patching FFMPEG"
export FFMPEG_INSTALL_DIR="$SCRIPT_DIR"/ffmpeg_source
export FFMPEG_PATCH_DIR="$SCRIPT_DIR"/ffmpeg_patch

chmod +x "$FFMPEG_PATCH_DIR"/patch.sh
"$FFMPEG_PATCH_DIR"/patch.sh || exit 1

echo "Configuring FFMPEG"
cd "${SCRIPT_DIR}"/ffmpeg_source || exit 1
chmod +x ./configure || exit 1
./configure \
  --prefix="${SCRIPT_DIR}"/ffmpeg_install \
  --pkg-config-flags="--static" \
  --extra-cflags="-I${SCRIPT_DIR}/ffmpeg_install/include -static" \
  --extra-ldflags="-L${SCRIPT_DIR}/ffmpeg_install/lib -static" \
  --extra-libs="-lpthread -lm" \
  --bindir="${SCRIPT_DIR}/ffmpeg_install/bin" \
  --enable-pic \
  --disable-doc \
  --disable-yasm \
  --disable-programs

echo "Compiling FFMPEG"
make -j || exit 1
make install

 