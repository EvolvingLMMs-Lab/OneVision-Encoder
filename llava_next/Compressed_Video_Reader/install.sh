#!/bin/bash

SCRIPT_DIR=$(realpath $(dirname -- "$0"))
cd "${SCRIPT_DIR}" || exit 1

bash ffmpeg/install_ffmpeg.sh

pip3 install . || exit 1

# test if installation is successful
cv_reader -h || (echo "Installation failed!" && exit 1)
