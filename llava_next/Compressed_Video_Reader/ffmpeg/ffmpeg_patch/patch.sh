# before running the script, set FFMPEG_PATCH_DIR and FFMPEG_INSTALL_DIR

if [[ -z "${FFMPEG_INSTALL_DIR}" ]] || [[ -z "${FFMPEG_PATCH_DIR}" ]]; then
    echo "Please set the environment variables FFMPEG_INSTALL_DIR and FFMPEG_PATCH_DIR to an appropiate value"
    exit 1
else
    yes | cp -rf "$FFMPEG_PATCH_DIR"/h264idct_template.c "$FFMPEG_INSTALL_DIR"/libavcodec
    yes | cp -rf "$FFMPEG_PATCH_DIR"/yuv2rgb.c "$FFMPEG_INSTALL_DIR"/libswscale
    yes | cp -rf "$FFMPEG_PATCH_DIR"/hevcdsp_template.c "$FFMPEG_INSTALL_DIR"/libavcodec
    yes | cp -rf "$FFMPEG_PATCH_DIR"/hevcdec.c "$FFMPEG_INSTALL_DIR"/libavcodec
    yes | cp -rf "$FFMPEG_PATCH_DIR"/hevcdec.h "$FFMPEG_INSTALL_DIR"/libavcodec
    yes | cp -rf "$FFMPEG_PATCH_DIR"/hevc_refs.c "$FFMPEG_INSTALL_DIR"/libavcodec
    # yes | cp -rf "$FFMPEG_PATCH_DIR"/h264_mb_template.c "$FFMPEG_INSTALL_DIR"/libavcodec
fi