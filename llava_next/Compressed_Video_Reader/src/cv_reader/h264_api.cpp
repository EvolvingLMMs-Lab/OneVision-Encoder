#include <ctime>
#include <string>
#include <queue>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <cstdlib>

extern "C" {
#include <libswscale/swscale.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavutil/motion_vector.h>
#include <libavutil/rational.h>
}

#include <Python.h>
#include "numpy/arrayobject.h"

// ---------------- Frame selector (optional) ----------------
// Supports frame-id selection and duplicated frame ids (e.g. uniform sampling with rounding).
struct FrameSelector {
    std::vector<int> uniq;                 // sorted unique target frame indices
    std::unordered_map<int, int> cnt;      // frame_idx -> how many times to emit

    bool enabled() const { return !uniq.empty(); }

    bool want(int idx) const {
        if (!enabled()) return true;
        return cnt.find(idx) != cnt.end();
    }

    int times(int idx) const {
        if (!enabled()) return 1;
        auto it = cnt.find(idx);
        return (it == cnt.end()) ? 0 : it->second;
    }

    bool done(int idx) const {
        if (!enabled()) return false;
        return idx >= uniq.back();
    }
};

static bool parse_frame_ids(PyObject *obj, FrameSelector &sel) {
    if (!obj || obj == Py_None) return true;

    PyObject *seq = PySequence_Fast(obj, "frame_ids must be a sequence");
    if (!seq) return false;

    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    sel.uniq.reserve((size_t) n);

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        long v = PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return false;
        }
        if (v < 0) continue;
        sel.cnt[(int) v] += 1;
        sel.uniq.push_back((int) v);
    }

    Py_DECREF(seq);

    if (sel.uniq.empty()) return true;
    std::sort(sel.uniq.begin(), sel.uniq.end());
    sel.uniq.erase(std::unique(sel.uniq.begin(), sel.uniq.end()), sel.uniq.end());
    return true;
}
// ---------------- Callback-based (streaming) decode ----------------
// Calls a Python callback for each selected frame; does NOT accumulate results in a list.
// callback(frame_dict) -> True/None to continue, False to stop early.
static int
decode_h264_packet_cb(AVPacket *pkt,
                      AVCodecContext *dec_ctx,
                      AVFrame *frame,
                      int *decoded_count,
                      PyObject *py_callback,
                      bool with_residual,
                      int max_frames,
                      int decode_len,
                      FrameSelector *selector,
                      struct SwsContext **psws_ctx,
                      AVRational st_time_base,
                      AVRational fr,
                      bool debug,
                      int *emit_counter) {
    int ret = avcodec_send_packet(dec_ctx, pkt);
    if (ret < 0) {
        fprintf(stderr, "Error while sending a packet to the decoder: \n");
        return ret;
    }

    while (ret >= 0) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            fprintf(stderr, "Error while receiving a frame from the decoder: \n");
            return ret;
        }

        // decoded_idx: number of frames received from decoder since current start (or since seek)
        int decoded_idx = *decoded_count;
        (*decoded_count)++;

        // Optional hard stop by number of decoded frames (useful after seek)
        if (decode_len > 0 && decoded_idx >= decode_len) {
            av_frame_unref(frame);
            return AVERROR_EOF;
        }

        // Backward-compatible hard stop: number of decoded frames from current start
        if (max_frames > 0 && decoded_idx >= max_frames) {
            av_frame_unref(frame);
            return AVERROR_EOF;
        }

        // frame_idx: try to recover the *global* frame index from PTS (works well for CFR videos)
        int frame_idx = decoded_idx;
        int64_t best_ts = frame->best_effort_timestamp;
        if (best_ts != AV_NOPTS_VALUE && fr.num > 0 && fr.den > 0) {
            // Convert timestamp (stream time_base) -> frame index (units of 1 frame)
            frame_idx = (int)av_rescale_q(best_ts, st_time_base, av_inv_q(fr));
        }

        // Skip frames we don't want (no numpy allocations, no callback)
        if (selector && selector->enabled() && !selector->want(frame_idx)) {
            av_frame_unref(frame);
            continue;
        }

        // Build result dict for Python
        PyObject *result = PyDict_New();

        std::string slice_type_str = "?";
        if (frame->pict_type == AV_PICTURE_TYPE_I) slice_type_str = "I";
        else if (frame->pict_type == AV_PICTURE_TYPE_P) slice_type_str = "P";
        else if (frame->pict_type == AV_PICTURE_TYPE_B) slice_type_str = "B";

        const int height = frame->height;
        const int width = frame->width;

        // codec_name
        const char *codec_name = dec_ctx->codec ? dec_ctx->codec->name : "unknown";
        PyObject *py_codec_name = PyUnicode_FromString(codec_name);
        PyDict_SetItemString(result, "codec_name", py_codec_name);
        Py_DECREF(py_codec_name);

        PyObject *py_width = PyLong_FromLong(width);
        PyDict_SetItemString(result, "width", py_width);
        Py_DECREF(py_width);

        PyObject *py_height = PyLong_FromLong(height);
        PyDict_SetItemString(result, "height", py_height);
        Py_DECREF(py_height);

        PyObject *py_frame_idx = PyLong_FromLong(frame_idx);
        PyDict_SetItemString(result, "frame_idx", py_frame_idx);
        Py_DECREF(py_frame_idx);

        PyObject *py_pict_type = PyUnicode_FromString(slice_type_str.c_str());
        PyDict_SetItemString(result, "pict_type", py_pict_type);
        Py_DECREF(py_pict_type);

        // MV array: (H/4, W/4, 4)
        npy_intp mv_dims[3];
        mv_dims[0] = (height / 4);
        mv_dims[1] = (width / 4);
        mv_dims[2] = 4;
        PyObject *mv_obj = PyArray_ZEROS(3, mv_dims, NPY_INT32, 0);
        PyArrayObject *mv_arr = reinterpret_cast<PyArrayObject *>(mv_obj);

        // Residual: keep existing RGB conversion as "residual" for backward compatibility,
        // and also provide raw Y plane as "residual_y" (useful for 128+residual Y output).
        PyArrayObject *residual_arr = nullptr;
        if (with_residual) {
            // residual_y: (H, W) uint8 copied from frame->data[0]
            npy_intp y_dims[2];
            y_dims[0] = height;
            y_dims[1] = width;
            PyObject *res_y_obj = PyArray_ZEROS(2, y_dims, NPY_UINT8, 0);
            uint8_t *dst_y = reinterpret_cast<uint8_t *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(res_y_obj)));
            for (int r = 0; r < height; ++r) {
                std::memcpy(dst_y + r * width, frame->data[0] + r * frame->linesize[0], width);
            }
            PyDict_SetItemString(result, "residual_y", res_y_obj);
            Py_DECREF(res_y_obj);

            // residual (RGB24): (H, W, 3)
            npy_intp residual_dims[3];
            residual_dims[0] = height;
            residual_dims[1] = width;
            residual_dims[2] = 3;
            residual_arr = (PyArrayObject *) PyArray_ZEROS(3, residual_dims, NPY_UINT8, 0);

            if (*psws_ctx == nullptr) {
                *psws_ctx = sws_getContext(dec_ctx->width, dec_ctx->height, AV_PIX_FMT_YUV420P,
                                           dec_ctx->width, dec_ctx->height, AV_PIX_FMT_RGB24,
                                           SWS_BILINEAR, nullptr, nullptr, nullptr);
            }

            uint8_t *res_data0 = reinterpret_cast<uint8_t *>(PyArray_DATA(residual_arr));
            uint8_t *res_data[1] = {res_data0};
            int dst_stride[1] = {width * 3};
            sws_scale(*psws_ctx, frame->data, frame->linesize, 0, dec_ctx->height, res_data, dst_stride);
        }

        // Fill MV from side-data
        AVFrameSideData *sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
        if (!sd && dec_ctx->codec_id == AV_CODEC_ID_HEVC) {
            fprintf(stderr, "[cv_reader] HEVC frame %d: no AV_FRAME_DATA_MOTION_VECTORS side-data\n", frame_idx);
        }
        if (sd) {
            const auto *mvs = (const AVMotionVector *) sd->data;
            int nb_mvs = sd->size / sizeof(*mvs);

            if (dec_ctx->codec_id == AV_CODEC_ID_HEVC) {
                for (int i = 0; i < nb_mvs; i++) {
                    const AVMotionVector *mv = &mvs[i];

                    int mvx = mv->src_x - mv->dst_x;
                    int mvy = mv->src_y - mv->dst_y;

                    int p_dst_x = mv->dst_x / 4;
                    int p_dst_y = mv->dst_y / 4;

                    if (mv->dst_x % 4 != 0 || mv->dst_y % 4 != 0) {
                        fprintf(stderr, "[cv_reader] HEVC dst_x/dst_y not divisible by 4: %d %d\n",
                                mv->dst_x, mv->dst_y);
                    }

                    if (p_dst_y >= 0 && p_dst_y < height / 4 && p_dst_x >= 0 && p_dst_x < width / 4) {
                        // Dual-direction HEVC export:
                        //   L0: source in [0..15]
                        //   L1: source in [16..31] (ref_idx + 16)
                        //   missing: source == -1
                        if (mv->source < 0) {
                            continue;
                        } else if (mv->source >= 16) {
                            *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 2)) = mvx; // L1_x
                            *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 3)) = mvy; // L1_y
                        } else {
                            *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 0)) = mvx; // L0_x
                            *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 1)) = mvy; // L0_y
                        }
                    }
                }
            } else {
                for (int i = 0; i < nb_mvs; i++) {
                    const AVMotionVector *mv = &mvs[i];
                    for (int x_start = (-1 * mv->w / 8); x_start < mv->w / 8; ++x_start) {
                        for (int y_start = (-1 * mv->h / 8); y_start < mv->h / 8; ++y_start) {
                            int mvx = mv->src_x - mv->dst_x;
                            int mvy = mv->src_y - mv->dst_y;

                            int p_dst_x = mv->dst_x / 4 + x_start;
                            int p_dst_y = mv->dst_y / 4 + y_start;

                            if (p_dst_y >= 0 && p_dst_y < height / 4 && p_dst_x >= 0 && p_dst_x < width / 4) {
                                bool is_L0 = (mv->source == 0 || mv->source < 0);
                                if (is_L0) {
                                    *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 0)) = mvx;
                                    *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 1)) = mvy;
                                } else {
                                    *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 2)) = mvx;
                                    *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 3)) = mvy;
                                }
                            }
                        }
                    }
                }
            }
        }

        PyDict_SetItemString(result, "motion_vector", mv_obj);
        Py_DECREF(mv_obj);

        if (with_residual && residual_arr) {
            PyDict_SetItemString(result, "residual", (PyObject *) residual_arr);
            Py_DECREF(residual_arr);
        }

        // Call Python callback (support duplicated frame ids)
        int times = selector ? selector->times(frame_idx) : 1;
        if (debug && emit_counter && *emit_counter < 5) {
            fprintf(stderr,
                    "[cv_reader] emit frame_idx=%d decoded_idx=%d best_ts=%lld\n",
                    frame_idx, decoded_idx, (long long)best_ts);
            (*emit_counter)++;
        }
        for (int k = 0; k < times; ++k) {
            PyObject *cb_ret = PyObject_CallFunctionObjArgs(py_callback, result, nullptr);
            if (!cb_ret) {
                Py_DECREF(result);
                av_frame_unref(frame);
                return AVERROR_EXTERNAL; // Python exception set
            }

            bool stop = false;
            if (cb_ret != Py_None) {
                // stop if callback returns False
                stop = (PyObject_IsTrue(cb_ret) == 0);
            }
            Py_DECREF(cb_ret);

            if (stop) {
                Py_DECREF(result);
                av_frame_unref(frame);
                return AVERROR_EOF;
            }
        }

        Py_DECREF(result);
        av_frame_unref(frame);

        // Early exit once the last requested frame is processed
        if (selector && selector->enabled() && selector->done(frame_idx)) {
            return AVERROR_EOF;
        }
    }

    return 0;
}


static int
decode_h264_packet(AVPacket *pkt, AVCodecContext *dec_ctx, AVFrame *frame, int *frame_count, PyObject *results,
                   bool with_residual, int max_frames) {
    struct SwsContext *sws_ctx = nullptr;

    int ret = avcodec_send_packet(dec_ctx, pkt);
    if (ret < 0) {
        fprintf(stderr, "Error while sending a packet to the decoder: \n");
        return ret;
    }

    while (ret >= 0) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            printf("Error while receiving a frame from the decoder: \n");
            fprintf(stderr, "Error while receiving a frame from the decoder: \n");
            return ret;
        }

        if (ret >= 0) {
            PyObject *result = PyDict_New();
            std::string slice_type_str = "?";
            if (frame->pict_type == AV_PICTURE_TYPE_I) {
                slice_type_str = "I";
            } else if (frame->pict_type == AV_PICTURE_TYPE_P) {
                slice_type_str = "P";
            } else if (frame->pict_type == AV_PICTURE_TYPE_B) {
                slice_type_str = "B";
            }

            const int height = frame->height;
            const int width = frame->width;
            // Add codec_name to result dict
            const char *codec_name = dec_ctx->codec ? dec_ctx->codec->name : "unknown";
            PyObject *py_codec_name = PyUnicode_FromString(codec_name);
            PyDict_SetItemString(result, "codec_name", py_codec_name);
            Py_DECREF(py_codec_name);

            // Channels: L0_x, L0_y, L1_x, L1_y  (refs/pred mode not stored here)
            npy_intp mv_dims[3];
            mv_dims[0] = (height / 4);
            mv_dims[1] = (width / 4);
            mv_dims[2] = 4;
            // PyObject holder for returning to Python, and PyArrayObject* for C-API macros
            PyObject *mv_obj = PyArray_ZEROS(3, mv_dims, NPY_INT32, 0);
            PyArrayObject *mv_arr = reinterpret_cast<PyArrayObject *>(mv_obj);
            npy_intp residual_dims[3];
            residual_dims[0] = height;
            residual_dims[1] = width;
            residual_dims[2] = 3;
            PyArrayObject *residual_arr = nullptr;
            if (with_residual) {
                residual_arr = (PyArrayObject *) PyArray_ZEROS(3, residual_dims, NPY_UINT8, 0);
                sws_ctx = sws_getContext(dec_ctx->width, dec_ctx->height, AV_PIX_FMT_YUV420P,
                                         dec_ctx->width, dec_ctx->height, AV_PIX_FMT_RGB24,
                                         SWS_BILINEAR, nullptr, nullptr, nullptr);
                uint8_t *res_data0 = reinterpret_cast<uint8_t *>(PyArray_DATA(residual_arr));
                uint8_t *res_data[1] = {res_data0};
                int dst_stride[1] = {width * 3};

                sws_scale(sws_ctx, frame->data, frame->linesize, 0, dec_ctx->height, res_data, dst_stride);
            }

            int i;
            AVFrameSideData *sd;

            (*frame_count)++;
            // If a frame limit is set (max_frames > 0), stop decoding once we reach it.
            if (max_frames > 0 && *frame_count > max_frames) {
                av_frame_unref(frame);
                // Signal end-of-stream to the outer loop so it stops reading packets.
                return AVERROR_EOF;
            }

            sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
            if (!sd && dec_ctx->codec_id == AV_CODEC_ID_HEVC) {
                fprintf(stderr, "[cv_reader] HEVC frame %d: no AV_FRAME_DATA_MOTION_VECTORS side-data\n",
                        *frame_count - 1);
            }
            if (sd) {
                const auto *mvs = (const AVMotionVector *) sd->data;
                int mvx, mvy;
                int nb_mvs = sd->size / sizeof(*mvs);

                if (dec_ctx->codec_id == AV_CODEC_ID_HEVC) {
                    // HEVC path: our FFmpeg patch already exports one AVMotionVector per min PU (usually 4x4),
                    // with dst_x/dst_y aligned to the 4-pixel grid. We can map directly to the H/4 x W/4 grid.
                    for (i = 0; i < nb_mvs; i++) {
                        const AVMotionVector *mv = &mvs[i];

                        // We defined src_x/src_y in the HEVC patch as: dst + (motion >> 2),
                        // so src_x - dst_x gives integer-pixel displacement.
                        mvx = mv->src_x - mv->dst_x;
                        mvy = mv->src_y - mv->dst_y;

                        int p_dst_x = mv->dst_x / 4;
                        int p_dst_y = mv->dst_y / 4;

                        if (mv->dst_x % 4 != 0 || mv->dst_y % 4 != 0) {
                            fprintf(stderr, "[cv_reader] HEVC dst_x/dst_y not divisible by 4: %d %d\n",
                                    mv->dst_x, mv->dst_y);
                        }

                        if (p_dst_y >= 0 && p_dst_y < height / 4 &&
                            p_dst_x >= 0 && p_dst_x < width / 4) {
                            // Dual-direction HEVC export:
                            //   L0: source in [0..15]
                            //   L1: source in [16..31] (ref_idx + 16)
                            //   missing: source == -1
                            if (mv->source < 0) {
                                continue;
                            } else if (mv->source >= 16) {
                                *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 2)) = mvx; // L1_x
                                *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 3)) = mvy; // L1_y
                            } else {
                                *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 0)) = mvx; // L0_x
                                *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 1)) = mvy; // L0_y
                            }
                        }
                    }
                } else {
                    // Original H.264/MPEG4 path: smear each motion vector over its covered block,
                    // filling a 4x4 grid inside the block using w/h.
                    int p_dst_x, p_dst_y;
                    for (i = 0; i < nb_mvs; i++) {
                        const AVMotionVector *mv = &mvs[i];
                        for (int x_start = (-1 * mv->w / 8); x_start < mv->w / 8; ++x_start) {
                            for (int y_start = (-1 * mv->h / 8); y_start < mv->h / 8; ++y_start) {
                                mvx = mv->src_x - mv->dst_x;
                                mvy = mv->src_y - mv->dst_y;

                                p_dst_x = mv->dst_x / 4 + x_start;
                                p_dst_y = mv->dst_y / 4 + y_start;
                                if (mv->dst_x % 4 != 0 || mv->dst_y % 4 != 0) {
                                    fprintf(stderr, "dst_x or dst_y is not divisible by 4: %d %d",
                                            mv->dst_x, mv->dst_y);
                                }

                                if (p_dst_y >= 0 && p_dst_y < height / 4 &&
                                    p_dst_x >= 0 && p_dst_x < width / 4) {
                                    // For H.264, some implementations use source < 0 to indicate L0;
                                    // for our HEVC patch, source == 0 is L0 and source == 1 is L1.
                                    bool is_L0 = (mv->source == 0 || mv->source < 0);
                                    if (is_L0) {
                                        *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 0)) = mvx; // L0_x
                                        *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 1)) = mvy; // L0_y
                                    } else {
                                        *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 2)) = mvx; // L1_x
                                        *reinterpret_cast<int32_t *>(PyArray_GETPTR3(mv_arr, p_dst_y, p_dst_x, 3)) = mvy; // L1_y
                                    }
                                }
                            }
                        }
                    }
                }
            }

            PyObject *py_width = PyLong_FromLong(width);
            PyDict_SetItemString(result, "width", py_width);
            Py_DECREF(py_width);

            PyObject *py_height = PyLong_FromLong(height);
            PyDict_SetItemString(result, "height", py_height);
            Py_DECREF(py_height);


            PyObject *py_frame_idx = PyLong_FromLong(*frame_count - 1);
            PyDict_SetItemString(result, "frame_idx", py_frame_idx);
            Py_DECREF(py_frame_idx);

            PyObject *py_pict_type = PyUnicode_FromString(slice_type_str.c_str());
            PyDict_SetItemString(result, "pict_type", py_pict_type);
            Py_DECREF(py_pict_type);


            PyDict_SetItemString(result, "motion_vector", mv_obj);
            Py_DECREF(mv_obj);

            if (with_residual) {
                PyDict_SetItemString(result, "residual", (PyObject *) residual_arr);
                Py_DECREF(residual_arr);
            }

            PyList_Append(results, result);
            Py_DECREF(result);

            av_frame_unref(frame);
        }
    }

    if (sws_ctx)
        sws_freeContext(sws_ctx);

    return 0;
}

static PyObject *decode_h264(AVFormatContext *fmt_ctx, bool with_residual, int max_frames) {
    int ret;
    AVStream *st;
    AVCodecContext *dec_ctx = nullptr;
    const AVCodec *dec = nullptr;
    AVDictionary *opts = nullptr;
    AVPacket *pkt = nullptr;
    AVFrame *frame = nullptr;
    int frame_count = 0;
    PyObject *results = PyList_New(0);

    bool reached_limit = false;

    int stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &dec, 0);
    st = fmt_ctx->streams[stream_idx];

    dec_ctx = avcodec_alloc_context3(dec);
    if (!dec_ctx) {
        fprintf(stderr, "Failed to allocate codec\n");
        goto end;
    }

    ret = avcodec_parameters_to_context(dec_ctx, st->codecpar);
    if (ret < 0) {
        fprintf(stderr, "Failed to copy codec parameters to codec context\n");
        goto end;
    }

    /* Init the video decoder */
    av_dict_set(&opts, "flags2", "+export_mvs", 0);
    // Disable in-loop filtering (H.264: deblock; HEVC: deblock/SAO) to keep residual buffer unmodified.
    // av_dict_set(&opts, "skip_loop_filter", "all", 0);

    ret = avcodec_open2(dec_ctx, dec, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        fprintf(stderr, "Failed to open video codec\n");
        goto end;
    }
    // Extra safety: make sure decoder context also skips loop filter.
    // dec_ctx->skip_loop_filter = AVDISCARD_ALL;
    // Ensure export_mvs is enabled on the codec context (for H.264 / HEVC)
    dec_ctx->flags2 |= AV_CODEC_FLAG2_EXPORT_MVS;


    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate frame\n");
        goto end;
    }

    pkt = av_packet_alloc();
    if (!pkt) {
        fprintf(stderr, "Could not allocate AVPacket\n");
        goto end;
    }

    /* read frames from the file */
    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == stream_idx) {
            ret = decode_h264_packet(pkt, dec_ctx, frame, &frame_count, results, with_residual, max_frames);
        }
        av_packet_unref(pkt);
        if (ret == AVERROR_EOF) {
            // We reached the requested max_frames; stop reading further packets.
            reached_limit = true;
            break;
        } else if (ret < 0) {
            break;
        }
    }

    /* flush cached frames (only if we did not hit a hard frame limit) */
    if (!reached_limit) {
        decode_h264_packet(nullptr, dec_ctx, frame, &frame_count, results, with_residual, max_frames);
    }

    end:
    avcodec_free_context(&dec_ctx);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    return results;
}

static PyObject *decode_h264_cb(AVFormatContext *fmt_ctx,
                                PyObject *py_callback,
                                bool with_residual,
                                int max_frames,
                                FrameSelector *selector,
                                int seek_to_frame,
                                int decode_len,
                                bool debug) {
    int ret;
    AVStream *st;
    AVCodecContext *dec_ctx = nullptr;
    const AVCodec *dec = nullptr;
    AVDictionary *opts = nullptr;
    AVPacket *pkt = nullptr;
    AVFrame *frame = nullptr;
    int frame_count = 0;
    struct SwsContext *sws_ctx = nullptr; // reused across frames (only used when with_residual=1)
    int emit_counter = 0;

    bool reached_limit = false;

    int stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &dec, 0);
    if (stream_idx < 0) {
        PyErr_SetString(PyExc_SystemError, "Could not find best video stream");
        return nullptr;
    }

    st = fmt_ctx->streams[stream_idx];

    // Estimate FPS for PTS->frame-index conversion and seek
    AVRational fr = av_guess_frame_rate(fmt_ctx, st, nullptr);
    if (fr.num <= 0 || fr.den <= 0) {
        fr = st->avg_frame_rate;
    }
    if (fr.num <= 0 || fr.den <= 0) {
        fr = AVRational{30, 1};
    }

    if (debug) {
        fprintf(stderr, "[cv_reader] fps guess: %d/%d time_base: %d/%d\n",
                fr.num, fr.den, st->time_base.num, st->time_base.den);
        if (selector && selector->enabled() && !selector->uniq.empty()) {
            fprintf(stderr, "[cv_reader] selector: uniq=%zu first=%d last=%d\n",
                    selector->uniq.size(), selector->uniq.front(), selector->uniq.back());
        }
    }

    dec_ctx = avcodec_alloc_context3(dec);
    if (!dec_ctx) {
        PyErr_SetString(PyExc_SystemError, "Failed to allocate codec");
        return nullptr;
    }

    ret = avcodec_parameters_to_context(dec_ctx, st->codecpar);
    if (ret < 0) {
        PyErr_SetString(PyExc_SystemError, "Failed to copy codec parameters to codec context");
        goto end;
    }

    av_dict_set(&opts, "flags2", "+export_mvs", 0);
    // Disable in-loop filtering (H.264: deblock; HEVC: deblock/SAO) to keep residual buffer unmodified.
    // av_dict_set(&opts, "skip_loop_filter", "all", 0);

    ret = avcodec_open2(dec_ctx, dec, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        PyErr_SetString(PyExc_SystemError, "Failed to open video codec");
        goto end;
    }

    // Extra safety: make sure decoder context also skips loop filter.
    // dec_ctx->skip_loop_filter = AVDISCARD_ALL;

    dec_ctx->flags2 |= AV_CODEC_FLAG2_EXPORT_MVS;

    // Optional: seek near a target frame to avoid decoding from the beginning.
    // NOTE: seeking goes to a keyframe at/before the target; we rely on PTS->frame_idx conversion.
    if (seek_to_frame >= 0) {
        int64_t ts = av_rescale_q((int64_t)seek_to_frame, av_inv_q(fr), st->time_base);
        if (ts < 0) ts = 0;
        int sret = av_seek_frame(fmt_ctx, stream_idx, ts, AVSEEK_FLAG_BACKWARD);
        if (debug) {
            fprintf(stderr, "[cv_reader] seek_to_frame=%d -> ts=%lld ret=%d\n",
                    seek_to_frame, (long long)ts, sret);
        }
        // Flush decoder state after seeking
        avcodec_flush_buffers(dec_ctx);
    }

    frame = av_frame_alloc();
    if (!frame) {
        PyErr_SetString(PyExc_SystemError, "Could not allocate frame");
        goto end;
    }

    pkt = av_packet_alloc();
    if (!pkt) {
        PyErr_SetString(PyExc_SystemError, "Could not allocate AVPacket");
        goto end;
    }


    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == stream_idx) {
            ret = decode_h264_packet_cb(pkt, dec_ctx, frame, &frame_count,
                                        py_callback, with_residual, max_frames,
                                        decode_len,
                                        selector, &sws_ctx,
                                        st->time_base, fr,
                                        debug, &emit_counter);
        }
        av_packet_unref(pkt);

        if (ret == AVERROR_EOF) {
            reached_limit = true;
            break;
        } else if (ret < 0) {
            break;
        }
    }

    if (!reached_limit) {
        decode_h264_packet_cb(nullptr, dec_ctx, frame, &frame_count,
                              py_callback, with_residual, max_frames,
                              decode_len,
                              selector, &sws_ctx,
                              st->time_base, fr,
                              debug, &emit_counter);
    }

end:
    if (sws_ctx) sws_freeContext(sws_ctx);
    avcodec_free_context(&dec_ctx);
    av_frame_free(&frame);
    av_packet_free(&pkt);

    if (PyErr_Occurred()) return nullptr;
    Py_RETURN_NONE;
}
static PyObject *read_video_cb(PyObject *self, PyObject *args) {
    char *path = nullptr;
    PyObject *callback = nullptr;
    int without_residual = 0;
    int max_frames = 0;
    PyObject *frame_ids_obj = Py_None;
    int seek_to_frame = -1;
    int decode_len = 0;

    try {
        // read_video_cb(path, callback, without_residual=0, max_frames=0, frame_ids=None, seek_to_frame=-1, decode_len=0)
        if (!PyArg_ParseTuple(args, "sO|iiOii", &path, &callback, &without_residual, &max_frames, &frame_ids_obj, &seek_to_frame, &decode_len)) {
            PyErr_SetObject(PyExc_SystemError, PyBytes_FromString("Argument Error"));
            return nullptr;
        }

        if (!PyCallable_Check(callback)) {
            PyErr_SetString(PyExc_TypeError, "callback must be callable");
            return nullptr;
        }

        bool debug = false;
        const char *dbg = std::getenv("CV_READER_DEBUG");
        if (dbg && std::atoi(dbg) != 0) debug = true;

        FrameSelector selector;
        if (!parse_frame_ids(frame_ids_obj, selector)) {
            // parse_frame_ids sets Python exception
            return nullptr;
        }

        AVFormatContext *fmt_ctx = nullptr;
        int stream_idx;

        if (avformat_open_input(&fmt_ctx, path, nullptr, nullptr) < 0) {
            std::string error_msg = "Could not open source file";
            error_msg += path;
            PyErr_SetObject(PyExc_SystemError, PyBytes_FromString(error_msg.c_str()));
            return nullptr;
        }

        if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
            if (fmt_ctx) avformat_close_input(&fmt_ctx);
            PyErr_SetObject(PyExc_SystemError, PyBytes_FromString("Could not find stream information"));
            return nullptr;
        }

        stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream_idx < 0) {
            if (fmt_ctx) avformat_close_input(&fmt_ctx);
            PyErr_SetObject(PyExc_SystemError,
                            PyBytes_FromString("Could not find video stream in the input, aborting"));
            return nullptr;
        }

        enum AVCodecID codec_id = fmt_ctx->streams[stream_idx]->codecpar->codec_id;
        if (codec_id == AV_CODEC_ID_H264 || codec_id == AV_CODEC_ID_HEVC ) {
            // Keep a strong ref to callback during decode
            Py_INCREF(callback);
            PyObject *ret = decode_h264_cb(fmt_ctx,
                                           callback,
                                           !without_residual,
                                           max_frames,
                                           (selector.enabled() ? &selector : nullptr),
                                           seek_to_frame,
                                           decode_len,
                                           debug);
            Py_DECREF(callback);

            if (fmt_ctx) avformat_close_input(&fmt_ctx);
            return ret;
        } else {
            if (fmt_ctx) avformat_close_input(&fmt_ctx);
            PyErr_SetObject(PyExc_SystemError,
                            PyBytes_FromString("Only support H264/HEVC video, aborting"));
            return nullptr;
        }
    }
    catch (...) {
        std::string error_msg = "Unexpected exception in C++ while reading video files ";
        error_msg += path;
        PyErr_SetObject(PyExc_SystemError, PyBytes_FromString(error_msg.c_str()));
        return nullptr;
    }
}

static PyObject *read_video(PyObject *self, PyObject *args) {
    char *path = nullptr;
    int without_residual = 0;
    int max_frames = 0;

    try {
        /* Parse arguments */
        if (!PyArg_ParseTuple(args, "s|ii", &path, &without_residual, &max_frames)) {
            PyErr_SetObject(PyExc_SystemError, PyBytes_FromString("Argument Error"));
            return nullptr;
        }


        // for container file
        AVFormatContext *fmt_ctx = nullptr;
        int stream_idx;

        PyObject *results;

        if (avformat_open_input(&fmt_ctx, path, nullptr, nullptr) < 0) {
            std::string error_msg = "Could not open source file";
            error_msg += path;
            PyErr_SetObject(PyExc_SystemError, PyBytes_FromString(error_msg.c_str()));
            return nullptr;
        }

        if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
            PyErr_SetObject(PyExc_SystemError, PyBytes_FromString("Could not find stream information"));
            return nullptr;
        }

        stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream_idx < 0) {
            PyErr_SetObject(PyExc_SystemError,
                            PyBytes_FromString("Could not find video stream in the input, aborting"));
            return nullptr;
        }

        enum AVCodecID codec_id = fmt_ctx->streams[stream_idx]->codecpar->codec_id;
        if (codec_id == AV_CODEC_ID_H264 ||
            codec_id == AV_CODEC_ID_HEVC ) {
            // The decode function is generic: it finds the best video stream and opens the proper decoder.
            results = decode_h264(fmt_ctx, !without_residual, max_frames);
        } else {
            PyErr_SetObject(PyExc_SystemError,
                            PyBytes_FromString("Only support H264/HEVC video, aborting"));
            return nullptr;
        }

        if (fmt_ctx) avformat_close_input(&fmt_ctx);

        return results;
    }
    catch (...) {
        std::string error_msg = "Unexpected exception in C++ while reading video files ";
        error_msg += path;
        PyErr_SetObject(PyExc_SystemError, PyBytes_FromString(error_msg.c_str()));
        return nullptr;
    }
}

// Python module

static PyMethodDef CVMethods[] = {
        {"read_video", (PyCFunction) read_video, METH_VARARGS, "Read video"},
        {"read_video_cb", (PyCFunction) read_video_cb, METH_VARARGS, "Read video (streaming callback)"},
        {nullptr,      nullptr, 0,                             nullptr} // Sentinel
};

static struct PyModuleDef CVModule = {
        PyModuleDef_HEAD_INIT,
        "api",
        "Python interface for reading compressed video.",
        -1,
        CVMethods
};


PyMODINIT_FUNC PyInit_api() {
    PyObject *m = PyModule_Create(&CVModule);

    /* IMPORTANT: this must be called */
    import_array();

    return m;
}