#include <ctime>
#include <string>
#include <queue>

extern "C" {
#include <libswscale/swscale.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavutil/motion_vector.h>
#include <libavutil/frame.h>
}

#include <Python.h>
#include "numpy/arrayobject.h"


static int
decode_h264_packet(AVPacket *pkt, AVCodecContext *dec_ctx, AVFrame *frame, int *frame_count, PyObject *results,
                   bool with_residual) {

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

            // Channels: L0_x, L0_y, L1_x, L1_y  (refs/pred mode not stored here)
            npy_intp mv_dims[3];
            mv_dims[0] = (height / 4);
            mv_dims[1] = (width / 4);
            mv_dims[2] = 4;
            // PyObject holder for returning to Python, and PyArrayObject* for C-API macros
            PyObject *mv_obj = PyArray_ZEROS(3, mv_dims, NPY_INT32, 0);
            PyArrayObject *mv_arr = reinterpret_cast<PyArrayObject *>(mv_obj);
            npy_intp residual_dims[2];
            residual_dims[0] = height;
            residual_dims[1] = width;
            PyArrayObject *residual_arr = nullptr;
            if (with_residual) {
                AVFrameSideData *sd_res = av_frame_get_side_data(frame, AV_FRAME_DATA_USER_RESIDUAL);

                if (sd_res && sd_res->data) {
                    fprintf(stderr,
                            "[cv_reader] frame %d: USER_RESIDUAL side data FOUND, size=%d (expect >= %d)\n",
                            *frame_count,
                            (int)sd_res->size,
                            height * width);

                    int dump_n = (int)sd_res->size;
                    if (dump_n > 32) dump_n = 32;  // dump up to 32 bytes
                    fprintf(stderr, "[cv_reader]   first %d bytes:", dump_n);
                    for (int k = 0; k < dump_n; ++k) {
                        fprintf(stderr, " %d", sd_res->data[k]);
                    }
                    fprintf(stderr, "\n");
                } else {
                    fprintf(stderr,
                            "[cv_reader] frame %d: USER_RESIDUAL side data NOT FOUND, fallback to zeros\n",
                            *frame_count);
                }

                if (sd_res && sd_res->data && sd_res->size >= (size_t)(height * width)) {
                    residual_arr = (PyArrayObject *) PyArray_SimpleNew(2, residual_dims, NPY_UINT8);
                    if (residual_arr) {
                        uint8_t *dst = reinterpret_cast<uint8_t *>(PyArray_DATA(residual_arr));
                        memcpy(dst, sd_res->data, (size_t)(height * width));
                    }
                } else {
                    residual_arr = (PyArrayObject *) PyArray_ZEROS(2, residual_dims, NPY_UINT8, 0);
                }
            }

            int i;
            AVFrameSideData *sd;

            (*frame_count)++;
            sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
            if (sd) {
                const auto *mvs = (const AVMotionVector *) sd->data;
                int p_dst_x, p_dst_y, mvx, mvy;
                for (i = 0; i < sd->size / sizeof(*mvs); i++) {
                    const AVMotionVector *mv = &mvs[i];
                    for (int x_start = (-1 * mv->w / 8); x_start < mv->w / 8; ++x_start) {
                        for (int y_start = (-1 * mv->h / 8); y_start < mv->h / 8; ++y_start) {
                            mvx = mv->src_x - mv->dst_x;
                            mvy = mv->src_y - mv->dst_y;

                            p_dst_x = mv->dst_x / 4 + x_start;
                            p_dst_y = mv->dst_y / 4 + y_start;
                            if (mv->dst_x % 4 != 0 || mv->dst_y % 4 != 0) {
                                fprintf(stderr, "dst_x or dst_y is not divisible by 4: %d %d", mv->dst_x, mv->dst_y);
                            }

                            if (p_dst_y >= 0 && p_dst_y < height / 4 &&
                                p_dst_x >= 0 && p_dst_x < width / 4) {
                                if (mv->source < 0) {
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

    return 0;
}

static PyObject *decode_h264(AVFormatContext *fmt_ctx, bool with_residual) {
    int ret;
    AVStream *st;
    AVCodecContext *dec_ctx = nullptr;
    const AVCodec *dec = nullptr;
    AVDictionary *opts = nullptr;
    AVPacket *pkt = nullptr;
    AVFrame *frame = nullptr;
    int frame_count = 0;
    PyObject *results = PyList_New(0);

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
    ret = avcodec_open2(dec_ctx, dec, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        fprintf(stderr, "Failed to open video codec\n");
        goto end;
    }
    fprintf(stderr, "[cv_reader] opened decoder: %s (id=%d)\n",
            avcodec_get_name(dec_ctx->codec_id), dec_ctx->codec_id);


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
            ret = decode_h264_packet(pkt, dec_ctx, frame, &frame_count, results, with_residual);
        }
        av_packet_unref(pkt);
        if (ret < 0)
            break;
    }

    /* flush cached frames */
    decode_h264_packet(nullptr, dec_ctx, frame, &frame_count, results, with_residual);

    end:
    avcodec_free_context(&dec_ctx);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    return results;
}

static PyObject *read_video(PyObject *self, PyObject *args) {
    char *path = nullptr;
    int without_residual = 0;

    try {
        /* Parse arguments */
        if (!PyArg_ParseTuple(args, "s|i", &path, &without_residual)) {
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
            codec_id == AV_CODEC_ID_HEVC ||
            codec_id == AV_CODEC_ID_MPEG4) {
            // The decode function is generic: it finds the best video stream and opens the proper decoder.
            results = decode_h264(fmt_ctx, !without_residual);
        } else {
            PyErr_SetObject(PyExc_SystemError,
                            PyBytes_FromString("Only support H264/HEVC/MPEG4 video, aborting"));
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