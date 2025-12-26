/*
 * HEVC video Decoder
 *
 * Copyright (C) 2012 - 2013 Guillaume Martres
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef AVCODEC_HEVCPRED_H
#define AVCODEC_HEVCPRED_H

#include <stddef.h>
#include <stdint.h>

struct HEVCContext;

typedef struct HEVCPredContext {
    void (*intra_pred[4])(struct HEVCContext *s, int x0, int y0, int c_idx);

    void (*pred_planar[4])(uint8_t *src, const uint8_t *top,
                           const uint8_t *left, ptrdiff_t stride);
    void (*pred_dc)(uint8_t *src, const uint8_t *top, const uint8_t *left,
                    ptrdiff_t stride, int log2_size, int c_idx);
    void (*pred_angular[4])(uint8_t *src, const uint8_t *top,
                            const uint8_t *left, ptrdiff_t stride,
                            int c_idx, int mode);
} HEVCPredContext;
/* 从源代码中可以看出，HEVCPredContext中存储了4个汇编函数指针（数组）：
 * intra_pred[4]()：帧内预测的入口函数，该函数执行过程中调用了后面3个函数指针。数组中4个函数分别处理4x4，8x8，16x16，32x32几种块。
 * pred_planar[4]()：Planar预测模式函数。数组中4个函数分别处理4x4，8x8，16x16，32x32几种块。
 * pred_dc()：DC预测模式函数。
 * pred_angular[4]()：角度预测模式。数组中4个函数分别处理4x4，8x8，16x16，32x32几种块。

 * */


void ff_hevc_pred_init(HEVCPredContext *hpc, int bit_depth);
void ff_hevcpred_init_x86(HEVCPredContext *c, const int bit_depth);

#endif /* AVCODEC_HEVCPRED_H */
