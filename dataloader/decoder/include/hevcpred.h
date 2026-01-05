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
/* From the source code, HEVCPredContext stores 4 assembly function pointers (arrays):
 * intra_pred[4](): Entry function for intra prediction, which calls the following 3 function pointers during execution. The 4 functions in the array handle 4x4, 8x8, 16x16, 32x32 block sizes respectively.
 * pred_planar[4](): Planar prediction mode function. The 4 functions in the array handle 4x4, 8x8, 16x16, 32x32 block sizes respectively.
 * pred_dc(): DC prediction mode function.
 * pred_angular[4](): Angular prediction mode. The 4 functions in the array handle 4x4, 8x8, 16x16, 32x32 block sizes respectively.
 * */


void ff_hevc_pred_init(HEVCPredContext *hpc, int bit_depth);
void ff_hevcpred_init_x86(HEVCPredContext *c, const int bit_depth);

#endif /* AVCODEC_HEVCPRED_H */
