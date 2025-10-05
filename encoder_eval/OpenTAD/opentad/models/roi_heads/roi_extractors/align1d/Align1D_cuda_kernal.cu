// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
// Modifies by Frost for 1D ussage
#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <c10/cuda/impl/CUDATest.h>



// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__device__ T linear_interpolate(const T* bottom_data,
    const int height,
    T t,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (t < -1.0 || t > height) {
    //empty
    return 0;
  }

  if (t <= 0) t = 0;

  int t_low = (int) t;
  int t_high;

  // get closest integers to t
  if (t_low >= height - 1) {
    t_high = t_low = height - 1;
    t = (T) t_low;
  } else {
    t_high = t_low + 1;
  }

  // get the distance to t
  T lt = t - t_low;
  T ht = 1. - lt;

  // do linear interpolation
  T v1 = bottom_data[t_low];
  T v2 = bottom_data[t_high];
  T w1 = ht, w2 = lt;

  T val = (w1 * v1 + w2 * v2);
  // printf("Check Linear Interpolate: w1=%f, v1=%f, w2=%f, v2=%f \n", w1, v1, w2, v2);
  return val;
}

template <typename T>
__global__ void Align1DForward(const int nthreads, const T* bottom_data,
    const T spatial_scale, const int channels,
    const int height,
    const int pooled_height, 
    const int sampling_ratio,
    const T* bottom_rois, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, pt) is an element in the pooled output
    int pt = index % pooled_height;
    int c = (index / pooled_height) % channels;
    int n = index / pooled_height / channels;

    // printf("Debug Main Loop: get pt, c, n are %d, %d, %d \n", pt, c, n);

    const T* offset_bottom_rois = bottom_rois + n * 3;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start = offset_bottom_rois[1] * spatial_scale;
    T roi_end = offset_bottom_rois[2] * spatial_scale;
    // printf("Debug roi boundary: w1,  w2,  is  %f, %f \n", roi_start,roi_end,);

    // Force malformed ROIs to be 1x1
    T roi_height = max(roi_end- roi_start, (T)1.);
    T bin_size = static_cast<T>(roi_height) / static_cast<T>(pooled_height);

    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    // printf("sampling_ratio %d \n", roi_bin_grid);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid; // e.g. = 4

    T output_val = 0.;
    for (int it = 0; it < roi_bin_grid; it ++) // e.g., it = 0, 1
    {
      const T t = roi_start + pt * bin_size + static_cast<T>(it + .5f) * bin_size / static_cast<T>(roi_bin_grid); // e.g., 0.5, 1.5

      T val = linear_interpolate(offset_bottom_data, height, t, index);
      // printf("Debug linear_interpolate: input=height:%d, t:%f, ... ; output=val:%f \n", height, t, val);
      output_val += val;
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}


template <typename T>
__device__ void linear_interpolate_gradient(
    const int height, 
    T t,
    T & w1, T & w2,
    int & t_low, int & t_high, 
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (t < -1.0 || t > height) {
    //empty
    w1 = w2 = 0.;
    t_low = t_high = -1;
    return;
  }

  if (t <= 0) t = 0;

  t_low = (int) t;

  if (t_low >= height - 1) {
    t_high = t_low = height - 1;
    t = (T) t_low;
  } else {
    t_high = t_low + 1;
  }

  T lt = t - t_low;
  T ht = 1. - lt;

  // T val = (w1 * v1 + w2 * v2);
  // T w1 = ht, w2 = lt;
  w1 = ht , w2 = lt;

  return;
}

template <typename T>
__global__ void Align1DBackwardFeature(const int nthreads, const T* top_diff,
    const int num_rois, const T spatial_scale,
    const int channels, const int height,
    const int pooled_height,
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, pt) is an element in the pooled output
    int pt = (index ) % pooled_height;
    int c = (index / pooled_height) % channels;
    int n = index / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 3;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start= offset_bottom_rois[1] * spatial_scale;
    T roi_end= offset_bottom_rois[2] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_height = max(roi_end- roi_start, (T)1.);
    T bin_size = static_cast<T>(roi_height) / static_cast<T>(pooled_height);

    T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height;

    int top_offset    = (n * channels + c) * pooled_height;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[pt];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid= (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid; // e.g. = 4

    for (int it = 0; it < roi_bin_grid; it ++) // e.g., iy = 0, 1
    {
      const T t = roi_start+ pt * bin_size+ static_cast<T>(it + .5f) * bin_size/ static_cast<T>(roi_bin_grid); // e.g., 0.5, 1.5

      T w1, w2;
      int t_low, t_high;

      linear_interpolate_gradient(height, t, w1, w2, t_low, t_high, index);

      T g1 = top_diff_this_bin * w1 / count;
      T g2 = top_diff_this_bin * w2 / count;

      if (t_low >= 0 && t_high >= 0)
      {
          atomicAdd(offset_bottom_diff + t_low, static_cast<T>(g1));
          atomicAdd(offset_bottom_diff + t_high, static_cast<T>(g2));
      } // if
    } // it
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward


at::Tensor Align_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int sampling_ratio) {
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");
  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);

  auto output = at::empty({num_rois, channels, pooled_height}, input.options());
  auto output_size = num_rois * pooled_height * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(at::ceil_div((long)output_size, 512L), 4096L));
  dim3 block(512);

  // printf("Debug main function: height:%d\n", height);

  if (output.numel() == 0) {
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "Align1D_forward", [&] {
    Align1DForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         input.contiguous().data<scalar_t>(),
         spatial_scale,
         channels,
         height,
         pooled_height,
         sampling_ratio,
         rois.contiguous().data<scalar_t>(),
         output.data<scalar_t>());
  });
  C10_CUDA_CHECK(cudaGetLastError());
  return output;
}

// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor Align_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int batch_size,
                                  const int channels,
                                  const int height,
                                  const int sampling_ratio) {
  AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto grad_input = at::zeros({batch_size, channels, height}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(at::ceil_div((long)grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    C10_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "ROIAlign_backward", [&] {
    Align1DBackwardFeature<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel(),
         grad.contiguous().data<scalar_t>(),
         num_rois,
         spatial_scale,
         channels,
         height,
         pooled_height,
         sampling_ratio,
         grad_input.data<scalar_t>(),
         rois.contiguous().data<scalar_t>());
  });
  C10_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}