#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
at::Tensor Align_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int sampling_ratio);

at::Tensor Align_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int batch_size,
                                  const int channels,
                                  const int height,
                                  const int sampling_ratio);

// C++ interface
at::Tensor Align_forward(const at::Tensor& input, // (bs,ch,t)
                                 const at::Tensor& rois, // (bs, start, end)
                                 const int pooled_height,
                                 const int sampling_ratio){
    return Align_forward_cuda( input, rois, 1.0, pooled_height, sampling_ratio);
                                     }

at::Tensor Align_backward(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const int pooled_height,
                                  const int batch_size,
                                  const int channels,
                                  const int height,
                                  const int sampling_ratio){
    return Align_backward_cuda(grad, rois, 1.0, pooled_height, batch_size, channels, height, sampling_ratio);
                                      }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &Align_forward, "Align forward (CUDA)");
  m.def("backward", &Align_backward, "Align backward (CUDA)");
}
