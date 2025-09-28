#include <torch/extension.h>
#include <vector>


int boundary_max_pooling_cuda_forward(
        const at::Tensor& input,
        const at::Tensor& segments,
        const at::Tensor& output
);

int boundary_max_pooling_cuda_backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        const at::Tensor& segments,
        const at::Tensor& grad_input
);

at::Tensor boundary_max_pooling_forward(
        const at::Tensor& input,
        const at::Tensor& segments) {

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int seg_num = segments.size(1);

    auto output = torch::zeros({batch_size, channels, seg_num}, input.options());
    boundary_max_pooling_cuda_forward(input, segments, output);
    return output;
}

at::Tensor boundary_max_pooling_backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        const at::Tensor& segments) {

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int t_dim = input.size(2);

    auto grad_input = torch::zeros({batch_size, channels, t_dim}, grad_output.options());
    boundary_max_pooling_cuda_backward(grad_output, input, segments, grad_input);
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &boundary_max_pooling_forward, "Boundary max pooling forward (CUDA)");
  m.def("backward", &boundary_max_pooling_backward, "Boundary max pooling backward (CUDA)");
}