#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}


template <typename scalar_t>
__global__ void BoundaryPoolingForward(
        const int nthreads,
        const scalar_t* input,
        const scalar_t* segments,
        scalar_t* output,
        const int channels,
        const int tscale,
        const int seg_num) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        const int k = index % seg_num;
        const int c = (index / seg_num) % channels;
        const int n = index / seg_num / channels;
        const int seg_type = c / (channels / 2);
        const int seg_index = n * seg_num * 4 + k * 4 + seg_type * 2;
        scalar_t maxn, val;
        int l = static_cast<int>(segments[seg_index]);
        int r = static_cast<int>(segments[seg_index + 1]);
        l = min(max(0, l), tscale - 1);
        r = min(max(0, r), tscale - 1);
        maxn = input[n * channels * tscale + c * tscale + l];
        for (int i = l + 1; i <= r; i++) {
            val = input[n * channels * tscale + c * tscale + i];
            if (val > maxn) {
                maxn = val;
            }
        }
        output[index] = maxn;
    }
}

template <typename scalar_t>
__global__ void BoundaryPoolingBackward(
        const int nthreads,
        const scalar_t* grad_output,
        const scalar_t* input,
        const scalar_t* segments,
        scalar_t* grad_input,
        const int channels,
        const int tscale,
        const int seg_num) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        const int k = index % seg_num;
        const int c = (index / seg_num) % channels;
        const int n = index / seg_num / channels;
        const int seg_type = c / (channels / 2);
        const int seg_index = n * seg_num * 4 + k * 4 + seg_type * 2;
        scalar_t maxn, val;
        int argmax;
        int l = static_cast<int>(segments[seg_index]);
        int r = static_cast<int>(segments[seg_index + 1]);
        l = min(max(0, l), tscale - 1);
        r = min(max(0, r), tscale - 1);
        maxn = input[n * channels * tscale + c * tscale + l];
        argmax = l;
        for (int i = l + 1; i <= r; i++) {
            val = input[n * channels * tscale + c * tscale + i];
            if (val > maxn) {
                maxn = val;
                argmax = i;
            }
        }
        scalar_t grad = grad_output[index];
        atomicAdd(grad_input + n * channels * tscale + c * tscale + argmax, grad);
    }
}

int boundary_max_pooling_cuda_forward(
        const at::Tensor& input,
        const at::Tensor& segments,
        const at::Tensor& output) {
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(segments.type().is_cuda(), "segments must be a CUDA tensor");
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int tscale = input.size(2);
    const int seg_num = segments.size(1);
    const int output_size = batch_size * channels * seg_num;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "BoundaryMaxPoolingForward", ([&] {

        BoundaryPoolingForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            output_size,
            input.data_ptr<scalar_t>(),
            segments.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            channels,
            tscale,
            seg_num);
    }));

    C10_CUDA_CHECK(cudaGetLastError());
    return 1;
}

int boundary_max_pooling_cuda_backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        const at::Tensor& segments,
        const at::Tensor& grad_input) {
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(segments.type().is_cuda(), "segments must be a CUDA tensor");
    AT_ASSERTM(grad_output.type().is_cuda(), "grad_output must be a CUDA tensor");
    const int batch_size = grad_output.size(0);
    const int channels = grad_output.size(1);
    const int tscale = grad_output.size(2);
    const int seg_num = segments.size(1);

    const int output_size = batch_size * channels * seg_num;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "BoundaryMaxPoolingBackward", ([&] {

        BoundaryPoolingBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            output_size,
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            segments.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            channels,
            tscale,
            seg_num);
    }));

    C10_CUDA_CHECK(cudaGetLastError());
    return 1;
}