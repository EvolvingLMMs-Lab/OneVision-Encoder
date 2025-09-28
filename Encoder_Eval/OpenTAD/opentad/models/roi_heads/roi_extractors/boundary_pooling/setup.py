from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="AFSD",
    version="1.0",
    description="Learning Salient Boundary Feature for Anchor-free " "Temporal Action Localization",
    author="Chuming Lin, Chengming Xu",
    author_email="chuminglin@tencent.com, cmxu18@fudan.edu.cn",
    ext_modules=[
        CUDAExtension(
            name="boundary_max_pooling_cuda",
            sources=[
                "boundary_max_pooling_cuda.cpp",
                "boundary_max_pooling_kernel.cu",
            ],
            extra_compile_args={"cxx": [], "nvcc": ["--expt-relaxed-constexpr", "-allow-unsupported-compiler"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
