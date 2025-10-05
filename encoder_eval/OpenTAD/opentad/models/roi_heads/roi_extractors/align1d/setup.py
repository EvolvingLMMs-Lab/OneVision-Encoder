from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="Align1D",
    version="2.5.0",
    author="Frost Mengmeng Xu",
    author_email="xu.frost@gmail.com",
    description="A small package for 1d aligment in cuda",
    long_description="Update: support pytorch 1.11",
    long_description_content_type="text/markdown",
    url="https://github.com/Frostinassiky/G-TAD",
    ext_modules=[
        CUDAExtension(
            name="Align1D",
            sources=[
                "Align1D_cuda.cpp",
                "Align1D_cuda_kernal.cu",
            ],
            extra_compile_args={"cxx": [], "nvcc": ["--expt-relaxed-constexpr", "-allow-unsupported-compiler"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
