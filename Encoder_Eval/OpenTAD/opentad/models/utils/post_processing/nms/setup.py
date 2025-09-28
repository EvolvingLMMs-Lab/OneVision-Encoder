from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="nms_1d_cpu",
    version="0.0.1",
    ext_modules=[CppExtension(name="nms_1d_cpu", sources=["./nms_cpu.cpp"], extra_compile_args=["-fopenmp"])],
    cmdclass={"build_ext": BuildExtension},
)
