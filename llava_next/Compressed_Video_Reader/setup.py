import os
from setuptools import find_packages, setup, Extension
import pkgconfig
import numpy as np

project_root = os.path.dirname(os.path.realpath(__file__))
ffmpeg_pkg_config_path = str(os.path.join(project_root, "ffmpeg/ffmpeg_install/lib/pkgconfig"))
os.environ["PKG_CONFIG_PATH"] = (
    os.environ["PKG_CONFIG_PATH"] + f":{ffmpeg_pkg_config_path}"
    if ("PKG_CONFIG_PATH" in os.environ)
    else str(ffmpeg_pkg_config_path)
)

d = pkgconfig.parse("libavformat libswscale libavcodec libavcodec libavutil")

reader = Extension(
    "cv_reader.api",
    include_dirs=[*d["include_dirs"], np.get_include()],
    library_dirs=d["library_dirs"],
    libraries=d["libraries"],
    sources=[
        "src/cv_reader/h264_api.cpp",
    ],
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-fPIC", "-Wl,-Bsymbolic"],
)

setup(
    name="cv-reader",
    author="",
    author_email="",
    version="1.1.0",
    license="MIT",
    description="API for reading H.264/H.265 residuals and motion vectors",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    keywords=["h.264/h.265", "motion vector", "residual"],
    ext_modules=[reader],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={"console_scripts": ["cv_reader=cv_reader.__main__:main"]},
    python_requires=">=3.7, <4",
    install_requires=["pkgconfig>=1.5.1", "numpy>=1.23,<2.0.0", "flow-vis>=0.1", "opencv-python>=4.0"],
    url="https://github.com/AcherStyx/Compressed-Video-Reader",
    project_urls={
        "Source": "https://github.com/AcherStyx/Compressed-Video-Reader",
        "Tracker": "https://github.com/AcherStyx/Compressed-Video-Reader/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)

# install_requires=['pkgconfig>=1.5.1', 'numpy>=1.17.0', 'flow-vis>=0.1', 'opencv-python>=4.0'],
