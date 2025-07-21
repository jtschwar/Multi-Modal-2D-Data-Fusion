from setuptools import setup, Extension

# Simple approach - just require pybind11 to be installed first
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "multimodal_fusion.ctvlib",
        sources=[
            "regularization/ctvlib.cpp",
            "regularization/bindings.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "regularization",
            "thirdparty/eigen",
        ],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    name="multimodal-fusion",
    version="0.1.0",
    author="Jonathan Schwartz",
    author_email="jtschwar@gmail.com",
    url="https://github.com/jtschwar/Multi-Modal-2D-Data-Fusion",
    description="2D Fused Multi-Modal Electron Microscopy",
    packages=["multimodal_fusion"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy",
        "matplotlib", 
        "h5py",
    ],
    setup_requires=[
        "pybind11>=2.10.0",
    ],
)