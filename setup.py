from setuptools import setup, Extension
import pybind11

cpp_args = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']

sfc_module = Extension(
    'perception_cpp',
    sources=['src/perception/cpp/bindings.cpp', 'src/perception/cpp/point_cloud_processor.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=cpp_args,
    )

setup(
    name='perception_cpp',
    version='1.0',
    description='C++ perception library for Text2Wheel',
    ext_modules=[sfc_module],
)
