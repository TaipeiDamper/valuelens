from __future__ import annotations

import os
import sys

import numpy as np
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


compile_args: list[str] = ["-O3"]
link_args: list[str] = []

if os.name == "nt":
    compile_args = ["/O2", "/std:c++17", "/EHsc"]
else:
    compile_args.extend(["-std=c++17"])

ext_modules = [
    Pybind11Extension(
        "valuelens_native",
        ["src/valuelens_native.cpp"],
        include_dirs=[np.get_include()],
        cxx_std=17,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

setup(
    name="valuelens-native",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

