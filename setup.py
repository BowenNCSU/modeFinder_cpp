import glob
import os
import sys

from setuptools import Extension, setup

if sys.platform == "win32":
    extra_compile_args = []
    libraries = []
else:
    extra_compile_args = ["-std=c11", "-O2"]
    libraries = ["m"]

setup(
    ext_modules=[
        Extension(
            "modefinder._modefinder",
            sources=["modefinder/_modefinder.c"] + sorted(glob.glob("csrc/*.c")),
            include_dirs=["csrc"],
            depends=sorted(glob.glob(os.path.join("csrc", "*.h"))),
            extra_compile_args=extra_compile_args,
            libraries=libraries,
        )
    ],
)
