from setuptools import setup
from setuptools import Extension
import sys

setup(
    name="pydatacopy",
    version="0.1",
    ext_modules=[
        Extension(
            "pydatacopy",
            sources=[],
        )
    ],
    package_data={'': ['build/pydatacopy.cpython-312-x86_64-linux-gnu.so']},
    include_package_data=True,
)