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
            # 这里不需要源码，因为你已经有 .so 文件
        )
    ],
    # 指定已经编译好的 .so 文件路径
    package_data={'': ['build/pydatacopy.cpython-312-x86_64-linux-gnu.so']},
    include_package_data=True,
)