import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))

# DEBUG=1 时生成可供 cuda-gdb 单步调试的设备代码（-G 会关闭设备端优化，性能显著下降）
DEBUG = os.getenv("DEBUG", "0") == "1"

# csrc/ 下所有 .cpp / .cu 自动参与编译，新增算子无需修改本文件
sources = sorted(
    glob.glob("csrc/**/*.cpp", recursive=True) + glob.glob("csrc/**/*.cu", recursive=True)
)

if DEBUG:
    cxx_flags = ["-O0", "-g"]
    nvcc_flags = ["-O0", "-g", "-G"]
else:
    cxx_flags = ["-O3"]
    # -lineinfo 让 Nsight Compute 能把性能指标对应到 .cu 源码行
    nvcc_flags = ["-O3", "-lineinfo"]

setup(
    ext_modules=[
        CUDAExtension(
            name="cudalab._C",
            sources=sources,
            include_dirs=[os.path.join(ROOT, "csrc")],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
