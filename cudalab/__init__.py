import torch  # noqa: F401  必须先于 _C 导入，_C.so 依赖 libtorch 的符号

from . import _C  # noqa: F401  导入即触发 C++ 侧 TORCH_LIBRARY 的算子注册
from .ops import vector_add

__all__ = ["vector_add"]
