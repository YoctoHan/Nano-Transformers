import torch
from torch import Tensor


def vector_add(a: Tensor, b: Tensor) -> Tensor:
    """逐元素相加 ``out = a + b``。

    要求 a、b 位于同一块 GPU，形状、dtype 相同且内存连续。
    """
    return torch.ops.cudalab.vector_add.default(a, b)
