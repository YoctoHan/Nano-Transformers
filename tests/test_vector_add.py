import pytest
import torch

import cudalab

DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]
SIZES = [1, 7, 256, 1000, 1 << 20, (1 << 20) + 3]


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).removeprefix("torch."))
@pytest.mark.parametrize("n", SIZES)
def test_matches_torch(n, dtype):
    a = torch.randn(n, device="cuda", dtype=dtype)
    b = torch.randn(n, device="cuda", dtype=dtype)
    torch.testing.assert_close(cudalab.vector_add(a, b), a + b)


def test_multi_dim():
    a = torch.randn(4, 5, 6, device="cuda")
    b = torch.randn(4, 5, 6, device="cuda")
    out = cudalab.vector_add(a, b)
    assert out.shape == a.shape
    torch.testing.assert_close(out, a + b)


def test_empty():
    a = torch.empty(0, device="cuda")
    assert cudalab.vector_add(a, a).numel() == 0


def test_rejects_cpu_tensor():
    # 只注册了 CUDA 实现，CPU 张量会被 dispatcher 直接拒绝
    a = torch.randn(8)
    with pytest.raises(NotImplementedError):
        cudalab.vector_add(a, a)


def test_rejects_non_contiguous():
    a = torch.randn(16, 8, device="cuda").t()
    with pytest.raises(RuntimeError, match="contiguous"):
        cudalab.vector_add(a, a)


def test_rejects_shape_mismatch():
    a = torch.randn(8, device="cuda")
    b = torch.randn(9, device="cuda")
    with pytest.raises(RuntimeError, match="same shape"):
        cudalab.vector_add(a, b)
