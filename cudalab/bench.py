from typing import Callable

import torch


def benchmark(fn: Callable, *args, warmup: int = 10, iters: int = 100) -> float:
    """返回 ``fn(*args)`` 单次调用的平均 GPU 耗时（毫秒）。

    kernel 发射是异步的，CPU 计时只能测到发射开销，所以这里用 CUDA event 计时。
    """
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters
