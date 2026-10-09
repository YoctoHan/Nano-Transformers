import torch

import cudalab
from cudalab.bench import benchmark


def main():
    print(f"device: {torch.cuda.get_device_name()}")
    header = f"{'N':>12} {'dtype':>9} | {'cudalab ms':>10} {'GB/s':>7} | {'torch ms':>10} {'GB/s':>7}"
    print(header)
    print("-" * len(header))

    for dtype in (torch.float32, torch.float16):
        for n in (1 << 16, 1 << 20, 1 << 24, 1 << 28):
            a = torch.randn(n, device="cuda", dtype=dtype)
            b = torch.randn(n, device="cuda", dtype=dtype)
            # 读 a、b + 写 out，共 3 次全量访存
            nbytes = 3 * n * a.element_size()

            ms_ours = benchmark(cudalab.vector_add, a, b)
            ms_ref = benchmark(torch.add, a, b)
            print(
                f"{n:>12} {str(dtype).removeprefix('torch.'):>9} | "
                f"{ms_ours:>10.4f} {nbytes / ms_ours / 1e6:>7.1f} | "
                f"{ms_ref:>10.4f} {nbytes / ms_ref / 1e6:>7.1f}"
            )


if __name__ == "__main__":
    main()
