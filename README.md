# cudalab

系统学习 CUDA 编程的练习仓库。Python 负责搭模型、写测试和 benchmark；C++ 通过 libtorch 接收张量并做参数检查；CUDA kernel 负责真正的计算。

```
Python    cudalab.vector_add(a, b)                 模型 / 测试 / benchmark
   │
   ▼
PyTorch   torch.ops.cudalab.vector_add             dispatcher 按张量所在设备分发
   │
   ▼
C++       vector_add_cuda(const at::Tensor&, ...)  检查输入、分配输出、按 dtype 分派、发射 kernel
   │
   ▼
CUDA      vector_add_kernel<<<grid, block>>>       在 GPU 上执行
```

## 目录结构

```
.
├── csrc/                         C++ / CUDA 源码，编译为 cudalab/_C.so
│   ├── registration.cpp          所有算子的 schema 声明 + Python 模块入口
│   ├── common.cuh                公共宏与工具函数（输入检查、ceil_div）
│   └── kernels/
│       └── vector_add/
│           └── vector_add.cu     kernel + host 函数 + CUDA 实现注册
├── cudalab/                      Python 包
│   ├── __init__.py               加载 _C.so 并导出算子
│   ├── ops.py                    算子的 Python 封装
│   └── bench.py                  基于 CUDA event 的计时工具
├── tests/                        正确性测试（与 PyTorch 原生实现对比）
├── benchmarks/                   性能测试脚本
├── scripts/setup_env.sh          一键搭建 / 恢复环境
├── setup.py                      扩展编译配置（torch.utils.cpp_extension）
├── pyproject.toml
└── Makefile
```

## 快速开始

```bash
make env                      # 首次或容器重建后执行：安装 Python 3.12 + torch 2.11 (cu128) 并编译扩展
source .venv/bin/activate
make test
python benchmarks/bench_vector_add.py
```

修改 `csrc/` 下的代码后执行 `make build`（ninja 增量编译，只重编改动过的文件），Python 侧的修改无需重新编译。

这台机器的 GPU 是共享的，跑 benchmark 前先用 `nvidia-smi` 看一下占用，再通过 `CUDA_VISIBLE_DEVICES=<空闲卡号>` 指定 GPU，否则测出来的数据会有噪声。

## 新增一个算子

以 `relu` 为例：

1. 新建 `csrc/kernels/relu/relu.cu`，写 kernel 和 host 函数 `relu_cuda`，文件末尾注册实现：
   ```cpp
   TORCH_LIBRARY_IMPL(cudalab, CUDA, m) { m.impl("relu", &relu_cuda); }
   ```
2. 在 `csrc/registration.cpp` 中声明 schema：`m.def("relu(Tensor x) -> Tensor");`
3. 在 `cudalab/ops.py` 中加 Python 封装，并在 `cudalab/__init__.py` 中导出。
4. 新建 `tests/test_relu.py`，与 `torch.relu` 对比结果。
5. `make build && make test`

`setup.py` 会自动收集 `csrc/` 下所有 `.cpp` / `.cu`，不需要修改。

## 调试与性能分析

- 越界访问 / 数据竞争检查：`compute-sanitizer python -m pytest tests/test_vector_add.py -x`
- 单步调试 kernel：`make debug`（以 `-G` 编译），然后 `cuda-gdb --args python your_script.py`；调试完执行 `make build` 即可恢复优化版本（编译参数变化后 ninja 会自动重编）
- Nsight Compute 分析 kernel：`ncu -k regex:vector_add -c 1 python benchmarks/bench_vector_add.py`，加 `--set full -o report` 可导出报告到宿主机上用 GUI 查看。release 编译带了 `-lineinfo`，报告里可以对应到源码行
  - 注意：宿主机上运行着 `dcgm-exporter`，它占用 GPU 性能计数器，会导致 ncu 报 `driver resource was unavailable`。需要在宿主机上暂停它的 profiling 采集（或临时停掉该服务）后才能使用 ncu

## 学习路线

- [x] 01 vector_add：thread / block / grid、全局索引、边界检查
- [ ] 02 访存优化：合并访存、向量化读写（float4）、grid-stride loop
- [ ] 03 reduction：shared memory、bank conflict、warp shuffle
- [ ] 04 softmax / layernorm：按行归约、kernel 融合、online softmax
- [ ] 05 GEMM：naive → shared memory tiling → register tiling → Tensor Core
- [ ] 06 autograd：为自定义算子注册反向传播，用它们在 Python 中搭一个小模型
- [ ] 07 进阶：FlashAttention 等
