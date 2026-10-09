#!/usr/bin/env bash
# 搭建 / 恢复开发环境。容器重建后重新执行一次即可：bash scripts/setup_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
# 需与容器内 nvcc 的 CUDA 版本（12.8）一致
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

# 解释器放在仓库内（NFS 持久化），否则容器重建后 .venv 里的 python 软链接会失效
export UV_PYTHON_INSTALL_DIR="${REPO_ROOT}/.uv/python"
# uv 缓存在容器本地盘，和 .venv 不在同一文件系统，无法硬链接
export UV_LINK_MODE=copy

if ! command -v uv >/dev/null 2>&1; then
    echo ">>> 安装 uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

if [ ! -x .venv/bin/python ]; then
    echo ">>> 创建虚拟环境 .venv (Python ${PYTHON_VERSION})"
    uv venv --python "${PYTHON_VERSION}" .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">>> 安装 torch==${TORCH_VERSION} (${TORCH_INDEX_URL})"
uv pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"

echo ">>> 安装构建工具与开发依赖"
uv pip install setuptools wheel ninja
uv pip install -e ".[dev]" --no-build-isolation

echo ">>> 环境自检"
python - <<'EOF'
import torch
import cudalab

print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | GPUs: {torch.cuda.device_count()}")
a = torch.randn(1024, device="cuda")
b = torch.randn(1024, device="cuda")
torch.testing.assert_close(cudalab.vector_add(a, b), a + b)
print("cudalab.vector_add OK")
EOF

echo
echo "完成。激活环境：source .venv/bin/activate"
