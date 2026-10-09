PYTHON ?= .venv/bin/python

.PHONY: help env build debug test clean

help:
	@echo "make env    创建/恢复 Python 环境（首次或容器重建后执行）"
	@echo "make build  增量编译 C++/CUDA 扩展（修改 csrc/ 后执行）"
	@echo "make debug  以 -G 编译扩展，供 cuda-gdb 调试"
	@echo "make test   运行全部测试"
	@echo "make clean  清理编译产物"

env:
	bash scripts/setup_env.sh

build:
	$(PYTHON) setup.py build_ext --inplace

debug:
	DEBUG=1 $(PYTHON) setup.py build_ext --inplace

test:
	$(PYTHON) -m pytest

clean:
	rm -rf build *.egg-info cudalab/*.so
