import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA is not available")
    for item in items:
        item.add_marker(skip)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
