# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Any

import matplotlib
import pytest
import torch
import torchvision
from pytest import MonkeyPatch

HEAVY_TESTS = (
    'TestCROMALarge',
    'TestViTHuge14',
    'TestDOFAHuge14',
    'TestScaleMAE',
    'aurora_swin_unet',
    'croma_large',
    'vit_huge',
    'dofa_huge',
    'scalemae_large',
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if any(name in item.nodeid for name in HEAVY_TESTS):
            item.add_marker(pytest.mark.xdist_group('heavy'))


def load(*args: Any, progress: bool = False, **kwargs: Any) -> Any:
    return torch.load(*args, **kwargs)


@pytest.fixture
def load_state_dict_from_url(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(torchvision.models._api, 'load_state_dict_from_url', load)


@pytest.fixture(autouse=True, scope='session')
def matplotlib_backend() -> None:
    matplotlib.use('agg')


@pytest.fixture(autouse=True)
def torch_hub(tmp_path: Path) -> None:
    torch.hub.set_dir(tmp_path)
