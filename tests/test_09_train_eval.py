"""09_train_eval_nograd.py のテスト"""

import torch
import torch.nn as nn


class DemoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.bn = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


def test_training_flag():
    """train()/eval()でtrainingフラグが変わるか"""
    model = DemoModel()

    model.train()
    assert model.training is True

    model.eval()
    assert model.training is False


def test_eval_deterministic():
    """eval()モードで出力が決定的か"""
    model = DemoModel()
    model.eval()

    x = torch.randn(5, 10)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)


def test_no_grad_disables_gradient():
    """no_grad()で勾配が無効化されるか"""
    model = DemoModel()
    model.eval()

    x = torch.randn(5, 10)

    with torch.no_grad():
        out = model(x)
    assert not out.requires_grad


def test_grad_enabled_by_default():
    """デフォルトで勾配が有効か"""
    x = torch.randn(5, 10, requires_grad=True)
    model = DemoModel()
    model.eval()
    out = model(x)
    assert out.requires_grad
