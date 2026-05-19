"""02_autograd_basics.py のテスト"""

import torch


def test_basic_grad():
    """基本的な微分 y = x^2 + 3x + 1, dy/dx = 2x + 3"""
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2 + 3 * x + 1
    y.backward()
    assert torch.isclose(x.grad, torch.tensor(7.0))


def test_multivar_grad():
    """多変数の偏微分"""
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    z = x ** 2 * y + y ** 3
    z.backward()
    assert torch.isclose(x.grad, torch.tensor(4.0))
    assert torch.isclose(y.grad, torch.tensor(13.0))


def test_grad_accumulation():
    """勾配の蓄積と初期化"""
    x = torch.tensor(3.0, requires_grad=True)
    y1 = x ** 2
    y1.backward()
    first_grad = x.grad.clone()

    y2 = x ** 3
    y2.backward()
    assert x.grad.item() == first_grad.item() + 27.0

    x.grad.zero_()
    y3 = x ** 3
    y3.backward()
    assert torch.isclose(x.grad, torch.tensor(27.0))


def test_no_grad():
    """torch.no_grad()で勾配計算が無効化される"""
    x = torch.tensor(2.0, requires_grad=True)
    with torch.no_grad():
        y = x ** 2
    assert not y.requires_grad


def test_detach():
    """detach()で計算グラフから切り離す"""
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2
    y_detached = y.detach()
    assert not y_detached.requires_grad
