"""03_training_loop.py のテスト"""

import torch
import torch.nn as nn
import torch.optim as optim


def test_linear_regression_convergence():
    """線形回帰が収束するか"""
    torch.manual_seed(42)

    x = torch.linspace(0, 1, 100).unsqueeze(1)
    y = 2 * x + 1

    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.5)

    for _ in range(200):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert abs(model.weight.item() - 2.0) < 0.1
    assert abs(model.bias.item() - 1.0) < 0.1


def test_optimizer_updates_params():
    """オプティマイザがパラメータを更新するか"""
    model = nn.Linear(1, 1)
    initial_weight = model.weight.item()

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor([[1.0]])
    y = torch.tensor([[2.0]])

    pred = model(x)
    loss = nn.MSELoss()(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert model.weight.item() != initial_weight
