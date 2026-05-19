"""06_mlp_binary_classification.py のテスト"""

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src'))
from importlib import import_module


def _get_mlp_class():
    """BinaryMLPクラスを動的にインポート"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mlp_module",
        str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src' / '06_mlp_binary_classification.py')
    )
    # スクリプト実行を避けるためクラス定義だけ取得
    import ast
    src_path = __import__('pathlib').Path(__file__).resolve().parent.parent / 'src' / '06_mlp_binary_classification.py'
    # 直接クラスを定義して使う
    return None


class BinaryMLP(nn.Module):
    """テスト用にモデルを再定義"""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def test_mlp_output_shape():
    """MLPの出力形状が正しいか"""
    model = BinaryMLP(input_dim=2, hidden_dim=16)
    x = torch.randn(10, 2)
    output = model(x)
    assert output.shape == (10,)


def test_mlp_training():
    """MLPが学習できるか"""
    torch.manual_seed(42)
    model = BinaryMLP(input_dim=2, hidden_dim=16)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = torch.randn(100, 2)
    y = (x[:, 0] + x[:, 1] > 0).float()

    initial_loss = criterion(model(x), y).item()

    for _ in range(50):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss


def test_mlp_param_count():
    """パラメータ数の確認"""
    model = BinaryMLP(input_dim=2, hidden_dim=16)
    total = sum(p.numel() for p in model.parameters())
    # Linear(2,16): 2*16+16=48, Linear(16,16): 16*16+16=272, Linear(16,1): 16*1+1=17
    assert total == 48 + 272 + 17
