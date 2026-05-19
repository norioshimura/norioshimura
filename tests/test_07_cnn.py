"""07_cnn_image_classification.py のテスト"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """テスト用にモデルを再定義"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test_cnn_output_shape():
    """CNNの出力形状が正しいか"""
    model = SimpleCNN()
    x = torch.randn(4, 1, 28, 28)
    output = model(x)
    assert output.shape == (4, 10)


def test_conv2d_output_size():
    """畳み込み層の出力サイズ計算"""
    conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
    x = torch.randn(1, 1, 8, 8)
    out = conv(x)
    assert out.shape == (1, 4, 8, 8)


def test_maxpool_halves_size():
    """MaxPoolがサイズを半分にするか"""
    pool = nn.MaxPool2d(2, 2)
    x = torch.randn(1, 1, 8, 8)
    out = pool(x)
    assert out.shape == (1, 1, 4, 4)


def test_cnn_training():
    """CNNが学習できるか"""
    torch.manual_seed(42)
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))

    initial_loss = criterion(model(x), y).item()

    for _ in range(10):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < initial_loss
