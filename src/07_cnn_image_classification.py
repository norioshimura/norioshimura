"""
07. CNNによる画像分類
====================
CNN（畳み込みニューラルネットワーク）で画像分類を実装する。
MNISTデータセットを使用した手書き数字の分類。

E資格ポイント:
- Conv2dの引数: in_channels, out_channels, kernel_size, stride, padding
- 出力サイズ: H_out = (H_in + 2*padding - kernel_size) / stride + 1
- MaxPool2d: 特徴マップのダウンサンプリング
- 畳み込み層 → プーリング層 → 全結合層の構成
- パラメータ数の計算方法
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ============================================================
# 1. 畳み込み演算の基礎
# ============================================================

# 入力: 1チャンネル、8x8の画像（バッチサイズ1）
input_image = torch.randn(1, 1, 8, 8)

# 畳み込み層: 1入力チャンネル → 4出力チャンネル、3x3カーネル
conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
output = conv(input_image)

print("=== 畳み込み演算の基礎 ===")
print(f"入力: {input_image.shape}")     # [1, 1, 8, 8]
print(f"出力: {output.shape}")          # [1, 4, 8, 8]
print(f"カーネルの形状: {conv.weight.shape}")  # [4, 1, 3, 3]
print(f"パラメータ数: {conv.weight.numel() + conv.bias.numel()}")

# プーリング
pool = nn.MaxPool2d(kernel_size=2, stride=2)
pooled = pool(output)
print(f"プーリング後: {pooled.shape}")  # [1, 4, 4, 4]

# ============================================================
# 2. ダミーMNISTデータの生成
# ============================================================
# 実際のMNISTダウンロードを避けるため、ダミーデータを使用

torch.manual_seed(42)

n_train, n_test = 1000, 200
n_classes = 10

# ダミー画像（28x28, 1チャンネル）とラベル
X_train = torch.randn(n_train, 1, 28, 28)
y_train = torch.randint(0, n_classes, (n_train,))
X_test = torch.randn(n_test, 1, 28, 28)
y_test = torch.randint(0, n_classes, (n_test,))

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=64, shuffle=True
)

# ============================================================
# 3. CNNモデルの定義
# ============================================================


class SimpleCNN(nn.Module):
    """シンプルなCNNモデル（MNIST分類用）"""

    def __init__(self) -> None:
        super().__init__()
        # 畳み込み層
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 28x28 → 28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 14x14 → 14x14
        self.pool = nn.MaxPool2d(2, 2)                             # サイズを半分に

        # 全結合層（28→14→7, 32チャンネル → 32*7*7 = 1568）
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 畳み込み + ReLU + プーリング
        x = self.pool(F.relu(self.conv1(x)))   # [B, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 32, 7, 7]

        # Flatten
        x = x.view(x.size(0), -1)  # [B, 1568]

        # 全結合層
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN()
print(f"\n=== CNNモデル ===")
print(model)

# パラメータ数の確認
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"総パラメータ数: {total_params:,}")
print(f"学習可能パラメータ数: {trainable_params:,}")

# ============================================================
# 4. 学習
# ============================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

print(f"\n=== 学習開始 ===")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(logits, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    avg_loss = total_loss / n_train
    accuracy = correct / total
    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2%}")

# ============================================================
# 5. 評価
# ============================================================

model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    _, test_preds = torch.max(test_logits, 1)
    test_accuracy = (test_preds == y_test).float().mean()
    print(f"\nテスト精度: {test_accuracy.item():.2%}")
    print("（ダミーデータのためランダム精度に近い値になります）")

# ============================================================
# 6. 特徴マップの確認
# ============================================================

sample = X_test[:1]  # 1枚の画像
with torch.no_grad():
    feat1 = F.relu(model.conv1(sample))
    feat2 = model.pool(feat1)
    print(f"\nconv1後の特徴マップ: {feat1.shape}")  # [1, 16, 28, 28]
    print(f"pool1後の特徴マップ: {feat2.shape}")    # [1, 16, 14, 14]

print("\n=== 07_cnn_image_classification.py 完了 ===")
