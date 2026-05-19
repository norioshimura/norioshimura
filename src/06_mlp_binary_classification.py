"""
06. MLPによる2値分類
====================
MLP（多層パーセプトロン）で2値分類を実装する。

E資格ポイント:
- MLPは全結合層（Linear）と活性化関数の組み合わせ
- 2値分類ではBCEWithLogitsLossを使用（sigmoid + BCE）
- 活性化関数: ReLU, Sigmoid, Tanh の特徴と使い分け
- 過学習防止にDropoutを使用
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 1. データ生成（月型データを模倣）
# ============================================================

torch.manual_seed(42)

n_samples = 500

# クラス0: 中心(0, 0)の正規分布
x0 = torch.randn(n_samples // 2, 2) + torch.tensor([0.0, 0.0])
# クラス1: 中心(2, 2)の正規分布
x1 = torch.randn(n_samples // 2, 2) + torch.tensor([2.0, 2.0])

X = torch.cat([x0, x1], dim=0)
y = torch.cat([torch.zeros(n_samples // 2),
               torch.ones(n_samples // 2)], dim=0)

# 訓練データとテストデータに分割（80:20）
n_train = int(0.8 * n_samples)
indices = torch.randperm(n_samples)
X_train, X_test = X[indices[:n_train]], X[indices[n_train:]]
y_train, y_test = y[indices[:n_train]], y[indices[n_train:]]

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ============================================================
# 2. MLPモデルの定義
# ============================================================


class BinaryMLP(nn.Module):
    """2値分類用の多層パーセプトロン"""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 入力層 → 隠れ層
            nn.ReLU(),                           # 活性化関数
            nn.Linear(hidden_dim, hidden_dim),   # 隠れ層 → 隠れ層
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),            # 隠れ層 → 出力層（logit）
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


model = BinaryMLP(input_dim=2, hidden_dim=16)
print(f"モデル構造:\n{model}")
print(f"パラメータ数: {sum(p.numel() for p in model.parameters())}")

# ============================================================
# 3. 損失関数とオプティマイザ
# ============================================================

# BCEWithLogitsLoss: Sigmoid + BinaryCrossEntropy
# L = -[y * log(σ(x)) + (1-y) * log(1 - σ(x))]
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ============================================================
# 4. 学習ループ
# ============================================================

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_X, batch_y in train_loader:
        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)

    avg_loss = total_loss / n_train
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] Loss: {avg_loss:.4f}")

# ============================================================
# 5. 評価
# ============================================================

model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_preds = (torch.sigmoid(test_logits) > 0.5).float()
    accuracy = (test_preds == y_test).float().mean()
    print(f"\nテスト精度: {accuracy.item():.2%}")

    # 混同行列の簡易表示
    tp = ((test_preds == 1) & (y_test == 1)).sum().item()
    fp = ((test_preds == 1) & (y_test == 0)).sum().item()
    fn = ((test_preds == 0) & (y_test == 1)).sum().item()
    tn = ((test_preds == 0) & (y_test == 0)).sum().item()

    print(f"混同行列: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"Precision: {precision:.2%}")
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"Recall: {recall:.2%}")

print("\n=== 06_mlp_binary_classification.py 完了 ===")
