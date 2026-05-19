"""
08. DropoutとBatchNormの比較
============================
正則化・正規化テクニックであるDropoutとBatchNormalizationを比較する。

E資格ポイント:
- Dropout: 学習時にランダムにユニットを無効化（過学習防止）
  - 学習時: 確率pでユニットを0にし、残りを1/(1-p)倍（逆Dropout）
  - 推論時: 全ユニットを使用（スケーリング済み）
- BatchNorm: ミニバッチ内で正規化（学習の安定化・高速化）
  - x̂ = (x - μ_B) / √(σ_B² + ε)
  - y = γx̂ + β （学習可能なパラメータ）
  - 学習時: バッチ統計を使用、推論時: 移動平均を使用
"""

import torch
import torch.nn as nn

# ============================================================
# 1. Dropoutの動作
# ============================================================

print("=== Dropout ===")

dropout = nn.Dropout(p=0.5)  # 50%の確率でドロップ

x = torch.ones(1, 10)

# 学習モードと推論モードで動作が異なる
dropout.train()
print(f"学習モード（複数回実行で結果が変わる）:")
for i in range(3):
    out = dropout(x)
    print(f"  試行{i+1}: {out}")

dropout.eval()
out_eval = dropout(x)
print(f"推論モード（全て通過）: {out_eval}")

# ============================================================
# 2. BatchNormの動作
# ============================================================

print("\n=== BatchNorm ===")

bn = nn.BatchNorm1d(4)  # 4次元の特徴量

# バッチサイズ3、特徴量4のデータ
x = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 4.0, 5.0],
    [3.0, 4.0, 5.0, 6.0],
])

bn.train()
out_train = bn(x)
print(f"入力:\n{x}")
print(f"BatchNorm後（学習モード）:\n{out_train}")
print(f"γ (weight): {bn.weight.data}")
print(f"β (bias): {bn.bias.data}")
print(f"移動平均μ: {bn.running_mean}")
print(f"移動平均σ²: {bn.running_var}")

bn.eval()
out_eval = bn(x)
print(f"\nBatchNorm後（推論モード）:\n{out_eval}")
print("（推論時は移動平均を使うため、学習時と値が異なる）")

# ============================================================
# 3. モデルでの比較
# ============================================================

print("\n=== モデルでの比較 ===")


class ModelWithoutRegularization(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelWithDropout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelWithBatchNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelWithBoth(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 4. 学習と比較
# ============================================================

torch.manual_seed(42)

# ダミーデータ
n_samples = 300
X = torch.randn(n_samples, 10)
y = torch.randint(0, 2, (n_samples,))
X_test = torch.randn(100, 10)
y_test = torch.randint(0, 2, (100,))

models = {
    "正則化なし": ModelWithoutRegularization(),
    "Dropout": ModelWithDropout(),
    "BatchNorm": ModelWithBatchNorm(),
    "両方": ModelWithBoth(),
}

criterion = nn.CrossEntropyLoss()

print(f"\n{'モデル':<12} {'最終Loss':>10} {'Train Acc':>10} {'Test Acc':>10}")
print("-" * 46)

for name, model in models.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 学習
    model.train()
    for epoch in range(50):
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 評価
    model.eval()
    with torch.no_grad():
        train_preds = model(X).argmax(dim=1)
        train_acc = (train_preds == y).float().mean()

        test_preds = model(X_test).argmax(dim=1)
        test_acc = (test_preds == y_test).float().mean()

    print(f"{name:<12} {loss.item():>10.4f} {train_acc.item():>10.2%} {test_acc.item():>10.2%}")

print("（ダミーデータのため、正則化の効果は限定的です）")

print("\n=== 08_dropout_batchnorm.py 完了 ===")
