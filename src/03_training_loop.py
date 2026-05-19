"""
03. 学習ループ
==============
PyTorchの基本的な学習ループ（forward → loss → backward → update）を学ぶ。

E資格ポイント:
- 学習ループの4ステップ: 順伝播→損失計算→逆伝播→パラメータ更新
- optimizer.zero_grad() → loss.backward() → optimizer.step() の順序
- SGD, Adam等のオプティマイザの違い
- 学習率（learning rate）の影響
"""

import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 1. 問題設定: y = 2x + 1 の線形回帰
# ============================================================

torch.manual_seed(42)

# 訓練データの生成
x_train = torch.linspace(0, 1, 100).unsqueeze(1)  # (100, 1)
y_train = 2 * x_train + 1 + 0.1 * torch.randn(100, 1)  # ノイズ付き

# ============================================================
# 2. モデルの定義
# ============================================================

model = nn.Linear(1, 1)  # 1入力 → 1出力の線形層
print(f"初期パラメータ: weight={model.weight.item():.4f}, bias={model.bias.item():.4f}")

# ============================================================
# 3. 損失関数とオプティマイザ
# ============================================================

criterion = nn.MSELoss()  # 平均二乗誤差: L = (1/N) Σ(y_pred - y)^2
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 確率的勾配降下法

# ============================================================
# 4. 学習ループ
# ============================================================

num_epochs = 100

for epoch in range(num_epochs):
    # ステップ1: 順伝播（forward pass）
    y_pred = model(x_train)

    # ステップ2: 損失の計算
    loss = criterion(y_pred, y_train)

    # ステップ3: 勾配の初期化 → 逆伝播
    optimizer.zero_grad()  # 勾配をゼロに初期化（蓄積防止）
    loss.backward()        # 逆伝播で勾配を計算

    # ステップ4: パラメータの更新
    optimizer.step()       # 勾配に基づいてパラメータを更新

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] Loss: {loss.item():.4f}")

# ============================================================
# 5. 学習結果の確認
# ============================================================

print(f"\n学習後: weight={model.weight.item():.4f}, bias={model.bias.item():.4f}")
print(f"期待値: weight≈2.0, bias≈1.0")

# ============================================================
# 6. オプティマイザの比較
# ============================================================

print("\n--- オプティマイザの比較 ---")

for opt_name, opt_class, lr in [("SGD", optim.SGD, 0.1),
                                  ("Adam", optim.Adam, 0.01)]:
    model2 = nn.Linear(1, 1)
    torch.manual_seed(0)
    optimizer2 = opt_class(model2.parameters(), lr=lr)
    criterion2 = nn.MSELoss()

    for epoch in range(100):
        y_pred = model2(x_train)
        loss = criterion2(y_pred, y_train)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

    print(f"{opt_name:5s}: weight={model2.weight.item():.4f}, "
          f"bias={model2.bias.item():.4f}, final_loss={loss.item():.4f}")

print("\n=== 03_training_loop.py 完了 ===")
