"""
02. Autograd基礎
================
PyTorchの自動微分（Autograd）の仕組みを学ぶ。

E資格ポイント:
- requires_grad=True で勾配追跡を有効にする
- backward() で逆伝播を実行し .grad に勾配が格納される
- 計算グラフは動的に構築される（Define-by-Run）
- grad_fn は各テンソルがどの演算で作られたかを保持する
"""

import torch

# ============================================================
# 1. 基本的な自動微分
# ============================================================
# y = x^2 + 3x + 1 の x=2 における微分を計算
# dy/dx = 2x + 3 = 7

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

print(f"x = {x}")
print(f"y = x^2 + 3x + 1 = {y}")
print(f"y.grad_fn = {y.grad_fn}")

# 逆伝播の実行
y.backward()
print(f"dy/dx = 2x + 3 = {x.grad}")  # 7.0

# ============================================================
# 2. 多変数の勾配
# ============================================================
# z = x^2 * y + y^3 の偏微分
# ∂z/∂x = 2xy, ∂z/∂y = x^2 + 3y^2

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

z = x ** 2 * y + y ** 3
z.backward()

print(f"\nz = x^2 * y + y^3")
print(f"∂z/∂x = 2xy = {x.grad}")     # 2*1*2 = 4.0
print(f"∂z/∂y = x^2 + 3y^2 = {y.grad}")  # 1 + 12 = 13.0

# ============================================================
# 3. ベクトルの勾配
# ============================================================
# スカラーでない出力の場合、backward()にgradient引数が必要

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # [1, 4, 9]

# ヤコビアン・ベクトル積
v = torch.tensor([1.0, 1.0, 1.0])
y.backward(gradient=v)
print(f"\ny = x^2, dy/dx = 2x = {x.grad}")  # [2, 4, 6]

# ============================================================
# 4. 勾配の蓄積と初期化
# ============================================================
# PyTorchは勾配を蓄積する（加算される）ため、
# 学習ループでは毎回 zero_grad() が必要

x = torch.tensor(3.0, requires_grad=True)

# 1回目
y1 = x ** 2
y1.backward()
print(f"\n1回目の勾配: {x.grad}")  # 6.0

# 2回目（蓄積される）
y2 = x ** 3
y2.backward()
print(f"2回目の勾配（蓄積）: {x.grad}")  # 6.0 + 27.0 = 33.0

# 初期化してやり直し
x.grad.zero_()
y3 = x ** 3
y3.backward()
print(f"初期化後の勾配: {x.grad}")  # 27.0

# ============================================================
# 5. 勾配計算の無効化
# ============================================================
# 推論時は勾配計算を無効化してメモリ・速度を最適化

x = torch.tensor(2.0, requires_grad=True)

# detach(): 計算グラフから切り離す
y = x ** 2
y_detached = y.detach()
print(f"\ndetach後: requires_grad={y_detached.requires_grad}")

# torch.no_grad(): ブロック内の勾配計算を無効化
with torch.no_grad():
    z = x ** 2
    print(f"no_grad内: requires_grad={z.requires_grad}")

print("\n=== 02_autograd_basics.py 完了 ===")
