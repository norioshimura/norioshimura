"""
01. Tensor基礎
==============
PyTorchの基本データ構造であるTensorの生成・操作を学ぶ。

E資格ポイント:
- TensorはNumpy配列に似た多次元配列で、GPU計算が可能
- dtype, device, shape（size）の概念を理解する
- ブロードキャスト演算を理解する
"""

import torch

# ============================================================
# 1. Tensorの生成
# ============================================================

# リストから生成
a = torch.tensor([1, 2, 3])
print(f"リストから生成: {a}, dtype={a.dtype}")

# ゼロ・イチ・乱数で生成
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
rand = torch.rand(2, 3)  # 一様分布 [0, 1)
randn = torch.randn(2, 3)  # 標準正規分布

print(f"zeros:\n{zeros}")
print(f"ones:\n{ones}")
print(f"rand:\n{rand}")
print(f"randn:\n{randn}")

# dtypeの指定
float_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
int_tensor = torch.tensor([1, 2], dtype=torch.int64)
print(f"float32: {float_tensor.dtype}, int64: {int_tensor.dtype}")

# ============================================================
# 2. Tensorの形状操作
# ============================================================

x = torch.arange(12)  # 0〜11
print(f"\n元のTensor: {x}, shape={x.shape}")

# reshape: 形状を変更（メモリ連続でなくても可）
reshaped = x.reshape(3, 4)
print(f"reshape(3,4):\n{reshaped}")

# view: 形状を変更（メモリ連続が必要）
viewed = x.view(4, 3)
print(f"view(4,3):\n{viewed}")

# unsqueeze / squeeze: 次元を追加・削除
y = torch.tensor([1, 2, 3])
print(f"\nunsqueeze(0): {y.unsqueeze(0).shape}")  # (1, 3)
print(f"unsqueeze(1): {y.unsqueeze(1).shape}")  # (3, 1)

y2 = torch.zeros(1, 3, 1)
print(f"squeeze前: {y2.shape}, squeeze後: {y2.squeeze().shape}")  # (3,)

# ============================================================
# 3. インデクシングとスライシング
# ============================================================

m = torch.arange(12).reshape(3, 4)
print(f"\n行列:\n{m}")
print(f"m[0]: {m[0]}")         # 0行目
print(f"m[:, 1]: {m[:, 1]}")   # 1列目
print(f"m[1, 2]: {m[1, 2]}")   # 1行2列の要素

# boolインデクシング
mask = m > 5
print(f"m > 5 の要素: {m[mask]}")

# ============================================================
# 4. 演算
# ============================================================

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 要素ごとの演算
print(f"\na + b = {a + b}")
print(f"a * b = {a * b}")
print(f"a ** 2 = {a ** 2}")

# 行列積
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = A @ B  # torch.matmul(A, B) と同じ
print(f"\n行列積 A(2x3) @ B(3x4) -> C{C.shape}")

# ============================================================
# 5. ブロードキャスト
# ============================================================
# 形状の異なるTensor間で自動的に形状を合わせて演算する仕組み

x = torch.ones(3, 4)
bias = torch.tensor([1.0, 2.0, 3.0, 4.0])  # (4,)
result = x + bias  # biasが(1,4)→(3,4)に拡張される
print(f"\nブロードキャスト: ones(3,4) + bias(4,) =\n{result}")

# ============================================================
# 6. NumPyとの相互変換
# ============================================================
import numpy as np

np_arr = np.array([1.0, 2.0, 3.0])
tensor_from_np = torch.from_numpy(np_arr)
np_from_tensor = tensor_from_np.numpy()

print(f"\nNumPy -> Tensor: {tensor_from_np}")
print(f"Tensor -> NumPy: {np_from_tensor}")

# 注意: メモリを共有するため、片方を変更するともう片方も変わる
np_arr[0] = 100
print(f"NumPy変更後のTensor: {tensor_from_np}")

print("\n=== 01_tensor_basics.py 完了 ===")
