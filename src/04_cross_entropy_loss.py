"""
04. CrossEntropyLoss
====================
多クラス分類で使われるCrossEntropyLossの仕組みを学ぶ。

E資格ポイント:
- CrossEntropyLoss = LogSoftmax + NLLLoss
- 入力はlogits（softmax前の生の出力）、ターゲットはクラスインデックス
- 数式: L = -Σ y_i * log(softmax(x_i))
       = -log(exp(x_c) / Σ_j exp(x_j))  （cは正解クラス）
- softmaxの数値安定性のためlog_softmaxを使う
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. Softmax関数
# ============================================================
# softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
# 出力を確率分布に変換する

logits = torch.tensor([2.0, 1.0, 0.1])
softmax_probs = F.softmax(logits, dim=0)
print(f"Logits: {logits}")
print(f"Softmax: {softmax_probs}")
print(f"合計: {softmax_probs.sum():.4f}")  # 1.0になる

# ============================================================
# 2. CrossEntropyLossの手動計算
# ============================================================

logits = torch.tensor([[2.0, 1.0, 0.1]])  # バッチサイズ1、3クラス
target = torch.tensor([0])  # 正解はクラス0

# 手動計算
log_softmax = F.log_softmax(logits, dim=1)
manual_loss = -log_softmax[0, target[0]]
print(f"\n手動計算のCE Loss: {manual_loss.item():.4f}")

# PyTorchのCrossEntropyLoss
criterion = nn.CrossEntropyLoss()
pytorch_loss = criterion(logits, target)
print(f"PyTorchのCE Loss:  {pytorch_loss.item():.4f}")
print(f"一致確認: {torch.allclose(manual_loss, pytorch_loss)}")

# ============================================================
# 3. バッチでの計算
# ============================================================

batch_logits = torch.tensor([
    [2.0, 1.0, 0.1],  # サンプル1
    [0.5, 2.5, 0.3],  # サンプル2
    [0.1, 0.3, 3.0],  # サンプル3
])
batch_targets = torch.tensor([0, 1, 2])  # 各サンプルの正解クラス

loss = criterion(batch_logits, batch_targets)
print(f"\nバッチのCE Loss: {loss.item():.4f}")

# 予測の確認
predictions = torch.argmax(batch_logits, dim=1)
print(f"予測: {predictions.tolist()}, 正解: {batch_targets.tolist()}")
accuracy = (predictions == batch_targets).float().mean()
print(f"正解率: {accuracy.item():.2%}")

# ============================================================
# 4. ラベルスムージング
# ============================================================
# 過学習を防ぐテクニック。正解ラベルを1.0ではなく
# 1 - ε にして、残りを他クラスに分配

criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
loss_smooth = criterion_smooth(batch_logits, batch_targets)
print(f"\nラベルスムージング(ε=0.1)後のLoss: {loss_smooth.item():.4f}")
print(f"通常のLoss:                        {loss.item():.4f}")

# ============================================================
# 5. 重み付きCrossEntropyLoss
# ============================================================
# クラス不均衡がある場合に使用

weights = torch.tensor([1.0, 2.0, 1.0])  # クラス1を重視
criterion_weighted = nn.CrossEntropyLoss(weight=weights)
loss_weighted = criterion_weighted(batch_logits, batch_targets)
print(f"\n重み付きLoss: {loss_weighted.item():.4f}")

print("\n=== 04_cross_entropy_loss.py 完了 ===")
