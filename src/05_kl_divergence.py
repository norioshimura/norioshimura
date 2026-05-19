"""
05. KLダイバージェンス
=====================
KLダイバージェンス（Kullback-Leibler Divergence）を学ぶ。

E資格ポイント:
- KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
- 2つの確率分布P, Qの「距離」を測る（非対称）
- KL(P||Q) ≠ KL(Q||P)（対称ではない）
- KL(P||Q) ≥ 0（ギブスの不等式）
- VAE（変分オートエンコーダ）の損失関数で使用
- 知識蒸留（Knowledge Distillation）でも使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. KLダイバージェンスの手動計算
# ============================================================

# 2つの確率分布
P = torch.tensor([0.4, 0.3, 0.2, 0.1])  # 真の分布
Q = torch.tensor([0.25, 0.25, 0.25, 0.25])  # 近似分布（一様分布）

# 手動計算: KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
kl_manual = (P * torch.log(P / Q)).sum()
print(f"手動計算 KL(P || Q) = {kl_manual.item():.4f}")

# ============================================================
# 2. PyTorchのKLDivLoss
# ============================================================
# 注意: PyTorchのKLDivLossは入力にlog確率を期待する

kl_loss = nn.KLDivLoss(reduction='sum')

# 入力: Q のlog確率、ターゲット: P の確率
loss_pytorch = kl_loss(torch.log(Q), P)
print(f"PyTorch KL(P || Q) = {loss_pytorch.item():.4f}")
print(f"一致確認: {torch.allclose(kl_manual, loss_pytorch)}")

# ============================================================
# 3. KLダイバージェンスの非対称性
# ============================================================

kl_pq = (P * torch.log(P / Q)).sum()
kl_qp = (Q * torch.log(Q / P)).sum()

print(f"\nKL(P || Q) = {kl_pq.item():.4f}")
print(f"KL(Q || P) = {kl_qp.item():.4f}")
print(f"非対称: KL(P||Q) ≠ KL(Q||P)")

# ============================================================
# 4. 同一分布のKLダイバージェンス
# ============================================================

kl_pp = (P * torch.log(P / P)).sum()
print(f"\nKL(P || P) = {kl_pp.item():.4f}")  # 0.0

# ============================================================
# 5. 知識蒸留での活用例
# ============================================================
# 教師モデルと生徒モデルの出力分布を近づける

temperature = 2.0

# 教師モデルと生徒モデルのlogits
teacher_logits = torch.tensor([[3.0, 1.0, 0.1]])
student_logits = torch.tensor([[2.5, 1.2, 0.3]])

# ソフトターゲット（温度付きsoftmax）
teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
student_soft = F.log_softmax(student_logits / temperature, dim=1)

# 蒸留損失
distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
print(f"\n知識蒸留のKL Loss (T={temperature}): {distill_loss.item():.4f}")

# ============================================================
# 6. VAEでのKLダイバージェンス
# ============================================================
# VAEでは潜在変数の事後分布q(z|x)を標準正規分布N(0,1)に近づける
# KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ^2) - μ^2 - σ^2)

mu = torch.tensor([0.5, -0.3, 0.8])
log_var = torch.tensor([-0.2, 0.1, -0.5])

kl_vae = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
print(f"\nVAEのKL Loss: {kl_vae.item():.4f}")

print("\n=== 05_kl_divergence.py 完了 ===")
