"""
09. train() / eval() / torch.no_grad() の違い
==============================================
モデルの学習モードと推論モードの違いを理解する。

E資格ポイント:
- model.train(): 学習モード（Dropout有効、BatchNormはバッチ統計使用）
- model.eval(): 推論モード（Dropout無効、BatchNormは移動平均使用）
- torch.no_grad(): 勾配計算を無効化（メモリ節約・速度向上）
- eval()とno_grad()は独立した概念（両方使うのが推論時のベストプラクティス）
"""

import torch
import torch.nn as nn

# ============================================================
# 1. train()とeval()の違い
# ============================================================

print("=== train() vs eval() ===")


class DemoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.bn = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


model = DemoModel()

# 同じ入力で比較
torch.manual_seed(0)
x = torch.randn(5, 10)

# 学習モード
model.train()
print(f"model.training = {model.training}")
out_train1 = model(x)
out_train2 = model(x)
print(f"学習モード出力1の合計: {out_train1.sum().item():.4f}")
print(f"学習モード出力2の合計: {out_train2.sum().item():.4f}")
print("（Dropoutにより毎回異なる出力）")

# 推論モード
model.eval()
print(f"\nmodel.training = {model.training}")
out_eval1 = model(x)
out_eval2 = model(x)
print(f"推論モード出力1の合計: {out_eval1.sum().item():.4f}")
print(f"推論モード出力2の合計: {out_eval2.sum().item():.4f}")
print("（Dropout無効のため同じ出力）")

# ============================================================
# 2. torch.no_grad()の効果
# ============================================================

print("\n=== torch.no_grad() ===")

x = torch.randn(3, 10, requires_grad=True)

# no_gradなし
y1 = model(x)
print(f"no_gradなし: requires_grad={y1.requires_grad}, grad_fn={y1.grad_fn is not None}")

# no_gradあり
with torch.no_grad():
    y2 = model(x)
    print(f"no_gradあり: requires_grad={y2.requires_grad}, grad_fn={y2.grad_fn is not None}")

# ============================================================
# 3. メモリ効率の比較
# ============================================================

print("\n=== メモリ効率 ===")

large_model = nn.Sequential(
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 10),
)

x = torch.randn(100, 1000)

# 勾配ありの場合
large_model.train()
y = large_model(x)
loss = y.sum()
loss.backward()

grad_params = sum(
    p.grad.numel() for p in large_model.parameters() if p.grad is not None
)
print(f"勾配を持つパラメータの要素数: {grad_params:,}")

# 勾配なしの場合（推論時の推奨パターン）
large_model.eval()
with torch.no_grad():
    y = large_model(x)
    # この場合、勾配は計算されないためメモリ効率が良い
    print(f"出力のrequires_grad: {y.requires_grad}")  # False

# ============================================================
# 4. 推論時のベストプラクティス
# ============================================================

print("\n=== 推論のベストプラクティス ===")


def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """推論時の正しいパターン"""
    model.eval()  # Dropout/BatchNormを推論モードに
    with torch.no_grad():  # 勾配計算を無効化
        output = model(x)
    return output


model = DemoModel()

# 学習フェーズ
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x_train = torch.randn(5, 10)

for i in range(3):
    output = model(x_train)
    loss = output.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"学習ステップ{i+1}: loss={loss.item():.4f}")

# 推論フェーズ
x_test = torch.randn(5, 10)
result = predict(model, x_test)
print(f"\n推論結果の形状: {result.shape}")
print(f"推論結果のrequires_grad: {result.requires_grad}")  # False

# ============================================================
# 5. 各層の動作モードまとめ
# ============================================================

print("\n=== 各層の動作モードまとめ ===")
print(f"{'層':<20} {'train()の動作':<25} {'eval()の動作'}")
print("-" * 70)
print(f"{'Dropout':<20} {'確率pで無効化+スケール':<25} {'全ユニット使用'}")
print(f"{'BatchNorm':<20} {'バッチ統計で正規化':<25} {'移動平均で正規化'}")
print(f"{'Linear/Conv':<20} {'通常計算':<25} {'通常計算（変化なし）'}")
print(f"{'ReLU等の活性化':<20} {'通常計算':<25} {'通常計算（変化なし）'}")

print(f"\n{'torch.no_grad()':<20} {'目的: 勾配計算の無効化（メモリ節約・高速化）'}")
print(f"{'eval()':<20} {'目的: Dropout/BNの動作切替'}")
print("→ 推論時は eval() + no_grad() を併用するのがベストプラクティス")

print("\n=== 09_train_eval_nograd.py 完了 ===")
