"""04_cross_entropy_loss.py のテスト"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_softmax_sums_to_one():
    """softmaxの出力が合計1になるか"""
    logits = torch.tensor([2.0, 1.0, 0.1])
    probs = F.softmax(logits, dim=0)
    assert torch.isclose(probs.sum(), torch.tensor(1.0))


def test_cross_entropy_manual_vs_pytorch():
    """手動計算とPyTorchのCELossが一致するか"""
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    target = torch.tensor([0])

    manual_loss = -F.log_softmax(logits, dim=1)[0, target[0]]
    pytorch_loss = nn.CrossEntropyLoss()(logits, target)

    assert torch.allclose(manual_loss, pytorch_loss)


def test_cross_entropy_perfect_prediction():
    """完璧な予測のlossが低いか"""
    logits_good = torch.tensor([[10.0, 0.0, 0.0]])
    logits_bad = torch.tensor([[0.0, 0.0, 10.0]])
    target = torch.tensor([0])

    criterion = nn.CrossEntropyLoss()
    loss_good = criterion(logits_good, target)
    loss_bad = criterion(logits_bad, target)

    assert loss_good < loss_bad


def test_label_smoothing():
    """ラベルスムージングのlossが通常より大きいか"""
    logits = torch.tensor([[3.0, 1.0, 0.1]])
    target = torch.tensor([0])

    loss_normal = nn.CrossEntropyLoss()(logits, target)
    loss_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, target)

    assert loss_smooth > loss_normal
