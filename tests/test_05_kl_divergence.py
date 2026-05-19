"""05_kl_divergence.py のテスト"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_kl_manual_matches_pytorch():
    """手動計算とPyTorchのKLDivLossが一致するか"""
    P = torch.tensor([0.4, 0.3, 0.2, 0.1])
    Q = torch.tensor([0.25, 0.25, 0.25, 0.25])

    kl_manual = (P * torch.log(P / Q)).sum()
    kl_pytorch = nn.KLDivLoss(reduction='sum')(torch.log(Q), P)

    assert torch.allclose(kl_manual, kl_pytorch, atol=1e-6)


def test_kl_non_negative():
    """KLダイバージェンスが非負か"""
    P = torch.tensor([0.5, 0.3, 0.2])
    Q = torch.tensor([0.3, 0.4, 0.3])

    kl = (P * torch.log(P / Q)).sum()
    assert kl.item() >= 0


def test_kl_same_distribution():
    """同一分布のKLが0か"""
    P = torch.tensor([0.5, 0.3, 0.2])
    kl = (P * torch.log(P / P)).sum()
    assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)


def test_kl_asymmetric():
    """KLダイバージェンスが非対称か"""
    P = torch.tensor([0.4, 0.3, 0.2, 0.1])
    Q = torch.tensor([0.25, 0.25, 0.25, 0.25])

    kl_pq = (P * torch.log(P / Q)).sum()
    kl_qp = (Q * torch.log(Q / P)).sum()

    assert not torch.isclose(kl_pq, kl_qp)


def test_vae_kl_loss():
    """VAEのKL Lossが計算できるか"""
    mu = torch.zeros(5)
    log_var = torch.zeros(5)
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)
