"""08_dropout_batchnorm.py のテスト"""

import torch
import torch.nn as nn


def test_dropout_train_vs_eval():
    """Dropoutが学習・推論モードで異なる動作をするか"""
    dropout = nn.Dropout(p=0.5)
    x = torch.ones(1, 100)

    dropout.train()
    out_train = dropout(x)
    # 学習時: 一部がゼロになる
    assert out_train.sum().item() < x.sum().item() * 1.5

    dropout.eval()
    out_eval = dropout(x)
    # 推論時: 全て通過
    assert torch.allclose(out_eval, x)


def test_batchnorm_train_vs_eval():
    """BatchNormが学習・推論モードで異なる動作をするか"""
    bn = nn.BatchNorm1d(4)

    x = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
    ])

    bn.train()
    out_train = bn(x)
    # 学習時: バッチ内で正規化される（平均≈0、分散≈1）
    assert abs(out_train.mean().item()) < 0.1

    bn.eval()
    out_eval = bn(x)
    # 推論時: 移動平均を使用（学習時と異なる値）
    assert not torch.allclose(out_train, out_eval)


def test_batchnorm_running_stats():
    """BatchNormの移動平均が更新されるか"""
    bn = nn.BatchNorm1d(4)
    assert torch.allclose(bn.running_mean, torch.zeros(4))

    bn.train()
    x = torch.randn(8, 4) + 5.0
    _ = bn(x)

    # 移動平均が更新される
    assert not torch.allclose(bn.running_mean, torch.zeros(4))
