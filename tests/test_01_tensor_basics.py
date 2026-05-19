"""01_tensor_basics.py のテスト"""

import torch
import numpy as np


def test_tensor_creation():
    """Tensorの生成が正しく動作するか"""
    a = torch.tensor([1, 2, 3])
    assert a.shape == (3,)
    assert a.dtype == torch.int64

    zeros = torch.zeros(2, 3)
    assert zeros.shape == (2, 3)
    assert zeros.sum().item() == 0.0

    ones = torch.ones(2, 3)
    assert ones.sum().item() == 6.0


def test_tensor_reshape():
    """形状操作が正しく動作するか"""
    x = torch.arange(12)
    reshaped = x.reshape(3, 4)
    assert reshaped.shape == (3, 4)

    viewed = x.view(4, 3)
    assert viewed.shape == (4, 3)


def test_unsqueeze_squeeze():
    """次元の追加・削除"""
    y = torch.tensor([1, 2, 3])
    assert y.unsqueeze(0).shape == (1, 3)
    assert y.unsqueeze(1).shape == (3, 1)

    y2 = torch.zeros(1, 3, 1)
    assert y2.squeeze().shape == (3,)


def test_arithmetic():
    """基本演算"""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    assert torch.allclose(a + b, torch.tensor([5.0, 7.0, 9.0]))
    assert torch.allclose(a * b, torch.tensor([4.0, 10.0, 18.0]))


def test_matmul():
    """行列積"""
    A = torch.ones(2, 3)
    B = torch.ones(3, 4)
    C = A @ B
    assert C.shape == (2, 4)
    assert C[0, 0].item() == 3.0


def test_broadcast():
    """ブロードキャスト"""
    x = torch.ones(3, 4)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = x + bias
    assert result.shape == (3, 4)
    assert result[0, 0].item() == 2.0
    assert result[0, 3].item() == 5.0


def test_numpy_conversion():
    """NumPy変換"""
    np_arr = np.array([1.0, 2.0, 3.0])
    tensor = torch.from_numpy(np_arr)
    assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    back_to_np = tensor.numpy()
    assert np.allclose(back_to_np, np_arr)
