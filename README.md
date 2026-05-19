# PyTorch E資格対策 学習リポジトリ

JDLA E資格（エンジニア資格）で必要なPyTorch実装力を身につけるための学習リポジトリです。

## 学習手順

以下の順番でコードを読み・実行し、PyTorchの基礎から応用までを段階的に学んでください。

| # | ファイル | テーマ | 前提知識 |
|---|---------|--------|---------|
| 1 | `src/01_tensor_basics.py` | Tensor基礎 | Python, NumPy |
| 2 | `src/02_autograd_basics.py` | Autograd基礎 | 微分の基本 |
| 3 | `src/03_training_loop.py` | 学習ループ | 01, 02 |
| 4 | `src/04_cross_entropy_loss.py` | CrossEntropyLoss | 03, 確率の基本 |
| 5 | `src/05_kl_divergence.py` | KLダイバージェンス | 04, 情報理論の基本 |
| 6 | `src/06_mlp_binary_classification.py` | MLPによる2値分類 | 03, 04 |
| 7 | `src/07_cnn_image_classification.py` | CNNによる画像分類 | 06 |
| 8 | `src/08_dropout_batchnorm.py` | DropoutとBatchNormの比較 | 06 |
| 9 | `src/09_train_eval_nograd.py` | train()/eval()/no_grad()の違い | 08 |

## 実行方法

```bash
# 必要なパッケージのインストール
pip install torch torchvision numpy pytest

# 各コードの実行（例: Tensor基礎）
python src/01_tensor_basics.py

# テストの実行
pytest tests/ -v
```

## E資格 数式・PyTorch API・ポイント対応表

### 1. Tensor基礎

| 数式・概念 | PyTorch API | E資格ポイント |
|-----------|-------------|-------------|
| テンソルの生成 | `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.rand()` | 多次元配列の生成方法を理解する |
| 形状変換 | `reshape()`, `view()`, `unsqueeze()`, `squeeze()` | `view`はメモリ連続が必要、`reshape`は不要 |
| 行列積 $C = AB$ | `torch.matmul()`, `@` 演算子 | 行列の次元の対応関係を理解する |
| ブロードキャスト | 自動適用 | 形状の異なるテンソル間の演算規則 |

### 2. Autograd（自動微分）

| 数式・概念 | PyTorch API | E資格ポイント |
|-----------|-------------|-------------|
| $\frac{\partial y}{\partial x}$ | `y.backward()`, `x.grad` | 計算グラフから自動的に勾配を計算 |
| 勾配の蓄積 | `optimizer.zero_grad()` | PyTorchは勾配を蓄積するため初期化が必要 |
| 勾配計算の無効化 | `torch.no_grad()`, `detach()` | 推論時のメモリ効率・速度向上 |
| ヤコビアン・ベクトル積 | `backward(gradient=v)` | 非スカラー出力の逆伝播 |

### 3. 学習ループ

| 数式・概念 | PyTorch API | E資格ポイント |
|-----------|-------------|-------------|
| 順伝播 $\hat{y} = f(x; \theta)$ | `model(x)` | `nn.Module`の`forward()`が呼ばれる |
| 損失計算 $L = \frac{1}{N}\sum(y - \hat{y})^2$ | `nn.MSELoss()` | 回帰: MSE、分類: CrossEntropy |
| 逆伝播 $\frac{\partial L}{\partial \theta}$ | `loss.backward()` | 連鎖律で全パラメータの勾配を計算 |
| パラメータ更新 $\theta \leftarrow \theta - \eta \nabla L$ | `optimizer.step()` | SGD, Adam等の更新則の違い |

### 4. CrossEntropyLoss

| 数式・概念 | PyTorch API | E資格ポイント |
|-----------|-------------|-------------|
| $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | `F.softmax(x, dim)` | 出力を確率分布に変換 |
| $L = -\log\frac{e^{x_c}}{\sum_j e^{x_j}}$ | `nn.CrossEntropyLoss()` | LogSoftmax + NLLLoss を内部で実行 |
| ラベルスムージング | `CrossEntropyLoss(label_smoothing=ε)` | 正解を $1-\varepsilon$ にし残りを分配 |
| クラス重み付け | `CrossEntropyLoss(weight=w)` | クラス不均衡対策 |

### 5. KLダイバージェンス

| 数式・概念 | PyTorch API | E資格ポイント |
|-----------|-------------|-------------|
| $D_{KL}(P \|\| Q) = \sum P(x) \log\frac{P(x)}{Q(x)}$ | `nn.KLDivLoss()` | 入力はlog確率、非対称 ($D_{KL}(P \|\| Q) \neq D_{KL}(Q \|\| P)$) |
| $D_{KL} \geq 0$ | - | ギブスの不等式（等号は $P = Q$ のとき） |
| VAEのKL: $-\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$ | 手動計算 | 事後分布を標準正規分布に近づける |
| 知識蒸留 | `F.kl_div(student, teacher)` | 温度付きsoftmaxでソフトターゲットを生成 |

### 6. MLP（多層パーセプトロン）

| 数式・概念 | PyTorch API | E資格ポイント |
|-----------|-------------|-------------|
| 全結合層 $y = Wx + b$ | `nn.Linear(in, out)` | パラメータ数 = in × out + out |
| ReLU $f(x) = \max(0, x)$ | `nn.ReLU()` | 勾配消失問題の緩和、計算が高速 |
| Sigmoid $\sigma(x) = \frac{1}{1+e^{-x}}$ | `torch.sigmoid()` | 出力を[0,1]に変換（2値分類の出力層） |
| BCE Loss $-[y\log p + (1-y)\log(1-p)]$ | `nn.BCEWithLogitsLoss()` | Sigmoid + BCEを統合（数値安定） |

### 7. CNN（畳み込みニューラルネットワーク）

| 数式・概念 | PyTorch API | E資格ポイント |
|-----------|-------------|-------------|
| 畳み込み演算 | `nn.Conv2d(in_ch, out_ch, kernel_size)` | 出力サイズ: $H_{out} = \lfloor\frac{H_{in} + 2p - k}{s}\rfloor + 1$ |
| プーリング | `nn.MaxPool2d(kernel_size)` | 特徴マップのダウンサンプリング |
| パラメータ数 | - | Conv2d: $k^2 \times C_{in} \times C_{out} + C_{out}$ |
| Flatten | `x.view(x.size(0), -1)` | 畳み込み出力を全結合層に接続 |

### 8. Dropout / BatchNorm

| 数式・概念 | PyTorch API | E資格ポイント |
|-----------|-------------|-------------|
| Dropout | `nn.Dropout(p)` | 学習時: 確率pで無効化 + $\frac{1}{1-p}$倍スケール |
| BatchNorm $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$ | `nn.BatchNorm1d(n)`, `nn.BatchNorm2d(n)` | 学習時: バッチ統計、推論時: 移動平均 |
| アフィン変換 $y = \gamma\hat{x} + \beta$ | BN内部パラメータ | $\gamma, \beta$は学習可能 |

### 9. train() / eval() / no_grad()

| 概念 | PyTorch API | E資格ポイント |
|------|-------------|-------------|
| 学習モード | `model.train()` | Dropout有効、BNはバッチ統計使用 |
| 推論モード | `model.eval()` | Dropout無効、BNは移動平均使用 |
| 勾配無効化 | `torch.no_grad()` | メモリ節約・速度向上（推論時は必須） |
| 推論のベストプラクティス | `model.eval()` + `torch.no_grad()` | 両方を併用する（独立した概念） |

## E資格頻出キーワード一覧

| カテゴリ | キーワード |
|---------|----------|
| 損失関数 | MSELoss, CrossEntropyLoss, BCELoss, KLDivLoss, NLLLoss |
| オプティマイザ | SGD, Adam, AdaGrad, RMSprop, 学習率スケジューラ |
| 活性化関数 | ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, GELU |
| 正則化 | Dropout, BatchNorm, LayerNorm, Weight Decay (L2正則化) |
| CNN関連 | Conv2d, MaxPool2d, AvgPool2d, stride, padding, dilation |
| データ処理 | DataLoader, Dataset, transforms, バッチサイズ |
| モデル管理 | train(), eval(), no_grad(), state_dict(), save/load |

## テスト

```bash
# 全テスト実行
pytest tests/ -v

# 特定のテストファイルを実行
pytest tests/test_01_tensor_basics.py -v
```

## ディレクトリ構成

```
.
├── README.md                          # 本ファイル（学習ガイド）
├── requirements.txt                   # 依存パッケージ
├── src/
│   ├── 01_tensor_basics.py           # Tensor基礎
│   ├── 02_autograd_basics.py         # Autograd基礎
│   ├── 03_training_loop.py           # 学習ループ
│   ├── 04_cross_entropy_loss.py      # CrossEntropyLoss
│   ├── 05_kl_divergence.py           # KLダイバージェンス
│   ├── 06_mlp_binary_classification.py # MLPによる2値分類
│   ├── 07_cnn_image_classification.py  # CNNによる画像分類
│   ├── 08_dropout_batchnorm.py       # DropoutとBatchNormの比較
│   └── 09_train_eval_nograd.py       # train()/eval()/no_grad()の違い
└── tests/
    ├── test_01_tensor_basics.py
    ├── test_02_autograd.py
    ├── test_03_training_loop.py
    ├── test_04_cross_entropy.py
    ├── test_05_kl_divergence.py
    ├── test_06_mlp.py
    ├── test_07_cnn.py
    ├── test_08_dropout_batchnorm.py
    └── test_09_train_eval.py
```

## 参考リンク

- [PyTorch公式ドキュメント](https://pytorch.org/docs/stable/)
- [PyTorch公式チュートリアル](https://pytorch.org/tutorials/)
- [JDLA E資格](https://www.jdla.org/certificate/engineer/)
