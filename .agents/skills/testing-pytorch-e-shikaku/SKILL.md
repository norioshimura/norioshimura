---
name: testing-pytorch-e-shikaku
description: Test the PyTorch E資格 learning repository. Use when verifying that learning scripts run correctly and tests pass.
---

# Testing PyTorch E資格 Learning Repository

## Prerequisites

- Python 3.10+
- PyTorch (CPU version is sufficient): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- pytest: `pip install pytest`

## Quick Verification

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v
# Expected: 37 passed

# Run all scripts independently
for f in src/0*.py; do python "$f"; done
# Each script should end with "=== 0X_*.py 完了 ==="
```

## Adversarial Assertions

When testing changes to the scripts, verify these concrete values:

| Script | Assertion | Expected |
|--------|-----------|----------|
| 02_autograd | dy/dx for y=x²+3x+1 at x=2 | 7.0 |
| 02_autograd | ∂z/∂x for z=x²y+y³ at (1,2) | 4.0 |
| 02_autograd | ∂z/∂y for z=x²y+y³ at (1,2) | 13.0 |
| 04_cross_entropy | Manual CE == PyTorch CE | `一致確認: True` |
| 04_cross_entropy | Batch prediction accuracy | 100.00% |
| 05_kl_divergence | KL(P ‖ P) | 0.0000 |
| 05_kl_divergence | KL(P‖Q) ≠ KL(Q‖P) | Non-symmetric |
| 06_mlp | Test accuracy on separable data | ≥ 80% |
| 09_train_eval | eval() mode output determinism | Two identical sums |
| 09_train_eval | no_grad requires_grad | False |

## Notes

- All testing is shell-based (no GUI recording needed)
- CNN script (07) uses dummy data, so test accuracy will be near random (~10%) — this is expected
- Scripts have no cross-dependencies; each can be tested in isolation
- MLP test accuracy may vary by seed but should consistently exceed 80% on the separable 2-class data
