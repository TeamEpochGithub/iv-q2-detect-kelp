
# Model configuration

## Model

## Preprocessing


### Feature transformation

### Feature columns




## Pretrain


### Steps

#### GBDT

Here for possible sweeping, make sure that the ? is replaced with a number, that aligns with the stepnumber in the pretrain pipeline.

model_loop_pipeline.pretrain_pipeline.pretrain_steps.?.max_images: 500 (0 - 5635)
model_loop_pipeline.pretrain_pipeline.pretrain_steps.?.type: Catboost [Catboost, LightGBM, XGBoost]
model_loop_pipeline.pretrain_pipeline.pretrain_steps.?.early_stopping_split: 0.2 (0 - 1)

```python
  - _target_: src.pipeline.model.model_loop.pretrain.gbdt.GBDT
    type: XGBoost
    max_images: 1000
    early_stopping_split: 0.2
```

### Loss functions

```python
import src.modules.loss.focal_tversky_loss

criterion:
_target_: src.modules.loss.dice_loss.DiceLoss

criterion:
_target_: torch.nn.BCELoss

criterion:
_target_: torch.nn.WeightedBCELoss
positive_freq: 0.0067

criterion:
_target_: src.modules.loss.dice_bce_loss.DiceBCELoss

criterion:
_target_: src.modules.loss.shrinkage_loss.ShrinkageLoss

criterion:
_target_: src.modules.loss.focal_loss.FocalLoss
alpha: ??? (0.25)
gamma: ??? (2.0)

criterion:
_target_: src.modules.loss.jaccard_loss.JaccardLoss

criterion:
_target_: src.modules.loss.tversky_loss.TverskyLoss
alpha: ??? (0.5)
beta: ??? (0.5)

criterion:
_target_: src.modules.loss.focal_tversky_loss.FocalTverskyLoss
alpha: ??? (0.5)
beta: ??? (0.5)
gamma: ??? (1.0)

criterion:
_target_: src.modules.loss.lovasz_hinge_loss.LovaszHingeLoss



```
