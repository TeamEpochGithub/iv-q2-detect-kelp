
# Model configuration

## Model

## Preprocessing

### Feature transformations

```python
transformations:
# ToZero
- _target_: src.pipeline.model.feature.transformation.to_zero.SetOutsideRange
range_to_zero: [[6750, 11000], [7000, 12250], [7000, 11750], [6000, 11250], [6750, 12000], [0.1, 1.1], [-1, 5]]  # [SWIR, NIR, RED, GREEN, BLUE, CLOUD, ELEVATION]
nan_to_zero: True
nan_value: -32768
- _target_: src.pipeline.model.feature.transformation.clip.Clip
feature_ranges: [[6250, 12000], [6500, 13250], [6500, 12000], [5750, 11250], [6500, 12000], [0, 1], [0, 5]]  # [SWIR, NIR, RED, GREEN, BLUE, CLOUD, ELEVATION]
  # ToZero
- _target_: src.pipeline.model.feature.transformation.to_zero.ToZero
range_to_zero: [[6750, 11000], [7000, 12250], [7000, 11750], [6000, 11250], [6750, 12000], [0.1,1.1], [-1, 5]]  #[SWIR, NIR, RED, GREEN, BLUE, CLOUD, ELEVATION]
nan_to_zero: True
nan_value: -32768
- _target_: src.pipeline.model.feature.transformation.clip.Clip
feature_ranges: [ [ 6250, 12000 ], [ 6500, 13250 ], [ 6500, 12000 ], [ 5750, 11250 ], [ 6500, 12000 ], [ 0, 1 ], [ 0, 5 ] ]  #[SWIR, NIR, RED, GREEN, BLUE, CLOUD, ELEVATION]

```

### Feature columns

```python
target_pipeline:
  processed_path: data/processed/target
  transformation_pipeline:
    transformations:
      - _target_: src.pipeline.model.feature.transformation.gaussian_blur.GaussianBlur
        sigma: 2
  column_pipeline:
    columns: []
```

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
import torch.nn
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

criterion:
_target_: torch.nn.KLDivLoss


```
