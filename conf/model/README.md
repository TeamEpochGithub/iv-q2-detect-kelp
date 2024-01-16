### Available loss functions:


```
criterion:
  _target_: src.modules.loss.dice_loss.DiceLoss
  
criterion:
  _target_: torch.nn.BCELoss

criterion: 
  _target_: torch.nn.WeightedBCELoss
  positive_freq: 0.0067

criterion:
  _target_: src.modules.loss.bce_dice_loss.BCEDiceLoss
  
criterion:
  _target_: src.modules.loss.shrinkage_loss.ShrinkageLoss
  
criterion:
  _target_: src.modules.loss.focal_loss.FocalLoss
  alpha: ??? (0.25)
  gamma: ??? (2.0)
    
criterion:
  _target_: src.modules.loss.jaccard.JaccardLoss
  
criterion:
  _target_: src.modules.loss.tversky.TverskyLoss
  alpha: ??? (0.5)
  beta: ??? (0.5)
  
    
``` 