### Available loss functions:


```
criterion:
  _target_: src.modules.loss.shrinkage_loss.ShrinkageLoss
  
criterion:
  _target_: src.modules.loss.dice_loss.DiceLoss
  
criterion:
  _target_: torch.nn.BCELoss
  
criterion: 
  _target_: torch.nn.WeightedBCELoss
  positive_freq: 0.0067

criterion:
  _target_: src.modules.loss.focal_loss.FocalLoss
  alpha: ???
  gamma: ???
    
criterion:
  _target_: src.modules.loss.jaccard.JaccardLoss
  
  
    
``` 