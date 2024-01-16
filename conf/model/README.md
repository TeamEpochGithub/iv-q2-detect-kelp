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
  _target_: src.modules.loss.jaccard_loss.JaccardLoss
  
criterion:
  _target_: src.modules.loss.tversky_loss.TverskyLoss
  alpha: ??? (0.5)
  beta: ??? (0.5)
  
criterion:
  _target_: src.modules.loss.tversky_focal.TverskyFocalLoss
  alpha: ??? (0.5)
  beta: ??? (0.5)
  gamma: ??? (1.0)
  
criterion:
  _target_: src.modules.loss.lovasz_hinge_loss.LovaszHingeLoss
  per_image: ??? (False)
 
  
     
``` 