
import torch
d = torch.load('data/cloud_cloud_ac_DC.pt', weights_only=False)
print('XYZ min:', d['xyz'].min(dim=0).values)
print('XYZ max:', d['xyz'].max(dim=0).values)
print('XYZ mean:', d['xyz'].mean(dim=0))
print('RGB min:', d['rgb'].min())
print('RGB max:', d['rgb'].max())
print('N points:', d['xyz'].shape[0])
