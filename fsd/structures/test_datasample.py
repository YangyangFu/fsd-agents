from mmengine.structures import InstanceData, PixelData
import random 
import torch 

metainfo = dict(
    img_id = 1,
    img_shape = (400, 600)
)

img = torch.randint(0, 255, (4, 20, 40))
featmap = torch.randint(0, 255, (10, 20, 40))

pdata = PixelData(metainfo=metainfo, image=img, featmap=featmap)
print(pdata.shape)
print(dir(pdata))