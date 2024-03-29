from .vit import vit
from .resnet import resnet
from .swin_transformer_v2 import swin_v2

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        return vit(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'resnet':
        return resnet(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'swin_v2':
        return swin_v2(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')