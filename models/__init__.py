from .resnet import *
from .densenet import *
from .densenet3 import *
from .mobilenetV2 import *
from .efficientnet_our import *

def load_model(name, num_classes=10, pretrained=False, **kwargs):
    model_dict = globals()
    model = model_dict[name](pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model
