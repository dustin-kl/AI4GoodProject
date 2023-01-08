from .baseline import DeepLabv3_plus
from .deeplabv3attention import DeepLabV3PlusAttention
from .unet import UNet
from .transunet.transunet import TransUNet
from .hyperparameters import hyper_parameters


def get_model(model_name):
    params = hyper_parameters[model_name]
    if model_name == "unet":
        return UNet(params)
    elif model_name == "baseline":
        return DeepLabv3_plus(nInputChannels=4, n_classes=3, _print=False)
    elif model_name == "attention":
        return DeepLabV3PlusAttention(3)
    elif model_name == "transunet":
        return TransUNet(params, 4)
