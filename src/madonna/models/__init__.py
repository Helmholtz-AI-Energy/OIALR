import torchvision.models as models

from . import components


def get_tv_model(config):
    if config["pretrained"]:
        print(f"=> using pre-trained model '{config.model.name}'")
        model = models.__dict__[config.model.name](pretrained=True)
    else:
        print(f"=> creating model '{config.model.name}'")
        model = models.__dict__[config.model.name]()
    return model
