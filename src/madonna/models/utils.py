import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    try:
        model = model.ddp_model
    except AttributeError:
        model = model

    def _disable(module):
        # try:
        #     module.track_running_stats = False  # if falis here, wont continue
        #     module.backup_momentum = module.momentum
        #     module.momentum = 0
        # except AttributeError:
        #     pass
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
            module.track_running_stats = False

    model.apply(_disable)


def enable_running_stats(model):
    try:
        model = model.ddp_model
    except AttributeError:
        model = model

    def _enable(module):
        # try:
        #     module.track_running_stats = True
        #     module.momentum = module.backup_momentum
        # except AttributeError:
        #     pass
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.track_running_stats = True
            module.momentum = module.backup_momentum

    model.apply(_enable)
