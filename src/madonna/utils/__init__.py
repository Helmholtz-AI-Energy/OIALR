from __future__ import annotations

import hydra
import torch.nn as nn
import torch.optim

from . import comm, datasets, tracking
from .utils import *


def get_model(config):
    model: nn.Module = hydra.utils.instantiate(config.model)
    # send model to devices
    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model


def get_criterion(config):
    criterion: nn.Module = hydra.utils.instantiate(config.training.criterion, _recursive=False)
    if torch.cuda.is_available():
        return criterion.cuda()
    else:
        return criterion


def get_optimizer(config, network):
    optimizer = hydra.utils.instantiate(config.training.optimizer)
    return optimizer(network.parameters(), lr=config.training.lr)


def get_lr_schedules(config, optim, len_ds=None):
    """
    Get learning rate schedules from config files

    Parameters
    ----------
    config
    optim
    len_ds

    Returns
    -------

    """
    sched_name = config.training.lr_schedule._target_.split(".")[0]

    # sched_params = config["lr_schedule"]["params"]
    if sched_name == "ExponentialLR":
        # sched_params["last_epoch"] = config["epochs"] - config["start_epoch"]
        config.training.lr_schedule.last_epoch = config.training.epochs - config.training.start_epoch
    elif sched_name == "CosineAnnealingLR":
        # sched_params["last_epoch"] = config['epochs'] - config['start_epoch']
        config.training.lr_schedule.T_max = len_ds
    elif sched_name == "CosineAnnealingWarmRestarts":
        config.training.lr_schedule.T_0 = len_ds
    elif sched_name == "CyclicLR":
        config.training.lr_schedule.max_lr = config.training.lr
        config.training.lr_schedule.step_size_up = len_ds

    scheduler = hydra.utils.instantiate(config.training.lr_schedule)
    scheduler = scheduler(optim)
    warmup_scheduler = hydra.utils.instantiate(config.training.lr_warmup)
    warmup_scheduler = warmup_scheduler(optim)

    return scheduler, warmup_scheduler
