from __future__ import annotations

import hydra
import torch.nn as nn
import torch.optim

from . import comm, datasets, tracking
from .utils import *


def get_model(config):
    if config.model is None:
        raise ValueError("model must be specified")
    if config.model.name.startswith("vit"):
        config.model.model.image_size = config.data.train_crop_size
    model: nn.Module = hydra.utils.instantiate(config.model.model)
    # send model to devices
    if torch.cuda.is_available() and not config.cpu_training:
        return model.cuda()
    else:
        return model


def get_criterion(config):
    if config.training.criterion is None:
        raise ValueError("Training criterion must be specified")
    criterion: nn.Module = hydra.utils.instantiate(config.training.criterion)
    if torch.cuda.is_available() and not config.cpu_training:
        return criterion.cuda()
    else:
        return criterion


def get_optimizer(config, network, lr=None):
    if config.training.optimizer is None:
        raise ValueError("Optimizer must be specified")
    optimizer = hydra.utils.instantiate(config.training.optimizer)
    kwargs = {}
    if config.training.init_opt_with_model:
        first = network
    else:
        first = network.parameters()

    if lr is None and config.training.lr is not None:
        kwargs["lr"] = config.training.lr
    elif lr is not None:
        kwargs["lr"] = lr
    return optimizer(first, **kwargs)


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
    if config.training.lr_schedule is None:
        return None, None

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
    if not config.training.lr_warmup:
        return scheduler, None

    warmup_scheduler = hydra.utils.instantiate(config.training.lr_warmup)
    warmup_scheduler = warmup_scheduler(optim)
    return scheduler, warmup_scheduler


def get_sigma_lr_schedules(config, sigma_optim, len_ds=None):
    if config.baseline:
        return None, None
    if config.training.lr_schedule is None:
        return None, None

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
    scheduler = scheduler(sigma_optim)
    if not config.training.lr_warmup:
        return scheduler, None

    warmup_scheduler = hydra.utils.instantiate(config.training.sigma_warmup)
    warmup_scheduler = warmup_scheduler(sigma_optim)
    return scheduler, warmup_scheduler
