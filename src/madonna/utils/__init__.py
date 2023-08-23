from __future__ import annotations

import hydra
import torch.nn as nn
import torch.optim
from omegaconf import open_dict
from timm.scheduler import create_scheduler

from . import comm, datasets, tracking
from .utils import *


def get_model(config):
    if config.model is None:
        raise ValueError("model must be specified")
    if config.model.name.startswith("vit"):
        config.model.model.image_size = config.data.train_crop_size
    config.model.model.num_classes = config.data.classes
    # if config.model.name.startswith("resnetrs"):
    #     config.model.model.image_size = config.data.train_crop_size
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
    # for adam-style otpimzers, need to collect the betas into a tuple
    betas = []
    if "beta1" in config.training.optimizer:
        beta1 = config.training.optimizer.beta1
        betas.append(beta1)
        del config.training.optimizer.beta1
    if "beta2" in config.training.optimizer:
        beta2 = config.training.optimizer.beta2
        betas.append(beta2)
        del config.training.optimizer.beta2
    if len(betas) > 0:
        betas = tuple(betas)
        with open_dict(config):
            config.training.optimizer.betas = betas

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

    if config.training.lr_schedule._target_ is None:
        # Using timm lr schedulers if its None..
        with open_dict(config):
            if "epochs" not in config.training.lr_schedule:
                epochs = config.training.epochs - config.training.lr_schedule.warmup_epochs
                config.training.lr_schedule.epochs = epochs
        scheduler, _ = create_scheduler(args=config.training.lr_schedule, optimizer=optim, updates_per_epoch=len_ds)
        return scheduler, None

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
    if config.baseline or "sigma_warmup" not in config.training.keys():
        return None, None

    if config.training.sigma_warmup is None:
        return None, None

    sched_name = config.training.sigma_warmup._target_.split(".")[0]

    # sched_params = config["lr_schedule"]["params"]
    if sched_name == "ExponentialLR":
        # sched_params["last_epoch"] = config["epochs"] - config["start_epoch"]
        config.training.sigma_warmup.last_epoch = config.training.epochs - config.training.start_epoch
    elif sched_name == "CosineAnnealingLR":
        # sched_params["last_epoch"] = config['epochs'] - config['start_epoch']
        config.training.sigma_warmup.T_max = len_ds
    elif sched_name == "CosineAnnealingWarmRestarts":
        config.training.sigma_warmup.T_0 = len_ds
    elif sched_name == "CyclicLR":
        config.training.sigma_warmup.max_lr = config.training.lr
        config.training.sigma_warmup.step_size_up = len_ds

    scheduler = hydra.utils.instantiate(config.training.sigma_warmup)
    scheduler = scheduler(sigma_optim)
    if not config.training.lr_warmup:
        return scheduler, None

    warmup_scheduler = hydra.utils.instantiate(config.training.sigma_warmup)
    warmup_scheduler = warmup_scheduler(sigma_optim)
    return scheduler, warmup_scheduler


def get_refactory_lr_warmup(config, opt, lr):
    if config.baseline:
        return None
    # TODO: need to reset the optimizer to the max LR before initializing the warmup scheduler
    for group in opt.param_groups:
        group["lr"] = lr

    warmup_scheduler = hydra.utils.instantiate(config.training.refactory_warmup)
    warmup_scheduler = warmup_scheduler(opt)
    return warmup_scheduler
