import logging
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm

log = logging.getLogger(__name__)


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


def change_adam_shapes(optimizer):
    """
    reset the shapes of the Adam optimizer buffers to be the same shape as the model parameters

    if `reset_buffers_zero`: reset the buffer to zero after reshaping it
    """
    resettime = time.perf_counter()
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    # instead of resetting optimizer, slice off bits of the saved states
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            if len(list(state.keys())) > 0:
                for k in ["exp_avg", "exp_avg_sq"]:
                    if state[k].shape != p.shape:
                        sl = []
                        for d in range(p.ndim):
                            sl.append(slice(0, p.shape[d]))
                        # print(type(state[k]))
                        state[k] = state[k][tuple(sl)]
                if group["amsgrad"]:
                    if state["max_exp_avg_sq"].shape != p.shape:
                        sl = []
                        for d in range(p.ndim):
                            sl.append(slice(0, p.shape[d]))
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"][tuple(sl)]
    if rank == 0:
        log.info(f"Reset Optimizer time: {time.perf_counter() - resettime}")


def change_sgd_shapes(optimizer):
    """
    reset the shapes of the SGD optimizer buffers to be the same shape as the model parameters

    if `reset_buffers_zero`: reset the buffer to zero after reshaping it
    """
    resettime = time.perf_counter()
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    # instead of resetting optimizer, slice off bits of the saved states
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            if len(list(state.keys())) > 0:
                # only need to change "momentum_buffer"
                if state["momentum_buffer"].shape != p.shape:
                    sl = []
                    for d in range(p.ndim):
                        sl.append(slice(0, p.shape[d]))
                    # print(type(state[k]))
                    state["momentum_buffer"] = state["momentum_buffer"][tuple(sl)]
    if rank == 0:
        log.info(f"Reset Optimizer time: {time.perf_counter() - resettime}")


def create_svd_param_groups(optimizer: optim.Optimizer, model, individual_groups=False):
    # make 3 optimizer groups within an optimizer
    #   non-2d, weights, svd
    params_non2d, params_weights, params_sigma = [], [], []
    sigma_param_index = []
    c = 0

    # self._create_param_lists(model)
    for c, (n, p) in enumerate(model.named_parameters()):
        if n.endswith(("_s", ".s")):
            params_sigma.append(p)
            sigma_param_index.append(c)
        elif n.endswith(("_u", ".u", "_vh", ".vh")):
            continue
        elif n.endswith("weight") and (p.ndim == 2 or p.ndim == 4):  # 4 for convs
            params_weights.append(p)
        else:
            params_non2d.append(p)
    # if config.training.fixing_method.keep_last_layer:
    #     last = params_weights.pop()
    #     params_non2d.append(last)

    # get optimizer kwargs from config
    opt_kwargs = optimizer.param_groups[0]
    opt_kwargs.pop("params")
    # opt_kwargs = dict(config.training.optimizer)
    # try:  # remove the target and partial flags - Hydra specific stuff
    #     del opt_kwargs["_target_"]
    #     del opt_kwargs["_partial_"]
    # except AttributeError:
    #     pass
    # delete the current parameter groups
    # opt_kwargs["lr"] = config.training.lr
    optimizer.param_groups = []
    # add the groups 0 -> non2d
    optimizer.add_param_group({"params": params_non2d, **opt_kwargs})
    optimizer.add_param_group({"params": params_weights, **opt_kwargs})
    optimizer.add_param_group({"params": params_sigma, **opt_kwargs})
