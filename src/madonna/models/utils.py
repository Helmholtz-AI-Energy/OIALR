import logging
import time

import torch
import torch.distributed as dist
import torch.nn as nn
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
