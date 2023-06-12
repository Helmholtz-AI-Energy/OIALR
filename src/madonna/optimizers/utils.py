import logging
import time
from typing import List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

# from mpi4py import MPI

# from ..utils import comm, utils

log = logging.getLogger(__name__)


# need function to reset the sizes of the optimizer buffers


def change_adam_shapes(optimizer, model, reset_buffers_zero: bool = False, param_indices=None):
    """
    reset the shapes of the Adam optimizer buffers to be the same shape as the model parameters

    if `reset_buffers_zero`: reset the buffer to zero after reshaping it
    """
    resettime = time.perf_counter()
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    # instead of resetting optimizer, slice off bits of the saved states
    # for group in optimizer.param_groups:
    for c, (n, p) in enumerate(model.named_parameters()):
        if param_indices is not None and c not in param_indices:
            continue
        # if dist.get_rank() == 0:
        #     print(n, optimizer.param_groups[0]["params"][c].shape, p.shape)
        settozero = False
        if n.endswith(".s") and p.requires_grad and reset_buffers_zero:
            settozero = True

        state = optimizer.state[p]
        if len(list(state.keys())) > 0:
            for k in ["exp_avg", "exp_avg_sq"]:
                if state[k].shape != p.shape:
                    sl = []
                    for d in range(p.ndim):
                        sl.append(slice(0, p.shape[d]))
                    # print(type(state[k]))
                    state[k] = state[k][tuple(sl)]
                if settozero:
                    state[k].zero_()
            if optimizer.param_groups[0]["amsgrad"]:
                if state["max_exp_avg_sq"].shape != p.shape:
                    sl = []
                    for d in range(p.ndim):
                        sl.append(slice(0, p.shape[d]))
                    state["max_exp_avg_sq"] = state["max_exp_avg_sq"][tuple(sl)]
                if settozero:
                    state["max_exp_avg_sq"].zero_()

        # if optimizer.param_groups[0]["params"][c].shape != p.shape:
        #     sl = []
        #     for d in range(p.ndim):
        #         sl.append(slice(0, p.shape[d]))
        #     optimizer.param_groups[0]["params"][c] = optimizer.param_groups[0]["params"][c][tuple(sl)]
        #     if settozero:
        #         optimizer.param_groups[0]["params"][c].zero_()
    if rank == 0:
        log.info(f"Reset Optimizer time: {time.perf_counter() - resettime}")


def change_sgd_shapes(optimizer, model, reset_buffers_zero: bool = False, param_indices=None):
    """
    reset the shapes of the SGD optimizer buffers to be the same shape as the model parameters

    if `reset_buffers_zero`: reset the buffer to zero after reshaping it
    """
    resettime = time.perf_counter()
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    # print(list(optimizer.param_groups[0].keys()))
    # raise ValueError
    for c, (n, p) in enumerate(model.named_parameters()):
        if param_indices is not None and c not in param_indices:
            continue
        # if dist.get_rank() == 0:
        #     print(n, optimizer.param_groups[0]["params"][c].shape, p.shape)
        settozero = False
        if n.endswith(".s") and p.requires_grad and reset_buffers_zero:
            settozero = True

        state = optimizer.state[p]
        # print(list(state.keys()))
        if len(list(state.keys())) > 0:
            # if len(optimizer.param_groups[0]["momentum_buffer"]) > 0:
            if state["momentum_buffer"].shape != p.shape:
                sl = []
                for d in range(p.ndim):
                    sl.append(slice(0, p.shape[d]))
                state["momentum_buffer"] = state["momentum_buffer"][tuple(sl)]
            if settozero:
                state["momentum_buffer"].zero_()

        # if optimizer.param_groups[0]["params"][c].shape != p.shape:
        #     sl = []
        #     for d in range(p.ndim):
        #         sl.append(slice(0, p.shape[d]))
        #     optimizer.param_groups[0]["params"][c] = optimizer.param_groups[0]["params"][c][tuple(sl)]
        #     optimizer.param_groups[0]["params"][c].zero_()
    if rank == 0:
        log.info(f"Reset Optimizer time: {time.perf_counter() - resettime}")
