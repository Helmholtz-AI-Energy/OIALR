import torch
import torch.distributed as dist
import torch.nn as nn


def init_methods(model, config):
    pass


"""
Methods for initializing models to use low rank best
1. random init
2. same seed random init
3. random init + trade topn

callbacks for init:
- sync largest singular values
- ???
"""


def random_init(model):
    pass


def same_seed(model):
    pass


def trade_largest_singulars(model: nn.Module, topn: int):
    # trade the largest topn singular values
    if not dist.is_initialized():
        return
    for n, p in model.named_parameters():
        if not p.requires_grad():
            # if they dont require grads, then they shouldnt be
            continue
