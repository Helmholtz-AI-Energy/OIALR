from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn


def only_on_rank_n(run_on_rank: int = 0):
    # run a command only on a specified rank
    # this is a decorator: i.e.: @only_on_rank_n("rank1") will run only on rank 1

    # three options: run on all, run on 0, run on target
    rank = 0  # start with run on all - stick with this if not distributed
    target_rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
        target_rank = run_on_rank
        if run_on_rank < 0:  # run on all
            target_rank = rank

    def rank_zero_only(fn: Callable) -> Callable:
        """Wrap a function to call internal function only in rank zero.
        Function that can be used as a decorator to enable a function/method
        being called only on global rank 0.
        """

        @wraps(fn)
        def wrapped_fn(*args: Any, **kwargs: Any) -> Any | None:
            # rank = getattr(rank_zero_only, "rank", None)
            # if rank is None:
            #     raise RuntimeError("torch distributed not initialized yet")
            if rank == target_rank:
                return fn(*args, **kwargs)
            return None

        return wrapped_fn

    return rank_zero_only


def print0(*args, sep=" ", end="\n", file=None):
    if dist.is_initialized() and dist.get_rank() == 0:
        print(*args, sep=sep, end=end, file=file)
    elif dist.is_initialized():
        return
    else:
        print(*args, sep=sep, end=end, file=file)


@only_on_rank_n(0)
def log0(msg, level, logger=None, *args, **kwargs):
    if logger is None:
        logger = logging.getLogger(__name__)
    getattr(logger, level)(msg, *args, **kwargs)


def change_batchnorm_tracking(model: nn.Module, tracking=False):
    for child in model.children():
        if hasattr(child, "track_running_stats"):
            child.track_running_stats = tracking
        change_batchnorm_tracking(child, tracking)


def roll_orthogonal_values(
    shape: Union[Tuple, List, torch.Size],
    dtype: torch.dtype,
    device: torch.device,
    scale_factor: float = 0.1,
):
    """
    References
    ----------
    .. [1] F. Mezzadri, "How to generate random matrices from the classical
       compact groups", :arXiv:`math-ph/0609050v2`.

    Parameters
    ----------
    shape
    dtype
    device
    scale_factor

    Returns
    -------

    """
    # roll new "gradients" to search the local area
    # the gradients are orthogonal matrices, normally between -1 and 1
    # TODO: since the parameters are typically on the same order of magnitude, it may be
    #  needed to scale the gradients. TBD.
    if len(shape) == 1:
        return torch.randn(shape, dtype=dtype, device=device) * scale_factor

    z = torch.randn(shape, dtype=dtype, device=device)
    if shape[-1] > shape[-2]:
        # need to transpose? or just do complete, then slice of the bad bits
        if len(shape) > 2:
            hold = torch.arange(len(shape) - 2)
            x_perm = z.permute(*hold, -1, -2)
            q, r = torch.linalg.qr(x_perm, mode="reduced")
            d = r.diagonal()
            ret = q * (d / d.abs()).unsqueeze(-2)
            ret = ret[..., : x_perm.shape[-1]]
            ret = ret.permute(*hold, -1, -2)
        else:
            x_perm = z.permute(-1, -2)
            q, r = torch.linalg.qr(x_perm, mode="reduced")
            d = r.diagonal()
            # print('h', q.shape, d.shape)
            ret = q * (d / d.abs()).unsqueeze(-2)
            ret = ret[..., : x_perm.shape[-1]]
            ret = ret.permute(-1, -2)
    else:
        z = torch.randn(shape, dtype=dtype, device=device)
        q, r = torch.linalg.qr(z, mode="reduced")
        d = r.diagonal()
        ret = q * (d / d.abs()).unsqueeze(-2)
    return ret * scale_factor
