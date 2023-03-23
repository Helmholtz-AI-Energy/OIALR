from __future__ import annotations

from functools import wraps
from typing import Any, Callable
import torch.nn as nn

import torch.distributed as dist


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
        Function that can be used as a decorator to enable a function/method being called only on global rank 0.
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


def change_batchnorm_tracking(model: nn.Module, tracking=False):
    for child in model.children():
        if hasattr(child, "track_running_stats"):
            child.track_running_stats = tracking
        change_batchnorm_tracking(child, tracking)
