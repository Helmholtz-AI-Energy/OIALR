import logging
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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


def change_adam_shapes(optimizer, update_from_svd=False):
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
            if len(list(state.keys())) == 0:
                continue
            for k in ["exp_avg", "exp_avg_sq"]:
                if state[k].shape == p.shape:
                    continue
                # this will only happen for Sigma matrices!
                # therefore, we dont need to worry about shapes/transposes
                # st = state[k].to(torch.float32)
                # _u, s, _vh = torch.linalg.svd(st, full_matrices=False)
                sl = []
                pad = []
                for d in range(p.ndim):
                    sl.append(slice(0, p.shape[d]))
                    if state[k].shape[d] < p.shape[d]:
                        pad = [0, p.shape[d] - state[k].shape[d]] + pad

                state[k] = state[k][tuple(sl)]
                if len(pad) > 0:
                    state[k] = F.pad(state[k], pad, "constant", 0)
                    state[k].mul_(0)

            if group["amsgrad"] and state["max_exp_avg_sq"].shape != p.shape:
                sl = []
                pad = []
                for d in range(p.ndim):
                    sl.append(slice(0, p.shape[d]))
                    if state["max_exp_avg_sq"].shape[d] < p.shape[d]:
                        pad = [0, p.shape[d] - state["max_exp_avg_sq"].shape[d]] + pad

                state["max_exp_avg_sq"] = state["max_exp_avg_sq"][tuple(sl)]
                if len(pad) > 0:
                    state["max_exp_avg_sq"] = F.pad(state["max_exp_avg_sq"], pad, "constant", 0)
                    state["max_exp_avg_sq"].mul_(0)
                # st = state["max_exp_avg_sq"].to(torch.float32)
                # _u, s, _vh = torch.linalg.svd(st, full_matrices=False)

                # state["max_exp_avg_sq"] *= 0
                # state["max_exp_avg_sq"] = torch.diag(s)[tuple(sl)].to(state["max_exp_avg_sq"].dtype)
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
                if state["momentum_buffer"] is None:
                    continue
                if state["momentum_buffer"].shape != p.shape:
                    sl = []
                    k = "momentum_buffer"
                    pad = []
                    for d in range(p.ndim):
                        sl.append(slice(0, p.shape[d]))
                        if state[k].shape[d] < p.shape[d]:
                            pad = [0, p.shape[d] - state[k].shape[d]] + pad

                    state[k] = state[k][tuple(sl)]
                    if len(pad) > 0:
                        state[k] = F.pad(state[k], pad, "constant", 0)
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


def replace_opt_state_with_svd_adam(optimizer: optim.Optimizer, replacement_dict):
    # replacement_dict: [full_rank_param] -> low_rank param
    for group in optimizer.param_groups:
        replace_idx = 0
        for p in group["params"]:
            if id(p) not in replacement_dict:
                replace_idx += 1
                continue
            new_p = replacement_dict[id(p)][0]
            layer_type = replacement_dict[id(p)][1]
            # change the state info to the svd
            state = optimizer.state[p]
            if len(list(state.keys())) > 0:
                for k in ["exp_avg", "exp_avg_sq"]:
                    st = state[k].to(torch.float32)
                    if layer_type in ["lin", "attn"]:
                        if st.shape[0] < st.shape[1]:
                            st = st.T
                    elif layer_type == "conv":
                        m = st.shape[0]
                        n = int(st.numel() / st.shape[0])
                        # svd shapes: [m, n] -> u[m, k] * s[k, k] * vh [k, n]    k-> min(m, n)
                        st = st.view(m, n)
                        if m < n:
                            hold = m
                            n = m
                            m = hold
                            st = st.T
                    min_s = min(tuple(st.shape))

                    # this will only happen for Sigma matrices!
                    # therefore, we dont need to worry about shapes/transposes
                    # _u, s, _vh = torch.linalg.svd(st, full_matrices=False)
                    # sl = []
                    # for d in range(p.ndim):
                    #     sl.append(slice(0, p.shape[d]))
                    # state[k] = state[k][tuple(sl)]
                    # state[k] = torch.diag(s).to(state[k].dtype)
                    state[k] = torch.zeros((min_s, min_s), dtype=state[k].dtype, device=state[k].device)

                if group["amsgrad"]:
                    # sl = []
                    # for d in range(p.ndim):
                    #     sl.append(slice(0, p.shape[d]))
                    st = state["max_exp_avg_sq"].to(torch.float32)
                    if layer_type in ["lin", "attn"]:
                        if st.shape[0] < st.shape[1]:
                            st = st.T
                    elif layer_type == "conv":
                        m = st.shape[0]
                        n = int(st.numel() / st.shape[0])
                        # svd shapes: [m, n] -> u[m, k] * s[k, k] * vh [k, n]    k-> min(m, n)
                        st = st.view(m, n)
                        if m < n:
                            hold = m
                            n = m
                            m = hold
                            st = st.T
                    # _u, s, _vh = torch.linalg.svd(st, full_matrices=False)
                    # state["max_exp_avg_sq"] = state["max_exp_avg_sq"][tuple(sl)]
                    min_s = min(tuple(st.shape))
                    # state["max_exp_avg_sq"] = torch.diag(s).to(state["max_exp_avg_sq"].dtype)
                    state["max_exp_avg_sq"] = torch.zeros(
                        (min_s, min_s),
                        dtype=state["max_exp_avg_sq"].dtype,
                        device=state["max_exp_avg_sq"].device,
                    )
            # replace the reference in the group dict
            optimizer.state[new_p] = state
            del optimizer.state[p]
            # change the state KEY to the svd param
            # del group["params"][id(p)]
            group["params"][replace_idx] = new_p
            # group["params"].append(new_p)
            replace_idx += 1


def replace_opt_state_with_svd_sgd(optimizer: optim.Optimizer, replacement_dict):
    # replacement_dict: [full_rank_param] -> low_rank param
    for group in optimizer.param_groups:
        replace_idx = 0
        for p in group["params"]:
            if p not in replacement_dict:
                replace_idx += 1
                continue
            new_p = replacement_dict[p][0]
            layer_type = replacement_dict[p][1]
            # change the state info to the svd
            state = optimizer.state[p]
            if len(list(state.keys())) > 0:
                for k in ["momentum_buffer"]:
                    st = state[k].to(torch.float32)
                    if layer_type in ["lin", "attn"]:
                        if st.shape[0] < st.shape[1]:
                            st = st.T
                    elif layer_type == "conv":
                        m = st.shape[0]
                        n = int(st.numel() / st.shape[0])
                        # svd shapes: [m, n] -> u[m, k] * s[k, k] * vh [k, n]    k-> min(m, n)
                        st = st.view(m, n)
                        if m < n:
                            hold = m
                            n = m
                            m = hold
                            st = st.T
                    min_s = min(tuple(st.shape))

                    # this will only happen for Sigma matrices!
                    # therefore, we dont need to worry about shapes/transposes
                    # _u, s, _vh = torch.linalg.svd(st, full_matrices=False)
                    # sl = []
                    # for d in range(p.ndim):
                    #     sl.append(slice(0, p.shape[d]))
                    # state[k] = state[k][tuple(sl)]
                    # state[k] = torch.diag(s).to(state[k].dtype)
                    state[k] = torch.zeros((min_s, min_s), dtype=state[k].dtype, device=state[k].device)

            # replace the reference in the group dict
            optimizer.state[new_p] = state
            del optimizer.state[p]
            # change the state KEY to the svd param
            # del group["params"][id(p)]
            group["params"][replace_idx] = new_p
            # group["params"].append(new_p)
            replace_idx += 1
