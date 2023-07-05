import contextlib
import logging
import time
from collections import defaultdict
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from ..models.utils import disable_running_stats, enable_running_stats

log = logging.getLogger(__name__)

"""
Notes on SAM:
- needs 2 forward/backward passes on each batch
- most implementations have it in the training loop:
```
# first f/b pass
loss = loss_fn(image_preds, image_labels)
loss.backward()
optimizer.first_step(zero_grad=True)

# second forward-backward pass
loss_fn(model(imgs), image_labels).backward()
optimizer.second_step(zero_grad=True)
```

since we have the other logic to use about the freezing, we need to call the SVD checking at the end of the step itself.

"""


def change_optimizer_group_for_svd(optimizer: optim.Optimizer, model, config):
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
        elif n.endswith("weight") and p.ndim == 2:
            params_weights.append(p)
        else:
            params_non2d.append(p)
    if config.training.fixing_method.keep_last_layer:
        last = params_weights.pop()
        params_non2d.append(last)

    # get optimizer kwargs from config
    opt_kwargs = dict(config.training.optimizer)
    try:  # remove the target and partial flags - Hydra specific stuff
        del opt_kwargs["_target_"]
        del opt_kwargs["_partial_"]
    except AttributeError:
        pass
    # delete the current parameter groups
    opt_kwargs["lr"] = config.training.min_lr
    optimizer.param_groups = []
    # add the groups 0 -> non2d
    optimizer.add_param_group({"params": params_non2d, **opt_kwargs})
    optimizer.add_param_group({"params": params_weights, **opt_kwargs})
    if config.training.sigma_optimizer.min_lr is not None:
        opt_kwargs["lr"] = config.training.sigma_optimizer.min_lr
    optimizer.add_param_group({"params": params_sigma, **opt_kwargs})


def reset_sigma_opt_shapes(optimizer: optim.Optimizer):
    resettime = time.perf_counter()
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    # slice off bits of the saved states
    # optimizer.param_groups[-1] -> should always be sigma lise
    for p in enumerate(optimizer.param_groups[-1]):
        # if dist.get_rank() == 0:
        #     print(n, optimizer.param_groups[0]["params"][c].shape, p.shape)
        state = optimizer.state[p]
        if len(list(state.keys())) > 0:
            for k in ["exp_avg", "exp_avg_sq"]:
                if state[k].shape != p.shape:
                    sl = []
                    for d in range(p.ndim):
                        sl.append(slice(0, p.shape[d]))
                    # print(type(state[k]))
                    state[k] = state[k][tuple(sl)]
            if optimizer.param_groups[-1]["amsgrad"]:
                if state["max_exp_avg_sq"].shape != p.shape:
                    sl = []
                    for d in range(p.ndim):
                        sl.append(slice(0, p.shape[d]))
                    state["max_exp_avg_sq"] = state["max_exp_avg_sq"][tuple(sl)]
    if rank == 0:
        log.info(f"Reset Optimizer time: {time.perf_counter() - resettime}")


class SAM(optim.Optimizer):
    """
    @inproceedings{foret2021sharpnessaware,
        title={Sharpness-aware Minimization for Efficiently Improving Generalization},
        author={Pierre Foret and Ariel Kleiner and Hossein Mobahi and Behnam Neyshabur},
        booktitle={International Conference on Learning Representations},
        year={2021},
        url={https://openreview.net/forum?id=6Tm1mposlrM}
    }
    @inproceesings{pmlr-v139-kwon21b,
        title={ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks},
        author={Kwon, Jungmin and Kim, Jeongseop and Park, Hyunseo and Choi, In Kwon},
        booktitle ={Proceedings of the 38th International Conference on Machine Learning},
        pages={5905--5914},
        year={2021},
        editor={Meila, Marina and Zhang, Tong},
        volume={139},
        series={Proceedings of Machine Learning Research},
        month={18--24 Jul},
        publisher ={PMLR},
        pdf={http://proceedings.mlr.press/v139/kwon21b/kwon21b.pdf},
        url={https://proceedings.mlr.press/v139/kwon21b.html},
    }

    usage:
    # first forward-backward step
    enable_running_stats(model)  # <- this is the important line
    predictions = model(inputs)
    loss = smooth_crossentropy(predictions, targets)
    loss.mean().backward()
    optimizer.first_step(zero_grad=True)

    # second forward-backward step
    disable_running_stats(model)  # <- this is the important line
    smooth_crossentropy(model(inputs), targets).mean().backward()
    optimizer.second_step(zero_grad=True)
    """

    def __init__(self, params, base_optimizer, model, config, **kwargs):
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.model.autocast)
        self.amp = config.model.autocast
        rho = config.training.sam.rho
        adaptive = config.training.sam.adaptive
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.model = model
        self.max_grad_norm = -1

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                # 2 things: another repo only did this for weights and NOT biases
                # do we need to change any shapes here? - No?
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for p in self.state.keys():
            if p.grad is None:
                continue
            p.data = self.state[p]["old_p"]
        # for group in self.param_groups:
        #     for p in group["params"]:
        #         if p.grad is None:
        #             continue
        #         p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        if self.max_grad_norm != -1:
            self.scaler.unscale_(self.param_groups)
            torch.nn.utils.clip_grad_norm_(self.param_groups, self.max_grad_norm)
        self.scaler.step(self.base_optimizer)
        self.scaler.update()

        if zero_grad:
            self.zero_grad(set_to_none=True)

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    # @torch.no_grad()
    # def step(self, closure=None):
    #     assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
    #     closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

    #     self.first_step(zero_grad=True)
    #     closure()
    #     self.second_step()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets, **kwargs)

                self.scaler.scale(loss).backward()
                loss_value = loss

            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None):
        # if closure:
        #     get_grad = closure
        # else:
        get_grad = self.forward_backward_func
        # outputs, loss_value = get_grad()

        # enable running stats
        with self.maybe_no_sync():
            enable_running_stats(self.model)
            # get gradient
            outputs, loss_value = get_grad()
            self.first_step(zero_grad=True)
            # disable running stats for second pass
        disable_running_stats(self.model)
        # get gradient at perturbed weights
        get_grad()
        self.second_step(zero_grad=True)

        return outputs, loss_value

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ],
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
