import logging
from collections import OrderedDict
from collections import abc as container_abcs
from collections import defaultdict
from typing import Optional

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.adam import adam

from ..models.svd import SVDLinear, SVDMultiheadAttention
from .utils import change_adam_shapes, change_sgd_shapes

log = logging.getLogger(__name__)


# taken from torch's optimizer stuff
def _use_grad_for_differentiable(func):
    def _use_grad(self, *args, **kwargs):
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults["differentiable"])
            ret = func(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(prev_grad)
        return ret

    return _use_grad


class MixedSVDOpt(object):
    def __init__(self, model: torch.nn.Module, config):
        # TODO: config must have the following:
        #   optimizer -> target + all the other stuff as normal
        #   lr -> learning rate for the normal optimizer
        #   sigma_optimizer -> optimizer specific for sigma - will not touch the other params
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.baseline = config.baseline
        # the optimizers will be passed via hydra, so they will have a target
        # TODO: removed during debug! but back in later
        # if config.training.optimizer is None:
        #     raise ValueError("Optimizer must be specified")

        self.params_non2d, self.params_weights, self.params_sigma = [], [], []
        self.sigma_param_index = []
        c = 0

        # self._create_param_lists(model)
        for c, (n, p) in enumerate(model.named_parameters()):
            if n.endswith(("_s", ".s")):
                # if self.rank == 0:
                #     print(f"sigma: {n} {p.shape}")
                self.params_sigma.append(p)
                self.sigma_param_index.append(c)
            elif n.endswith(("_u", ".u", "_vh", ".vh")):
                # if self.rank == 0:
                #     print(f"u/vh (skipping - not in any list): {n} {p.shape}")
                continue
            elif n.endswith("weight") and p.ndim == 2:
                # if self.rank == 0:
                #     print(f"normal weights: {n} {p.shape}")
                self.params_weights.append(p)
            else:
                # if self.rank == 0:
                #     print(f"non-2d weights: {n} {p.shape}")
                self.params_non2d.append(p)
        if config.training.fixing_method.keep_last_layer:
            last = self.params_weights.pop()
            self.params_non2d.append(last)
        # raise ValueError

        optimizer1 = hydra.utils.instantiate(config.training.optimizer)
        # create the list of different parametes
        self.opt1: optim.Optimizer = optimizer1(self.params_non2d, lr=config.training.lr)
        # add a second group to the first optimier with the weigth matrices
        self.opt1.add_param_group({"params": self.params_weights})
        self.train_opt1 = True

        if self.baseline:
            self.sigma_opt = None
            return
        # else:
        # in sigma_optimizer it should have all the parameters
        sigma_opt = hydra.utils.instantiate(config.training.sigma_optimizer)
        self.sigma_opt: optim.Optimizer = sigma_opt(self.params_sigma)
        self.sigma_opt_is_adam = isinstance(self.sigma_opt, torch.optim.Adam)
        self.sigma_opt_is_sgd = isinstance(self.sigma_opt, torch.optim.SGD)

    def _create_param_lists(self, module):
        # self.params_non2d, self.params_weights, self.params_sigma = [], [], []
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                self.params_non2d.append(module.bias)
            self.params_weights.append(module.weight)
        elif isinstance(module, SVDLinear):
            if module.bias is not None:
                self.params_non2d.append(module.bias)
            self.params_weights.append(module.weight)
            self.params_sigma.append(module.s)
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                self.params_weights.append(module.in_proj_weight)
                self.params_sigma.append(module.in_proj_s)
            else:
                self.params_weights.append(module.q_proj_weight)
                self.params_weights.append(module.k_proj_weight)
                self.params_weights.append(module.v_proj_weight)
            if module.in_proj_bias is not None:
                self.params_non2d.append(module.in_proj_bias)
            if module.bias_k is not None:
                self.params_non2d.append(module.bias_k)
            if module.bias_v is not None:
                self.params_non2d.append(module.bias_v)
        elif isinstance(module, SVDMultiheadAttention):
            if module.in_proj_weight is not None:
                self.params_weights.append(module.in_proj_weight)
                self.params_sigma.append(module.in_proj_s)
            else:
                self.params_weights.append(module.q_proj_weight)
                self.params_sigma.append(module.q_s)
                self.params_weights.append(module.k_proj_weight)
                self.params_sigma.append(module.k_s)
                self.params_weights.append(module.v_proj_weight)
                self.params_sigma.append(module.v_s)
            if module.in_proj_bias is not None:
                self.params_non2d.append(module.in_proj_bias)
            if module.bias_k is not None:
                self.params_non2d.append(module.bias_k)
            if module.bias_v is not None:
                self.params_non2d.append(module.bias_v)

        for n, child in module.named_children():
            self._create_param_lists(child)
        return

    def zero_grad(self, set_to_none: bool = False, reset_sigma=True):
        self.opt1.zero_grad(set_to_none=set_to_none)
        if self.baseline or not reset_sigma:
            return
        self.sigma_opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None, step_sigma=False, step_opt1=True):
        if step_opt1:
            self.opt1.step(closure=closure)
        if self.baseline:
            return
        if step_sigma:
            if self.rank == 0:
                log.info("Running step on Sigma optimizer")
            self.sigma_opt.step(closure=closure)

    # remove the weight group once all the sigma params have gradients
    # needs to be handled one step heigher
    # (in the training script with a cue from the stability function)
    def remove_full_rank_weights(self):
        if self.baseline:
            return
        # THIS IS DANGEROUS! it might become an issue latter....maybe?
        try:
            # for p in self.opt1.param_groups[1]:
            # if i delete 1 params from self.opt1.param_groups[1]["params"], i also need to go into the state to trimm it off???
            del self.opt1.param_groups[1]
            if self.rank == 0:
                log.info("Removing the full rank part of the base optimizer")
        except IndexError:
            # if there is only 1 group on opt1 then move on, faster to try and fail then check
            pass

    def reset_shapes_of_sigma(self, model):
        if self.sigma_opt_is_adam:
            change_adam_shapes(
                optimizer=self.sigma_opt,
                model=model,
                param_indices=self.sigma_param_index,
            )
        elif self.sigma_opt_is_sgd:
            change_sgd_shapes(
                optimizer=self.sigma_opt,
                model=model,
                param_indices=self.sigma_param_index,
            )

    def disable_train_non2d(self):
        for p in self.params_non2d:
            p.requires_grad = False
        self.train_opt1 = False

    def enable_train_non2d(self):
        for p in self.params_non2d:
            p.requires_grad = True
        self.train_opt1 = True


# class SVDAdam(optim.Optimizer):
#     def __init__(
#         self,
#         named_params,
#         lr=1e-3,
#         betas=(0.9, 0.999),
#         eps=1e-8,
#         weight_decay=0,
#         amsgrad=False,
#         *,
#         foreach: Optional[bool] = None,
#         maximize: bool = False,
#         capturable: bool = False,
#         differentiable: bool = False,
#         fused: Optional[bool] = None,
#     ):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
#         if not 0.0 <= weight_decay:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

#         defaults = dict(
#             lr=lr,
#             betas=betas,
#             eps=eps,
#             weight_decay=weight_decay,
#             amsgrad=amsgrad,
#             maximize=maximize,
#             foreach=foreach,
#             capturable=capturable,
#             differentiable=differentiable,
#             fused=fused,
#         )
#         group0, group1, group2 = [], [], []
#         for n, p in named_params:
#             if n.endswith(("_s", ".s")):
#                 group2.append(p)
#             elif n.endswith(("_u", ".u", "_vh", ".vh")):
#                 continue
#             elif n.endswith("weight"):
#                 group1.append(p)
#             else:
#                 group0.append(p)
#         super().__init__(group0, defaults)
#         self.add_param_group({"params": group1})
#         self.add_param_group({"params": group2})

#         # need to reset the different parameter groups
#         # group 0 -> non-2D things (bias, norms, etc)
#         # group 1 -> weight matrices used at the top of training
#         # group 2 -> sigma matrices

#         self.rank = dist.get_rank() if dist.is_initialized() else 0
#         self.skip_parameters = []

#         if fused:
#             if differentiable:
#                 raise RuntimeError("`fused` does not support `differentiable`")
#             self._step_supports_amp_scaling = True
#             # TODO(crcrpar): [low prec params & their higher prec copy]
#             # Suppor AMP with FP16/BF16 model params which would need
#             # higher prec copy of params to do update math in higher prec to
#             # alleviate the loss of information.
#             if not all(
#                 p.is_cuda and torch.is_floating_point(p) for pg in self.param_groups for p in pg["params"]
#             ):
#                 raise RuntimeError("`fused=True` requires all the params to be CUDA, floating point Tensor")
#             if foreach:
#                 raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

#     def __setstate__(self, state):
#         super().__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault("amsgrad", False)
#             group.setdefault("maximize", False)
#             group.setdefault("foreach", None)
#             group.setdefault("capturable", False)
#             group.setdefault("differentiable", False)
#             group.setdefault("fused", None)
#         state_values = list(self.state.values())
#         step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]["step"])
#         if not step_is_tensor:
#             for s in state_values:
#                 s["step"] = torch.tensor(float(s["step"]))

#     def seperate_sigma_group(self, model: torch.nn.Module, skip_key: tuple = (".s", "_s")):
#         new_group = []
#         for c, (n, p) in enumerate(model.named_parameters()):
#             if p.requires_grad and n.endswith(skip_key):
#                 self.skip_parameters.append(c)

#     def zero_grad(self, set_to_none: bool = True):
#         r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

#         Args:
#             set_to_none (bool): instead of setting to zero, set the grads to None.
#                 This will in general have lower memory footprint, and can modestly improve performance.
#                 However, it changes certain behaviors. For example:
#                 1. When the user tries to access a gradient and perform manual ops on it,
#                 a None attribute or a Tensor full of 0s will behave differently.
#                 2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
#                 are guaranteed to be None for params that did not receive a gradient.
#                 3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
#                 (in one case it does the step with a gradient of 0 and in the other it skips
#                 the step altogether).
#         """
#         foreach = self.defaults.get("foreach", False)

#         if not hasattr(self, "_zero_grad_profile_name"):
#             self._patch_step_function()
#         if foreach:
#             per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
#         with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
#             for group in self.param_groups:
#                 for c, p in enumerate(group["params"]):
#                     if p.grad is not None and c not in self.skip_parameters:
#                         if set_to_none:
#                             p.grad = None
#                         else:
#                             if p.grad.grad_fn is not None:
#                                 p.grad.detach_()
#                             else:
#                                 p.grad.requires_grad_(False)
#                             if not foreach or p.grad.is_sparse:
#                                 p.grad.zero_()
#                             else:
#                                 per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad)
#             if foreach:
#                 for _, per_dtype_grads in per_device_and_dtype_grads.items():
#                     for grads in per_dtype_grads.values():
#                         torch._foreach_zero_(grads)

#     def _init_group(
#         self,
#         group,
#         params_with_grad,
#         grads,
#         exp_avgs,
#         exp_avg_sqs,
#         max_exp_avg_sqs,
#         state_steps,
#     ):
#         # NOTE: this is where we need to remove the S params from the
#         for c, p in enumerate(group["params"]):
#             if p.grad is not None and c not in self.skip_parameters:
#                 params_with_grad.append(p)
#                 if p.grad.is_sparse:
#                     raise RuntimeError(
#                         "Adam does not support sparse gradients, please consider SparseAdam instead",
#                     )
#                 grads.append(p.grad)

#                 state = self.state[p]
#                 # Lazy state initialization
#                 if len(state) == 0:
#                     state["step"] = (
#                         torch.zeros((1,), dtype=torch.float, device=p.device)
#                         if group["capturable"] or group["fused"]
#                         else torch.tensor(0.0)
#                     )
#                     # Exponential moving average of gradient values
#                     state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                     # Exponential moving average of squared gradient values
#                     state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                     if group["amsgrad"]:
#                         # Maintains max of all exp. moving avg. of sq. grad. values
#                         state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

#                 exp_avgs.append(state["exp_avg"])
#                 exp_avg_sqs.append(state["exp_avg_sq"])

#                 if group["amsgrad"]:
#                     max_exp_avg_sqs.append(state["max_exp_avg_sq"])
#                 if group["differentiable"] and state["step"].requires_grad:
#                     raise RuntimeError("`requires_grad` is not supported for `step` in differentiable mode")
#                 state_steps.append(state["step"])

#     @_use_grad_for_differentiable
#     def step(self, closure=None):
#         """Performs a single optimization step.

#         Args:
#             closure (Callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         self._cuda_graph_capture_health_check()

#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             params_with_grad = []
#             grads = []
#             exp_avgs = []
#             exp_avg_sqs = []
#             max_exp_avg_sqs = []
#             state_steps = []
#             beta1, beta2 = group["betas"]

#             self._init_group(
#                 group,
#                 params_with_grad,
#                 grads,
#                 exp_avgs,
#                 exp_avg_sqs,
#                 max_exp_avg_sqs,
#                 state_steps,
#             )

#             adam(
#                 params_with_grad,
#                 grads,
#                 exp_avgs,
#                 exp_avg_sqs,
#                 max_exp_avg_sqs,
#                 state_steps,
#                 amsgrad=group["amsgrad"],
#                 beta1=beta1,
#                 beta2=beta2,
#                 lr=group["lr"],
#                 weight_decay=group["weight_decay"],
#                 eps=group["eps"],
#                 maximize=group["maximize"],
#                 foreach=group["foreach"],
#                 capturable=group["capturable"],
#                 differentiable=group["differentiable"],
#                 fused=group["fused"],
#                 grad_scale=getattr(self, "grad_scale", None),
#                 found_inf=getattr(self, "found_inf", None),
#             )

#         return loss
