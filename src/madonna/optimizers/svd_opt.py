from collections import OrderedDict
from collections import abc as container_abcs
from collections import defaultdict
from typing import Optional

import torch
import torch.distributed as dist
import torch.optim as optim

# from torch.optim.optimizer import _use_grad_for_differentiable


optim.Adam


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


class SVDAdam(optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.skip_parameters = []

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            # TODO(crcrpar): [low prec params & their higher prec copy]
            # Suppor AMP with FP16/BF16 model params which would need
            # higher prec copy of params to do update math in higher prec to
            # alleviate the loss of information.
            if not all(
                p.is_cuda and torch.is_floating_point(p) for pg in self.param_groups for p in pg["params"]
            ):
                raise RuntimeError("`fused=True` requires all the params to be CUDA, floating point Tensor")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("fused", None)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]["step"])
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def set_skip_parameters(self, model: torch.nn.Module, skip_key: tuple = (".s", "_s")):
        for c, (n, p) in enumerate(model.named_parameters()):
            if p.requires_grad and n.endswith(skip_key):
                self.skip_parameters.append(c)

    def zero_grad(self, set_to_none: bool = True):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        foreach = self.defaults.get("foreach", False)

        if not hasattr(self, "_zero_grad_profile_name"):
            self._patch_step_function()
        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                for c, p in enumerate(group["params"]):
                    if p.grad is not None and c not in self.skip_parameters:
                        if set_to_none:
                            p.grad = None
                        else:
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            else:
                                p.grad.requires_grad_(False)
                            if not foreach or p.grad.is_sparse:
                                p.grad.zero_()
                            else:
                                per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad)
            if foreach:
                for _, per_dtype_grads in per_device_and_dtype_grads.items():
                    for grads in per_dtype_grads.values():
                        torch._foreach_zero_(grads)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        # NOTE: this is where we need to remove the S params from the
        for c, p in enumerate(group["params"]):
            if p.grad is not None and c not in self.skip_parameters:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead",
                    )
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0)
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                if group["differentiable"] and state["step"].requires_grad:
                    raise RuntimeError("`requires_grad` is not supported for `step` in differentiable mode")
                state_steps.append(state["step"])

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            optim.adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss
