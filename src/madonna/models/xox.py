import logging
import math
from copy import copy, deepcopy
from typing import Callable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch._torch_docs import reproducibility_notes
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import parametrizations, parametrize

from ..utils import utils

log = logging.getLogger(__name__)


class XOXModel(nn.Module):
    def __init__(
        self,
        existing_model: nn.Module,
        stability_frequency: int = 10,
        delay: int = 100,
        weight_activation: Callable = torch.heaviside,
        weight_activation_kwargs=None,
        train_xt: bool = True,
        perc_params: float = 0.8,
    ):
        super().__init__()
        if weight_activation_kwargs is None:
            weight_activation_kwargs = {"values": 0}
        else:
            self.weight_activation_kwargs = weight_activation_kwargs
        self.weight_activation = weight_activation
        self.train_xt = train_xt
        self.perc_params = perc_params

        self.target_model = self._replace_layers(existing_model)
        if dist.is_initialized():
            if dist.get_rank() == 0:
                log.info("Initializing DDP")
            self.target_model = DDP(self.target_model, find_unused_parameters=True)
        try:
            if dist.get_rank() == 0:
                print(self.target_model)
        except RuntimeError:  # dist is not initialized
            print(self.target_model)

        # self.target_model = torch.compile(self.target_model)
        # raise ValueError("")
        self.stability_frequency = stability_frequency
        self.call_count = 0
        self.skip_stability = False
        self.delay = delay
        self.stable_list = []

    def _replace_layers(self, module, name=None, process_group=None):
        module_output = module
        if isinstance(module, nn.Linear):
            module_output = XOXLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                weight_activation=self.weight_activation,
                weight_activation_kwargs=self.weight_activation_kwargs,
                train_xt=self.train_xt,
                perc_params=self.perc_params,
            ).to(device=module.weight.device, dtype=module.weight.dtype)
            # module_output = parametrizations.orthogonal(module_output, name="u")
            # module_output = parametrizations.orthogonal(module_output, name="vh")  # TODO: trans?
            # module_output = torch.compile(module_output)
        # SKIPPING CONV2D Layers!!
        for n, child in module.named_children():
            module_output.add_module(
                f"{n}",
                self._replace_layers(
                    child,
                    name=f"{n}",
                    process_group=process_group,
                ),
            )
        del module
        return module_output

    @torch.no_grad()
    def test_basis_stability_all_layers(self, module):
        # self.sync_models()
        if self.skip_stability:
            return
        rank = dist.get_rank()
        if rank == 0:
            log.info("Testing Stability")
        all_stable = True
        num_stable, total = 0, 0
        for c, (name, mod) in enumerate(module.named_modules()):
            if hasattr(mod, "test_q_stability"):
                total += 1
                uchanging, vhchaning, k = mod.test_stability()

                if rank == 0:
                    log.info(f"Layer: {name}: U: {uchanging}, Vh: {vhchaning}, k: {k}")
                if uchanging >= 1 and vhchaning >= 1:
                    num_stable += 1
                else:
                    all_stable = False

        if dist.get_rank() == 0:
            log.info(
                f"Stablity stats: {num_stable} of {total} layers with fixed U + Vh -> "
                f"{100 * num_stable / total:.4f}%",
            )
        if all_stable:
            self.skip_stability = True

    def forward(self, inputs):
        self.call_count += 1
        if (
            self.target_model.training
            and self.call_count % self.stability_frequency == self.stability_frequency - 1
            and self.call_count >= self.delay
        ):
            self.test_basis_stability_all_layers(module=self.target_model)
            # self.train()
        return self.target_model(inputs)

    @torch.no_grad()
    def sync_models(self):
        if not dist.is_initialized():
            return
        rank = dist.get_rank()
        if rank == 0:
            log.info("Syncing layer.weight on all layers")
        sz = dist.get_world_size()
        waits = []
        for n, p in self.target_model.named_parameters():
            if p.requires_grad:
                p /= sz
                waits.append(dist.all_reduce(p, async_op=True))  # sum
        for w in waits:
            w.wait()


class Triu(nn.Module):
    def forward(self, x):
        a = x.triu()
        return a


class XOXLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        weight_activation: Callable = torch.heaviside,
        weight_activation_kwargs: dict = None,
        train_xt: bool = True,
        perc_params: float = 0.8,
    ) -> None:
        if weight_activation_kwargs is None:
            weight_activation_kwargs = {"values": 0}
        self.weight_activation_kwargs = weight_activation_kwargs
        self.weight_activation = weight_activation
        self.train_xt = train_xt
        factory_kwargs = {"device": device, "dtype": dtype}
        super(XOXLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        a = 1
        b = in_features + out_features
        c = -perc_params * in_features * out_features
        pos = (-b + torch.sqrt(b**2 - 4 * a * c)) / 2 * a
        neg = (-b - torch.sqrt(b**2 - 4 * a * c)) / 2 * a
        inner_dim = max(pos, neg)

        # self.weight = torch.empty((out_features, in_features), **factory_kwargs)
        if out_features >= in_features:  # simplest case (no transpose)
            self.x0 = torch.zeros((out_features, inner_dim), **factory_kwargs)
            self.o = torch.zeros((inner_dim, inner_dim), **factory_kwargs)
            self.x1 = torch.zeros((inner_dim, in_features), **factory_kwargs)
            self.trans = False
        else:
            self.x0 = torch.zeros((in_features, inner_dim), **factory_kwargs)
            self.o = torch.zeros((inner_dim, inner_dim), **factory_kwargs)
            self.x1 = torch.zeros((inner_dim, out_features), **factory_kwargs)
            self.trans = True

        self.x0 = nn.Parameter(self.u)
        self.o = nn.Parameter(self.s)
        self.x1 = nn.Parameter(self.vh)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109

        # TODO: should this be orthogonal init?
        nn.init.orthogonal_(self.x0)
        nn.init.orthogonal_(self.x1)
        nn.init.kaiming_uniform_(self.o, a=math.sqrt(5))

        if self.bias is not None:
            w = (self.x0 @ self.o @ self.x1).T if self.trans else self.x0 @ self.o @ self.x1
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # @torch.compile()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = (self.x0 @ self.o @ self.x1).T if self.trans else self.x0 @ self.o @ self.x1
        w = self.weight_activation(w, **self.weight_activation_kwargs)
        return F.linear(input, w, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
        )
