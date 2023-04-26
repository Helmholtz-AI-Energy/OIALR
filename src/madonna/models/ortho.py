import logging
import math
from copy import copy, deepcopy
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch._torch_docs import reproducibility_notes
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import parametrizations, parametrize

from ..utils import utils

log = logging.getLogger(__name__)


class QROrthoFixingModel(nn.Module):
    def __init__(
        self,
        existing_model: nn.Module,
        stability_frequency: int = 10,
        delay: int = 100,
        qthreshold: float = 0.999,
    ):
        super().__init__()
        self.track_stab_lst = {}
        self.qthreshold = qthreshold
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
        # raise ValueError("")
        self.stability_frequency = stability_frequency
        self.call_count = 0
        self.skip_stability = False
        self.delay = delay
        self.stable_list = []

    def _replace_layers(self, module, name=None, process_group=None):
        module_output = module
        if isinstance(module, nn.Linear):
            module_output = QROrthoLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                qthreshold=self.qthreshold,
            ).to(device=module.weight.device, dtype=module.weight.dtype)
            module_output = parametrizations.orthogonal(module_output, name="q")
            module_output = parametrize.register_parametrization(
                module_output,
                tensor_name="r",
                parametrization=Triu(),
            )
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
        self.sync_models()
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
                changing = mod.test_q_stability()

                if changing == 1:
                    if rank == 0:
                        log.info(f"Fixing Q for layer: {name} - step count: {self.call_count}")
                    num_stable += 1
                elif changing < 1:
                    if rank == 0:
                        log.info(f"Training normally for layer: {name}, stabiltiy: {changing}")
                    all_stable = False
                else:
                    num_stable += 1
                    # log.debug(f"Q was fixed previously for layer: {name}")  # TODO: remove!
            # else:
            #     # if rank == 0:
            #     #     log.info(f"Syncing params of {name}")
        if dist.get_rank() == 0:
            log.info(
                f"Stablity stats: {num_stable} of {total} layers with fixed Q -> {100 * num_stable / total:.4f}%",
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

    # @torch.no_grad()
    # def sync_models(self):
    #     if not dist.is_initialized():
    #         return
    #     sz = dist.get_world_size()
    #     waits = []
    #     for n, p in self.target_model.named_parameters():
    #         if p.requires_grad:
    #             p /= sz
    #             waits.append(dist.all_reduce(p, async_op=True))  # sum
    #     for w in waits:
    #         w.wait()


class Triu(nn.Module):
    def forward(self, x):
        a = x.triu()
        return a


class QROrthoLinear(nn.Module):
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
        qthreshold: float = 0.9,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(QROrthoLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # IDEA: replace Q and R with normal torch layers
        #           y = xA.T + b
        #           y = x * (Q * R).T + b
        #           y = x * R * Q + b
        # if most things are TS, should I even worry about the SF case?
        if out_features >= in_features:  # simplest case (no transpose)
            self.q = nn.Parameter(torch.zeros((out_features, in_features), **factory_kwargs))
            self.r = nn.Parameter(torch.zeros((in_features, in_features), **factory_kwargs))
            self.trans = False
        else:
            self.q = nn.Parameter(torch.zeros((in_features, out_features), **factory_kwargs))
            self.r = nn.Parameter(torch.zeros((out_features, out_features), **factory_kwargs))
            self.trans = True

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.cossim = nn.CosineSimilarity(dim=0)
        self.q_fixed = False
        self.q_prev = None
        self.qthreshold = qthreshold
        # del self.weight

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109

        # nn.init.uniform_(self.r)
        nn.init.orthogonal_(self.q)
        nn.init.kaiming_uniform_(self.r, a=math.sqrt(5))
        self.r.triu_()
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            w = (self.qlayer @ self.r).T if self.trans else self.qlayer @ self.r
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # @torch.compile()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.q_fixed:
            # with torch.no_grad():
            self.q.requires_grad = False
            # self.q.grad = None
        w = (self.q @ self.r).T if self.trans else self.q @ self.r
        return F.linear(input, w, self.bias)

    @torch.no_grad()
    # @torch.compile()
    def test_q_stability(self):
        if self.q_fixed:
            # only do the switch once!
            # TODO: testing syncing R only here. remove later?
            # if dist.is_initialized():
            #     sz = dist.get_world_size()
            #     self.r /= sz
            #     dist.all_reduce(self.r, async_op=False)
            return 2

        if self.q_prev is None:
            self.q_prev = self.q.data.clone().detach()
            return 0

        csim = self.cossim(self.q_prev, self.qlayer.weight)
        csmean, _ = csim.mean(), csim.std()

        if csmean > self.qthreshold:
            self.q_fixed = 1

        # continue training normally
        return csmean

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
        )

    # def train(self: nn.Module, mode: bool = True) -> nn.Module:
    #     r"""Sets the module in training mode.
    #
    #     This has any effect only on certain modules. See documentations of
    #     particular modules for details of their behaviors in training/evaluation
    #     mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
    #     etc.
    #
    #     Args:
    #         mode (bool): whether to set training mode (``True``) or evaluation
    #                      mode (``False``). Default: ``True``.
    #
    #     Returns:
    #         Module: self
    #     """
    #     if not isinstance(mode, bool):
    #         raise ValueError("training mode is expected to be boolean")
    #
    #     # with torch.no_grad():
    #     #     if dist.is_initialized():
    #     #         self.r /= dist.get_world_size()
    #     #         dist.all_reduce(self.r)
    #     #     # self.r.triu_()
    #     #     # self.weight.set_(self.get_weight())
    #     #     # self.weight.set_((self.q @ self.r).T if self.trans else self.q @ self.r)
    #
    #     self.training = mode
    #     for module in self.children():
    #         module.train(mode)
    #     return self
