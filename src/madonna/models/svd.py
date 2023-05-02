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


class SVDFixingModel(nn.Module):
    def __init__(
        self,
        existing_model: nn.Module,
        stability_frequency: int = 10,
        delay: int = 100,
        uvthreshold: float = 0.999,
        sigma_cutoff_fraction: float = 0.1,
        sync_usv: bool = False,
        train_full_first: bool = False,
        full_rank_sigma: bool = False,
    ):
        super().__init__()
        self.uvthreshold = uvthreshold
        self.sigma_cutoff_fraction = sigma_cutoff_fraction
        self.sync_usv = sync_usv
        self.train_full_first = train_full_first
        self.full_rank_sigma = full_rank_sigma
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
            module_output = SVDLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                uvthreshold=self.uvthreshold,
                sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                sync_usv=self.sync_usv,
                train_full_first=self.train_full_first,
                full_rank_sigma=self.full_rank_sigma,
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


class SVDLinear(nn.Module):
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
        uvthreshold: float = 0.9,
        sigma_cutoff_fraction: float = 0.1,
        sync_usv: bool = False,
        train_full_first: bool = False,
        full_rank_sigma: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SVDLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.full_rank_sigma = full_rank_sigma

        self.weight = torch.empty((out_features, in_features), **factory_kwargs)
        if out_features >= in_features:  # simplest case (no transpose)
            self.u = torch.zeros((out_features, in_features), **factory_kwargs)
            if full_rank_sigma:
                self.s = torch.zeros((in_features, in_features), **factory_kwargs)
            else:
                self.s = torch.zeros(in_features, **factory_kwargs)
            self.vh = torch.zeros((in_features, in_features), **factory_kwargs)
            self.trans = False
        else:
            self.u = torch.zeros((in_features, out_features), **factory_kwargs)
            if full_rank_sigma:
                self.s = torch.zeros((out_features, out_features), **factory_kwargs)
            else:
                self.s = torch.zeros(out_features, **factory_kwargs)
            self.vh = torch.zeros((out_features, out_features), **factory_kwargs)
            self.trans = True

        # if train_full_first:
        #     self.weight = nn.Parameter(self.weight)
        #     self.s = nn.Parameter(self.s)
        #     self.vh = nn.Parameter(self.vh)
        # else:
        # TODO: set up training without full rank weights
        self.weight = nn.Parameter(self.weight)
        self.u = nn.Parameter(self.u)
        self.s = nn.Parameter(self.s)
        self.vh = nn.Parameter(self.vh)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.cossim = nn.CosineSimilarity(dim=0)
        self.sigma_cutoff_fraction = sigma_cutoff_fraction
        self.sync_usv = sync_usv
        self.u_fixed = False
        self.u_prev = None
        self.vh_fixed = False
        self.vh_prev = None
        self.k = min(in_features, out_features)
        self.uthreshold = uvthreshold
        self.vhthreshold = uvthreshold
        # if not train_full_first:
        #     del self.weight

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109

        # nn.init.uniform_(self.r)
        nn.init.orthogonal_(self.u)
        nn.init.orthogonal_(self.vh.T)
        nn.init.kaiming_uniform_(self.s, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            # w = (self.q @ self.r).T if self.trans else self.q @ self.r
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def get_weight(self):
        if not (self.u_fixed and self.vh_fixed):  # if both are not fixed -> normal training
            return self.weight
        # detach sets 'requires grad' to False
        u = self.u.detach() if self.u_fixed else self.u
        vh = self.vh.detach() if self.vh_fixed else self.vh
        if not self.full_rank_sigma:
            w = u[:, self.k] @ torch.diag(self.s[: self.k]) @ vh[: self.k]
        else:
            w = u[:, self.k] @ self.s[: self.k, : self.k] @ vh[: self.k]
        return w.T if self.trans else w

    # @torch.compile()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.get_weight()
        return F.linear(input, w, self.bias)

    @torch.no_grad()
    # @torch.compile()
    def test_stability(self):
        # TODO: should we make sure to have S be the same across processes?
        rank = dist.get_rank() if dist.is_initialized() else 0

        if self.u_fixed and self.vh_fixed:
            return 2, 2, self.k

        # TODO: should the weights be synced before this?
        u, s, vh = torch.linalg.svd(self.weight, full_matrices=False)

        if not self.u_fixed:
            # can test columns here since U has orthogonal columns
            if self.u_prev is None:
                self.u_prev = u.clone().detach()
                retu = 0
            else:
                ucsim = self.cossim(self.u_prev, u)
                ucsmean, _ = ucsim.mean(), ucsim.std()
                retu = ucsmean
                self.u_prev = u.clone().detach()
                if ucsmean > self.uthreshold:
                    self.u_fixed = 1
        else:
            retu = 1
        if not self.vh_fixed:
            # need to test on transpose here since ROWS of V.H are orthogonal
            if self.vh_prev is None:
                self.vh_prev = vh.clone().detach()
                retvh = 0
            else:
                vhcsim = self.cossim(self.vh_prev.T, vh.T)
                vhcsmean, _ = vhcsim.mean(), vhcsim.std()
                retvh = vhcsmean
                self.vh_prev = vh.clone().detach()
                if vhcsmean > self.vhthreshold:
                    self.vh_fixed = 1
        else:
            retvh = 1

        # once everything is fixed, adjust K
        if self.u_fixed and self.vh_fixed:
            cutoff = s[0] * self.sigma_cutoff_fraction
            self.k = torch.nonzero(s < cutoff)[0].item()
            if rank == 0:
                log.info("Fixing K in linear layer (name unknown)")
        return retu, retvh, self.k

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
        )

    def train(self: nn.Module, mode: bool = True) -> nn.Module:
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")

        if dist.is_initialized() and self.sync_usv and (self.u_fixed or self.vh_fixed):
            with torch.no_grad():
                self.u /= dist.get_world_size()
                uwait = dist.all_reduce(self.u, async_op=True)
                self.vh /= dist.get_world_size()
                vhwait = dist.all_reduce(self.vh, async_op=True)
                self.s /= dist.get_world_size()
                swait = dist.all_reduce(self.s, async_op=True)
                uwait.wait()
                vhwait.wait()
                swait.wait()
            # self.r.triu_()
            # self.weight.set_(self.get_weight())
            # self.weight.set_((self.q @ self.r).T if self.trans else self.q @ self.r)

        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
