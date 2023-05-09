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
        full_rank_sigma: bool = False,
        keep_fist_layer: bool = False,
    ):
        super().__init__()
        self.uvthreshold = uvthreshold
        self.sigma_cutoff_fraction = sigma_cutoff_fraction
        self.sync_usv = sync_usv
        self.full_rank_sigma = full_rank_sigma
        # print('before replace layers')
        self.first_layer = keep_fist_layer
        self.local_model = self._replace_layers(existing_model)
        if dist.is_initialized():
            if dist.get_rank() == 0:
                log.info("Initializing DDP")
            self.ddp_model = DDP(self.local_model, find_unused_parameters=True)
        try:
            if dist.get_rank() == 0:
                print(self.ddp_model)
        except RuntimeError:  # dist is not initialized
            self.ddp_model = self.local_model
            print(self.local_model)

        # self.compiled_model = torch.compile(self.ddp_model)
        # raise ValueError("")
        self.stability_frequency = stability_frequency
        self.call_count = 0
        self.skip_stability = False
        self.delay = delay
        self.stable_list = []

    def _replace_layers(self, module, name=None, process_group=None):
        module_output = module
        # print(f'wrapping {name} {module}')
        if isinstance(module, nn.Linear):
            if not self.first_layer:
                module_output = SVDLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    uvthreshold=self.uvthreshold,
                    sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                    sync_usv=self.sync_usv,
                    full_rank_sigma=self.full_rank_sigma,
                ).to(device=module.weight.device, dtype=module.weight.dtype)
                # module_output = parametrizations.orthogonal(module_output, name="u")
                # module_output = parametrizations.orthogonal(module_output, name="vh")  # TODO: trans?
                # module_output = torch.compile(module_output)
            else:
                self.first_layer = False
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
        # if self.skip_stability:
        #     return
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        if rank == 0:
            log.info("Testing Stability")
        all_stable = True
        num_stable, total = 0, 0
        for c, (name, mod) in enumerate(module.named_modules()):
            # try:
            if hasattr(mod, "test_stability"):
                uchanging, k, perc, stable = mod.test_stability()
                total += 1
                if rank == 0:
                    log.info(f"Layer: {name}: UVh: {uchanging:.3f}, k: {k}, % params: {perc*100:.3f}")
                if stable:
                    num_stable += 1

        if rank == 0:
            log.info(
                f"Stablity stats: {num_stable} of {total} layers with fixed U + Vh -> "
                f"{100 * num_stable / total:.4f}%",
            )
        if all_stable:
            self.skip_stability = True

    @torch.no_grad()
    def get_perc_params_all_layers(self, module):
        # self.sync_models()
        # if self.skip_stability:
        #     return
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        percs, actives, normals = [], [], []
        full_active = 0
        full_normal = 0
        for c, (name, mod) in enumerate(module.named_modules()):
            if hasattr(mod, "get_perc_params"):
                perc_params, active_params, normal_params = mod.get_perc_params()
                full_active += active_params
                full_normal += normal_params
            else:
                # TODO: need to get the parameters of the other modules??
                pass

        if rank == 0:
            log.info(
                f"Active Params: {100 * (full_active / full_normal):.4f}%",
            )
        return 100 * (full_active / full_normal), full_active, full_normal

    def forward(self, inputs):
        if self.ddp_model.training:
            self.call_count += 1
            if (
                self.call_count % self.stability_frequency == self.stability_frequency - 1
                and self.call_count >= self.delay
            ):
                self.test_basis_stability_all_layers(module=self.ddp_model)
            # self.train()
        # if self.call_count > (self.stability_frequency * 4 + self.delay):
        #     return self.local_model(inputs)
        return self.ddp_model(inputs)

    @torch.no_grad()
    def sync_models(self, verbose=True):
        if not dist.is_initialized():
            return
        rank = dist.get_rank()
        if rank == 0:
            log.info("Syncing layer.weight on all layers")
        sz = dist.get_world_size()
        waits = []
        for n, p in self.local_model.named_parameters():
            if p.requires_grad:
                if verbose and n.endswith(".s"):
                    log.info(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
                if not p.is_contiguous():
                    p.set_(p.contiguous())
                p /= sz
                waits.append(dist.all_reduce(p, async_op=True))  # sum
        for w in waits:
            w.wait()


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
        full_rank_sigma: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SVDLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.full_rank_sigma = full_rank_sigma

        self.weight = torch.empty((out_features, in_features), **factory_kwargs)
        if out_features >= in_features:  # simplest case (no transpose)
            self.trans = False
            w = self.weight
        else:
            self.trans = True
            w = self.weight.T
            
        # u, s, vh = torch.linalg.svd(w, full_matrices=False)
        k = min(tuple(w.shape))
        self.u = torch.empty((w.shape[0], k), **factory_kwargs)
        self.vh = torch.empty((k, w.shape[1]), **factory_kwargs)
        self.s = torch.empty((k, k) if self.full_rank_sigma else k, **factory_kwargs)
        # print(f"first shapes w: {tuple(w.shape)} u: {tuple(self.u.shape)} s: {tuple(self.s.shape)} vh: {tuple(self.vh.shape)}")

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
        self.s_prev = None
        self.k = min(in_features, out_features)
        self.uthreshold = uvthreshold
        self.vhthreshold = uvthreshold

        self.prev_uvh = None

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            # w = (self.q @ self.r).T if self.trans else self.q @ self.r
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.compile()
    def get_weight(self):
        if not self.u_fixed and not self.vh_fixed:
            # if both are not fixed -> normal training
            # if dist.get_rank() == 0:
            #     log.info("Using self.weight")
            return self.weight
        # detach sets 'requires grad' to False
        # if self.training:
        #     self.s.requires_grad = True
        #     self.u.requires_grad = False if self.u_fixed else True
        #     self.vh.requires_grad = False if self.u_fixed else True
        # u, vh = self.u, self.vh
        u = self.u.detach() if self.u_fixed else self.u
        vh = self.vh.detach() if self.vh_fixed else self.vh

        s = self.s if self.full_rank_sigma else torch.diag(self.s)

        w = torch.linalg.multi_dot([u, s, vh])
        ret = w.T if self.trans else w
        with torch.no_grad():
            self.weight *= 0
            self.weight += ret
        # if dist.get_rank() == 0:
        #     log.info("Using USV")
        return ret

    # @torch.compile()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.get_weight()
        return F.linear(input, w, self.bias)

    @torch.no_grad()
    def test_stability(self):
        # TODO: should we make sure to have S be the same across processes?
        rank = dist.get_rank() if dist.is_initialized() else 0

        # updating usv in full rank is different!

        # if self.u_fixed and self.vh_fixed:
        #     sdiff = self.s_prev - self.s
        #     self.s_prev = self.s.data.clone()
        #     if rank == 0:
        #         print(f"s diff: {sdiff.mean():.4f}, {sdiff.min():.4f}, {sdiff.max():.4f}")
        #     # switch back to check on the SVD stuff
        #     # self.u_fixed = False
        #     normal_params = self.weight.numel()
        #     total_params = (self.u.shape[0] * self.k) + (self.k ** 2) + (self.k + self.v.shape[1])
        #     perc_params = total_params / normal_params
        #     return 2, self.k, perc_params, True
        set_usvh = True
        if self.full_rank_sigma and self.u_fixed:
            if rank == 0:
                log.info(f"in full rank sigma update of usvh")
            self._update_usv()
            retu, retvh = 2, 2
            # self.u_fixed = False
            # self._update_k()
            set_usvh = False
            u, s, vh = self.u, self.s, self.vh
            uvh = u @ vh
        else:
            # w = self.weight.T if self.trans else self.weight
            w = self.get_weight()
            w = w.T if self.trans else w
            dtp = w.dtype
            u, s, vh = torch.linalg.svd(w.to(torch.float32), full_matrices=False)  # , driver="gesvd")
            u = u.to(dtp)
            s = s.to(dtp)
            vh = vh.to(dtp)
            uvh = u @ vh
            if self.prev_uvh is None:
                if rank == 0:
                    log.info("in first stability update")
                self.prev_uvh = uvh
                self.u.zero_()
                self.u.add_(u)
                self.s.zero_()
                self.s.add_(torch.diag(s) if self.full_rank_sigma else s)
                self.vh.zero_()
                self.vh.add_(vh)
                return 0, self.k, 1., self.u_fixed
            self.prev_uvh = uvh
            if rank == 0:
                log.info("in normal stability update")

        # use cosine similarity (dot product for orthonormal) to determine similarity
        csim = self.cossim(self.prev_uvh, uvh)
        self.prev_uvh = uvh
        csmean, _ = csim.mean(), csim.std()
        if csmean > self.uthreshold:
            # if dist.get_rank() == 0:
            #     udiff = u - self.u
            #     vhdiff = vh - self.vh
            #     print(f"{udiff.mean():.4f}, {udiff.max():.4f}, {udiff.mean():.4f}")
            #     # print(f"{udiff.mean():.4f}, {udiff.max():.4f}, {udiff.mean():.4f}")
            #     print(f"{vhdiff.mean():.4f}, {vhdiff.max():.4f}, {vhdiff.mean():.4f}")
                
            self.u_fixed, self.vh_fixed = True, True
            if set_usvh:
                self.u.zero_()
                self.u.add_(u)
                self.vh.zero_()
                self.vh.add_(vh)
                if self.full_rank_sigma:
                    self.s.zero_()
                    self.s[:self.k, :self.k].add_(torch.diag(s[:self.k]))
                else:
                    self.s.zero_()
                    self.s[:self.k].add_(s[:self.k])

            # if dist.get_rank() == 0:
            #     print(f"u: {self.u.mean():.4f}, {self.u.min():.4f}, {self.u.max():.4f}, {self.u.std():.4f}")
            #     print(f"s: {self.s.mean():.4f}, {self.s.min():.4f}, {self.s.max():.4f}, {self.s.std():.4f}")
            #     print(f"vh: {self.vh.mean():.4f}, {self.vh.min():.4f}, {self.vh.max():.4f}, {self.vh.std():.4f}")
        retu = csmean
        # retvh = csmean

        self._update_k()
        self.s_prev = self.s.data.clone()
        perc_params, _, _ = self.get_perc_params()
        
        return retu, self.k, perc_params, self.u_fixed
    
    @torch.no_grad()
    def _update_usv(self):
        if not self.full_rank_sigma and not self.u_fixed:
            raise ValueError("this function is only for full-rank sigma with usvh is fixed")
        # NOTE: no slicing because need the shapes to line up. self.s[self.k:, self.k:] should be 0?
        usig, sig, vhsig = torch.linalg.svd(self.s)  # square mat, full mat arg doesnt matter
        usig[torch.abs(usig) < 1e-5] *= 0
        vhsig[torch.abs(vhsig) < 1e-5] *= 0
        sig[torch.abs(sig) < 1e-6] *= 0

        # cutoff vectors with only small changes --------------------------------------
        # in this case, remove the irrelevant vectors here not after multi!
        cutoff = sig[0] * self.sigma_cutoff_fraction
        nz = torch.nonzero(sig < cutoff)
        if len(nz) == 0:
            # In this case ALL of the basis vectors are useful
            k = sig.shape[0]
        else:
            k = nz[0].item()

        usig[:, k:].mul_(0)
        vhsig[k:].mul_(0)
        sig[k:].mul_(0)
        # -----------------------------------------------------------------------------

        holdu = self.u @ usig
        self.u.zero_()
        self.u.add_(holdu.contiguous())
        holdvh = vhsig @ self.vh
        self.vh.zero_()
        self.vh.add_(holdvh.contiguous())
        self.s.zero_()
        self.s.add_(torch.diag(sig))
        # to be safe: set weight to be the same here too
        w = torch.linalg.multi_dot([self.u, self.s, self.vh])
        ret = w.T if self.trans else w
        self.weight *= 0
        self.weight += ret
        # let this get handled by other function?
        # if dist.get_rank() == 0:
        #     print(f"u: {usig.mean():.4f}, {usig.min():.4f}, {usig.max():.4f}, {usig.std():.4f}")
        #     print(f"s: {sig.mean():.4f}, {sig.min():.4f}, {sig.max():.4f}, {sig.std():.4f}")
        #     print(f"vh: {vhsig.mean():.4f}, {vhsig.min():.4f}, {vhsig.max():.4f}, {vhsig.std():.4f}")

    @torch.no_grad()
    def _update_k(self):
        # adjust K to slice of less important singular values
        s = torch.diag(self.s) if self.full_rank_sigma else self.s
        if self.u_fixed and self.vh_fixed:
            cutoff = s[0] * self.sigma_cutoff_fraction
            nz = torch.nonzero(s < cutoff)
            if len(nz) == 0:
                # In this case ALL of the basis vectors are useful
                self.k = s.shape[0]
            else:
                self.k = nz[0].item()

        self.u[:, self.k:] *= 0
        self.vh[self.k:] *= 0
        if self.full_rank_sigma:
            self.s[self.k:, self.k:].mul_(0)
        else:
            self.s[self.k:].mul_(0)

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

        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def get_perc_params(self):
        normal_params = self.weight.numel()
        if self.u_fixed and self.vh_fixed:
            active_params = (self.u.shape[0] * self.k) + (self.k ** 2) + (self.k + self.vh.shape[1])
        else:
            active_params = normal_params
        perc_params = active_params / normal_params
        return perc_params, active_params, normal_params
