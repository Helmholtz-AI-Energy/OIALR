import logging
import math
from copy import copy, deepcopy
from typing import Optional, Union, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
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
        keep_first_layer: bool = False,
        keep_last_layer: bool = True,
        update_from_simga: bool = True,
    ):
        super().__init__()
        self.uvthreshold = uvthreshold
        self.sigma_cutoff_fraction = sigma_cutoff_fraction
        self.sync_usv = sync_usv
        self.full_rank_sigma = full_rank_sigma
        self.update_from_simga = update_from_simga
        # print('before replace layers')
        self.first_layer = keep_first_layer
        self.keep_last_layer = keep_last_layer
        self.last_layer = None
        self.local_model = self._replace_layers(existing_model)
        if keep_last_layer:
            self._reset_last_layer(self.local_model)
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
        # for n, p in self.local_model.named_parameters():
        #     print(f"{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")

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
                    start_weight=module.weight,
                    start_bias=module.bias,
                    update_from_simga=self.update_from_simga,
                ).to(device=module.weight.device, dtype=module.weight.dtype)
                # module_output = parametrizations.orthogonal(module_output, name="u")
                # module_output = parametrizations.orthogonal(module_output, name="vh")  # TODO: trans?
                # module_output = torch.compile(module_output)
                self.last_layer = [module, name]
            else:
                self.first_layer = False
        elif isinstance(module, nn.MultiheadAttention):
            if not self.first_layer:
                module_output = SVDMultiheadAttention(
                    embed_dim=module.embed_dim,
                    num_heads=module.num_heads,
                    dropout=module.dropout,
                    bias=module.in_proj_bias is not None,
                    add_bias_kv=module.bias_k is not None,
                    add_zero_attn=module.add_zero_attn,
                    kdim=module.kdim,
                    vdim=module.vdim,
                    batch_first=module.batch_first,
                    # device=module., get from module.out_proj.weight!
                    # dtype=None,
                    uvh_threshold=self.uvthreshold,
                    sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                    sync_usv=self.sync_usv,  # TODO: should this even be here? are we letting them drift?
                    full_rank_sigma=self.full_rank_sigma,
                    start_q=module.q_proj_weight,
                    start_k=module.k_proj_weight,
                    start_v=module.v_proj_weight,
                    start_in_proj=module.in_proj_weight,
                    start_k_bias=module.bias_k,
                    start_v_bias=module.bias_v,
                    start_in_proj_bias=module.in_proj_bias,
                    update_from_simga=self.update_from_simga,
                ).to(device=module.out_proj.weight.device, dtype=module.out_proj.weight.dtype)
                self.last_layer = [module, name]
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
    
    def _reset_last_layer(self, module, name=None):
        # if dist.get_rank() == 0:
        #     print("replace", name)
        module_output = module
        if name == self.last_layer[1]:
            device = module.weight.device
            dtype = module.weight.dtype
            module_output = self.last_layer[0].to(device=device, dtype=dtype)
        for name, child in module.named_children():
            module_output.add_module(name, self._reset_last_layer(child, name))
        # del module
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
                    try:
                        uchange = f"{uchanging:.3f}"
                    except TypeError:  # in list return (qkv from attention...)
                        uchange = ""
                        for u in uchanging:
                            uchange += f"{u:.3f}, "
                    try:
                        percs = f"{perc * 100:.3f}"
                    except TypeError:  # in list return (qkv from attention...)
                        percs = ""
                        for p in perc:
                            percs += f"{p:.3f}, "
                    log.info(f"{name}: UVh: {uchange} - k: {k} - % active params: {percs}")
        # removing stability stuff, pain in the ass
        #         try: 
        #             stable[0]
        #             for s in stable:
        #                 if s:
        #                     num_stable += 1
        #         if stable:
        #             num_stable += 1

        # if rank == 0 and total > 0:
        #     log.info(
        #         f"Stablity stats: {num_stable} of {total} layers with fixed U + Vh -> "
        #         f"{100 * num_stable / total:.4f}%",
        #     )
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
        
        if full_normal == 0:
            full_normal = 1
            full_active = 1
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
        start_weight=None,
        start_bias=None,
        update_from_simga=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SVDLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.full_rank_sigma = full_rank_sigma
        self.update_from_simga = full_rank_sigma and update_from_simga

        if start_weight is not None:
            self.weight = start_weight
        else:
            self.weight = torch.empty((out_features, in_features), **factory_kwargs)
        self.weight = nn.Parameter(self.weight)

        if out_features >= in_features:  # simplest case (no transpose)
            self.trans = False
            w = self.weight
        else:
            self.trans = True
            w = self.weight.T
            
        # u, s, vh = torch.linalg.svd(w, full_matrices=False)
        k = min(tuple(w.shape))
        self.u = torch.zeros((w.shape[0], k), **factory_kwargs)
        self.vh = torch.zeros((k, w.shape[1]), **factory_kwargs)
        self.s = torch.zeros((k, k) if self.full_rank_sigma else k, **factory_kwargs)

        self.u = nn.Parameter(self.u)
        self.s = nn.Parameter(self.s, requires_grad=False)
        self.vh = nn.Parameter(self.vh)

        if bias:
            if start_bias is None:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.bias = nn.Parameter(start_bias)
        else:
            self.register_parameter("bias", None)
        
        if start_weight is None:
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
        #     # sdiff = self.s_prev - self.s
        #     # self.s_prev = self.s.data.clone()
        #     # if rank == 0:
        #     #     print(f"s diff: {sdiff.mean():.4f}, {sdiff.min():.4f}, {sdiff.max():.4f}")
        #     # switch back to check on the SVD stuff??
        #     # self.u_fixed = False
        #     perc_params, _, _ = self.get_perc_params()
        #     return 2, self.k, perc_params, True
        set_usvh = True
        if self.full_rank_sigma and self.u_fixed and self.update_from_simga:
            if rank == 0:
                log.info("in full rank sigma update of usvh")
            self._update_usv()
            # self.u_fixed = False
            # self._update_k()
            set_usvh = False
            u, s, vh = self.u, self.s, self.vh
            uvh = u @ vh
            # perc_params, _, _ = self.get_perc_params()
            # return 2, self.k, perc_params, True
        else:
            w = self.weight.T if self.trans else self.weight
            # w = self.get_weight()
            # w = w.T if self.trans else w
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
        self.prev_uvh = uvh
        if csmean > self.uthreshold:
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

            # self.u[torch.abs(self.u) < 1e-5] *= 0
            # self.vh[torch.abs(self.vh) < 1e-5] *= 0
            # self.s[torch.abs(self.s) < 1e-6] *= 0

            self._update_k()
        perc_params, _, _ = self.get_perc_params()
        
        return csmean, self.k, perc_params, self.u_fixed

    @torch.no_grad()
    @torch.compile()
    def _update_usv(self):
        if not self.full_rank_sigma and not self.u_fixed:
            raise ValueError("this function is only for full-rank sigma with usvh is fixed")
        # NOTE: no slicing because need the shapes to line up. self.s[self.k:, self.k:] should be 0?
        usig, sig, vhsig = torch.linalg.svd(self.s)  # square mat, full mat arg doesnt matter
        # usig[torch.abs(usig) < 1e-5] *= 0
        # vhsig[torch.abs(vhsig) < 1e-5] *= 0
        # sig[torch.abs(sig) < 1e-6] *= 0

        # # cutoff vectors with only small changes --------------------------------------
        # # in this case, remove the irrelevant vectors here not after multi!
        # cutoff = sig[0] * self.sigma_cutoff_fraction
        # nz = torch.nonzero(sig < cutoff)
        # if len(nz) == 0:
        #     # In this case ALL of the basis vectors are useful
        #     k = sig.shape[0]
        # else:
        #     k = nz[0].item()

        # usig[:, k:].mul_(0)
        # vhsig[k:].mul_(0)
        # sig[k:].mul_(0)
        # # -----------------------------------------------------------------------------

        holdu = self.u @ usig
        self.u.zero_()
        self.u.add_(holdu)
        holdvh = vhsig @ self.vh
        self.vh.zero_()
        self.vh.add_(holdvh)
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
        prevk = self.k
        if self.u_fixed and self.vh_fixed:
            cutoff = s[0] * self.sigma_cutoff_fraction
            nz = torch.nonzero(s < cutoff)
            if len(nz) == 0:
                # In this case ALL of the basis vectors are useful
                newk = s.shape[0]
            else:
                newk = nz[0].item()
        self.k = newk
        # if newk < 0.75 * prevk:
        #     self.k = int(prevk * 0.75)
        #     log.debug(f"values of S after dropping slice value by only 75% of suggestion: {s[:5]}")

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

    @torch.compile()
    def get_perc_params(self):
        normal_params = self.weight.numel()
        if self.u_fixed and self.vh_fixed:
            # active_params = (self.u.shape[0] * self.k) + (self.k ** 2) + (self.k + self.vh.shape[1])
            trainable_params = self.k ** 2
        else:
            trainable_params = normal_params
        perc_params = trainable_params / normal_params
        return perc_params, trainable_params, normal_params


class SVDMultiheadAttention(nn.Module):
    r"""
    Almost all of this class is based on the torch standard MultiheadAttention class
    I have changed things to work with the SVD abstractions, but that is it.


    Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``forward()`` will use the optimized implementations of
    ``scaled_dot_product_attention()``.

    In addition to support for the new ``scaled_dot_product_attention()``
    function, for speeding up Inference, MHA will use
    fastpath inference with support for Nested Tensors, iff:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor.
    - inputs are batched (3D) with ``batch_first==True``
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``batch_first`` is ``True`` and the input is batched
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed
    - autocast is disabled

    If the optimized inference fastpath implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            device=None,
            dtype=None,
            uvh_threshold=0.9,
            sigma_cutoff_fraction=0.1,
            sync_usv=False,  # TODO: should this even be here? are we letting them drift?
            full_rank_sigma=True,
            start_q=None,
            start_k=None,
            start_v=None,
            start_in_proj=None,
            start_k_bias=None,
            start_v_bias=None,
            start_in_proj_bias=None,
            update_from_simga: bool = True,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.update_from_simga = full_rank_sigma and update_from_simga
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.full_rank_sigma = full_rank_sigma
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.qu = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            if self.full_rank_sigma:
                self.qs = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            else:
                self.qs = Parameter(torch.empty((embed_dim), **factory_kwargs))
            self.qvh = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.q_trans = False
            self.q_slice = embed_dim

            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            if self.kdim > embed_dim:
                # u - kdim x embed, s - embed x embed, vh - embed x embed -> after trans is embed x kdim
                self.k_trans = True
                self.k_u = Parameter(torch.empty((self.kdim, embed_dim), **factory_kwargs))
                if not self.full_rank_sigma:
                    self.k_s = Parameter(torch.empty((embed_dim), **factory_kwargs))
                else:
                    self.k_s = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
                self.k_vh = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
                self.k_slice = embed_dim
            else:
                # u - embed x kdim, s - kdim x kdim, vh - kdim x kdim
                self.k_trans = False
                self.k_u = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
                if not self.full_rank_sigma:
                    self.k_s = Parameter(torch.empty((self.kdim), **factory_kwargs))
                else:
                    self.k_s = Parameter(torch.empty((self.kdim, self.kdim), **factory_kwargs))
                self.k_vh = Parameter(torch.empty((self.kdim, self.kdim), **factory_kwargs))
                self.k_slice = self.kdim

            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            if self.vdim > embed_dim:
                # u - vdim x embed, s - embed x embed, vh - embed x embed -> after trans is embed x vdim
                self.v_trans = True
                self.v_u = Parameter(torch.empty((self.vdim, embed_dim), **factory_kwargs))
                if not self.full_rank_sigma:
                    self.v_s = Parameter(torch.empty((embed_dim), **factory_kwargs))
                else:
                    self.v_s = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
                self.v_vh = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
                self.v_slice = embed_dim
            else:
                # u - embed x vdim, s - vdim x vdim, vh - vdim x vdim
                self.v_trans = False
                self.v_u = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
                if not self.full_rank_sigma:
                    self.v_s = Parameter(torch.empty((self.vdim), **factory_kwargs))
                else:
                    self.v_s = Parameter(torch.empty((self.vdim, self.vdim), **factory_kwargs))
                self.v_vh = Parameter(torch.empty((self.vdim, self.vdim), **factory_kwargs))
                self.v_slice = self.vdim
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            # in_proj is always TS
            self.in_proj_u = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            if not self.full_rank_sigma:
                self.in_proj_s = Parameter(torch.empty((embed_dim), **factory_kwargs))
            else:
                self.in_proj_s = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.in_proj_vh = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.in_proj_trans = False
            self.in_proj_slice = embed_dim

            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()
        self.sigma_cutoff_fraction = sigma_cutoff_fraction
        self.uvh_fixed_q = False
        self.uvh_fixed_k = False
        self.uvh_fixed_v = False
        self.uvh_fixed_in_proj = False
        if not self._qkv_same_embed_dim:
            self.prev_uvh_q = None
            self.prev_uvh_k = None
            self.prev_uvh_v = None
        else:
            self.prev_uvh_in_proj = None

        self.cossim = nn.CosineSimilarity(dim=0)
        self.uvh_threshold = uvh_threshold

        with torch.no_grad():  # set class params from existing
            if not self._qkv_same_embed_dim:  # in this case, have q, k, v and bias_k and bias_v
                factory = {"device": start_q.device, "dtype": start_q.dtype}
                if start_q is not None:
                    self.q_proj_weight.zero_()
                    self.q_proj_weight.data = self.q_proj_weight.data.to(**factory)
                    self.q_proj_weight.add_(start_q)
                if start_k is not None:
                    self.k_proj_weight.zero_()
                    self.k_proj_weight.data = self.k_proj_weight.data.to(**factory)
                    self.k_proj_weight.add_(start_k)
                if start_v is not None:
                    self.v_proj_weight.zero_()
                    self.v_proj_weight.data = self.v_proj_weight.data.to(**factory)
                    self.v_proj_weight.add_(start_v)
                if add_bias_kv:
                    self.bias_k.zero_()
                    self.bias_k.data = self.bias_k.data.to(**factory)
                    self.bias_k.add_(start_k_bias)
                    self.bias_v.zero_()
                    self.bias_v.data = self.bias_v.data.to(**factory)
                    self.bias_v.add_(start_v_bias)
            else:
                if start_in_proj is not None:
                    factory = {"device": start_in_proj.device, "dtype": start_in_proj.dtype}
                    self.in_proj_weight.data = self.in_proj_weight.data.to(**factory)
                    self.in_proj_weight.zero_()
                    self.in_proj_weight.add_(start_in_proj)
            if bias and start_in_proj_bias is not None:
                factory = {"device": start_in_proj_bias.device, "dtype": start_in_proj_bias.dtype}
                self.in_proj_bias.zero_()
                self.in_proj_bias.data = self.in_proj_bias.data.to(**factory)
                self.in_proj_bias.add_(start_in_proj_bias)

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    @torch.compile()
    def _get_q(self):
        if self.q_proj_weight is None:
            return self.q_proj_weight
        if not self.uvh_fixed_q:
            return self.q_proj_weight
        u = self.q_u.detach()
        vh = self.q_vh.detach()

        s = self.q_s if self.full_rank_sigma else torch.diag(self.q_s)

        ret = torch.linalg.multi_dot([u, s, vh])
        # no transpose for q - square matrix
        with torch.no_grad():
            self.q_proj_weight *= 0
            self.q_proj_weight += ret
        return ret

    @torch.compile()
    def _get_k(self):
        if self.k_proj_weight is None:
            return self.k_proj_weight
        if not self.uvh_fixed_k:
            return self.k_proj_weight
        u = self.k_u.detach()
        vh = self.k_vh.detach()

        s = self.k_s if self.full_rank_sigma else torch.diag(self.k_s)

        w = torch.linalg.multi_dot([u, s, vh])
        ret = w.T if self.k_trans else w
        with torch.no_grad():
            self.k_proj_weight *= 0
            self.k_proj_weight += ret
        return ret

    @torch.compile()
    def _get_v(self):
        if self.v_proj_weight is None:
            return self.v_proj_weight
        if not self.uvh_fixed_v:
            return self.v_proj_weight
        u = self.v_u.detach()
        vh = self.v_vh.detach()

        s = self.v_s if self.full_rank_sigma else torch.diag(self.v_s)

        w = torch.linalg.multi_dot([u, s, vh])
        ret = w.T if self.v_trans else w
        with torch.no_grad():
            self.v_proj_weight *= 0
            self.v_proj_weight += ret
        return ret

    # @torch.compile()
    def _get_in_proj(self) -> Tensor:
        if not self.uvh_fixed_in_proj:
            # print('uvh not fixed', type(self.in_proj_weight))
            return self.in_proj_weight
        u = self.in_proj_u.detach()
        vh = self.in_proj_vh.detach()

        s = self.in_proj_s if self.full_rank_sigma else torch.diag(self.in_proj_s)

        ret = torch.linalg.multi_dot([u, s, vh])
        # No need for transpose, in_proj is always TS (be definition)
        with torch.no_grad():
            self.in_proj_weight *= 0
            self.in_proj_weight += ret
        return ret

    @torch.compile()
    def get_weight(self):
        if not self._qkv_same_embed_dim:  # get qkv
            q = self._get_q()
            k = self._get_k()
            v = self._get_v()
            return q, k, v
        else:
            return self._get_in_proj()

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and float masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.
                If both attn_mask and key_padding_mask are supplied, their types should match.
            is_causal: If specified, applies a causal mask as attention mask.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``attn_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
                heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
                effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
            :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
            where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
            embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
            returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
            :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
            :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
            head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

            .. note::
                `batch_first` argument is ignored for unbatched inputs.
        """

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self._get_in_proj(),  # self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self._get_in_proj(),  # self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    merged_mask,
                    need_weights,
                    average_attn_weights,
                    mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            in_proj = self._get_in_proj()  # self.in_proj_weight,
            q = self._get_q()
            k = self._get_k()
            v = self._get_v()

            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                in_proj, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=q, k_proj_weight=k,
                v_proj_weight=v,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        else:
            in_proj = self._get_in_proj()  # self.in_proj_weight,
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                in_proj, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                    query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
        r"""
        Determine mask type and combine masks if necessary. If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type

    # TODO: compile?
    @torch.no_grad()
    def _test_stability_general(self, qkvin):
        # TODO: move me!
        # # if self._qkv_same_embed_dim: -> training on in_proj and qkv are none
        # if self.in_proj_weight is not None:
        #     return

        """
        This is going to do the qkv updates for q, k, v, or in_proj based on the value of qkvin

        uvh_fixed_[in_proj, q, k, v]
        [in_proj, q, k, v]_u
        [in_proj, q, k, v]_s
        [in_proj, q, k, v]_vh
        [in_proj, q, k, v]_trans
        prev_uvh_[in_proj, q, k, v]
        _get_[in_proj, q, k, v]
        [in_proj, q, k, v]_slice
        base weights:
            [q, k, v]_proj_weight
            in_proj_weight
        """
        # if getattr(self, f"uvh_fixed_{qkvin}") and not self.update_from_simga:
        #     perc_params, _, _ = self.get_perc_params()
        #     return 2., getattr(self, f"{qkvin}_slice"), perc_params, getattr(self, f"uvh_fixed_{qkvin}")

        rank = dist.get_rank() if dist.is_initialized() else 0

        set_usvh = True  # if true: skip the update of USVH (only false for full_rank which had a different update logic)
        if self.full_rank_sigma and getattr(self, f"uvh_fixed_{qkvin}") and self.update_from_simga:
            # updating U/Vh from full-rank sigma
            if rank == 0:
                log.info("Full rank sigma update of usvh")
            # TODO: update _update_usv!
            self._update_usv(qkvin)
            set_usvh = False
            u = getattr(self, f"{qkvin}_u")
            s = getattr(self, f"{qkvin}_s")
            vh = getattr(self, f"{qkvin}_vh")
            uvh = u @ vh
            # perc_params, _, _ = self.get_perc_params()
            # return 2., getattr(self, f"{qkvin}_slice"), perc_params, getattr(self, f"uvh_fixed_{qkvin}")
        else:
            # w = self.weight.T if self.trans else self.weight
            w = getattr(self, f"{qkvin}_proj_weight") if qkvin in "qkv" else self.in_proj_weight
            w = w.T if getattr(self, f"{qkvin}_trans") else w
            # w = getattr(self, f"_get_{qkvin}")()
            dtp = w.dtype
            u, s, vh = torch.linalg.svd(w.to(torch.float32), full_matrices=False)  # , driver="gesvd")
            u = u.to(dtp)
            s = s.to(dtp)
            vh = vh.to(dtp)
            uvh = u @ vh
            prev_uvh = getattr(self, f"prev_uvh_{qkvin}")
            if prev_uvh is None:  # first iteration 
                if rank == 0:
                    log.info("First stability update")
                setattr(self, f"prev_uvh_{qkvin}", uvh)
                uself = getattr(self, f"{qkvin}_u")
                sself = getattr(self, f"{qkvin}_s")
                vhself = getattr(self, f"{qkvin}_vh")
                uself.zero_()
                uself.add_(u)
                sself.zero_()
                # print(f"{sself.shape}, {s.shape}, {self.full_rank_sigma}")
                sself.add_(torch.diag(s) if self.full_rank_sigma else s)
                vhself.zero_()
                vhself.add_(vh)
                return 0, getattr(self, f"{qkvin}_slice"), 1., getattr(self, f"uvh_fixed_{qkvin}")
            if rank == 0:
                log.info("in normal stability update")

        # use cosine similarity (dot product for orthonormal) to determine similarity
        prev_uvh = getattr(self, f"prev_uvh_{qkvin}")
        csim = self.cossim(prev_uvh, uvh)
        setattr(self, f"prev_uvh_{qkvin}", uvh)
        csmean, _ = csim.mean(), csim.std()
        if csmean > self.uvh_threshold:
            setattr(self, f"uvh_fixed_{qkvin}", True)
            uself = getattr(self, f"{qkvin}_u")
            sself = getattr(self, f"{qkvin}_s")
            vhself = getattr(self, f"{qkvin}_vh")
            gensl = getattr(self, f"{qkvin}_slice")
            if set_usvh:
                uself.zero_()
                uself.add_(u)
                vhself.zero_()
                vhself.add_(vh)

                if self.full_rank_sigma:
                    sself.zero_()
                    sself[:gensl, :gensl].add_(torch.diag(s[:gensl]))
                else:
                    sself.zero_()
                    sself[:gensl].add_(s[:gensl])
            # uself[torch.abs(uself) < 1e-5] *= 0
            # vhself[torch.abs(vhself) < 1e-5] *= 0
            # sself[torch.abs(sself) < 1e-6] *= 0

            self._update_k_slice(qkvin)

        if dist.get_rank() == 0:
            usig = getattr(self, f"{qkvin}_u")
            sig = getattr(self, f"{qkvin}_s")
            vhsig = getattr(self, f"{qkvin}_vh")
            print(f"u: {usig.mean():.4f}, {usig.min():.4f}, {usig.max():.4f}, {usig.std():.4f}")
            print(f"s: {sig.mean():.4f}, {sig.min():.4f}, {sig.max():.4f}, {sig.std():.4f}")
            print(f"vh: {vhsig.mean():.4f}, {vhsig.min():.4f}, {vhsig.max():.4f}, {vhsig.std():.4f}")
        # self.s_prev = self.s.data.clone()
        perc_params, _, _ = self.get_perc_params()
        return csmean, getattr(self, f"{qkvin}_slice"), perc_params, getattr(self, f"uvh_fixed_{qkvin}")

    @torch.no_grad()
    def test_stability(self):
        if self.in_proj_weight is not None:
            return self._test_stability_general(qkvin="in_proj")
        else:
            csmeanq, qslice, qparams, qfixed = self._test_stability_general(qkvin="q")
            csmeank, kslice, kparams, kfixed = self._test_stability_general(qkvin="k")
            csmeanv, vslice, vparams, vfixed = self._test_stability_general(qkvin="v")
            return [
                [csmeanq, csmeank, csmeanv],  # TODO: should this already be a string?
                [qslice, kslice, vslice],  # TODO: should this already be a string?
                [qparams, kparams, vparams],  # TODO: should this already be a string?
                [qfixed, kfixed, vfixed],  # TODO: should this already be a string?
            ]

    @torch.no_grad()
    # @torch.compile()  # TODO: fix compiling here: index error?
    def _update_usv(self, qkvin):
        if not self.full_rank_sigma and not self.u_fixed:
            raise ValueError("this function is only for full-rank sigma with usvh is fixed")
        # NOTE: no slicing because need the shapes to line up. self.s[self.k:, self.k:] should be 0?
        uself = getattr(self, f"{qkvin}_u")
        sself = getattr(self, f"{qkvin}_s")
        vhself = getattr(self, f"{qkvin}_vh")
        # gensl = getattr(self, f"{qkvin}_slice")
        usig, sig, vhsig = torch.linalg.svd(sself)  # square mat, full mat arg doesnt matter
        # usig[torch.abs(usig) < 1e-5] *= 0
        # vhsig[torch.abs(vhsig) < 1e-5] *= 0
        # sig[torch.abs(sig) < 1e-6] *= 0

        # # cutoff vectors with only small changes --------------------------------------
        # # in this case, remove the irrelevant vectors here not after multi!
        # cutoff = sig[0] * self.sigma_cutoff_fraction
        # nz = torch.nonzero(sig < cutoff)
        # if len(nz) == 0:
        #     # In this case ALL of the basis vectors are useful
        #     k = sig.shape[0]
        # else:
        #     k = nz[0].item()

        # usig[:, k:].mul_(0)
        # vhsig[k:].mul_(0)
        # sig[k:].mul_(0)
        # # -----------------------------------------------------------------------------

        holdu = uself @ usig
        uself.zero_()
        uself.add_(holdu)  # .contiguous())
        holdvh = vhsig @ vhself
        vhself.zero_()
        vhself.add_(holdvh)  # .contiguous())
        sself.zero_()
        sself.add_(torch.diag(sig))
        # to be safe: set weight to be the same here too
        w = torch.linalg.multi_dot([uself, sself, vhself])
        ret = w.T if getattr(self, f"{qkvin}_trans") else w
        if qkvin in ["q", "k", "v"]:
            weight = getattr(self, f"{qkvin}_proj_weight")
        else:
            weight = self.in_proj_weight
        weight *= 0
        weight += ret
        # let this get handled by other function?
        # if dist.get_rank() == 0:
        #     print(f"u: {usig.mean():.4f}, {usig.min():.4f}, {usig.max():.4f}, {usig.std():.4f}")
        #     print(f"s: {sig.mean():.4f}, {sig.min():.4f}, {sig.max():.4f}, {sig.std():.4f}")
        #     print(f"vh: {vhsig.mean():.4f}, {vhsig.min():.4f}, {vhsig.max():.4f}, {vhsig.std():.4f}")

    @torch.no_grad()
    def _update_k_slice(self, qkvin):
        uself = getattr(self, f"{qkvin}_u")
        sself = getattr(self, f"{qkvin}_s")
        vhself = getattr(self, f"{qkvin}_vh")
        gensl = getattr(self, f"{qkvin}_slice")
        # adjust K to slice of less important singular values
        # only want to compare the diagonal entries of sigma
        s = torch.diag(sself) if self.full_rank_sigma else sself
        prevsl = gensl
        if getattr(self, f"uvh_fixed_{qkvin}"):
            cutoff = s[0] * self.sigma_cutoff_fraction
            nz = torch.nonzero(s < cutoff)
            if len(nz) == 0:
                # In this case ALL of the basis vectors are useful
                newsl = s.shape[0]
            else:
                newsl = nz[0].item()
        # if newsl < 0.75 * prevsl:
        #     # TODO: log message?
        #     newsl = int(prevsl * 0.75)
        #     # print(s[:5])
        setattr(self, f"{qkvin}_slice", newsl)

        uself[:, newsl:] *= 0
        vhself[newsl:] *= 0
        if self.full_rank_sigma:
            sself[newsl:, newsl:].mul_(0)
        else:
            sself[newsl:].mul_(0)

    @torch.compile()
    def get_perc_params_select(self, qkvin):
        if qkvin in ["q", "k", "v"]:
            weight = getattr(self, f"{qkvin}_proj_weight")
        else:
            weight = self.in_proj_weight
        normal_params = weight.numel()
        if getattr(self, f"uvh_fixed_{qkvin}"):
            # active_params = (self.u.shape[0] * self.k) + (self.k ** 2) + (self.k + self.vh.shape[1])
            trainable_params = getattr(self, f"{qkvin}_slice") ** 2
        else:
            trainable_params = normal_params
        perc_params = trainable_params / normal_params
        return perc_params, trainable_params, normal_params

    def get_perc_params(self):
        if not self._qkv_same_embed_dim:  # get perc perams for all
            _, qtrain, qnormal = self.get_perc_params_select("q")
            _, ktrain, knormal = self.get_perc_params_select("k")
            _, vtrain, vnormal = self.get_perc_params_select("v")
            active = qtrain + ktrain + vtrain
            normal = qnormal + knormal + vnormal
            perc = active / normal
        else:
            perc, active, normal = self.get_perc_params_select("in_proj")
        return perc, active, normal
