import collections
import logging
import math
import random
from copy import copy
from itertools import repeat
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

from . import mixing, utils


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
# _triple = _ntuple(3, "_triple")
# _quadruple = _ntuple(4, "_quadruple")

log = logging.getLogger(__name__)

# TODO: make general convNd class to inherit from


class SVDSyncConv2d(nn.modules.conv._ConvNd):
    __doc__ = r"""
    TODO: link to torch conv2D docs
    add other docs about this function itself
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
        uvhthreshold: float = 0.9,
        sigma_cutoff_fraction: float = 0.1,
        start_weight=None,
        start_bias=None,
        update_from_simga=True,
        reinit_shapes=False,
        norm=None,
        activation=None,
        distributed_updates=True,
        inner_dim_init_ratio=1.0,
        random_sigma: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )
        self.inner_dim_init_ratio = inner_dim_init_ratio
        # detectron2 things...norm/activation is wrapped into one place here
        self.norm = norm
        self.activation = activation
        # new things ------------------------------------
        with torch.no_grad():
            if start_weight is not None:
                self.weight.zero_()
                self.weight.add_(start_weight)
            if start_bias is not None:
                self.bias.zero_()
                self.bias.add_(start_bias)

        self.update_from_simga = update_from_simga
        self.reinit_shapes = reinit_shapes
        self.distributed_updates = distributed_updates

        weight_shape = self.weight.shape
        self.base_weigh_shape = tuple(weight_shape)
        m = weight_shape[0]
        n = weight_shape[1] * weight_shape[2] * weight_shape[3]
        k = min(m, n)
        # svd shapes: [m, n] -> u[m, k] * s[k, k] * vh [k, n]    k-> min(m, n)
        w = self.weight.view(m, n)
        if m >= n:
            self.trans = False
        else:  # need to flip m and n
            self.trans = True
            hold = m
            n = m
            m = hold
            w = w.T

        try:
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
        except RuntimeError:
            log.info("Error doing SVD for given weights for Conv2D layer, rolling new weights")
            w2 = torch.empty_like(w)
            nn.init.kaiming_normal_(w2, a=math.sqrt(5))
            u, s, vh = torch.linalg.svd(w2, full_matrices=False)

        self.gen = torch.Generator(device=w.device)
        self.gen.manual_seed(random.randint(0, 68719476735))
        if random_sigma:
            log.debug(f"layer-local random seed: {self.gen.initial_seed()}")
            s = torch.rand(s.shape[0], device=s.device, dtype=s.dtype, generator=self.gen) * s.max()

        self.inner_dim = torch.tensor(k * self.inner_dim_init_ratio, dtype=torch.int)

        self.u = nn.Parameter(u[:, : self.inner_dim].clone(), requires_grad=False)
        # self.s = nn.Parameter(self.s, requires_grad=True)
        self.s = nn.Parameter(torch.diag(s[: self.inner_dim]), requires_grad=True)
        self.vh = nn.Parameter(vh[: self.inner_dim].clone(), requires_grad=False)

        self.p = nn.Parameter(torch.ones_like(s[: self.inner_dim], requires_grad=False))
        self.train_p = False

        self.sigma_cutoff_fraction = sigma_cutoff_fraction
        self.inner_dim_buffer = torch.empty(3)  # inner dim, csmean, changing k
        self.uvhthreshold = uvhthreshold
        self.wait_inner_dim, self.wait_s, self.wait_u, self.wait_vh = None, None, None, None

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.size = dist.get_world_size()
        else:
            self.rank = 0
            self.size = 1

        self.sigma_generator = torch.Generator(device=s.device)
        self.sigma_generator.manual_seed(random.randint(0, 68719476735))
        self.last_send_rank = None
        self.prev_uvh = None
        del self.weight

        # internal random generator
        seed = torch.seed()
        if dist.is_initialized():
            seed = seed + dist.get_rank() + 1000
        # log.info(f"Internal seed for linear layer: {seed}")
        self.gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        self.gen.manual_seed(seed)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # From nn.conv2d
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        x = F.conv2d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def get_weight(self, for_svd=False):
        # detach sets 'requires grad' to False
        if self.training:
            self.s.requires_grad = True
            self.u.requires_grad = False
            self.vh.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = True
        if self.train_p and self.training:
            self.p.requires_grad = True
            if self.bias is not None:
                self.bias.requires_grad = False
            w = torch.linalg.multi_dot([self.u, self.p.view(-1, 1) * self.s, self.vh])
        else:
            w = torch.linalg.multi_dot([self.u, self.s, self.vh])
        if self.trans:
            w = w.T

        ret = w.reshape(self.base_weigh_shape)
        return ret

    def forward(self, input: Tensor) -> Tensor:
        weight = self.get_weight()
        return self._conv_forward(input, weight, self.bias)

    @torch.no_grad()
    def _update_inner_dim_and_shapes(self):
        # adjust K to slice of less important singular values
        s = torch.diag(self.s) if self.s.ndim == 2 else self.s
        prevk = self.inner_dim.clone()
        # new plan for cutoff - instead of using s[0] use 1% of the minimum
        # this will enforce that the array never shrinks below 1%
        cutoff = s[0] * self.sigma_cutoff_fraction
        min_dim = int(self.vh.shape[-1] * 0.01)  # always TS
        # cutoff = s[min_dim] * self.sigma_cutoff_fraction
        nz = torch.nonzero(s < cutoff)
        if len(nz) == 0:
            # In this case ALL of the basis vectors are useful
            newk = s.shape[0]
        elif nz[0] < prevk * 0.5 and prevk * 0.5 > min_dim:
            # add breaking, dont let it cut off more than 50% of the values
            newk = int(prevk * 0.5)
        elif nz[0].item() < min_dim:
            newk = min_dim
        else:
            newk = nz[0].item()

        change_k = newk != prevk
        self.inner_dim.mul_(0)
        self.inner_dim.add_(newk)
        if change_k:
            self._update_usvh_shapes()
        return change_k

    @torch.no_grad()
    def _update_usvh_shapes(self):
        # either update the shapes of USVh or set the irrelevant values to 0
        # if self.reinit_shapes:
        self.u.set_(self.u[:, : self.inner_dim].contiguous())
        self.vh.set_(self.vh[: self.inner_dim].contiguous())
        self.s.set_(self.s[: self.inner_dim, : self.inner_dim].contiguous())
        self.p.set_(self.p[: self.inner_dim].contiguous())
        # else:
        #     self.u[:, self.inner_dim :] *= 0
        #     self.vh[self.inner_dim :] *= 0
        #     self.s[self.inner_dim :, self.inner_dim :].mul_(0)
        #     self.p[self.inner_dim :].mul_(0)

    @torch.no_grad()
    def _full_rank_update_usv(self):
        s = self.s.to(torch.float32)
        usig, sig, vhsig = torch.linalg.svd(s)  # square mat, full mat arg doesnt matter
        usig = usig.to(self.s.dtype)
        sig = sig.to(self.s.dtype)
        vhsig = vhsig.to(self.s.dtype)

        holdu = self.u @ usig
        self.u.zero_()
        self.u.add_(holdu)
        holdvh = vhsig @ self.vh
        self.vh.zero_()
        self.vh.add_(holdvh)
        self.s.zero_()
        self.s.add_(torch.diag(sig))
        csmean = 1.0

        change_k = False
        if csmean >= self.uvhthreshold:
            change_k = self._update_inner_dim_and_shapes()
        return {"csmean": csmean, "change_k": change_k}

    @torch.no_grad()
    def remove_smaller_vecs(self, name):
        self.name = name
        # no SVD here, just cutting off sigma vals based on cutoff
        self._update_inner_dim_and_shapes()
        perc, _, _ = self.get_perc_params()
        if self.rank == 0:
            log.info(
                f"{name[-30:]}: Update inner dim: params: {perc * 100:.2f}, "
                f"\t[{self.u.shape[0]} {self.s.shape[0]} {self.vh.shape[1]}]",
            )

    def get_interior_inner_dim(self):
        return {
            "weight": self.inner_dim,
        }

    def extra_repr(self) -> str:
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}" ", stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        s += f", inner_dim={self.inner_dim.item()}"
        return s.format(**self.__dict__)

    # @torch.compile()
    def get_perc_params(self):
        normal_params = self.u.shape[0] * self.vh.shape[1]  # self.s.numel()
        bias_params = 0 if self.bias is None else self.bias.numel()
        trainable_params = self.inner_dim**2
        trainable_params += bias_params
        normal_params += bias_params
        perc_params = trainable_params / normal_params
        return perc_params, trainable_params, normal_params

    def distribute_workload(self, min_size_fraction=0.05, name=None):
        utils.split_sigma_workload(
            collected_u=self.u,
            collected_s=self.s,
            collected_vh=self.vh,
            sigma_generator=self.sigma_generator,
            min_size_fraction=min_size_fraction,
            set_usvh=True,
            name=name,
            p=self.p,
        )

    def gather_distributed_factorizations(self, sigma_cutoff_fraction=0.2, name=None):
        utils.collect_lr_reps_from_all(
            local_u=self.u,
            local_s=self.s,
            local_vh=self.vh,
            set_usvh=True,
            name=name,
            p=self.p,
        )

    @torch.no_grad()
    def mix_sigma(self, method, *args, **kwargs):
        mixing.mix_sigma(*args, u=self.u, s=self.s, vh=self.vh, method=method, generator=self.gen, **kwargs)

    @torch.no_grad()
    def start_p_training(self):
        # set P to be 1/sz to emulate an average, will let the network learn how to use this
        self.p.zero_()
        sz = dist.get_world_size() if dist.is_initialized() else 1
        self.p.add_(1 / sz)
        self.p.requires_grad = True
        self.train_p = True
        self.s.requires_grad = False

    @torch.no_grad()
    def update_sp_train_normally(self):
        # revert to normal training for the layer
        # update sigma from P and start training normally
        hold = self.p.view(-1, 1) * self.s
        self.s.zero_()
        self.s.add_(hold)
        self.p.zero_()
        self.p.add_(1)
        self.p.requires_grad = False
        self.train_p = False
        self.s.requires_grad = True


class SVDSyncConv1d(nn.modules.conv._ConvNd):
    __doc__ = r"""
    TODO: link to torch conv2D docs
    add other docs about this function itself
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
        uvhthreshold: float = 0.9,
        sigma_cutoff_fraction: float = 0.1,
        start_weight=None,
        start_bias=None,
        update_from_simga=True,
        reinit_shapes=False,
        norm=None,
        activation=None,
        distributed_updates=True,
        inner_dim_init_ratio=1.0,
        random_sigma: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )
        self.inner_dim_init_ratio = inner_dim_init_ratio
        # detectron2 things...norm/activation is wrapped into one place here
        self.norm = norm
        self.activation = activation
        # new things ------------------------------------
        with torch.no_grad():
            if start_weight is not None:
                self.weight.zero_()
                self.weight.add_(start_weight)
            if start_bias is not None:
                self.bias.zero_()
                self.bias.add_(start_bias)

        self.update_from_simga = update_from_simga
        self.reinit_shapes = reinit_shapes

        weight_shape = self.weight.shape
        self.base_weigh_shape = tuple(weight_shape)
        self.distributed_updates = distributed_updates
        m = weight_shape[0]
        n = int(self.weight.numel() / m)
        k = min(m, n)
        # svd shapes: [m, n] -> u[m, k] * s[k, k] * vh [k, n]    k-> min(m, n)
        w = self.weight.view(m, n)
        if m >= n:
            self.trans = False
        else:  # need to flip m and n
            self.trans = True
            hold = m
            n = m
            m = hold
            w = w.T
        u, s, vh = torch.linalg.svd(w, full_matrices=False)

        self.gen = torch.Generator(device=w.device)
        self.gen.manual_seed(random.randint(0, 68719476735))
        if random_sigma:
            log.debug(f"layer-local random seed: {self.gen.initial_seed()}")
            s = torch.rand(s.shape[0], device=s.device, dtype=s.dtype, generator=self.gen) * s.max()

        self.inner_dim = torch.tensor(k * self.inner_dim_init_ratio, dtype=torch.int)

        self.u = nn.Parameter(u[:, : self.inner_dim].clone(), requires_grad=False)
        # self.s = nn.Parameter(self.s, requires_grad=True)
        self.s = nn.Parameter(torch.diag(s[: self.inner_dim]), requires_grad=True)
        self.vh = nn.Parameter(vh[: self.inner_dim].clone(), requires_grad=False)

        self.p = nn.Parameter(torch.ones_like(s[: self.inner_dim], requires_grad=False))
        self.train_p = False

        self.sigma_cutoff_fraction = sigma_cutoff_fraction
        self.inner_dim_buffer = torch.empty(3)  # inner dim, csmean, changing k
        self.uvhthreshold = uvhthreshold
        self.wait_inner_dim, self.wait_s, self.wait_u, self.wait_vh = None, None, None, None

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.size = dist.get_world_size()
        else:
            self.rank = 0
            self.size = 1

        self.sigma_generator = torch.Generator(device=s.device)
        self.sigma_generator.manual_seed(random.randint(0, 68719476735))

        self.first_distribute_workload = True
        self.last_send_rank = None
        self.prev_uvh = None
        del self.weight

        # internal random generator
        seed = torch.initial_seed()
        if dist.is_initialized():
            seed = seed + dist.get_rank() + 1000
        log.info(f"Internal seed for linear layer: {seed}")
        self.gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        self.gen.manual_seed(seed)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # From nn.conv2d
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                _single(0),
                self.dilation,
                self.groups,
            )
        x = F.conv1d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def get_weight(self, for_svd=False):
        # detach sets 'requires grad' to False
        if self.training:
            self.s.requires_grad = True
            self.u.requires_grad = False
            self.vh.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = True
        if self.train_p and self.training:
            self.p.requires_grad = True
            if self.bias is not None:
                self.bias.requires_grad = False
            w = torch.linalg.multi_dot([self.u, self.p.view(-1, 1) * self.s, self.vh])
        else:
            w = torch.linalg.multi_dot([self.u, self.s, self.vh])
        if self.trans:
            w = w.T

        ret = w.reshape(self.base_weigh_shape)
        return ret

    def forward(self, input: Tensor) -> Tensor:
        weight = self.get_weight()
        return self._conv_forward(input, weight, self.bias)

    @torch.no_grad()
    def _update_inner_dim_and_shapes(self):
        # adjust K to slice of less important singular values
        s = torch.diag(self.s) if self.s.ndim == 2 else self.s
        prevk = self.inner_dim.clone()
        # new plan for cutoff - instead of using s[0] use 1% of the minimum
        # this will enforce that the array never shrinks below 1%
        cutoff = s[0] * self.sigma_cutoff_fraction
        min_dim = int(self.vh.shape[-1] * 0.01)  # always TS
        # cutoff = s[min_dim] * self.sigma_cutoff_fraction
        nz = torch.nonzero(s < cutoff)
        if len(nz) == 0:
            # In this case ALL of the basis vectors are useful
            newk = s.shape[0]
        elif nz[0] < prevk * 0.5 and prevk * 0.5 > min_dim:
            # add breaking, dont let it cut off more than 50% of the values
            newk = int(prevk * 0.5)
        elif nz[0].item() < min_dim:
            newk = min_dim
        else:
            newk = nz[0].item()

        change_k = newk != prevk
        self.inner_dim.mul_(0)
        self.inner_dim.add_(newk)
        if change_k:
            self._update_usvh_shapes()
        return change_k

    @torch.no_grad()
    def _update_usvh_shapes(self):
        # either update the shapes of USVh or set the irrelevant values to 0
        # if self.reinit_shapes:
        self.u.set_(self.u[:, : self.inner_dim].contiguous())
        self.vh.set_(self.vh[: self.inner_dim].contiguous())
        self.s.set_(self.s[: self.inner_dim, : self.inner_dim].contiguous())
        self.p.set_(self.p[: self.inner_dim].contiguous())
        # else:
        #     self.u[:, self.inner_dim :] *= 0
        #     self.vh[self.inner_dim :] *= 0
        #     self.s[self.inner_dim :, self.inner_dim :].mul_(0)
        #     self.p[self.inner_dim :]

    @torch.no_grad()
    def _full_rank_update_usv(self):
        s = self.s.to(torch.float32)
        try:
            usig, sig, vhsig = torch.linalg.svd(s)  # square mat, full mat arg doesnt matter
        except torch._C._LinAlgError:
            return {"csmean": 1.0, "change_k": False}
        usig = usig.to(self.s.dtype)
        sig = sig.to(self.s.dtype)
        vhsig = vhsig.to(self.s.dtype)

        holdu = self.u @ usig
        self.u.zero_()
        self.u.add_(holdu)
        holdvh = vhsig @ self.vh
        self.vh.zero_()
        self.vh.add_(holdvh)
        self.s.zero_()
        self.s.add_(torch.diag(sig))
        csmean = 1.0

        change_k = False
        if csmean >= self.uvhthreshold:
            change_k = self._update_inner_dim_and_shapes()
        return {"csmean": csmean, "change_k": change_k}

    @torch.no_grad()
    def remove_smaller_vecs(self, name):
        self.name = name
        # no SVD here, just cutting off sigma vals based on cutoff
        self._update_inner_dim_and_shapes()
        perc, _, _ = self.get_perc_params()
        if self.rank == 0:
            log.info(
                f"{name[-30:]}: Update inner dim: params: {perc * 100:.2f}, "
                f"\t[{self.u.shape[0]} {self.s.shape[0]} {self.vh.shape[1]}]",
            )

    def get_interior_inner_dim(self):
        return {
            "weight": self.inner_dim,
        }

    def extra_repr(self) -> str:
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}" ", stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        s += f", inner_dim={self.inner_dim.item()}"
        return s.format(**self.__dict__)

    # @torch.compile()
    def get_perc_params(self):
        normal_params = self.u.shape[0] * self.vh.shape[1]  # self.s.numel()
        bias_params = 0 if self.bias is None else self.bias.numel()
        trainable_params = self.inner_dim**2
        trainable_params += bias_params
        normal_params += bias_params
        perc_params = trainable_params / normal_params
        return perc_params, trainable_params, normal_params

    def distribute_workload(self, min_size_fraction=0.05, name=None):
        utils.split_sigma_workload(
            collected_u=self.u,
            collected_s=self.s,
            collected_vh=self.vh,
            sigma_generator=self.sigma_generator,
            min_size_fraction=min_size_fraction,
            set_usvh=True,
            name=name,
            p=self.p,
        )

    def gather_distributed_factorizations(self, sigma_cutoff_fraction=0.2, name=None):
        utils.collect_lr_reps_from_all(
            local_u=self.u,
            local_s=self.s,
            local_vh=self.vh,
            sigma_cutoff_fraction=sigma_cutoff_fraction,
            set_usvh=True,
            name=name,
            p=self.p,
        )

    @torch.no_grad()
    def mix_sigma(self, method, *args, **kwargs):
        mixing.mix_sigma(*args, u=self.u, s=self.s, vh=self.vh, method=method, generator=self.gen, **kwargs)

    @torch.no_grad()
    def start_p_training(self):
        # set P to be 1/sz to emulate an average, will let the network learn how to use this
        self.p.zero_()
        sz = dist.get_world_size() if dist.is_initialized() else 1
        self.p.add_(1 / sz)
        self.p.requires_grad = True
        self.train_p = True
        self.s.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    @torch.no_grad()
    def update_sp_train_normally(self):
        # revert to normal training for the layer
        # update sigma from P and start training normally
        hold = self.p.view(-1, 1) * self.s
        self.s.zero_()
        self.s.add_(hold)
        self.p.zero_()
        self.p.add_(1)
        self.p.requires_grad = False
        self.train_p = False
        self.s.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True
