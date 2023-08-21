import collections
import logging
import math
from copy import copy
from itertools import repeat
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


# _single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
# _triple = _ntuple(3, "_triple")
# _quadruple = _ntuple(4, "_quadruple")

log = logging.getLogger(__name__)


class SVDConv2dUSVh(nn.modules.conv._ConvNd):
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
        full_rank_sigma: bool = False,
        start_weight=None,
        start_bias=None,
        update_from_simga=True,
        reinit_shapes=False,
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
        # new things ------------------------------------
        with torch.no_grad():
            if start_weight is not None:
                self.weight.zero_()
                self.weight.add_(start_weight)
            if start_bias is not None:
                self.bias.zero_()
                self.bias.add_(start_bias)

        self.full_rank_sigma = full_rank_sigma
        self.update_from_simga = full_rank_sigma and update_from_simga
        self.reinit_shapes = reinit_shapes

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

        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # w2 = self.weight.view(m, n)
        # if self.trans:
        #     w2 = w.T
        # u2, s2, vh2 = torch.linalg.svd(w2, full_matrices=False)  # TS matrix so its not a big deal
        # # u, _ = torch.linalg.qr(torch.randn_like(u), mode="reduced")
        # # vh, _ = torch.linalg.qr(torch.randn_like(vh), mode="reduced")
        # cutoff = s[0] * 0.4
        # cutoff_index = torch.nonzero(s < cutoff)
        # if len(cutoff_index) > 0:
        #     cutoff_index = cutoff_index[0]
        #     u = u[:, :cutoff_index].contiguous()
        #     vh = vh[:cutoff_index].contiguous()
        #     s = s[:cutoff_index]
        # cutoff2 = s2[0] * 0.4
        # cutoff_index2 = torch.nonzero(s2 < cutoff2)
        # if len(cutoff_index2) > 0:
        #     cutoff_index2 = cutoff_index2[0]
        #     u2 = u2[:, :cutoff_index2].contiguous()
        #     vh2 = vh2[:cutoff_index2].contiguous()
        #     s2 = s2[:cutoff_index2]
        # self.u = torch.cat([u, u2], dim=1)
        # self.vh = torch.cat([vh, vh2], dim=0)
        # self.s = torch.diag(torch.cat([s, s2]))

        self.u = nn.Parameter(u, requires_grad=False)
        # self.s = nn.Parameter(self.s, requires_grad=True)
        self.s = nn.Parameter(torch.diag(s), requires_grad=True)
        self.vh = nn.Parameter(vh, requires_grad=False)

        self.sigma_cutoff_fraction = sigma_cutoff_fraction
        self.inner_dim = torch.tensor(k, dtype=torch.int)
        self.inner_dim_buffer = torch.empty(3)  # inner dim, csmean, changing k
        self.uvhthreshold = uvhthreshold
        self.wait_inner_dim, self.wait_s, self.wait_u, self.wait_vh = None, None, None, None

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.size = dist.get_world_size()
        else:
            self.rank = 0
            self.size = 1
        self.last_send_rank = None
        self.prev_uvh = None
        del self.weight

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
        return F.conv2d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @torch.no_grad()
    def get_weight_for_svd(self):
        w = torch.linalg.multi_dot([self.u, self.s, self.vh])
        return w

    def get_weight(self, for_svd=False):
        # detach sets 'requires grad' to False
        if self.training:
            self.s.requires_grad = True
            self.u.requires_grad = False
            self.vh.requires_grad = False
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
        # min_dim = int(self.vh.shape[-1] * 0.01)  # always TS
        # cutoff = s[min_dim] * self.sigma_cutoff_fraction
        nz = torch.nonzero(s < cutoff)
        if len(nz) == 0:
            # In this case ALL of the basis vectors are useful
            newk = s.shape[0]
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
        if self.reinit_shapes:
            self.u.set_(self.u[:, : self.inner_dim].contiguous())
            self.vh.set_(self.vh[: self.inner_dim].contiguous())
            if self.full_rank_sigma:
                self.s.set_(self.s[: self.inner_dim, : self.inner_dim].contiguous())
            else:
                self.s.set_(self.s[: self.inner_dim])
        else:
            self.u[:, self.inner_dim :] *= 0
            self.vh[self.inner_dim :] *= 0
            if self.full_rank_sigma:
                self.s[self.inner_dim :, self.inner_dim :].mul_(0)
            else:
                self.s[self.inner_dim :].mul_(0)

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
    def test_stability_distributed(self, working_rank, name, nonblocking=True):
        self.name = name

        self.last_send_rank = working_rank
        if working_rank != self.rank:
            # move on for now, coming back later
            # make sure to save the rank which did this layer
            self.inner_dim_buffer = self.inner_dim_buffer.to(device=self.s.device, non_blocking=True)
            self.inner_dim = self.inner_dim.to(device=self.s.device, non_blocking=True)
            # if in the first iteration
            # self.bcast_usvh(src=working_rank, nonblocking=nonblocking)
            self.prev_uvh = torch.tensor(1, device=self.s.device)  # doing this to satisfy 'not None'
            # receive the wait_k from the working process
            self.wait_inner_dim = dist.broadcast(
                self.inner_dim_buffer,
                src=working_rank,
                async_op=nonblocking,
            )
            return

        # case 3: update the stable U and Vh from the full rank sigma matrix
        status = self._full_rank_update_usv()
        # shapes are updated within above (as is slice update)
        perc, _, _ = self.get_perc_params()
        # normalize self.s to max == 1
        # maxs = self.s.max()
        # self.s /= maxs
        # self.vh *= maxs
        log.info(
            f"{name[-30:]}: Full rank update, csmean: {status['csmean']:.3f}, params: {perc * 100:.2f}, "
            f"\t[{self.u.shape[0]} {self.s.shape[0]} {self.vh.shape[1]}]",
        )

        self.inner_dim_buffer[0] = self.inner_dim.to(torch.float)
        self.inner_dim_buffer[1] = status["csmean"]
        self.inner_dim_buffer[2] = status["change_k"]
        if not dist.is_initialized():
            return
        self.inner_dim_buffer = self.inner_dim_buffer.to(self.s.device)

        self.wait_inner_dim = dist.broadcast(self.inner_dim_buffer, src=working_rank, async_op=nonblocking)

    @torch.no_grad()
    def wait_inner_dim_reshape_bcast_usvh(self, nonblocking=True):
        # if wait_k is None -> K is the same -> optimizer is fine (shapes are the same)
        reset_optimizer = bool(self.inner_dim_buffer[2].item())
        if self.last_send_rank is None:  # skip all comms
            return reset_optimizer, True
        # self.wait_inner_dim = dist.broadcast(self.inner_dim_buffer, src=self.last_send_rank, async_op=False)
        if self.wait_inner_dim is not None:
            self.wait_inner_dim.wait()
            self.wait_inner_dim = None
            if self.rank != self.last_send_rank:
                self.inner_dim = self.inner_dim_buffer[0].to(torch.int)
                # print('before reshape')
                self._update_usvh_shapes()

        reset_optimizer = bool(self.inner_dim_buffer[2].item())
        stable = self.inner_dim_buffer[1] >= self.uvhthreshold
        self.inner_dim_buffer *= 0
        self.bcast_usvh(src=self.last_send_rank, nonblocking=nonblocking)
        return reset_optimizer, stable

    @torch.no_grad()
    def bcast_usvh(self, src, nonblocking=True):
        if not dist.is_initialized() or self.last_send_rank is None:
            return
        # self.wait_k = dist.broadcast(self.k, src=src, async_op=nonblocking)
        if not self.u.is_contiguous():
            self.u.set_(self.u.contiguous())
        if not self.s.is_contiguous():
            self.s.set_(self.s.contiguous())
        if not self.vh.is_contiguous():
            self.vh.set_(self.vh.contiguous())
        self.wait_u = dist.broadcast(self.u, src=src, async_op=nonblocking)
        self.wait_s = dist.broadcast(self.s, src=src, async_op=nonblocking)
        self.wait_vh = dist.broadcast(self.vh, src=src, async_op=nonblocking)

    @torch.no_grad()
    def wait_on_usvh(self):
        if self.wait_u is not None:
            self.wait_u.wait()
            self.wait_u = None
        if self.wait_s is not None:
            self.wait_s.wait()
            self.wait_s = None
        if self.wait_vh is not None:
            self.wait_vh.wait()
            self.wait_u = None

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