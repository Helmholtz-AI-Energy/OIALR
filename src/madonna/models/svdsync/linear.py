import logging
import math
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from . import mixing

log = logging.getLogger(__name__)


class SVDSyncLinear(nn.Module):
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
        start_weight=None,
        start_bias=None,
        distributed_updates=True,
        inner_dim_init_ratio=1.0,
        random_sigma: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SVDSyncLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.distributed_updates = distributed_updates
        self.inner_dim_init_ratio = inner_dim_init_ratio

        if start_weight is not None:
            self.weight = start_weight
        else:
            # changed from empty
            self.weight = torch.empty((out_features, in_features), **factory_kwargs)
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.weight = nn.Parameter(self.weight)

        if out_features >= in_features:  # simplest case (no transpose)
            self.trans = False
            w = self.weight
        else:
            self.trans = True
            w = self.weight.T

        # u, s, vh = torch.linalg.svd(w, full_matrices=False)
        # k = min(tuple(w.shape))
        u, s, vh = torch.linalg.svd(w, full_matrices=False)  # TS matrix so its not a big deal
        self.gen = torch.Generator(device=w.device)
        self.gen.manual_seed(random.randint(0, 68719476735))
        if random_sigma:
            log.debug(f"layer-local random seed: {self.gen.initial_seed()}")
            s = torch.rand(s.shape[0], device=s.device, dtype=s.dtype, generator=self.gen) * s.max()

        self.inner_dim = torch.tensor(int(s.shape[0] * self.inner_dim_init_ratio), dtype=torch.int)
        self.u = nn.Parameter(u[:, : self.inner_dim], requires_grad=False)
        # self.s = nn.Parameter(self.s, requires_grad=True)
        self.s = nn.Parameter(torch.diag(s[: self.inner_dim]), requires_grad=True)
        self.vh = nn.Parameter(vh[: self.inner_dim], requires_grad=False)

        if bias:
            if start_bias is None:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.bias = nn.Parameter(start_bias)
        else:
            self.register_parameter("bias", None)

        # self.sigma_cutoff_fraction = sigma_cutoff_fraction
        # self.prev_uvh = torch.tensor([1])
        del self.weight
        self.sigma_generator = torch.Generator(device=s.device)
        self.sigma_generator.manual_seed(random.randint(0, 68719476735))

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.first_distribute_workload = True

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

    def distribute_workload(self, min_size_fraction=0.05):
        # TODO: what happens when there are not enough vectors in
        # 1: collect sigmas from all ranks
        #   are the shapes known??
        # 2: join all sigmas together
        # 3: deterine number of vecs to send
        # 4: shuffle/sort sigma
        # 5: select new elements to send
        # 6: send/recv new vecs and update accordingly

        if not dist.is_initialized():
            return
        # unify the generator for the random permutation in step 4
        genstate = self.sigma_generator.get_state().to(device=self.s.device, dtype=torch.float)
        dist.broadcast(genstate, src=0)
        self.sigma_generator.set_state(genstate.to(torch.int8))

        # if we start with the same seed everywhere and the same sigma, then the first time
        #   through this is different. done need to do steps 1/2

        # assumption: sigma is diagonal and full-rank
        loc_s = self.s.diag()
        loc_u = self.u
        loc_vh = self.vh
        min_num_vecs = int(self.vh.shape[1] * min_size_fraction)
        # upper limit on number of vecs to send is the number of vecs across the process space
        #   in the frist iteration through this, that would be the number of diag elements in sigma
        total_vecs = loc_s.shape[0]
        if not self.first_distribute_workload:
            # 1. collect simgas
            #   send shapes around
            shapes = torch.zeros(self.world_size, device=self.s.device)
            shapes[self.rank] = self.s.shape[0]
            dist.all_reduce(shapes)  # sum everything up
            total_vecs = shapes.sum()

            # 2. join all sigmas together
            full_sigma = torch.zeros(total_vecs, dtype=self.s.dtype, device=self.s.device)
            shapes = [0] + shapes.tolist()
            full_sigma[shapes[self.rank] : shapes[self.rank + 1]] = loc_s
            wait_sigma = dist.all_reduce(full_sigma, async_op=True)  # sum op is default
            # 6 (pre work). getting U/Vh
            full_u = torch.zeros((loc_u.shape[0], total_vecs), dtype=loc_u.dtype, device=loc_u.device)
            full_vh = torch.zeros((total_vecs, loc_vh.shape[1]), dtype=loc_vh.dtype, device=loc_vh.device)
            full_u[:, shapes[self.rank] : shapes[self.rank + 1]] = loc_u
            full_vh[shapes[self.rank] : shapes[self.rank + 1]] = loc_vh
            wait_vh = dist.all_reduce(full_vh, async_op=True)  # smaller than U -> send first
            wait_u = dist.all_reduce(full_u, async_op=True)

        # 3 determine number of vecs to send
        num_vecs_to_get = total_vecs // self.world_size
        vecs_to_get_all = torch.zeros(self.world_size, device=loc_s.device, dtype=torch.int)
        vecs_to_get_all += num_vecs_to_get
        vecs_to_get_all[: total_vecs % self.world_size] += 1  # deal with remainer on lower ranks

        # 4: shuffle sigma -> no sort
        #       need to know where everything is. random works too (hopefully)
        # NOTE: will be effected if the seeds are different!!
        inds = torch.randperm(total_vecs, device=loc_s.device, dtype=torch.int, generator=self.sigma_generator)

        fact = {"device": self.s.device, "dtype": self.dtype}

        # 5: select new elements to send
        rank_inds_list = []
        rank_inds_list.append(inds[: vecs_to_get_all[0]])
        for r in range(1, self.world_size):
            rank_inds_list.append(inds[vecs_to_get_all[r - 1] : vecs_to_get_all[r]])
        # 6: send/recv new vecs and update accordingly
        # NOTE: using all-to-all is probably more efficient,
        #       but we that is more complicated and non-blocking all-reduce is probably just as fast,
        #       however it is more bytes to send
        if not self.first_distribute_workload:
            wait_sigma.wait()
            new_sigma = full_sigma[rank_inds_list[self.rank]].to(**fact)
            wait_vh.wait()
            new_vh = full_vh[rank_inds_list[self.rank]].to(**fact)
            wait_u.wait()
            new_u = full_u[:, rank_inds_list[self.rank]].to(**fact)
        else:
            new_sigma = loc_s[rank_inds_list[self.rank]].to(**fact)
            new_vh = loc_vh[rank_inds_list[self.rank]].to(**fact)
            new_u = loc_u[:, rank_inds_list[self.rank]].to(**fact)

        if new_sigma.shape[0] < min_num_vecs:
            log.info("Generating extra orthogonal vectors to fill in extra space")
            # in this case, there are not enough vectors (hyperparam)
            #   if there are < 10 vectors or <5% of posibilities, then issues can arrise (maybe)
            # TODO: test me / determine if worth it
            # need to add orthogonal vectors to u and vh, random values for sigma (use QR)
            to_gen = min_num_vecs - new_sigma.shape[0]
            holdu = torch.randn((loc_u.shape[0], to_gen), **fact)
            new_u_additional, _ = torch.linalg.qr(holdu)  # mode=reduced by default
            holdvh = torch.randn((to_gen, loc_vh.shape[0]), **fact)
            new_vh_additional, _ = torch.linalg.qr(holdvh)  # mode=reduced by default
            sigma_additional = torch.rand(to_gen, **fact)
            new_vh = torch.cat([new_vh, new_vh_additional], dim=0)
            new_sigma = torch.cat([new_sigma, sigma_additional], dim=0)
            new_u = torch.cat([new_u, new_u_additional], dim=1)

        self.vh.set_(new_vh)
        self.u.set_(new_u)
        self.s.set_(torch.diag(new_sigma))  # new_sigma is 1D
        self.first_distribute_workload = False

    # @torch.compile()  # TODO: compiling seems to be weird here...might need to compile two different functions?
    def get_weight(self):
        # detach sets 'requires grad' to False
        if self.training:
            self.s.requires_grad = True
            self.u.requires_grad = False
            self.vh.requires_grad = False
        u, vh = self.u, self.vh
        s = self.s
        # print(u.shape, s.shape, vh.shape)
        w = torch.linalg.multi_dot([u, s, vh])
        ret = w.T if self.trans else w
        return ret

    # @torch.compile()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.get_weight()
        return F.linear(input, w, self.bias)

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

        # if newk < 0.5 * prevk:
        #     newk = int(prevk * 0.5)
        #     log.info(f"Values of S after dropping slice value by only 50% of suggestion: {s[:5]}")
        # if newk < self.vh.shape[1] * 0.05:
        #     newk = int(self.vh.shape[1] * 0.05)
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
            self.s.set_(self.s[: self.inner_dim, : self.inner_dim].contiguous())
        else:
            self.u[:, self.inner_dim :] *= 0
            self.vh[self.inner_dim :] *= 0
            self.s[self.inner_dim :, self.inner_dim :].mul_(0)

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

        change_k = self._update_inner_dim_and_shapes()
        return {"csmean": 1.0, "change_k": change_k}

    @torch.no_grad()
    def update_inner_rank(self, name):
        self.name = name
        # case 3: update the stable U and Vh from the full rank sigma matrix
        status = self._full_rank_update_usv()
        # shapes are updated within above (as is slice update)
        perc, _, _ = self.get_perc_params()
        log.info(
            f"{name[-30:]}: Full rank update, csmean: {status['csmean']:.3f}, params: {perc * 100:.2f}, "
            f"\t[{self.u.shape[0]} {self.s.shape[0]} {self.vh.shape[1]}]",
        )
        # log.info(f"singular values: {torch.diag(self.s)[:50].tolist()}")
        # self.inner_dim_buffer[0] = self.inner_dim.to(dtype=torch.float)
        # self.inner_dim_buffer[1] = status["csmean"]
        # self.inner_dim_buffer[2] = status["change_k"]

    def get_interior_inner_dim(self):
        return {
            "weight": self.inner_dim,
        }

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, interier k={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.inner_dim.item(),
        )

    # @torch.compile()
    def get_perc_params(self):
        normal_params = self.u.shape[0] * self.vh.shape[1]  # self.s.numel()
        bias_params = 0 if self.bias is None else self.bias.numel()
        trainable_params = self.inner_dim**2
        trainable_params += bias_params
        normal_params += bias_params
        perc_params = trainable_params / normal_params
        return perc_params, trainable_params, normal_params

    @torch.no_grad()
    def mix_sigma(self, method, *args, **kwargs):
        mixing.mix_sigma(*args, u=self.u, s=self.s, vh=self.vh, method=method, generator=self.gen, **kwargs)
