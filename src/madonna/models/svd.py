import logging
import math
import time
from collections import defaultdict
from copy import copy, deepcopy
from time import perf_counter
from typing import Optional, Tuple, Union

import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch._torch_docs import reproducibility_notes
from torch.nn import Parameter
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import parametrizations, parametrize

# from ..utils import utils
# from ..optimizers.utils import change_adam_shapes

# from .. import optimizers

log = logging.getLogger(__name__)


def change_adam_shapes(optimizer):
    """
    reset the shapes of the Adam optimizer buffers to be the same shape as the model parameters

    if `reset_buffers_zero`: reset the buffer to zero after reshaping it
    """
    resettime = time.perf_counter()
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    # instead of resetting optimizer, slice off bits of the saved states
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            if len(list(state.keys())) > 0:
                for k in ["exp_avg", "exp_avg_sq"]:
                    if state[k].shape != p.shape:
                        sl = []
                        for d in range(p.ndim):
                            sl.append(slice(0, p.shape[d]))
                        # print(type(state[k]))
                        state[k] = state[k][tuple(sl)]
                if group["amsgrad"]:
                    if state["max_exp_avg_sq"].shape != p.shape:
                        sl = []
                        for d in range(p.ndim):
                            sl.append(slice(0, p.shape[d]))
                        state["max_exp_avg_sq"] = state["max_exp_avg_sq"][tuple(sl)]
    if rank == 0:
        log.info(f"Reset Optimizer time: {time.perf_counter() - resettime}")


class SVDFixingModel(nn.Module):
    def __init__(
        self,
        existing_model: nn.Module,
        stability_frequency: int = 10,
        delay: int = 100,
        uvhthreshold: float = 0.999,
        sigma_cutoff_fraction: float = 0.1,
        sync_usv: bool = False,
        full_rank_sigma: bool = False,
        keep_first_layer: bool = False,
        keep_last_layer: bool = True,
        update_from_simga: bool = True,
        reinit_shapes: bool = True,
    ):
        super().__init__()
        self.uvhthreshold = uvhthreshold
        self.sigma_cutoff_fraction = sigma_cutoff_fraction
        self.sync_usv = sync_usv
        self.full_rank_sigma = full_rank_sigma
        self.update_from_simga = update_from_simga
        self.first_layer = keep_first_layer
        self.keep_last_layer = keep_last_layer
        self.last_layer = None
        self.reinit_shapes = reinit_shapes

        self.local_model = self._replace_layers(existing_model)
        if keep_last_layer:
            self._reset_last_layer(self.local_model)
        self.rank = 0

        if dist.is_initialized():
            self.rank = dist.get_rank()

            if dist.get_rank() == 0:
                log.info("Initializing DDP")
            self.ddp_model = DDP(self.local_model, find_unused_parameters=False)
        try:
            if dist.get_rank() == 0:
                print(self.ddp_model)
        except RuntimeError:  # dist is not initialized
            self.ddp_model = self.local_model
            # print(self.local_model)

        # self.compiled_model = torch.compile(self.ddp_model)
        # raise ValueError("")
        self.stability_frequency = stability_frequency
        self.call_count = 0
        self.num_stability_layvers_to_check = 0
        self.call_count_stability = 0
        self.layer_count_selector = torch.Generator()
        self.layer_count_selector.manual_seed(123456)
        self.skip_stability = False
        self.delay = delay
        self.stable_list = []

        num = 0
        for n, p in self.ddp_model.named_parameters():
            if p.requires_grad:
                if n[-2:] not in [".s", "_s", "_u", ".u", "vh"]:
                    # print(f"{n}: {p.numel()}")
                    num += p.numel()

        self.base_trainable_parameters = num
        self.svd_modules = []
        calls = 0
        sz = 1 if not dist.is_initialized() else dist.get_world_size()
        for name, mod in self.ddp_model.named_modules():
            if hasattr(mod, "test_stability_distributed"):
                working_rank = calls % sz
                self.svd_modules.append((name, mod, working_rank))
                try:  # only a module in the attention layers and its just faster to try to get something
                    if mod._qkv_same_embed_dim:
                        calls += 1
                    else:
                        calls += 3
                except AttributeError:
                    calls += 1
        self.fib1, self.fib2 = 0, 1
        self.next_stability_iteration = self.delay + self.fib1
        # n1, n2 = 0, 1
        # start = 100 -> delay
        # offset = 10 -> frequency
        # fib_epochs = [n1 + start]
        # while n2 < 200:
        #     n1, n2 = n2, n1 + n2
        #     fib_epochs.append(start + (n2 * offset))
        #     print(fib_epochs)
        # # fib_epochs.pop()
        # # fib_epochs.append(200 - 1)
        # print(f"fib_epochs {fib_epochs}")
        # print(self.svd_modules)
        self.optimizer = None  # optimizers.MixedSVDOpt
        self.local_generator = torch.Generator()
        self.local_generator.manual_seed(self.rank)
        # TODO: add me to forward function !!
        # optimizer.zero_grad(set_to_none=True, reset_sigma=True)
        #     stabtime = time.perf_counter()
        #     reset_optimizer, all_layers_stable = model.test_basis_stability_all_layers()

        #     if all_layers_stable and not all_stable:
        #         # NOTE: this will only do something when it isnt baseline
        #         if rank == 0:
        #             log.info("Deleting the full rank weights from the base optimizer")
        #         optimizer.remove_full_rank_weights()
        #         all_stable = all_layers_stable
        #     if reset_optimizer:
        #         optimizer.reset_shapes_of_sigma(model)
        self.all_stable = False
        self.state_dict = self.ddp_model.state_dict
        self.parameters = self.ddp_model.parameters
        self.named_parameters = self.ddp_model.named_parameters
        self.named_buffers = self.ddp_model.named_buffers
        self.named_children = self.ddp_model.named_children
        self.children = self.ddp_model.children
        self.cuda = self.ddp_model.cuda

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer  # optimizers.MixedSVDOpt

    def _replace_layers(self, module, name=None, process_group=None):
        module_output = module
        # print(f'wrapping {name} {module}')
        if isinstance(module, nn.Linear):
            if not self.first_layer:
                module_output = SVDLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    uvhthreshold=self.uvhthreshold,
                    sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                    sync_usv=self.sync_usv,
                    full_rank_sigma=self.full_rank_sigma,
                    start_weight=module.weight,
                    start_bias=module.bias,
                    update_from_simga=self.update_from_simga,
                    reinit_shapes=self.reinit_shapes,
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
                    uvh_threshold=self.uvhthreshold,
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
                    reinit_shapes=self.reinit_shapes,
                ).to(device=module.out_proj.weight.device, dtype=module.out_proj.weight.dtype)
                self.last_layer = [module, name]
            else:
                self.first_layer = False
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
            try:
                device = module.weight.device
                dtype = module.weight.dtype
            except AttributeError:
                try:
                    device = module.in_proj_weight.device
                    dtype = module.in_proj_weight.dtype
                except AttributeError:
                    device = module.q_proj_weight.device
                    dtype = module.q_proj_weight.dtype
            module_output = self.last_layer[0].to(device=device, dtype=dtype)
        for name, child in module.named_children():
            module_output.add_module(name, self._reset_last_layer(child, name))
        # del module
        return module_output

    @torch.no_grad()
    def test_basis_stability_all_layers_old(self):
        # self.sync_models()
        # if self.skip_stability:
        #     return
        rank = dist.get_rank() if dist.is_initialized() else 0
        sz = dist.get_world_size() if dist.is_initialized() else 1
        if rank == 0:
            log.info("Testing Stability")
        all_stable = True
        total = 0
        reset_optimizer = False
        calls = 0
        for c, (name, mod) in enumerate(self.ddp_model.named_modules()):
            # try:
            if hasattr(mod, "test_stability"):
                dist.barrier()
                working_rank = calls % sz
                print(name, working_rank)
                uchanging, k, perc, stable, changing_k = mod.test_stability(working_rank)
                print(f"done with test_stability for {name}")
                mod.wait_on_kusvh()
                print(f"done with waiting for {name}")
                try:
                    if mod._qkv_same_embed_dim:
                        calls += 1
                    else:
                        calls += 3
                except AttributeError:
                    calls += 1
                total += 1
                if rank == 0:
                    try:
                        uchange = f"{uchanging:.4f}"
                        percs = f"{perc * 100:.4f}"
                    except TypeError:  # in list return (qkv from attention...)
                        uchange = ""
                        percs = ""
                        for u in uchanging:
                            uchange += f"{u:.4f}, "
                        for p in perc:
                            percs += f"{p * 100:.1f}, "
                    # log.info(f"{name[-30:]}: UVh: {uchange} - k: {k} - % active params: {percs}")
                if changing_k:
                    reset_optimizer = True

        for c, (name, mod) in enumerate(self.ddp_model.named_modules()):
            # try:
            if hasattr(mod, "wait_on_kusvh"):
                mod.wait_on_kusvh()

        if dist.is_initialized():  # and reset_optimizer:
            # this indicates that the shapes of the parameters changed
            # need to re-init DDP to have the correct buckets
            # TODO: this might be a preformance hit
            ddp_time = perf_counter()
            del self.ddp_model
            self.ddp_model = DDP(self.local_model, find_unused_parameters=False, static_graph=False)
            if dist.get_rank() == 0:
                log.info(f"Reinit DDP. Time takesn: {perf_counter() - ddp_time}")
        if all_stable:
            self.skip_stability = True
        return reset_optimizer

    @torch.no_grad()
    def test_basis_stability_all_layers(self):
        # self.sync_models()
        # if self.skip_stability:
        #     return
        rank = dist.get_rank() if dist.is_initialized() else 0
        # sz = dist.get_world_size() if dist.is_initialized() else 1
        if rank == 0:
            log.info("Testing Stability")
        all_stable = True
        reset_optimizer = False
        # want to select a number of layers to run the stability on
        # increate the number based on the num_stability_checks
        inds = torch.arange(len(self.svd_modules))
        self.num_stability_layvers_to_check += int(inds[-1].item() * 0.1)
        # work in factors of 5% of the network
        cutoff = self.num_stability_layvers_to_check
        inds = inds[-cutoff:]
        if self.rank == 0:
            log.info(f"num layers for svd: {len(self.svd_modules)} cutoff: {cutoff} {len(inds)}")
        # self.svd_modules -> name, module, working rank
        for i in inds:
            name, mod, working_rank = self.svd_modules[i.item()]
            mod.test_stability_distributed(name=name, working_rank=working_rank, nonblocking=True)
        # dist.barrier()
        for i in inds:
            name, mod, working_rank = self.svd_modules[i.item()]
            reset_opt, stable = mod.wait_inner_dim_reshape_bcast_usvh(nonblocking=True)
            if reset_opt:
                reset_optimizer = True
            if not stable:
                all_stable = False
        # dist.barrier()
        for i in inds:
            _, mod, working_rank = self.svd_modules[i.item()]
            mod.wait_on_usvh()

        if dist.is_initialized():  # and reset_optimizer:
            # this indicates that the shapes of the parameters changed
            # need to re-init DDP to have the correct buckets
            # TODO: this might be a preformance hit
            ddp_time = perf_counter()
            self.ddp_model = DDP(self.local_model, find_unused_parameters=False, static_graph=False)
            if dist.get_rank() == 0:
                log.info(f"Reinit DDP. Time taken: {perf_counter() - ddp_time}")
        # if all_stable:
        #     self.skip_stability = True
        return reset_optimizer, all_stable

    @torch.no_grad()
    def get_perc_params_all_layers(self, module):
        # self.sync_models()
        # if self.skip_stability:
        #     return
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        # percs, actives, normals = [], [], []
        full_active = 0
        # print("in perc params")
        for n, p in self.ddp_model.named_parameters():
            # print(f"{n} {p.requires_grad}")
            if p.requires_grad:
                # if n[-2:] not in [".s", "_s", "_u", ".u", "vh"]:
                full_active += p.numel()

        full_normal = self.base_trainable_parameters
        if rank == 0:
            log.info(
                f"Active Params: {100 * (full_active / full_normal):.4f}%",
            )
        return 100 * (full_active / full_normal), full_active, full_normal

    def check_stability_on_count(self, force=False):
        ret = False
        if self.ddp_model.training:
            self.call_count += 1
            if force:
                return self.test_basis_stability_all_layers()

            if (
                self.call_count % self.stability_frequency == self.stability_frequency - 1
                and self.call_count >= self.delay
            ):
                ret = self.test_basis_stability_all_layers()
                # self.train()
        return ret

    @staticmethod
    def reset_all_states(optimizer: optim.Optimizer):
        # reset op1 first
        # for group in self.opt1.param_groups:
        optimizer.state = defaultdict(dict)

    @torch.no_grad()
    def model_stability_tracking(self):
        # TODO: Should this be moved to the training script? might be easier...
        self.call_count += 1
        if self.call_count != self.next_stability_iteration:
            return False
        self.call_count_stability += 1

        stabtime = time.perf_counter()
        reset_optimizer, all_layers_stable = self.test_basis_stability_all_layers()

        if all_layers_stable and not self.all_stable:
            # NOTE: no need to remove the full_rank_weights, they will cause a few extra clock ticks but nothing more
            # self.optimizer.remove_full_rank_weights()
            # self.reset_all_states(self.optimizer)
            self.all_stable = all_layers_stable
        if reset_optimizer:
            change_adam_shapes(self.optimizer)
            # self.reset_all_states(self.optimizer)
            # self.insert_noise(noise_level=1e-2)
            # self.optimizer.reset_shapes_of_sigma(self)
        self.optimizer.zero_grad(set_to_none=True)

        # self.ddp_model = DDP(self.local_model, find_unused_parameters=False, static_graph=False)
        # dist.barrier()

        # self.fib1, self.fib2 = self.fib2, self.fib1 + self.fib2
        # self.next_stability_iteration = self.delay + (self.fib2 * self.stability_frequency)
        self.next_stability_iteration += self.stability_frequency
        self.next_stability_iteration = int(self.next_stability_iteration)

        if self.rank == 0:
            log.info(
                f"Stability time: {time.perf_counter() - stabtime}\t"
                f"Current iteration: {self.call_count}\t"
                f"Next iteration: {self.next_stability_iteration}",
            )
        return True

    @torch.no_grad()
    def insert_noise(self, noise_level=1e-1):
        for n, p in self.ddp_model.named_parameters():
            if n.endswith(".s") or n.endswith("_s") and p.requires_grad:
                # # TODO: remove later if not working
                # sdiag = torch.diag(self.s).clone()
                sdiag = torch.diag(p)
                # sdiag_diff1 = torch.diff(sdiag, n=1) * 0.001
                # sdiag_diff2 = torch.diff(sdiag, n=2) * 0.0001
                # # sdiag_diff3 = torch.diff(sdiag, n=3) * 0.001
                # for i in range(p.shape[0] - 1):
                #     p[i, i + 1] = sdiag_diff1[i]
                #     if i < p.shape[0] - 2:
                #         p[i, i + 2] = sdiag_diff2[i]

                # mask = torch.abs(p) <= 1e-5
                # umask = torch.abs(p) > 1e-7

                # print(f"{n} -> {torch.count_nonzero(mask)}")
                # p[mask] *= 0
                # p[mask] += noise_level * torch.rand_like(p[mask]) * sdiag.min()
                # rand = torch.rand(
                #     p[mask].shape, generator=self.local_generator, device=p.device, dtype=p.dtype
                # )
                # p[mask] += (1 / sdiag[0]) * rand * noise_level
                if self.local_generator.device != p.device:
                    self.local_generator = torch.Generator(device=p.device)
                    self.local_generator.manual_seed(self.rank + 10000)
                rand = torch.rand(
                    p.shape,
                    generator=self.local_generator,
                    device=p.device,
                    dtype=p.dtype,
                )
                p += (1 / sdiag[0]) * rand / (self.call_count_stability)

    def forward(self, inputs):
        return self.ddp_model(inputs)

    @torch.no_grad()
    def track_interior_slices_mlflow(self, config, epoch):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if not config.enable_tracking or rank != 0:
            return
        # for c, (name, mod) in enumerate(self.ddp_model.named_modules()):
        #     # try:
        #     if hasattr(mod, "test_stability"):
        #         slices = mod.get_interior_inner_dim()
        #         for sl in slices:
        #             mlflow.log_metric(name + f".{sl}", slices[sl], step=epoch)
        #             # print(f"logging interior slice for {name}.{sl}")

    @torch.no_grad()
    def sync_models(self, verbose=True):
        if not dist.is_initialized():
            return
        rank = dist.get_rank()
        if rank == 0:
            log.info("Syncing layers which require grad on all layers")
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
        uvhthreshold: float = 0.9,
        sigma_cutoff_fraction: float = 0.1,
        sync_usv: bool = False,
        full_rank_sigma: bool = False,
        start_weight=None,
        start_bias=None,
        update_from_simga=True,
        reinit_shapes=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SVDLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.full_rank_sigma = full_rank_sigma
        self.update_from_simga = full_rank_sigma and update_from_simga
        self.reinit_shapes = reinit_shapes

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

        self.u = nn.Parameter(self.u, requires_grad=False)
        self.s = nn.Parameter(self.s, requires_grad=False)
        self.vh = nn.Parameter(self.vh, requires_grad=False)

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
        # self.u_stable = False
        self.uvh_stable = False
        self.u_prev = None
        # self.vh_stable = False
        self.vh_prev = None
        self.s_prev = None
        self.inner_dim = torch.tensor(min(in_features, out_features), dtype=torch.int)
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

    # @torch.compile()  # TODO: compiling seems to be weird here...might need to compile two different functions?
    def get_weight(self):
        if not self.uvh_stable:
            # if both are not stable -> normal training
            # if dist.get_rank() == 0:
            #     log.info("Using self.weight")
            self.weight.requires_grad = True
            self.u.requires_grad = False
            self.vh.requires_grad = False
            self.s.requires_grad = False
            return self.weight
        # detach sets 'requires grad' to False
        if self.training:
            self.s.requires_grad = True
            if self.uvh_stable:
                self.u.requires_grad = False
                self.vh.requires_grad = False
                self.weight.requires_grad = False

                # self.bias.requires_grad = False
        u, vh = self.u, self.vh
        # u = self.u.detach() if self.u_stable else self.u
        # vh = self.vh.detach() if self.vh_stable else self.vh

        s = self.s if self.full_rank_sigma else torch.diag(self.s)

        w = torch.linalg.multi_dot([u, s, vh])
        ret = w.T if self.trans else w
        # eps = torch.finfo(ret.dtype).eps * 10
        # ret[torch.abs(ret) < eps] *= 0
        with torch.no_grad():
            self.weight *= 0
            self.weight += ret
        # if dist.get_rank() == 0:
        #     log.info("Using USV")
        return ret

    # @torch.compile()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.get_weight()
        # if self.bias is None:
        #     self.bias.requires_grad = False
        # print(f"bias is None? {self.bias is None}")
        return F.linear(input, w, self.bias)

    # @torch.no_grad()
    # def test_stability_old(self, working_rank=0, nonblocking=True):
    #     # TODO: should we make sure to have S be the same across processes?
    #     if self.rank != working_rank:
    #         self.inner_dim = self.inner_dim.to(self.weight.device)
    #         self.bcast_kusvh(src=working_rank)
    #         return 4, self.inner_dim, 0.0, True, True
    #     # updating usv in full rank is different!

    #     # if self.uvh_stable and not self.update_from_simga:
    #     #     # sdiff = self.s_prev - self.s
    #     #     # self.s_prev = self.s.data.clone()
    #     #     # if rank == 0:
    #     #     #     print(f"s diff: {sdiff.mean():.4f}, {sdiff.min():.4f}, {sdiff.max():.4f}")
    #     #     # switch back to check on the SVD stuff??
    #     #     # self.uvh_stable = False
    #     #     perc_params, _, _ = self.get_perc_params()
    #     #     return 3, self.k, perc_params, True, False
    #     set_usvh = True
    #     if self.full_rank_sigma and self.uvh_stable and self.update_from_simga:
    #         self._full_rank_update_usv()
    #         # self.uvh_stable = False
    #         # self._update_k()
    #         # set_usvh = False
    #         u, s, vh = self.u, self.s, self.vh
    #         uvh = u @ vh
    #         # self.update_from_simga = False
    #         perc_params, _, _ = self.get_perc_params()
    #         # return 2, self.k, perc_params, True, False
    #     else:
    #         w = self.weight.T if self.trans else self.weight
    #         # w = self.get_weight()
    #         # w = w.T if self.trans else w
    #         dtp = w.dtype
    #         u, s, vh = torch.linalg.svd(w.to(torch.float32), full_matrices=False)  # , driver="gesvd")
    #         u = u.to(dtp)
    #         s = s.to(dtp)
    #         vh = vh.to(dtp)
    #         uvh = u @ vh
    #         if self.prev_uvh is None:
    #             if self.rank == working_rank:
    #                 log.info("linear in first stability update")
    #             self.prev_uvh = uvh
    #             self.u.zero_()
    #             self.u.add_(u)
    #             self.s.zero_()
    #             self.s.add_(torch.diag(s) if self.full_rank_sigma else s)
    #             self.vh.zero_()
    #             self.vh.add_(vh)

    #             self.inner_dim = self.inner_dim.to(device=s.device)
    #             # send to other ranks
    #             self.bcast_kusvh(src=working_rank)
    #             return 0, self.inner_dim, 1.0, self.uvh_stable, False
    #         self.prev_uvh = uvh
    #         if self.rank == working_rank:
    #             log.info("linear in normal stability update")

    #     # use cosine similarity (dot product for orthonormal) to determine similarity
    #     csim = self.cossim(self.prev_uvh, uvh)
    #     self.prev_uvh = uvh
    #     csmean, _ = csim.mean(), csim.std()
    #     self.prev_uvh = uvh
    #     change_k = False
    #     if csmean > self.uvhthreshold:
    #         self.uvh_stable = True

    #         self.u.zero_()
    #         self.u.add_(u)
    #         self.vh.zero_()
    #         self.vh.add_(vh)
    #         if self.full_rank_sigma:
    #             self.s.zero_()
    #             self.s[: self.inner_dim, : self.inner_dim].add_(torch.diag(s[: self.inner_dim]))
    #         else:
    #             self.s.zero_()
    #             self.s[: self.inner_dim].add_(s[: self.inner_dim])

    #         self.weight.requires_grad = False
    #         self.u.requires_grad = False
    #         self.vh.requires_grad = False
    #         self.s.requires_grad = True

    #         change_k = self._update_inner_dim()
    #         self._update_usvh_shapes()
    #     perc_params, _, _ = self.get_perc_params()

    #     # send to other ranks
    #     self.bcast_kusvh(src=working_rank)
    #     return csmean, self.inner_dim, perc_params, self.uvh_stable, change_k

    @torch.no_grad()
    def _update_inner_dim_and_shapes(self):
        # adjust K to slice of less important singular values
        s = torch.diag(self.s) if self.s.ndim == 2 else self.s
        prevk = self.inner_dim.clone()
        # new plan for cutoff - instead of using s[0] use 1% of the minimum
        # this will enforce that the array never shrinks below 1%
        if self.uvh_stable:
            # cutoff = s[0] * self.sigma_cutoff_fraction
            min_dim = int(self.vh.shape[-1] * 0.01)  # always TS
            cutoff = s[min_dim] * self.sigma_cutoff_fraction
            nz = torch.nonzero(s < cutoff)
            if len(nz) == 0:
                # In this case ALL of the basis vectors are useful
                newk = s.shape[0]
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
    def _first_stability(self):
        log.debug("linear: first stability call")
        w = self.weight.T if self.trans else self.weight
        # w = self.get_weight()
        # w = w.T if self.trans else w
        dtp = w.dtype
        w = w.to(torch.float32)
        print(f"first stability: {w.mean():.4f} {w.min():.4f} {w.max():.4f}")
        u, s, vh = torch.linalg.svd(w, full_matrices=False)  # , driver="gesvd")

        u = u.to(dtp)
        s = s.to(dtp)
        vh = vh.to(dtp)
        uvh = u @ vh

        self.prev_uvh = uvh
        self.u.zero_()
        self.u.add_(u)
        self.s.zero_()
        self.s.add_(torch.diag(s) if self.full_rank_sigma else s)
        self.vh.zero_()
        self.vh.add_(vh)

        self.inner_dim = self.inner_dim.to(device=s.device)

    @torch.no_grad()
    def _full_rank_weight_stability(self):
        w = self.weight.T if self.trans else self.weight
        # w = self.get_weight()
        # w = w.T if self.trans else w
        dtp = w.dtype
        w = w.to(torch.float32)
        u, s, vh = torch.linalg.svd(w, full_matrices=False)  # , driver="gesvd")
        u = u.to(dtp)
        s = s.to(dtp)
        vh = vh.to(dtp)
        uvh = u @ vh
        csim = self.cossim(self.prev_uvh, uvh)
        csmean, _ = csim.mean(), csim.std()
        # comp = w - (u @ torch.diag(s) @ vh)
        print(
            f"normal stability update: {w.mean():.4f} {w.min():.4f} {w.max():.4f} similarity: {csim.mean():.4f} {csim.std():.4f}\t",
            # f"{comp.mean():.4f}\t{comp.min():.4f}\t{comp.max():.4f}"
        )
        self.prev_uvh.zero_()
        self.prev_uvh.add_(uvh)
        change_k = False
        if csmean >= self.uvhthreshold:
            self.uvh_stable = True

            self.u.zero_()
            self.u.add_(u)
            self.vh.zero_()
            self.vh.add_(vh)
            if self.full_rank_sigma:
                self.s.zero_()
                self.s[: self.inner_dim, : self.inner_dim].add_(torch.diag(s[: self.inner_dim]))
            else:
                self.s.zero_()
                self.s[: self.inner_dim].add_(s[: self.inner_dim])

            self.weight.requires_grad = False
            self.u.requires_grad = False
            self.vh.requires_grad = False
            self.s.requires_grad = True
            change_k = self._update_inner_dim_and_shapes()
        return {"csmean": csmean, "change_k": change_k}

    @torch.no_grad()
    def _full_rank_update_usv(self):
        if not self.full_rank_sigma and not self.uvh_stable:
            raise ValueError("this function is only for full-rank sigma with usvh is stable")

        # if self.rank == working_rank:
        # log.info("in full rank sigma update of usvh")
        # NOTE: no slicing because need the shapes to line up. self.s[self.k:, self.k:] should be 0?
        s = self.s.to(torch.float32)
        usig, sig, vhsig = torch.linalg.svd(s)  # square mat, full mat arg doesnt matter
        usig = usig.to(self.s.dtype)
        sig = sig.to(self.s.dtype)
        vhsig = vhsig.to(self.s.dtype)
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
        # w = torch.linalg.multi_dot([self.u, self.s, self.vh])
        # ret = w.T if self.trans else w
        # self.weight *= 0
        # self.weight += ret

        # normal update from cosine similarity stuff
        # uvh = self.u @ self.vh
        # csim = self.cossim(self.prev_uvh, uvh)
        # csmean, _ = csim.mean(), csim.std()
        # self.prev_uvh.zero_()
        # self.prev_uvh.add_(uvh)
        csmean = 1.0

        # rand_noise = torch.rand_like(self.s) * torch.diag(self.s).min() * 0.1
        # rand_noise.fill_diagonal_(0)
        # self.s.add(rand_noise)

        # csmean = 1.0
        # change_k = False
        self.uvh_stable = True
        self.weight.requires_grad = False
        self.u.requires_grad = False
        self.vh.requires_grad = False
        self.s.requires_grad = True
        change_k = False
        if csmean >= self.uvhthreshold:
            change_k = self._update_inner_dim_and_shapes()
        return {"csmean": csmean, "change_k": change_k}

    @torch.no_grad()
    def test_stability_distributed(self, working_rank, name, nonblocking=True):
        self.name = name

        # if we dont need to send anything as its all frozen and not touched, have early exit here
        if (
            self.prev_uvh is not None
            and not (self.full_rank_sigma and self.uvh_stable and self.update_from_simga)
            and self.uvh_stable
        ):
            # early out here without any communication
            self.last_send_rank = None
            if working_rank == self.rank:
                log.info(
                    f"{name[-30:]}: All Frozen, [{self.u.shape[0]}, {self.s.shape[0]},{self.vh.shape[1]}]",
                )
            return

        self.last_send_rank = working_rank
        if working_rank != self.rank:
            # move on for now, coming back later
            # make sure to save the rank which did this layer
            # print(f"{self.rank} != {working_rank} skipping computation, starting bcast ops")
            if self.prev_uvh is None:
                # if no prev_uvh -> first iteration: need to get usvh only, can start now
                self.inner_dim_buffer = self.inner_dim_buffer.to(device=self.weight.device, non_blocking=True)
                self.inner_dim = self.inner_dim.to(device=self.weight.device, non_blocking=True)
                # if in the first iteration
                # self.bcast_usvh(src=working_rank, nonblocking=nonblocking)
                self.prev_uvh = torch.tensor(1, device=self.weight.device)  # doing this to satisfy 'not None'
                # self.wait_inner_dim = None
                # return
            # receive the wait_k from the working process
            self.wait_inner_dim = dist.broadcast(
                self.inner_dim_buffer,
                src=working_rank,
                async_op=nonblocking,
            )
            return

        if self.prev_uvh is not None and self.prev_uvh.ndim == 0:  # just in case...
            # if somehow we get into this block, we will just go through the first stability check
            # also need to reset the stability of the layer
            # but would need to reset the stability on all ranks...
            self.prev_uvh = self.u @ self.vh

        if self.prev_uvh is None:
            self.inner_dim_buffer = self.inner_dim_buffer.to(device=self.weight.device, non_blocking=True)
            self.inner_dim = self.inner_dim.to(device=self.weight.device, non_blocking=True)
            # case 1: do SVD for the first time and calculate the basis
            self._first_stability()
            # print('after linear stability')
            self.wait_inner_dim = None
            status = {"csmean": 0.0, "change_k": 0.0}
            # change of plans: start USVh communcation in the wait function just like the other cases
            # self.bcast_usvh(src=working_rank, nonblocking=nonblocking)
            log.info(
                f"{name[-30:]}: 1st stability, csmean: None, params: 100%, [{self.weight.shape[0]}, {self.weight.shape[1]}]",
            )
            # return
        elif self.full_rank_sigma and self.uvh_stable and self.update_from_simga:
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
        elif not self.uvh_stable:
            # case 2: normal stability update
            status = self._full_rank_weight_stability()
            perc, _, _ = self.get_perc_params()
            # normalize self.s to max == 1
            # maxs = self.s.max()
            # self.s /= maxs
            # self.vh *= maxs
            log.info(
                f"{name[-30:]}: Normal stability, csmean: {status['csmean']:.3f}, params: {perc * 100:.2f}, "
                f"\t[{self.u.shape[0]} {self.s.shape[0]} {self.vh.shape[1]}]",
            )
        else:
            # case here is when uvh is frozen but we are not updating the bases
            # dont need to do any communication, dont need to computation
            # can just get the percentage of active parameters/whatever else needs to be returned for logs
            raise RuntimeError("something went wrong, stuck in 'else' in stability...")

        if not dist.is_initialized():
            return
        self.inner_dim_buffer[0] = self.inner_dim.to(torch.float)
        self.inner_dim_buffer[1] = status["csmean"]
        self.inner_dim_buffer[2] = status["change_k"]
        # if status["csmean"] >= self.uvhthreshold:
        #     self.uvh_stable = True

        #     self.weight.requires_grad = False
        #     self.weight.grad = None
        #     self.u.requires_grad = False
        #     self.vh.requires_grad = False
        #     self.s.requires_grad = True
        self.wait_inner_dim = dist.broadcast(self.inner_dim_buffer, src=working_rank, async_op=nonblocking)

    @torch.no_grad()
    def wait_inner_dim_reshape_bcast_usvh(self, nonblocking=True):
        # if wait_k is None -> K is the same -> optimizer is fine (shapes are the same)
        reset_optimizer = False
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

            if self.inner_dim_buffer[1] >= self.uvhthreshold:
                self.uvh_stable = True

                self.weight.requires_grad = False
                self.weight.grad = None
                self.u.requires_grad = False
                self.vh.requires_grad = False
                self.s.requires_grad = True
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
        return "in_features={}, out_features={}, bias={}, interier k={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.inner_dim.item(),
        )

    # @torch.compile()
    def get_perc_params(self):
        normal_params = self.weight.numel()
        bias_params = 0 if self.bias is None else self.bias.numel()
        if self.uvh_stable:
            trainable_params = self.inner_dim
            if self.full_rank_sigma:
                trainable_params = trainable_params**2
        else:
            trainable_params = normal_params
        trainable_params += bias_params
        normal_params += bias_params
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

    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
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
        reinit_shapes=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
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
        self.reinit_shapes = reinit_shapes
        assert self.head_dim * num_heads == self.embed_dim, "num_heads must be factor of embed_dim"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.q_u = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs), requires_grad=False)
            if self.full_rank_sigma:
                self.q_s = Parameter(
                    torch.empty((embed_dim, embed_dim), **factory_kwargs),
                    requires_grad=False,
                )
            else:
                self.q_s = Parameter(torch.empty((embed_dim), **factory_kwargs), requires_grad=False)
            self.q_vh = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs), requires_grad=False)
            self.q_trans = False
            self.q_inner_dim = torch.tensor(embed_dim, dtype=torch.int)

            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            if self.kdim > embed_dim:
                # u - kdim x embed, s - embed x embed, vh - embed x embed -> after trans is embed x kdim
                self.k_trans = True
                self.k_u = Parameter(
                    torch.empty((self.kdim, embed_dim), **factory_kwargs),
                    requires_grad=False,
                )
                if not self.full_rank_sigma:
                    self.k_s = Parameter(torch.empty((embed_dim), **factory_kwargs), requires_grad=False)
                else:
                    self.k_s = Parameter(
                        torch.empty((embed_dim, embed_dim), **factory_kwargs),
                        requires_grad=False,
                    )
                self.k_vh = Parameter(
                    torch.empty((embed_dim, embed_dim), **factory_kwargs),
                    requires_grad=False,
                )
                self.k_inner_dim = torch.tensor(embed_dim, dtype=torch.int)
            else:
                # u - embed x kdim, s - kdim x kdim, vh - kdim x kdim
                self.k_trans = False
                self.k_u = Parameter(
                    torch.empty((embed_dim, self.kdim), **factory_kwargs),
                    requires_grad=False,
                )
                if not self.full_rank_sigma:
                    self.k_s = Parameter(torch.empty((self.kdim), **factory_kwargs), requires_grad=False)
                else:
                    self.k_s = Parameter(
                        torch.empty((self.kdim, self.kdim), **factory_kwargs),
                        requires_grad=False,
                    )
                self.k_vh = Parameter(
                    torch.empty((self.kdim, self.kdim), **factory_kwargs),
                    requires_grad=False,
                )
                self.k_inner_dim = torch.tensor(self.kdim, dtype=torch.int)

            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            if self.vdim > embed_dim:
                # u - vdim x embed, s - embed x embed, vh - embed x embed -> after trans is embed x vdim
                self.v_trans = True
                self.v_u = Parameter(
                    torch.empty((self.vdim, embed_dim), **factory_kwargs),
                    requires_grad=False,
                )
                if not self.full_rank_sigma:
                    self.v_s = Parameter(torch.empty((embed_dim), **factory_kwargs), requires_grad=False)
                else:
                    self.v_s = Parameter(
                        torch.empty((embed_dim, embed_dim), **factory_kwargs),
                        requires_grad=False,
                    )
                self.v_vh = Parameter(
                    torch.empty((embed_dim, embed_dim), **factory_kwargs),
                    requires_grad=False,
                )
                self.v_inner_dim = torch.tensor(embed_dim, dtype=torch.int)
            else:
                # u - embed x vdim, s - vdim x vdim, vh - vdim x vdim
                self.v_trans = False
                self.v_u = Parameter(
                    torch.empty((embed_dim, self.vdim), **factory_kwargs),
                    requires_grad=False,
                )
                if not self.full_rank_sigma:
                    self.v_s = Parameter(torch.empty((self.vdim), **factory_kwargs), requires_grad=False)
                else:
                    self.v_s = Parameter(
                        torch.empty((self.vdim, self.vdim), **factory_kwargs),
                        requires_grad=False,
                    )
                self.v_vh = Parameter(
                    torch.empty((self.vdim, self.vdim), **factory_kwargs),
                    requires_grad=False,
                )
                self.v_inner_dim = torch.tensor(self.vdim, dtype=torch.int)
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            # in_proj is always TS
            self.in_proj_u = Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs),
                requires_grad=False,
            )
            if not self.full_rank_sigma:
                self.in_proj_s = Parameter(torch.empty((embed_dim), **factory_kwargs), requires_grad=False)
            else:
                self.in_proj_s = Parameter(
                    torch.empty((embed_dim, embed_dim), **factory_kwargs),
                    requires_grad=False,
                )
            self.in_proj_vh = Parameter(
                torch.empty((embed_dim, embed_dim), **factory_kwargs),
                requires_grad=False,
            )
            self.in_proj_trans = False
            self.in_proj_inner_dim = torch.tensor(embed_dim, dtype=torch.int)

            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()
        self.sigma_cutoff_fraction = sigma_cutoff_fraction
        self.uvh_stable_q = False
        self.uvh_stable_k = False
        self.uvh_stable_v = False
        self.uvh_stable_in_proj = False
        if not self._qkv_same_embed_dim:
            self.prev_uvh_q = None
            self.prev_uvh_k = None
            self.prev_uvh_v = None
        else:
            self.prev_uvh_in_proj = None

        self.cossim = nn.CosineSimilarity(dim=0)
        self.uvhthreshold = uvh_threshold

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.size = dist.get_world_size() if dist.is_initialized() else 1

        self.last_send_ranks = {"q": None, "k": None, "v": None, "in_proj": None}
        self.waits = {
            "q": {"u": None, "s": None, "vh": None, "inner_dim": None},
            "k": {"u": None, "s": None, "vh": None, "inner_dim": None},
            "v": {"u": None, "s": None, "vh": None, "inner_dim": None},
            "in_proj": {"u": None, "s": None, "vh": None, "inner_dim": None},
        }
        self.inner_dim_buffers = {
            "q": torch.zeros(3),
            "k": torch.zeros(3),
            "v": torch.zeros(3),
            "in_proj": torch.zeros(3),
        }

        with torch.no_grad():  # set class params from existing
            if not self._qkv_same_embed_dim:  # in this case, have q, k, v and bias_k and bias_v
                if start_in_proj is not None and start_q is None:
                    sh = start_in_proj.shape[0] // 3
                    start_q = start_in_proj[:sh]
                    start_k = start_in_proj[sh : sh * 2]
                    start_v = start_in_proj[sh * 2 : sh * 3]
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
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def _get_device(self):
        if self._qkv_same_embed_dim:
            return self.in_proj_weight.device
        else:
            return self.q_proj_weight.device

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super().__setstate__(state)

    # @torch.compile()
    def _get_q(self):
        if self.q_proj_weight is None:
            return self.q_proj_weight
        if not self.uvh_stable_q:
            self.q_u.requires_grad = False
            self.q_s.requires_grad = False
            self.q_vh.requires_grad = False
            return self.q_proj_weight

        if self.training:
            self.q_u.requires_grad = False
            self.q_s.requires_grad = True
            self.q_vh.requires_grad = False
            self.q_proj_weight.requires_grad = False
        u, vh = self.q_u, self.q_vh
        # u = self.q_u.detach()
        # vh = self.q_vh.detach()

        s = self.q_s if self.full_rank_sigma else torch.diag(self.q_s)

        ret = torch.linalg.multi_dot([u, s, vh])
        # no transpose for q - square matrix
        with torch.no_grad():
            self.q_proj_weight *= 0
            self.q_proj_weight += ret
        return ret

    # @torch.compile()
    def _get_k(self):
        if self.k_proj_weight is None:
            return self.k_proj_weight
        if not self.uvh_stable_k:
            self.k_u.requires_grad = False
            self.k_s.requires_grad = False
            self.k_vh.requires_grad = False
            return self.k_proj_weight
        if self.training:
            self.k_u.requires_grad = False
            self.k_s.requires_grad = True
            self.k_vh.requires_grad = False
            self.k_proj_weight.requires_grad = False
        u, vh = self.k_u, self.k_vh
        # u = self.k_u.detach()
        # vh = self.k_vh.detach()

        s = self.k_s if self.full_rank_sigma else torch.diag(self.k_s)

        w = torch.linalg.multi_dot([u, s, vh])
        ret = w.T if self.k_trans else w
        with torch.no_grad():
            self.k_proj_weight *= 0
            self.k_proj_weight += ret
        return ret

    # @torch.compile()
    def _get_v(self):
        if self.v_proj_weight is None:
            return self.v_proj_weight
        if not self.uvh_stable_v:
            self.v_u.requires_grad = False
            self.v_s.requires_grad = False
            self.v_vh.requires_grad = False
            return self.v_proj_weight
        if self.training:
            self.v_u.requires_grad = False
            self.v_s.requires_grad = True
            self.v_vh.requires_grad = False
            self.v_proj_weight.requires_grad = False
        u, vh = self.v_u, self.v_vh
        # u = self.v_u.detach()
        # vh = self.v_vh.detach()

        s = self.v_s if self.full_rank_sigma else torch.diag(self.v_s)

        w = torch.linalg.multi_dot([u, s, vh])
        ret = w.T if self.v_trans else w
        with torch.no_grad():
            self.v_proj_weight *= 0
            self.v_proj_weight += ret
        return ret

    # @torch.compile()
    def _get_in_proj(self) -> Tensor:
        if self.in_proj_weight is None:
            # if rank == 0:
            #     log.info("in_proj weight is None")
            return self.in_proj_weight
        if not self.uvh_stable_in_proj:
            # print('uvh not stable', type(self.in_proj_weight))
            # if rank == 0:
            #     log.info("Using default in_proj_weight")
            self.in_proj_u.requires_grad = False
            self.in_proj_s.requires_grad = False
            self.in_proj_vh.requires_grad = False
            return self.in_proj_weight
        if self.training:
            self.in_proj_u.requires_grad = False
            self.in_proj_vh.requires_grad = False
            self.in_proj_weight.requires_grad = False

        self.in_proj_s.requires_grad = True
        u = self.in_proj_u  # .detach()
        vh = self.in_proj_vh  # .detach()

        s = self.in_proj_s if self.full_rank_sigma else torch.diag(self.in_proj_s)

        ret = torch.linalg.multi_dot([u, s, vh])
        # eps = torch.finfo(ret.dtype).eps * 10
        # ret[torch.abs(ret) < eps] *= 0
        # No need for transpose, in_proj is always TS (be definition)
        with torch.no_grad():
            self.in_proj_weight *= 0
            self.in_proj_weight += ret
        # if rank == 0:
        #     log.info("Using USVh in_proj_weight")
        return ret

    # @torch.compile()
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
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`,
                shape should be :math:`(S)`. Binary and float masks are supported.
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
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        why_not_fast_path = ""
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = (
                f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
            )
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = (
                f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
            )
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

        try:
            out_proj_weights = self.out_proj.get_weight()
        except AttributeError:
            out_proj_weights = self.out_proj.weight

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self._get_in_proj(),  # self.in_proj_weight,
                self.in_proj_bias,
                out_proj_weights,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x is None or x.is_cuda or "cpu" in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
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
                    out_proj_weights,
                    self.out_proj.bias,
                    merged_mask,
                    need_weights,
                    average_attn_weights,
                    mask_type,
                )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )

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
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                in_proj,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                out_proj_weights,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=q,
                k_proj_weight=k,
                v_proj_weight=v,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            in_proj = self._get_in_proj()  # self.in_proj_weight,
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                in_proj,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                out_proj_weights,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def merge_masks(
        self,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        query: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[int]]:
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
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(
                    batch_size,
                    self.num_heads,
                    -1,
                    -1,
                )
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(
                    -1,
                    self.num_heads,
                    -1,
                    -1,
                )
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type

    # # TODO: compile?
    # @torch.no_grad()
    # def _test_stability_general_old(self, qkvin, working_rank=0, nonblocking=True):
    #     # TODO: move me!
    #     # # if self._qkv_same_embed_dim: -> training on in_proj and qkv are none
    #     # if self.in_proj_weight is not None:
    #     #     return

    #     """
    #     This is going to do the qkv updates for q, k, v, or in_proj based on the value of qkvin

    #     uvh_stable_[in_proj, q, k, v]
    #     [in_proj, q, k, v]_u
    #     [in_proj, q, k, v]_s
    #     [in_proj, q, k, v]_vh
    #     [in_proj, q, k, v]_trans
    #     prev_uvh_[in_proj, q, k, v]
    #     _get_[in_proj, q, k, v]
    #     [in_proj, q, k, v]_inner_dim
    #     base weights:
    #         [q, k, v]_proj_weight
    #         in_proj_weight
    #     """
    #     # if getattr(self, f"uvh_stable_{qkvin}") and not self.update_from_simga:
    #     #     perc_params, _, _ = self.get_perc_params()
    #     #     self._bcast_kusvh_abs(qkvin, working_rank, nonblocking)
    #     #     return 3., getattr(self, f"{qkvin}_inner_dim"), perc_params, getattr(self, f"uvh_stable_{qkvin}"), False

    #     rank = dist.get_rank() if dist.is_initialized() else 0
    #     # print(f"{qkvin} {rank} {working_rank}")
    #     # if rank != working_rank:
    #     #     sl = getattr(self, f"{qkvin}_inner_dim")
    #     #     sl = sl.to(device=self._get_device())
    #     #     setattr(self, f"{qkvin}_inner_dim", sl)
    #     #     self._bcast_kusvh_abs(qkvin, working_rank, nonblocking)
    #     #     print(f"rank {rank} exiting moving on for {qkvin}")
    #     #     return 4., getattr(self, f"{qkvin}_inner_dim"), -1., getattr(self, f"uvh_stable_{qkvin}"), False
    #     print(f"rank {rank} doing normal logic {qkvin}")

    #     set_usvh = True  # if true: skip the update of USVH (only false for full_rank which had a different update logic)
    #     if self.full_rank_sigma and getattr(self, f"uvh_stable_{qkvin}") and self.update_from_simga:
    #         # updating U/Vh from full-rank sigma
    #         if rank == working_rank:
    #             log.debug(f"Full rank sigma update of usvh: {qkvin}")
    #         # TODO: update _update_usv!
    #         self._update_usv(qkvin)
    #         set_usvh = False
    #         u, s, vh, gensl = self._get_usvh_from_qkvin(qkvin)
    #         uvh = u @ vh
    #         # self.update_from_simga = False
    #         perc_params, _, _ = self.get_perc_params()
    #         # self._bcast_kusvh_abs(qkvin, working_rank, nonblocking)
    #         # return 2., getattr(self, f"{qkvin}_inner_dim"), perc_params, getattr(self, f"uvh_stable_{qkvin}"), False
    #     else:
    #         # w = self.weight.T if self.trans else self.weight
    #         w = getattr(self, f"{qkvin}_proj_weight") if qkvin in "qkv" else self.in_proj_weight
    #         w = w.T if getattr(self, f"{qkvin}_trans") else w
    #         # w = getattr(self, f"_get_{qkvin}")()
    #         dtp = w.dtype
    #         u, s, vh = torch.linalg.svd(w.to(torch.float32), full_matrices=False)  # , driver="gesvd")
    #         u = u.to(dtp)
    #         s = s.to(dtp)
    #         vh = vh.to(dtp)
    #         uvh = u @ vh
    #         prev_uvh = getattr(self, f"prev_uvh_{qkvin}")
    #         if prev_uvh is None:  # first iteration
    #             if rank == working_rank:
    #                 log.info(f"First stability update: {qkvin}")
    #             setattr(self, f"prev_uvh_{qkvin}", uvh)
    #             selfu, selfs, selfvh, sl = self._get_usvh_from_qkvin(qkvin)
    #             selfu.zero_()
    #             selfu.add_(u)
    #             selfs.zero_()
    #             # print(f"{sself.shape}, {s.shape}, {self.full_rank_sigma}")
    #             selfs.add_(torch.diag(s) if self.full_rank_sigma else s)
    #             selfvh.zero_()
    #             selfvh.add_(vh)
    #             sl = sl.to(device=s.device)
    #             setattr(self, f"{qkvin}_inner_dim", sl)
    #             # self._bcast_kusvh_abs(qkvin, working_rank, nonblocking)
    #             return (
    #                 0,
    #                 getattr(self, f"{qkvin}_inner_dim"),
    #                 1.0,
    #                 getattr(self, f"uvh_stable_{qkvin}"),
    #                 False,
    #             )
    #         if rank == working_rank:
    #             log.info(f"in normal stability update: {qkvin}")
    #     # use cosine similarity (dot product for orthonormal) to determine similarity
    #     # if rank == working_rank:
    #     #     log.info(f"before cossim: {qkvin} {prev_uvh.shape}")
    #     csim = self.cossim(prev_uvh, uvh)
    #     # if rank == working_rank:
    #     #     log.info(f"after cossim: {qkvin} {csim}")
    #     setattr(self, f"prev_uvh_{qkvin}", uvh)
    #     csmean, _ = csim.mean(), csim.std()
    #     change_sl = False
    #     u, s, vh, gensl = self._get_usvh_from_qkvin(qkvin)
    #     if csmean >= self.uvhthreshold:
    #         setattr(self, f"uvh_stable_{qkvin}", True)

    #         if set_usvh:
    #             uself.zero_()
    #             uself.add_(u)
    #             vhself.zero_()
    #             vhself.add_(vh)

    #             if self.full_rank_sigma:
    #                 sself.zero_()
    #                 sself[:gensl, :gensl].add_(torch.diag(s[:gensl]))
    #             else:
    #                 sself.zero_()
    #                 sself[:gensl].add_(s[:gensl])
    #         # eps = torch.finfo(uself.dtype).eps
    #         # uself[torch.abs(uself) < eps] *= 0
    #         # vhself[torch.abs(vhself) < eps] *= 0
    #         # sself[torch.abs(sself) < eps] *= 0

    #         change_sl = self._update_inner_dim(qkvin)
    #     perc_params, _, _ = self.get_perc_params()
    #     # w = torch.linalg.multi_dot([uself, sself, vhself])
    #     # w[torch.abs(w) < 1e-5] *= 0
    #     # if dist.get_rank() == 0:
    #     #     print(f"sparcity: {torch.count_nonzero(w) / w.numel()}")
    #     # if rank == working_rank:
    #     #     log.info(f"before contiguous and send: {qkvin}")
    #     # uself = getattr(self, f"{qkvin}_u")
    #     # sself = getattr(self, f"{qkvin}_s")
    #     # vhself = getattr(self, f"{qkvin}_vh")
    #     # uself.set_(uself.contiguous())
    #     # sself.set_(sself.contiguous())
    #     # vhself.set_(vhself.contiguous())
    #     # self._bcast_kusvh_abs(qkvin, working_rank, nonblocking)
    #     return (
    #         csmean,
    #         getattr(self, f"{qkvin}_inner_dim"),
    #         perc_params,
    #         getattr(self, f"uvh_stable_{qkvin}"),
    #         change_sl,
    #     )

    @torch.no_grad()
    def _test_stability_distributed_abs(self, qkvin, name, working_rank, nonblocking=True):
        """
        This is going to do the qkv updates for q, k, v, or in_proj based on the value of qkvin

        uvh_stable_[in_proj, q, k, v]
        [in_proj, q, k, v]_u
        [in_proj, q, k, v]_s
        [in_proj, q, k, v]_vh
        [in_proj, q, k, v]_trans
        prev_uvh_[in_proj, q, k, v]
        _get_[in_proj, q, k, v]
        [in_proj, q, k, v]_inner_dim
        base weights:
            [q, k, v]_proj_weight
            in_proj_weight
        """
        stable = getattr(self, f"uvh_stable_{qkvin}")
        prev_uvh = getattr(self, f"prev_uvh_{qkvin}")
        if prev_uvh is not None and not (self.full_rank_sigma and stable and self.update_from_simga) and stable:
            # early out here without any communication
            self.last_send_ranks[qkvin] = None
            u, s, vh, _ = self._get_usvh_from_qkvin(qkvin=qkvin)
            if working_rank == self.rank:
                log.info(f"{name[-30]} - {qkvin}: All shapes Frozen, [{u.shape[0]} {s.shape[0]} {vh.shape[1]}]")
            return

        self.last_send_ranks[qkvin] = working_rank
        if working_rank != self.rank:
            # move on for now, coming back later
            # make sure to save the rank which did this layer
            if prev_uvh is None:
                # if no prev_uvh -> first iteration: need to get usvh only, can start now
                dev = self._get_device()

                sl = getattr(self, f"{qkvin}_inner_dim")
                sl = sl.to(device=dev, non_blocking=True)
                setattr(self, f"{qkvin}_inner_dim", sl)
                self.inner_dim_buffers[qkvin] = self.inner_dim_buffers[qkvin].to(
                    device=dev,
                    non_blocking=True,
                )

                setattr(self, f"prev_uvh_{qkvin}", torch.tensor(1, device=dev))
                self.waits[qkvin]["inner_dim"] = None
                # if in the first iteration
                # return
            # receive the info from the working process
            self.waits[qkvin]["inner_dim"] = dist.broadcast(
                self.inner_dim_buffers[qkvin],
                src=working_rank,
                async_op=nonblocking,
            )
            # print('for waits')
            return

        if prev_uvh is not None and prev_uvh.ndim == 0:  # just in case...
            # if somehow we get into this block, we will just go through the first stability check
            # also need to reset the stability of the layer
            # but would need to reset the stability on all ranks...
            u, s, vh, _ = self._get_usvh_from_qkvin(qkvin)
            setattr(self, f"prev_uvh_{qkvin}", u @ vh)

        if prev_uvh is None:
            dev = self._get_device()
            sl = getattr(self, f"{qkvin}_inner_dim")
            sl = sl.to(device=dev, non_blocking=True)
            setattr(self, f"{qkvin}_inner_dim", sl)
            self.inner_dim_buffers[qkvin] = self.inner_dim_buffers[qkvin].to(device=dev, non_blocking=True)
            # case 1: do SVD for the first time and calculate the basis
            self._first_stability_abs(qkvin)
            status = {"csmean": 0.0, "change_k": 0.0}
            # change of plans: start USVh communcation in the wait function just like the other cases
            # self.bcast_usvh(src=working_rank, nonblocking=nonblocking)
            if qkvin in ["q", "k", "v"]:
                weight = getattr(self, f"{qkvin}_proj_weight")
            else:
                weight = self.in_proj_weight
            log.info(
                f"{name[-30:]} - {qkvin}: 1st stability, csmean: None, params: 100%, [{weight.shape[0]}, {weight.shape[1]}]",
            )
        elif self.full_rank_sigma and stable and self.update_from_simga:
            # case 3: update the stable U and Vh from the full rank sigma matrix
            status = self._full_rank_sigma_update_usv_abs(qkvin, working_rank)
            self.inner_dim_buffers[qkvin][0]
            # shapes are updated within above (as is slice update)
            perc, _, _ = self.get_perc_params()
            u, s, vh, sl = self._get_usvh_from_qkvin(qkvin)
            # normalize self.s to max == 1
            # maxs = s.max()
            # s /= maxs
            # vh *= maxs
            log.info(
                f"{name[-30:]}: Full rank update, csmean: {status['csmean']:.3f}, params: {perc * 100:.2f}, "
                f"\t[{u.shape[0]} {s.shape[0]} {vh.shape[1]}]",
            )
            # print(f"{torch.diag(s)[:5]}")

        elif not stable:
            # case 2: normal stability update
            status = self._weight_stability_abs(qkvin, working_rank)
            perc, _, _ = self.get_perc_params()
            u, s, vh, sl = self._get_usvh_from_qkvin(qkvin)
            # maxs = s.max()
            # s /= maxs
            # vh *= maxs
            log.info(
                f"{name[-30:]}: Normal stability, csmean: {status['csmean']:.3f}, params: {perc * 100:.2f}, "
                f"\t[{u.shape[0]} {s.shape[0]} {vh.shape[1]}]",
            )
        else:
            # case here is when uvh is frozen but we are not updating the bases
            # dont need to do any communication, dont need to computation
            # can just get the percentage of active parameters/whatever else needs to be returned for logs
            raise RuntimeError("something went wrong, stuck in 'else' in stability...")

        if not dist.is_initialized():
            return
        # send K if it has changed
        # [in_proj, q, k, v]_inner_dim
        u, s, vh, sl = self._get_usvh_from_qkvin(qkvin)
        self.inner_dim_buffers[qkvin][0] = sl.to(torch.float)
        self.inner_dim_buffers[qkvin][1] = status["csmean"]
        self.inner_dim_buffers[qkvin][2] = status["change_k"]

        if status["csmean"] >= self.uvhthreshold:
            setattr(self, f"uvh_stable_{qkvin}", True)
            try:
                getattr(self, f"{qkvin}_proj_weight").requires_grad = False
            except AttributeError:
                # for in_proj, we dont need the proj part
                getattr(self, f"{qkvin}_weight").requires_grad = False
            u.requires_grad = False
            s.requires_grad = True
            vh.requires_grad = False

        self.waits[qkvin]["inner_dim"] = dist.broadcast(
            self.inner_dim_buffers[qkvin],
            src=working_rank,
            async_op=nonblocking,
        )

    @torch.no_grad()
    def _first_stability_abs(self, qkvin):
        w = getattr(self, f"{qkvin}_proj_weight") if qkvin in "qkv" else self.in_proj_weight
        w = w.T if getattr(self, f"{qkvin}_trans") else w
        # w = getattr(self, f"_get_{qkvin}")()
        dtp = w.dtype
        # print(f"first stab {qkvin} before svd")
        w = w.to(torch.float32)
        u, s, vh = torch.linalg.svd(w, full_matrices=False)  # , driver="gesvd")
        # print(f"first stab {qkvin} {u.device} {u.dtype}")
        u = u.to(dtp)
        s = s.to(dtp)
        vh = vh.to(dtp)
        uvh = u @ vh
        # print(f"first stab end svd {qkvin}")

        setattr(self, f"prev_uvh_{qkvin}", uvh)
        u, s, vh, sl = self._get_usvh_from_qkvin(qkvin)
        # print(f"first stab after get usvh {qkvin}")
        u.zero_()
        u.add_(u)
        s.zero_()
        # print(f"{sself.shape}, {s.shape}, {self.full_rank_sigma}")
        s.add_(torch.diag(s) if self.full_rank_sigma else s)
        vh.zero_()
        vh.add_(vh)
        sl = sl.to(device=s.device)
        setattr(self, f"{qkvin}_inner_dim", sl)

    @torch.no_grad()
    def _weight_stability_abs(self, qkvin, working_rank):
        log.debug("normal weight stability test")
        w = getattr(self, f"{qkvin}_proj_weight") if qkvin in "qkv" else self.in_proj_weight
        w = w.T if getattr(self, f"{qkvin}_trans") else w
        # w = self.get_weight()
        # w = w.T if self.trans else w
        prev_uvh = getattr(self, f"prev_uvh_{qkvin}")
        dtp = w.dtype
        w = w.to(torch.float32)
        u, s, vh = torch.linalg.svd(w, full_matrices=False)  # , driver="gesvd")
        u = u.to(dtp)
        s = s.to(dtp)
        vh = vh.to(dtp)
        uvh = u @ vh
        csim = self.cossim(prev_uvh, uvh)
        csmean, _ = csim.mean(), csim.std()
        prev_uvh.set_(uvh)
        change_k = False
        if csmean >= self.uvhthreshold:
            self.uvh_stable = True

            selfu, selfs, selfvh, _ = self._get_usvh_from_qkvin(qkvin)
            selfu.zero_()
            selfu.add_(u)
            selfs.zero_()
            selfs.add_(torch.diag(s) if self.full_rank_sigma else s)
            selfvh.zero_()
            selfvh.add_(vh)

            change_k = self._update_inner_dim_and_shapes_abs(qkvin)
        return {"csmean": csmean, "change_k": change_k}

    @torch.no_grad()
    def _full_rank_sigma_update_usv_abs(self, qkvin, working_rank=0):
        if not self.full_rank_sigma:
            raise ValueError("this function is only for full-rank sigma with usvh is stable")

        if self.rank == working_rank:
            log.debug("in full rank sigma update of usvh")
        # NOTE: no slicing because need the shapes to line up. self.s[self.k:, self.k:] should be 0?
        u, s, vh, _ = self._get_usvh_from_qkvin(qkvin)
        dtp = s.dtype
        s = s.to(torch.float32)
        usig, sig, vhsig = torch.linalg.svd(s)  # square mat, full mat arg doesnt matter
        # usig[torch.abs(usig) < 1e-5] *= 0
        # vhsig[torch.abs(vhsig) < 1e-5] *= 0
        # sig[torch.abs(sig) < 1e-6] *= 0

        usig = usig.to(dtp)
        sig = sig.to(dtp)
        vhsig = vhsig.to(dtp)

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

        holdu = u @ usig
        u.zero_()
        u.add_(holdu)
        holdvh = vhsig @ vh
        vh.zero_()
        vh.add_(holdvh)
        s.zero_()
        s.add_(torch.diag(sig))
        # if self.rank == 0:
        # print(f"in full rank update: {torch.count_nonzero(torch.abs(s) < 1e-7)}")

        # normal update from cosine similarity stuff
        # uvh = u @ vh
        # prev_uvh = getattr(self, f"prev_uvh_{qkvin}")
        # csim = self.cossim(prev_uvh, uvh)
        # csmean, _ = csim.mean(), csim.std()
        csmean = 1.0
        # rand_noise = torch.rand_like(s) * sig.min() * 0.1
        # rand_noise.fill_diagonal_(0)
        # s.add(rand_noise)

        # setattr(self, f"prev_uvh_{qkvin}", uvh)
        # csmean = 1.0
        change_k = False
        if csmean >= self.uvhthreshold:
            setattr(self, f"uvh_stable_{qkvin}", True)

            change_k = self._update_inner_dim_and_shapes_abs(qkvin)
        return {"csmean": csmean, "change_k": change_k}

    @torch.no_grad()
    def _wait_inner_dim_reshape_bcast_usvh_abs(self, qkvin, nonblocking=True):
        # if wait_k is None -> K is the same -> optimizer is fine (shapes are the same)
        reset_optimizer = False
        if self.last_send_ranks[qkvin] is None:
            return reset_optimizer, True
        if self.waits[qkvin]["inner_dim"] is not None:
            self.waits[qkvin]["inner_dim"].wait()
            self.waits[qkvin]["inner_dim"] = None

            if self.rank != self.last_send_ranks[qkvin]:
                # [in_proj, q, k, v]_inner_dim
                setattr(self, f"{qkvin}_inner_dim", self.inner_dim_buffers[qkvin][0].to(torch.int))
                self._update_usvh_shapes(qkvin)

                if self.inner_dim_buffers[qkvin][1] >= self.uvhthreshold:
                    # stable = getattr(self, f"uvh_stable_{qkvin}")
                    setattr(self, f"uvh_stable_{qkvin}", True)
                    try:
                        getattr(self, f"{qkvin}_proj_weight").requires_grad = False
                        getattr(self, f"{qkvin}_proj_weight").grad = None
                    except AttributeError:
                        getattr(self, f"{qkvin}_weight").requires_grad = False
                        getattr(self, f"{qkvin}_weight").grad = None
                    u, s, vh, _ = self._get_usvh_from_qkvin(qkvin)
                    u.requires_grad = False
                    s.requires_grad = True
                    vh.requires_grad = False
        reset_optimizer = bool(self.inner_dim_buffers[qkvin][2].item())
        stable = self.inner_dim_buffers[qkvin][1] >= self.uvhthreshold
        self.inner_dim_buffers[qkvin] *= 0
        self._bcast_usvh_abs(qkvin=qkvin, src=self.last_send_ranks[qkvin], nonblocking=nonblocking)
        return reset_optimizer, stable

    def _bcast_usvh_abs(self, qkvin, src, nonblocking):
        if not dist.is_initialized():
            return
        u, s, vh, _ = self._get_usvh_from_qkvin(qkvin)
        self.waits[qkvin]["u"] = dist.broadcast(u.data, src=src, async_op=nonblocking)
        self.waits[qkvin]["s"] = dist.broadcast(s.data, src=src, async_op=nonblocking)
        self.waits[qkvin]["vh"] = dist.broadcast(vh.data, src=src, async_op=nonblocking)

    def _wait_on_usvh_abs(self, qkvin):
        for w in self.waits[qkvin]:
            if self.waits[qkvin][w] is not None:
                self.waits[qkvin][w].wait()
                self.waits[qkvin][w] = None

    def wait_on_usvh(self):
        if self._qkv_same_embed_dim:
            self._wait_on_usvh_abs("in_proj")
        for qkv in "qkv":
            self._wait_on_usvh_abs(qkv)

    @torch.no_grad()
    def test_stability_distributed(self, name, working_rank=0, nonblocking=True):
        if self.in_proj_weight is not None:
            return self._test_stability_distributed_abs(
                qkvin="in_proj",
                name=name,
                working_rank=working_rank,
                nonblocking=nonblocking,
            )
        else:
            sz = dist.get_world_size() if dist.is_initialized() else 1
            one, two, three = working_rank, (working_rank + 1) % sz, (working_rank + 2) % sz
            self._test_stability_distributed_abs(
                qkvin="q",
                name=name,
                working_rank=one,
                nonblocking=nonblocking,
            )
            self._test_stability_distributed_abs(
                qkvin="k",
                name=name,
                working_rank=two,
                nonblocking=nonblocking,
            )
            self._test_stability_distributed_abs(
                qkvin="v",
                name=name,
                working_rank=three,
                nonblocking=nonblocking,
            )

    @torch.no_grad()
    def wait_inner_dim_reshape_bcast_usvh(self, nonblocking=True):
        if self.in_proj_weight is not None:
            return self._wait_inner_dim_reshape_bcast_usvh_abs(qkvin="in_proj", nonblocking=nonblocking)
        else:
            resq, stabq = self._wait_inner_dim_reshape_bcast_usvh_abs(qkvin="q", nonblocking=nonblocking)
            resk, stabk = self._wait_inner_dim_reshape_bcast_usvh_abs(qkvin="k", nonblocking=nonblocking)
            resv, stabv = self._wait_inner_dim_reshape_bcast_usvh_abs(qkvin="v", nonblocking=nonblocking)
            return resq or resk or resv, stabk and stabq and stabv

    @torch.no_grad()
    # @torch.compile()
    def _update_inner_dim_and_shapes_abs(self, qkvin):
        u, s, vh, sl = self._get_usvh_from_qkvin(qkvin)
        # adjust K to slice of less important singular values
        # only want to compare the diagonal entries of sigma
        sdiag = torch.diag(s) if self.full_rank_sigma else s
        prevsl = sl.clone()
        # if getattr(self, f"uvh_stable_{qkvin}"):
        min_dim = int(vh.shape[-1] * 0.01)  # always TS
        cutoff = sdiag[min_dim] * self.sigma_cutoff_fraction
        nz = torch.nonzero(sdiag < cutoff)
        if len(nz) == 0:
            # In this case ALL of the basis vectors are useful
            newsl = s.shape[0]
        else:
            newsl = nz[0].item()

        # if newsl < 0.5 * prevsl:
        #     # TODO: log message?
        #     newsl = int(prevsl * 0.5)
        #     log.info(f"Values of S after dropping slice value by only 50% of suggestion: {sdiag[:5]}")
        sl.mul_(0)
        sl.add_(newsl)
        if prevsl != newsl:
            self._update_usvh_shapes(qkvin)
        return prevsl != newsl

    @torch.no_grad()
    def _update_usvh_shapes(self, qkvin):
        # either update the shapes of USVh or set the irrelevant values to 0
        u, s, vh, sl = self._get_usvh_from_qkvin(qkvin)
        if self.reinit_shapes:
            u.set_(u[:, :sl].contiguous())
            vh.set_(vh[:sl].contiguous())
            if self.full_rank_sigma:
                s.set_(s[:sl, :sl].contiguous())
            else:
                s.set_(s[:sl])
        else:
            u[:, sl:] *= 0
            vh[sl:] *= 0
            if self.full_rank_sigma:
                s[sl:, sl:].mul_(0)
            else:
                s[sl:].mul_(0)

    @torch.no_grad()
    def _get_usvh_from_qkvin(self, qkvin):
        u = getattr(self, f"{qkvin}_u")
        s = getattr(self, f"{qkvin}_s")
        vh = getattr(self, f"{qkvin}_vh")
        sl = getattr(self, f"{qkvin}_inner_dim")
        return u, s, vh, sl

    # @torch.compile()
    def get_perc_params_select(self, qkvin):
        if qkvin in ["q", "k", "v"]:
            weight = getattr(self, f"{qkvin}_proj_weight")
        else:
            weight = self.in_proj_weight
        normal_params = weight.numel()
        if getattr(self, f"uvh_stable_{qkvin}"):
            # active_params = (self.u.shape[0] * self.k) + (self.k ** 2) + (self.k + self.vh.shape[1])
            if self.full_rank_sigma:
                trainable_params = getattr(self, f"{qkvin}_inner_dim") ** 2
            else:
                trainable_params = getattr(self, f"{qkvin}_inner_dim")
        else:
            trainable_params = normal_params
        return trainable_params, normal_params

    def get_perc_params(self):
        if not self._qkv_same_embed_dim:  # get perc perams for all
            qtrain, qnormal = self.get_perc_params_select("q")
            ktrain, knormal = self.get_perc_params_select("k")
            vtrain, vnormal = self.get_perc_params_select("v")
            active = qtrain + ktrain + vtrain
            normal = qnormal + knormal + vnormal
        else:
            active, normal = self.get_perc_params_select("in_proj")

        in_bias = 0 if self.in_proj_bias is None else self.in_proj_bias.numel()
        k_bias = 0 if self.bias_k is None else self.bias_k.numel()
        v_bias = 0 if self.bias_v is None else self.bias_v.numel()
        normal += in_bias + k_bias + v_bias
        active += in_bias + k_bias + v_bias
        perc = active / normal
        return perc, active, normal

    def get_interior_dim(self):
        if not self._qkv_same_embed_dim:
            return {
                "q": self.q_inner_dim,
                "k": self.k_inner_dim,
                "v": self.v_inner_dim,
            }
        return {
            "in_proj": self.in_proj_inner_dim,
        }
