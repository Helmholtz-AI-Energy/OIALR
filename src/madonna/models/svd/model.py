import logging
import time
from collections import OrderedDict, defaultdict
from time import perf_counter

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import parametrizations, parametrize

from ..utils import (
    change_adam_shapes,
    change_sgd_shapes,
    create_svd_param_groups,
    replace_opt_state_with_svd_adam,
)
from .attention import SVDMultiheadAttention
from .attentionusvh import SVDMultiheadAttentionUSVh
from .attentionvh import SVDMultiheadAttentionVh
from .conv import SVDConv2d
from .convusvh import SVDConv1dUSVh, SVDConv2dUSVh
from .convvh import SVDConv2dVh
from .linear import SVDLinear
from .linearusvh import SVDLinearUSVh
from .linearvh import SVDLinearVh

# from ..utils import utils
# from ..optimizers.utils import change_adam_shapes

# from .. import optimizers

log = logging.getLogger(__name__)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class SVDFixingModel(nn.Module):
    def __init__(
        self,
        full_rank_model: nn.Module,
        stability_frequency: int = 10,
        delay: int = 100,
        uvhthreshold: float = 0.999,  # TODO: remove me!
        sigma_cutoff_fraction: float = 0.1,
        keep_first_layer: bool = False,
        keep_last_layer: bool = True,
        reinit_shapes: bool = True,
        stable_update_delay: int = 0,
        create_svd_param_group: str = None,  # options: 'one', 'many'
        step_on_forward: bool = False,
        full_rank_warmup: bool = True,
    ):
        super().__init__()

        num = 0
        full_params = 0
        for n, p in full_rank_model.named_parameters():
            if n[-2:] not in [".s", "_s", "_u", ".u", "vh"]:
                full_params += p.numel()
                if p.requires_grad:
                    # print(f"{n}: {p.numel()}")
                    num += p.numel()

        self.base_trainable_parameters = num
        self.base_all_parameters = full_params

        self.uvhthreshold = uvhthreshold
        self.sigma_cutoff_fraction = sigma_cutoff_fraction
        self.full_rank_sigma = True  # full_rank_sigma TODO: remove me!
        self.update_from_simga = True  # update_from_simga TODO: remove me!
        self.first_layer = keep_first_layer
        self.keep_last_layer = keep_last_layer
        self.last_layer = None
        self.reinit_shapes = reinit_shapes
        self.full_rank_warmup = full_rank_warmup
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()
        self.local_full_rank_model = full_rank_model
        self.low_rank_replacement_list = {}

        if full_rank_warmup and delay > 0:
            if self.rank == 0:
                log.info("Starting with training in full rank")
            self.model = self.local_full_rank_model
            if dist.is_initialized():
                if self.rank == 0:
                    log.info("Initializing DDP")
                self.model = DDP(full_rank_model, find_unused_parameters=False)
        else:
            self.setup_low_rank_training(skip_optimizer_init=True)

        # self.compiled_model = torch.compile(self.model)
        # raise ValueError("")
        self.step_on_forward = step_on_forward
        self.stability_frequency = stability_frequency
        self.call_count = 0
        self.num_stability_layvers_to_check = 0
        self.call_count_stability = 0
        self.layer_count_selector = torch.Generator()
        self.layer_count_selector.manual_seed(123456)
        self.skip_stability = False
        self.delay = delay
        self.stable_list = []
        self.stable_update_delay = stable_update_delay

        self.fib1, self.fib2 = 0, 1
        self.next_stability_iteration = self.delay + self.fib1

        self.optimizer = None  # optimizers.MixedSVDOpt
        self.local_generator = torch.Generator()
        self.local_generator.manual_seed(self.rank)

        self.all_stable = False
        self.state_dict = self.model.state_dict
        self.parameters = self.model.parameters
        self.named_parameters = self.model.named_parameters
        self.named_modules = self.model.named_modules
        self.named_buffers = self.model.named_buffers
        self.named_children = self.model.named_children
        self.children = self.model.children
        self.cuda = self.model.cuda
        # TODO: add other methods here??
        self.__repr__ = self.model.__repr__
        self.create_svd_param_group = create_svd_param_group
        self.train = self.model.train

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer  # optimizers.MixedSVDOpt
        if isinstance(optimizer, (optim.Adam, optim.AdamW)):
            self.reshape_opt_state_fn = change_adam_shapes
        elif isinstance(optimizer, optim.SGD):
            self.reshape_opt_state_fn = change_sgd_shapes

        # if self.create_svd_param_group is not None:
        #     create_svd_param_groups(optimizer, model=self.model, individual_groups=False)
        #     # after this groups are: [non2d, full rank weights, sigma weights]
        #     if "weight_decay" in optimizer.param_groups[2]:
        #         optimizer.param_groups[2]["weight_decay"] *= 0.

    @torch.no_grad()
    def setup_low_rank_training(self, skip_optimizer_init=False):
        if self.rank == 0:
            log.info("Starting with training in low rank")
        self.local_low_rank_model = self._replace_layers(self.local_full_rank_model)

        if self.keep_last_layer:
            self._reset_last_layer(self.local_low_rank_model)

        if dist.is_initialized():
            if self.rank == 0:
                log.info("Initializing DDP")
            self.model = DDP(self.local_low_rank_model, find_unused_parameters=False)
        else:
            self.model = self.local_low_rank_model
        self.svd_modules = {}
        self.layer_names = []
        calls = 0
        sz = 1 if not dist.is_initialized() else dist.get_world_size()
        for name, mod in self.model.named_modules():
            if hasattr(mod, "test_stability_distributed"):
                working_rank = calls % sz
                # self.svd_modules.append((name, mod, working_rank))
                self.svd_modules[name] = {"mod": mod, "working_rank": working_rank, "stable": False, "stable_delay": 0}
                self.layer_names.append(name)
                try:  # only a module in the attention layers and its just faster to try to get something
                    if mod._qkv_same_embed_dim:
                        calls += 1
                    else:
                        calls += 3
                except AttributeError:
                    calls += 1
        if not skip_optimizer_init:  # only need to do this if we start in full rank...i think
            # if self.create_svd_param_group is not None:
            #     create_svd_param_groups(self.optimizer, model=self.model, individual_groups=False)
            replace_opt_state_with_svd_adam(self.optimizer, self.low_rank_replacement_list)

    def _replace_layers(self, module, name=None, process_group=None):
        module_output = module
        # print(f'wrapping {name} {module}')
        if isinstance(module, nn.Linear) and min(module.weight.shape) > max(module.weight.shape) / 10:
            if not self.first_layer:
                module_output = SVDLinearUSVh(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    uvhthreshold=self.uvhthreshold,
                    sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                    full_rank_sigma=self.full_rank_sigma,
                    start_weight=module.weight,
                    start_bias=module.bias,
                    update_from_simga=self.update_from_simga,
                    reinit_shapes=self.reinit_shapes,
                ).to(device=module.weight.device, dtype=module.weight.dtype)
                # module_output = parametrizations.orthogonal(module_output, name="u")
                # module_output = parametrizations.orthogonal(module_output, name="vh")  # TODO: trans?
                # module_output = torch.compile(module_output)
                self.last_layer = [module, name, module.weight.dtype, module.weight.device]
                self.low_rank_replacement_list[module.weight] = [module_output.s, "lin"]
            else:
                self.first_layer = False
        elif isinstance(module, nn.MultiheadAttention):
            if not self.first_layer:
                module_output = SVDMultiheadAttentionUSVh(
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
                self.last_layer = [module, name, None, None]
                if module.in_proj_weight is not None:
                    self.low_rank_replacement_list[module.in_proj_weight] = [module_output.in_proj_s, "attn"]
                else:
                    self.low_rank_replacement_list[module.q_proj_weight] = [module_output.q_s, "attn"]
                    self.low_rank_replacement_list[module.k_proj_weight] = [module_output.k_s, "attn"]
                    self.low_rank_replacement_list[module.v_proj_weight] = [module_output.v_s, "attn"]
            else:
                self.first_layer = False
        elif isinstance(module, nn.Conv2d):
            wv = module.weight.view(module.weight.shape[0], -1)
            if wv.shape[0] < wv.shape[1]:
                wv.T
            if wv.shape[1] < wv.shape[0] / 10:
                pass  # skip this layer if there are not enough params
            elif not self.first_layer:
                module_output = SVDConv2dUSVh(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    padding_mode=module.padding_mode,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                    uvhthreshold=self.uvhthreshold,
                    sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                    full_rank_sigma=self.full_rank_sigma,
                    start_bias=module.bias,
                    start_weight=module.weight,
                    update_from_simga=self.update_from_simga,
                    reinit_shapes=self.reinit_shapes,
                    norm=module.norm if hasattr(module, "norm") else None,
                    activation=module.activation if hasattr(module, "activation") else None,
                )
                self.last_layer = [module, name, module.weight.dtype, module.weight.device]
                self.low_rank_replacement_list[module.weight] = [module_output.s, "conv"]
            else:
                self.first_layer = False
        elif isinstance(module, nn.Conv1d):
            wv = module.weight.view(module.weight.shape[0], -1)
            if wv.shape[0] < wv.shape[1]:
                wv.T
            if wv.shape[1] < wv.shape[0] / 10:
                pass  # skip this layer if there are not enough params
            elif not self.first_layer:
                module_output = SVDConv1dUSVh(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    padding_mode=module.padding_mode,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                    uvhthreshold=self.uvhthreshold,
                    sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                    full_rank_sigma=self.full_rank_sigma,
                    start_bias=module.bias,
                    start_weight=module.weight,
                    update_from_simga=self.update_from_simga,
                    reinit_shapes=self.reinit_shapes,
                    norm=module.norm if hasattr(module, "norm") else None,
                    activation=module.activation if hasattr(module, "activation") else None,
                )
                self.last_layer = [module, name, module.weight.dtype, module.weight.device]
                self.low_rank_replacement_list[module.weight] = [module_output.s, "conv"]
            else:
                self.first_layer = False
        for n, child in module.named_children():
            module_output.add_module(
                f"{n}",
                self._replace_layers(
                    child,
                    name=f"{name}.{n}" if name is not None else f"{n}",
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
            if self.last_layer[2] is None:
                try:
                    device = module.in_proj_s.device
                    dtype = module.in_proj_s.dtype
                except AttributeError:
                    device = module.q_s.device
                    dtype = module.q_s.dtype
            else:
                dtype = self.last_layer[2]
                device = self.last_layer[3]
            module_output = self.last_layer[0].to(device=device, dtype=dtype)
            del self.low_rank_replacement_list[self.last_layer[0].weight]
        for n, child in module.named_children():
            module_output.add_module(n, self._reset_last_layer(child, f"{name}.{n}" if name is not None else f"{n}"))
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
        for c, (name, mod) in enumerate(self.model.named_modules()):
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

        for c, (name, mod) in enumerate(self.model.named_modules()):
            # try:
            if hasattr(mod, "wait_on_kusvh"):
                mod.wait_on_kusvh()

        if dist.is_initialized():  # and reset_optimizer:
            # this indicates that the shapes of the parameters changed
            # need to re-init DDP to have the correct buckets
            # TODO: this might be a preformance hit
            ddp_time = perf_counter()
            # del self.model
            self.model = DDP(self.local_low_rank_model, find_unused_parameters=False, static_graph=False)
            if dist.get_rank() == 0:
                log.info(f"Reinit DDP. Time takesn: {perf_counter() - ddp_time}")
        if all_stable:
            self.skip_stability = True
        return reset_optimizer

    @torch.no_grad()
    def test_basis_stability_all_layers(self):
        # if self.skip_stability:
        #     return
        rank = dist.get_rank() if dist.is_initialized() else 0
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
        working_layers = self.layer_names[-cutoff:]
        # working_layers = self.layer_names[:]

        # inds = inds[-cutoff:]
        if self.rank == 0:
            log.info(f"num layers for svd: {len(working_layers)} cutoff: {cutoff} {len(self.layer_names)}")
        # self.svd_modules ->
        #   mod: module,
        #   working_rank: working rank,
        #   stable: stable,
        #   stable_delay: number of checks until checking stability again
        for layer in working_layers:
            if self.svd_modules[layer]["stable_delay"] == 0:
                mod, working_rank = self.svd_modules[layer]["mod"], self.svd_modules[layer]["working_rank"]
                mod.test_stability_distributed(name=layer, working_rank=working_rank, nonblocking=True)
        # dist.barrier()
        for layer in working_layers:
            if self.svd_modules[layer]["stable_delay"] == 0:
                mod, working_rank = self.svd_modules[layer]["mod"], self.svd_modules[layer]["working_rank"]
                reset_opt, stable = mod.wait_inner_dim_reshape_bcast_usvh(nonblocking=True)
                if stable:
                    self.svd_modules[layer]["stable"] = stable
                    self.svd_modules[layer]["stable_delay"] = self.stable_update_delay
                if reset_opt:
                    reset_optimizer = True
                if not stable:
                    all_stable = False
        # dist.barrier()
        for layer in working_layers:
            if self.svd_modules[layer]["stable_delay"] == 0:
                mod, working_rank = self.svd_modules[layer]["mod"], self.svd_modules[layer]["working_rank"]
                mod.wait_on_usvh()
            else:
                self.svd_modules[layer]["stable_delay"] -= 1

        return reset_optimizer, all_stable

    @torch.no_grad()
    def get_perc_params_all_layers(self):
        # self.sync_models()
        # if self.skip_stability:
        #     return
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        # percs, actives, normals = [], [], []
        trainable = 0
        untrainable = 0
        # print("in perc params")
        for n, p in self.model.named_parameters():
            # print(f"{n} {p.requires_grad}")
            if p.requires_grad:
                # if n[-2:] not in [".s", "_s", "_u", ".u", "vh"]:
                trainable += p.numel()
            else:
                untrainable += p.numel()

        full_normal = self.base_trainable_parameters
        full_active = trainable
        # full_deactivated = untrainable
        full_model = trainable + untrainable
        compression_perc = 100 * (full_model / self.base_all_parameters)
        if rank == 0:
            log.info(
                f"Active Params: {100 * (full_active / full_normal):.4f}% active: {full_active} "
                f"Full Rank: {full_normal} Low rank total: {full_model} compression: {compression_perc}",
            )
        return 100 * (full_active / full_normal), full_active, full_normal, compression_perc

    @staticmethod
    def reset_all_states(optimizer: optim.Optimizer):
        # reset op1 first
        # for group in self.opt1.param_groups:
        optimizer.state = defaultdict(dict)

    @torch.no_grad()  # The function is the main method of doing stability tracking
    def model_stability_tracking(self, force=False):
        """
        NOTE: should be called either every epoch, or after X steps!!!
            - dependent on config params

        This is the main function for tracking SVD of layers
        """
        # OOO:
        # 2. check if stability should be checked - early out
        # 3. check stability of layers (see self.test_basis_stability_all_layers)
        # 4. if the optimizer needs to be reset, do it (reshapes the buffers within it)
        self.call_count += 1
        # print(self.call_count, self.delay)
        if self.call_count == self.delay and self.full_rank_warmup:  # - self.stability_frequency:
            self.setup_low_rank_training()
            # return

        if not force and self.call_count != self.next_stability_iteration:
            return
        self.call_count_stability += 1

        stabtime = time.perf_counter()
        reset_optimizer, all_layers_stable = self.test_basis_stability_all_layers()

        if all_layers_stable and not self.all_stable:
            # NOTE: no need to remove the full_rank_weights, they will cause a few extra clock ticks but nothing more
            # self.optimizer.remove_full_rank_weights()
            # self.reset_all_states(self.optimizer)
            self.all_stable = all_layers_stable

        if dist.is_initialized():
            # this indicates that the shapes of the parameters changed
            # need to re-init DDP to have the correct buckets
            ddp_time = perf_counter()
            self.model = DDP(self.model.module, find_unused_parameters=False, static_graph=False)
            if dist.get_rank() == 0:
                log.info(f"Reinit DDP. Time taken: {perf_counter() - ddp_time}")
        if reset_optimizer:
            self.reshape_opt_state_fn(self.optimizer)
            # self.reset_all_states(self.optimizer)
            # self.insert_noise(noise_level=1e-2)
            # self.optimizer.reset_shapes_of_sigma(self)
        self.optimizer.zero_grad(set_to_none=True)

        # self.model = DDP(self.local_model, find_unused_parameters=False, static_graph=False)
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
            self.get_perc_params_all_layers()
        return True

    def forward(self, *args, **kwargs):
        if self.step_on_forward and self.model.training:
            # print("in stability tracking block")
            self.model_stability_tracking(force=False)
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def track_interior_slices_mlflow(self, config, epoch):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if not config.enable_tracking or rank != 0:
            return
        # for c, (name, mod) in enumerate(self.model.named_modules()):
        #     # try:
        #     if hasattr(mod, "test_stability"):
        #         slices = mod.get_interior_inner_dim()
        #         for sl in slices:
        #             mlflow.log_metric(name + f".{sl}", slices[sl], step=epoch)
        #             # print(f"logging interior slice for {name}.{sl}")

    @torch.no_grad()
    def load_state_dict(self, best_model_path):
        # print(state_dict.keys())
        # return ValueError
        import os

        lcl_rank = int(os.environ["PMIX_RANK"])
        state_dict = torch.load(best_model_path, map_location=f"cuda:{lcl_rank}")
        for n, p in self.local_low_rank_model.named_parameters():
            # if self.local_low_rank_model[k]
            loaded_param = state_dict[n]
            # loaded_param = loaded_param.to(dtype=p.dtype, device=p.device)
            # print(k, '\t', n)

            if loaded_param.shape != p.shape:
                # print(f"changing shape of {n}")
                p.set_(torch.zeros(loaded_param.shape, dtype=p.dtype, device=p.device))

        self.local_low_rank_model.load_state_dict(state_dict)
        self.get_perc_params_all_layers()
