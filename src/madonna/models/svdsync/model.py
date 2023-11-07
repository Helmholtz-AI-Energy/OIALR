import logging
import time
from collections import OrderedDict, defaultdict
from time import perf_counter

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils import (
    change_adam_shapes,
    change_sgd_shapes,
    create_svd_param_groups,
    replace_opt_state_with_svd_adam,
)
from .attention import SVDSyncMultiheadAttention
from .conv import SVDSyncConv1d, SVDSyncConv2d
from .linear import SVDSyncLinear

log = logging.getLogger(__name__)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class SVDSyncModel(nn.Module):
    def __init__(
        self,
        full_rank_model: nn.Module,
        full_rank_warmup: bool = False,  # TODO: remove me or deal with me somehow... not planed to use atm
        fixed_inner_dim: bool = True,
        inner_dim_init_ratio: float = 1.0,
        random_sigma: bool = False,
        sigma_cutoff_fraction: float = 0.1,
        fixed_full_rank_layers: list = None,
        # --------- blur params ----------------------
        mixing_method: str = "exp",
        mixing_options: dict = None,
        # --------- sync params ----------------------
        delay: int = 1000,
        distributed_steps: int = 1000,
        p_steps: int = 10,
        trade_method: str = "fib",
        vecs_to_trade: int = 100,
        ordering: str = "cat",
        sync_min_size_fraction: float = 0.1,
    ):
        """
        FIXME: do this stuffs

        This class is for the specific methods to sync SVD models
        """
        super().__init__()

        # ---------- model stuff --------------------
        num = 0
        full_params = 0
        for n, p in full_rank_model.named_parameters():
            if n[-2:] not in [".s", "_s", "_u", ".u", "vh"]:
                full_params += p.numel()
                if p.requires_grad:
                    # print(f"{n}: {p.numel()}")
                    num += p.numel()

        self.fixed_full_rank_layers = fixed_full_rank_layers

        self.base_trainable_parameters = num
        self.base_all_parameters = full_params
        self.fixed_inner_dim = fixed_inner_dim
        self.inner_dim_init_ratio = inner_dim_init_ratio

        self.update_from_sigma = True  # update_from_sigma TODO: remove me?
        self.last_layer = None
        self.full_rank_warmup = full_rank_warmup
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()
        self.local_full_rank_model = full_rank_model
        self.low_rank_replacement_list = {}
        self.local_low_rank_model = None
        # self.non_svd_params = []
        self.random_sigma = random_sigma

        if full_rank_warmup and delay > 0:
            if self.rank == 0:
                log.info("Starting with training in full rank")
            self.model = self.local_full_rank_model
        else:
            self.setup_low_rank_training(skip_optimizer_init=True)

        self.optimizer = None  # optimizers.MixedSVDOpt
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
        self.train = self.model.train
        self.sigma_cutoff_fraction = sigma_cutoff_fraction
        # ---------------- sync params -----------------------------------------
        self.train_count = 0
        self.num_stability_layvers_to_check = 0
        self.delay = delay
        self.distributed_steps = distributed_steps
        self.p_steps = p_steps

        # self.fib1, self.fib2 = 0, 1
        # self.next_sync_iteration = self.delay + self.fib1
        self.trade_method = trade_method  # method for trading the singular values and vectors
        self.vecs_to_trade = vecs_to_trade  # number of vectors to send each time (upper limit)
        self.ordering = ordering  # how to order the vals/vecs
        self.sync_min_size_fraction = sync_min_size_fraction
        self.train_p = False
        # ---------------- mixing params  ------------------------------------
        self.mixing_method = mixing_method
        if mixing_options is None:
            mixing_options = {}
        self.mixing_options = mixing_options
        # ------------------- other params ------------------------------------
        self.local_generator = torch.Generator()
        self.local_generator.manual_seed(self.rank)

    def set_optimizers(self, optimizer_normal, optimizer_p):
        # Set normal training optimizer
        self.optimizer_normal = optimizer_normal
        if isinstance(optimizer_normal, (optim.Adam, optim.AdamW)):
            self.reshape_opt_state_fn = change_adam_shapes
        elif isinstance(optimizer_normal, optim.SGD):
            self.reshape_opt_state_fn = change_sgd_shapes
        # set optimizer for p steps
        self.optimizer_p = optimizer_p
        if isinstance(optimizer_p, (optim.Adam, optim.AdamW)):
            self.reshape_opt_state_fn_p = change_adam_shapes
        elif isinstance(optimizer_p, optim.SGD):
            self.reshape_opt_state_fn_p = change_sgd_shapes

    @torch.no_grad()
    def setup_low_rank_training(self, skip_optimizer_init=False):
        """
        Sets up the low rank training.

        If the optimizer state is populated (full-rank training done before this is called),
        then skip_optimizer_init should be True. This will call another helper to change the
        optimizer states to align with the OIALR weight shapes.

        Args:
                skip_optimizer_init: If True don't initialize SVD
        """
        if self.rank == 0:
            log.info("Starting with training in low rank")
        self.local_low_rank_model = self._replace_layers(self.local_full_rank_model)

        # Reset the last layer to the last layer.
        # if self.keep_last_layer:
        #     self._reset_last_layer(self.local_low_rank_model)

        self.model = self.local_low_rank_model
        self.svd_modules = {}
        self.non_svd_modules = {}
        self.layer_names = []
        calls = 0
        # sz = 1 if not dist.is_initialized() else dist.get_world_size()

        ign_norm_train = []
        ign_p_train = []
        for name, mod in self.model.named_modules():
            if hasattr(mod, "gather_distributed_factorizations"):
                # if self.distributed_updates:
                #     working_rank = calls % sz
                # else:
                # working_rank = self.rank
                # self.svd_modules.append((name, mod, working_rank))
                self.svd_modules[name] = {"mod": mod}
                self.layer_names.append(name)
                try:
                    # only a module in the attention layers and its just faster to try to get something
                    if mod._qkv_same_embed_dim:  # using in_proj
                        calls += 1
                        # ign_norm_train.extend([f"{name}.in_proj_u", f"{name}.in_proj_s", f"{name}.in_proj_vh"])
                        ign_norm_train.extend(
                            [f"{name}.in_proj_u", f"{name}.in_proj_s", f"{name}.in_proj_vh", f"{name}.in_proj_p"],
                        )
                        ign_p_train.extend([f"{name}.in_proj_u", f"{name}.in_proj_s", f"{name}.in_proj_vh"])
                    else:
                        calls += 3
                        ign_norm_train.extend(
                            [
                                f"{name}.q_u",
                                f"{name}.q_s",
                                f"{name}.q_vh",
                                f"{name}.q_p",
                                f"{name}.k_u",
                                f"{name}.k_s",
                                f"{name}.k_vh",
                                f"{name}.k_p",
                                f"{name}.v_u",
                                f"{name}.v_s",
                                f"{name}.v_vh",
                                f"{name}.v_p",
                            ],
                        )
                        ign_norm_train.extend(
                            [
                                f"{name}.q_u",
                                f"{name}.q_s",
                                f"{name}.q_vh",
                                f"{name}.k_u",
                                f"{name}.k_s",
                                f"{name}.k_vh",
                                f"{name}.v_u",
                                f"{name}.v_s",
                                f"{name}.v_vh",
                            ],
                        )
                except AttributeError:
                    # non-attn case
                    calls += 1
                    # ign.extend([f"{name}.u", f"{name}.s", f"{name}.vh"])
                    ign_norm_train.extend([f"{name}.u", f"{name}.s", f"{name}.vh", f"{name}.p"])
                    ign_p_train.extend([f"{name}.u", f"{name}.s", f"{name}.vh", f"{name}.p"])
            elif len(list(mod.children())) == 0:
                self.non_svd_modules[name] = {"mod": mod}
                for n, p in mod.named_parameters():
                    # in P_training, need to ignore all the other parameters -> not training
                    ign_p_train.append(f"{name}.{n}")

        self.ignore_norm_train = ign_norm_train
        self.ignore_p_train = ign_p_train
        self.model._ddp_params_and_buffers_to_ignore = ign_norm_train
        if not skip_optimizer_init:  # only need to do this if we start in full rank
            replace_opt_state_with_svd_adam(self.optimizer, self.low_rank_replacement_list)

        # No reason not to do DDP training if we can ignore the low-rank parameters
        self.model = DDP(self.model)
        if dist.get_rank() == 0:
            ids = []
            dct = {}
            for n, p in self.model.named_parameters():
                # print(n, id(p))
                ids.append(id(p))
                dct[id(p)] = n
            for p in self.model._module_parameters:
                del dct[id(p)]
        #     print(dct)
        # raise ValueError
        # self.non_svd_params = []
        # for n, p in self.named_parameters():
        #     if not n.endswith((".s", ".u", ".vh")) and p.requires_grad:
        #         self.non_svd_params.append(p)

    def _replace_layers(self, module, name=None, process_group=None):
        module_output = module
        if isinstance(module, nn.Linear) and name not in self.fixed_full_rank_layers:
            # print(f"layer name: {name} {self.fixed_full_rank_layers}")
            # if not self.first_layer:
            module_output = SVDSyncLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                # sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                start_weight=module.weight,
                start_bias=module.bias,
                # update_from_sigma=self.update_from_sigma,
                # reinit_shapes=self.reinit_shapes,
                # distributed_updates=self.use_ddp,
                inner_dim_init_ratio=self.inner_dim_init_ratio,
                random_sigma=self.random_sigma,
            ).to(device=module.weight.device, dtype=module.weight.dtype)
            # self.last_layer = [module, name, module.weight.dtype, module.weight.device]
            self.low_rank_replacement_list[id(module.weight)] = [module_output.s, "lin"]
            # else:
            #     self.first_layer = False
        elif isinstance(module, nn.MultiheadAttention) and name not in self.fixed_full_rank_layers:
            # if not self.first_layer:
            module_output = SVDSyncMultiheadAttention(
                embed_dim=module.embed_dim,
                num_heads=module.num_heads,
                dropout=module.dropout,
                bias=module.in_proj_bias is not None,
                add_bias_kv=module.bias_k is not None,
                add_zero_attn=module.add_zero_attn,
                kdim=module.kdim,
                vdim=module.vdim,
                batch_first=module.batch_first,
                # uvh_threshold=self.uvhthreshold,
                # sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                start_q=module.q_proj_weight,
                start_k=module.k_proj_weight,
                start_v=module.v_proj_weight,
                start_in_proj=module.in_proj_weight,
                start_k_bias=module.bias_k,
                start_v_bias=module.bias_v,
                start_in_proj_bias=module.in_proj_bias,
                # update_from_sigma=self.update_from_sigma,
                # reinit_shapes=self.reinit_shapes,
                # distributed_updates=self.use_ddp,
                inner_dim_init_ratio=self.inner_dim_init_ratio,
                random_sigma=self.random_sigma,
            ).to(device=module.out_proj.weight.device, dtype=module.out_proj.weight.dtype)
            # self.last_layer = [module, name, None, None]
            if module.in_proj_weight is not None:
                self.low_rank_replacement_list[id(module.in_proj_weight)] = [module_output.in_proj_s, "attn"]
            else:
                self.low_rank_replacement_list[id(module.q_proj_weight)] = [module_output.q_s, "attn"]
                self.low_rank_replacement_list[id(module.k_proj_weight)] = [module_output.k_s, "attn"]
                self.low_rank_replacement_list[id(module.v_proj_weight)] = [module_output.v_s, "attn"]
            # else:
            #     self.first_layer = False
        elif isinstance(module, nn.Conv2d) and name not in self.fixed_full_rank_layers:
            wv = module.weight.view(module.weight.shape[0], -1)
            if wv.shape[0] < wv.shape[1]:
                wv.T
            if wv.shape[1] < wv.shape[0] / 10:
                pass  # skip this layer if there are not enough params
            else:  # if not self.first_layer:
                module_output = SVDSyncConv2d(
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
                    # uvhthreshold=self.uvhthreshold,
                    # sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                    start_bias=module.bias,
                    start_weight=module.weight,
                    # update_from_sigma=self.update_from_sigma,
                    # reinit_shapes=self.reinit_shapes,
                    norm=module.norm if hasattr(module, "norm") else None,
                    activation=module.activation if hasattr(module, "activation") else None,
                    # distributed_updates=self.use_ddp,
                    inner_dim_init_ratio=self.inner_dim_init_ratio,
                    random_sigma=self.random_sigma,
                )
                # self.last_layer = [module, name, module.weight.dtype, module.weight.device]
                self.low_rank_replacement_list[id(module.weight)] = [module_output.s, "conv"]
            # else:
            #     self.first_layer = False
        elif isinstance(module, nn.Conv1d) and name not in self.fixed_full_rank_layers:
            wv = module.weight.view(module.weight.shape[0], -1)
            if wv.shape[0] < wv.shape[1]:
                wv.T
            if wv.shape[1] < wv.shape[0] / 10:
                pass  # skip this layer if there are not enough params
            else:  # if not self.first_layer:
                module_output = SVDSyncConv1d(
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
                    # uvhthreshold=self.uvhthreshold,
                    # sigma_cutoff_fraction=self.sigma_cutoff_fraction,
                    start_bias=module.bias,
                    start_weight=module.weight,
                    # update_from_sigma=self.update_from_sigma,
                    # reinit_shapes=self.reinit_shapes,
                    norm=module.norm if hasattr(module, "norm") else None,
                    activation=module.activation if hasattr(module, "activation") else None,
                    # distributed_updates=self.use_ddp,
                    inner_dim_init_ratio=self.inner_dim_init_ratio,
                    random_sigma=self.random_sigma,
                )
                # self.last_layer = [module, name, module.weight.dtype, module.weight.device]
                self.low_rank_replacement_list[id(module.weight)] = [module_output.s, "conv"]
            # else:
            #     self.first_layer = False
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

    @torch.no_grad()
    def get_perc_params_all_layers(self):
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        # percs, actives, normals = [], [], []
        trainable = 0
        untrainable = 0
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

    def set_epoch_length(self, len_dataloader):
        self.epoch_len = len_dataloader
        self.distributed_steps -= self.distributed_steps % self.epoch_len
        self.distributed_steps -= self.p_steps
        self.last_step = self.distributed_steps + self.p_steps
        self.last_step += self.last_step % self.epoch_len  # add epoch len
        if self.distributed_steps < 10:
            raise ValueError(
                "Distributed steps must be > 10 (so it learns something), "
                f"distributed steps: {self.distributed_steps} epoch len: {self.epoch_len} p steps: {self.p_steps}",
            )

    @staticmethod
    def reset_all_opt_states(optimizer: optim.Optimizer):
        # reset op1 first
        # for group in self.opt1.param_groups:
        optimizer.state = defaultdict(dict)

    @torch.no_grad()  # The function is the main method of doing stability tracking
    def average_models(self):
        # TODO: make a buffer to hold all of the weights for this to make this work in a delayed way
        #   unsure if needed, might be fast enough, dont know about frequency
        if not dist.is_initialized():
            return
        # for nonblocking case, check if there are waits to wait for
        # if nonblocking, wait for the sent items then exit
        # if waits is not None:
        #     for w in waits:
        #         w.wait()
        #     if nonblocking:
        #         return None
        waits = []
        for n, p in self.named_parameters():
            if not p.requires_grad or n.endswith(("_u", ".u", "_vh", ".vh")):
                continue
            waits.append(dist.all_reduce(p, op=dist.ReduceOp.AVG, async_op=True))
        # if nonblocking, return the waits for later
        # if blocking, wait for the op to complete right now
        # if nonblocking:
        #     return waits
        # else:
        for w in waits:
            w.wait()

    @torch.no_grad()
    def gather_distributed_factorizations(self):
        for name in self.svd_modules:
            # self.svd_modules[name] = {"mod": mod, "working_rank": working_rank, "stable": False, "stable_delay": 0}
            self.svd_modules[name]["mod"].gather_distributed_factorizations(name=name)

    @torch.no_grad()
    def distribute_workload(self):
        for name in self.svd_modules:
            # self.svd_modules[name] = {"mod": mod, "working_rank": working_rank, "stable": False, "stable_delay": 0}
            self.svd_modules[name]["mod"].distribute_workload(min_size_fraction=self.sync_min_size_fraction, name=name)

    @torch.no_grad()
    def mix_svd_layers(self):
        log.info(f"Mixing sigma of SVD layers with {self.mixing_method}")
        for name in self.svd_modules:
            self.svd_modules[name]["mod"].mix_sigma(method=self.mixing_method, **self.mixing_options)

    def forward(self, *args, **kwargs):
        # TODO: if we want to run this every N steps, then we need to track all of that.
        #       also, need to make specific functions for the val iterations
        # if self.step_on_forward and self.model.training:
        #     # print("in stability tracking block")
        #     self.model_stability_tracking(force=False)
        if not self.training:
            pass
        elif self.train_count < self.delay:  # in delay before doing anything
            return_opt = self.optimizer_normal
        elif self.train_count == self.delay:
            # at delay point -> distribute workload and prep normal training methods
            self.distribute_workload()
            self.prep_normal_training()
            self.train_count, self.delay = 0, -1
            return_opt = self.optimizer_normal
        elif self.train_count < self.distributed_steps:
            # continue to train locally on SVD layers and globally on others
            return_opt = self.optimizer_normal
        elif self.train_count == self.distributed_steps:
            # collect all trained models, prepare to train P vectors
            self.gather_distributed_factorizations()
            self.prep_p_training()
            return_opt = self.optimizer_p
        elif self.distributed_steps < self.train_count < self.last_step - 1:
            # train P vectors on each layer, no training other layers
            return_opt = self.optimizer_p
        elif self.train_count == self.last_step - 1:
            # fold P values into sigma, resume normal training in parallal
            # NOTE: this will let the sigma values dift slightly before they get distributed....
            # FIXME: see above
            self.prep_normal_training()
            return_opt = self.optimizer_normal

        self.train_count += 1
        return self.model(*args, **kwargs), return_opt

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

    @torch.no_grad()
    def prep_validate(self):
        # average non-svd parameters
        # should the P values be folder in here as well?
        # TODO: fix the SVD layers
        ...

    @torch.no_grad()
    def freeze_non_p(self):
        # freeze the network's parameters which are not P (mixing vectors)
        for name in self.non_svd_modules:
            # self.svd_modules[name] = {"mod": mod, }
            self.non_svd_modules[name]["mod"].eval()

    @torch.no_grad()
    def unfreeze_non_p(self):
        # freeze the network's parameters which are not P (mixing vectors)
        for name in self.non_svd_modules:
            # self.svd_modules[name] = {"mod": mod, }
            self.non_svd_modules[name]["mod"].train()

    @torch.no_grad()
    def fold_p_into_simga(self):
        # fold P into sigma and switch to normal training
        for name in self.svd_modules:
            # self.svd_modules[name] = {"mod": mod, }
            self.svd_modules[name]["mod"].update_sp_train_normally()
        self.train_p = False

    @torch.no_grad()
    def prep_p_training(self):
        # freeze training other values and start training P
        self.train_p = True
        self.freeze_non_p()
        for name in self.svd_modules:
            # self.svd_modules[name] = {"mod": mod, }
            self.svd_modules[name]["mod"].start_p_training()
        self.train_p = True
        self.model._ddp_params_and_buffers_to_ignore = self.ignore_p_train
        self.ddp_model = DDP(self.model)

    @torch.no_grad()
    def prep_normal_training(self):
        # return to normal training -> no distributed sigma sync, but sync everything else
        self.train_p = False
        self.fold_p_into_simga()
        self.unfreeze_non_p()
        self.model._ddp_params_and_buffers_to_ignore = self.ignore_norm_train
        self.ddp_model = DDP(self.model)
