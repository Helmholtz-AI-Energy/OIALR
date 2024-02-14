import logging
from collections import OrderedDict
from collections import abc as container_abcs
from collections import defaultdict

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from ..models.oialr import SVDLinear, SVDMultiheadAttention
from .utils import change_adam_shapes, change_sgd_shapes

log = logging.getLogger(__name__)


class MixedSVDOpt(object):
    def __init__(self, model: torch.nn.Module, config):
        # TODO: config must have the following:
        #   optimizer -> target + all the other stuff as normal
        #   lr -> learning rate for the normal optimizer
        #   sigma_optimizer -> optimizer specific for sigma - will not touch the other params
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.baseline = config.baseline
        # the optimizers will be passed via hydra, so they will have a target
        if config.training.optimizer is None:
            raise ValueError("Optimizer must be specified")

        self.params_non2d, self.params_weights, self.params_sigma = [], [], []
        self.sigma_param_index = []
        c = 0

        # self._create_param_lists(model)
        for c, (n, p) in enumerate(model.named_parameters()):
            if n.endswith(("_s", ".s")):
                self.params_sigma.append(p)
                self.sigma_param_index.append(c)
            elif n.endswith(("_u", ".u", "_vh", ".vh")):
                continue
            elif n.endswith("weight") and p.ndim == 2:
                self.params_weights.append(p)
            else:
                self.params_non2d.append(p)
        if config.training.fixing_method.keep_last_layer:
            last = self.params_weights.pop()
            self.params_non2d.append(last)

        optimizer1 = hydra.utils.instantiate(config.training.optimizer)
        # create the list of different parametes
        self.opt1: optim.Optimizer = optimizer1(self.params_non2d, lr=config.training.lr)
        # add a second group to the first optimier with the weigth matrices
        self.opt1.add_param_group({"params": self.params_weights})
        self.train_opt1 = True

        if self.baseline:
            self.sigma_opt = None
            return
        # in sigma_optimizer it should have all the parameters
        sigma_opt = hydra.utils.instantiate(config.training.sigma_optimizer)
        self.sigma_opt: optim.Optimizer = sigma_opt(self.params_sigma)
        self.sigma_opt_is_adam = isinstance(self.sigma_opt, torch.optim.Adam)
        self.sigma_opt_is_sgd = isinstance(self.sigma_opt, torch.optim.SGD)

    def _create_param_lists(self, module):
        # self.params_non2d, self.params_weights, self.params_sigma = [], [], []
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                self.params_non2d.append(module.bias)
            self.params_weights.append(module.weight)
        elif isinstance(module, SVDLinear):
            if module.bias is not None:
                self.params_non2d.append(module.bias)
            self.params_weights.append(module.weight)
            self.params_sigma.append(module.s)
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                self.params_weights.append(module.in_proj_weight)
                self.params_sigma.append(module.in_proj_s)
            else:
                self.params_weights.append(module.q_proj_weight)
                self.params_weights.append(module.k_proj_weight)
                self.params_weights.append(module.v_proj_weight)
            if module.in_proj_bias is not None:
                self.params_non2d.append(module.in_proj_bias)
            if module.bias_k is not None:
                self.params_non2d.append(module.bias_k)
            if module.bias_v is not None:
                self.params_non2d.append(module.bias_v)
        elif isinstance(module, SVDMultiheadAttention):
            if module.in_proj_weight is not None:
                self.params_weights.append(module.in_proj_weight)
                self.params_sigma.append(module.in_proj_s)
            else:
                self.params_weights.append(module.q_proj_weight)
                self.params_sigma.append(module.q_s)
                self.params_weights.append(module.k_proj_weight)
                self.params_sigma.append(module.k_s)
                self.params_weights.append(module.v_proj_weight)
                self.params_sigma.append(module.v_s)
            if module.in_proj_bias is not None:
                self.params_non2d.append(module.in_proj_bias)
            if module.bias_k is not None:
                self.params_non2d.append(module.bias_k)
            if module.bias_v is not None:
                self.params_non2d.append(module.bias_v)

        for n, child in module.named_children():
            self._create_param_lists(child)
        return

    def zero_grad(self, set_to_none: bool = False, reset_sigma=True):
        self.opt1.zero_grad(set_to_none=set_to_none)
        if self.baseline or not reset_sigma:
            return
        self.sigma_opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None, step_sigma=False, step_opt1=True):
        if step_opt1:
            self.opt1.step(closure=closure)
        if self.baseline:
            return
        if step_sigma:
            if self.rank == 0:
                log.info("Running step on Sigma optimizer")
            self.sigma_opt.step(closure=closure)

    # remove the weight group once all the sigma params have gradients
    # needs to be handled one step heigher
    # (in the training script with a cue from the stability function)
    def remove_full_rank_weights(self):
        if self.baseline:
            return
        # THIS IS DANGEROUS! it might become an issue latter....maybe?
        try:
            del self.opt1.param_groups[1]
            if self.rank == 0:
                log.info("Removing the full rank part of the base optimizer")
        except IndexError:
            # if there is only 1 group on opt1 then move on, faster to try and fail then check
            pass

    def reset_all_states(self):
        # reset op1 first
        # for group in self.opt1.param_groups:
        self.opt1.state = defaultdict(dict)
        self.sigma_opt.state = defaultdict(dict)

    def reset_shapes_of_sigma(self, model):
        if self.sigma_opt_is_adam:
            change_adam_shapes(
                optimizer=self.sigma_opt,
                model=model,
                param_indices=self.sigma_param_index,
                reset_buffers_zero=True,
            )
        elif self.sigma_opt_is_sgd:
            change_sgd_shapes(
                optimizer=self.sigma_opt,
                model=model,
                param_indices=self.sigma_param_index,
            )

    def disable_train_non2d(self):
        for p in self.params_non2d:
            p.requires_grad = False
        self.train_opt1 = False

    def enable_train_non2d(self):
        for p in self.params_non2d:
            p.requires_grad = True
        self.train_opt1 = True

    def get_lrs(self):
        out_dict = {}
        out_dict["non2d"] = self.opt1.param_groups[0]["lr"]
        prnt_str = f"NonSVD Param LR: {out_dict['non2d']}"
        if not self.baseline:
            # out_dict["full-rank"] = self.opt1.param_groups[1]["lr"]
            out_dict["sigma"] = self.sigma_opt.param_groups[0]["lr"]
            prnt_str += f"\tSimga LR: {out_dict['sigma']}"
        return out_dict, prnt_str

    def set_momentum_from_weights(self):
        # replace the parts sigma with svd's of adams momentum
        if self.sigma_opt_is_adam:
            # there should be the same amount of elements in sigma's params and weights,
            # only need to go through one of them.
            opt1keys = self.params_weights
            # NOTE: run this AFTER switching to sigma training?
            for c, p in enumerate(self.sigma_opt.param_groups[0]["params"]):
                opt1_state = self.opt1.state[opt1keys[c]]
                # TODO: this may not exist!!
                sigma_state = self.sigma_opt.state[p]
                sigma_group = self.sigma_opt.param_groups[0]
                if len(sigma_state) == 0:
                    # taken from torch's Adam
                    sigma_state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if sigma_group["capturable"] or sigma_group["fused"]
                        else torch.tensor(0.0)
                    )
                    # Exponential moving average of gradient values
                    sigma_state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    sigma_state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if sigma_group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        sigma_state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if len(list(opt1_state.keys())) > 0:
                    d1 = p.shape[0]
                    for k in ["exp_avg", "exp_avg_sq"]:
                        # need to get svd of the cut down shape for sigma from opt1
                        _, s, vh = torch.linalg.svd(opt1_state[k], full_matrices=False)
                        sigma_state[k].zero_()
                        if sigma_state[k].shape == 2:  # full rank stuff
                            s = torch.diag(s)  # @ vh
                            sigma_state[k].add_(s[:d1, :d1])
                            # mask = sigma_state[k] < 1e-6
                            # noise = torch.rand_like(sigma_state[k][mask]) * s.min() * 0.1
                            # sigma_state[k][mask].add_(noise)
                        elif sigma_state[k].shape == 1:  # full rank stuff
                            sigma_state[k].add_(s[:d1])

                    if self.opt1.param_groups[1]["amsgrad"]:
                        _, s, vh = torch.linalg.svd(opt1_state["max_exp_avg_sq"], full_matrices=False)
                        sigma_state["max_exp_avg_sq"].zero_()
                        if sigma_state["max_exp_avg_sq"].shape == 2:  # full rank stuff
                            s = torch.diag(s)  # @ vh
                            sigma_state["max_exp_avg_sq"].add_(s[:d1, :d1])
                            # k = "max_exp_avg_sq"
                            # mask = sigma_state[k] < 1e-6
                            # noise = torch.rand_like(sigma_state[k][mask]) * s.min() * 0.1
                            # sigma_state[k][mask].add_(noise)
                        elif sigma_state["max_exp_avg_sq"].shape == 1:  # full rank stuff
                            sigma_state["max_exp_avg_sq"].add_(s[:d1])
