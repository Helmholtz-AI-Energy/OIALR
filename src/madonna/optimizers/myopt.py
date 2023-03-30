import logging
import time
from typing import List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from mpi4py import MPI

from ..utils import comm, utils

log = logging.getLogger(__name__)


class MyOpt(object):
    def __init__(
        self,
        model: nn.Module,
        # config,
        contract_group_size: int = 4,
        contract_method: str = "ddp",
        topk_sigma: int = -1,
        blocking: bool = False,
    ):
        self.model = model
        # self.config = config
        self.model_buffers = dict()
        self.model_buffers_waits = dict()
        self.best_parameters = dict()
        self.best_parameters_waits = dict()

        for np, p in self.model.named_parameters():
            # if p.requires_grad:
            self.model_buffers[f"{np}"] = torch.zeros_like(p.data)
            # self.model_buffers_waits[f"{np}"] = None
            self.best_parameters[f"{np}"] = torch.zeros_like(p.data)
            # self.best_parameters_waits[f"{np}"] = None

        self.global_rank, self.local_rank = dist.get_rank(), None
        self.global_size, self.local_size = dist.get_world_size(), None

        # ============================= init params ====================================
        # self.chaotic_factor = 4
        # self.init_models()
        # skip init models for now, models should already be initialized and should be different
        #   across the ranks

        # ============================= control params ====================================
        self.best_local_loss = None
        # TODO: add these to the parameter list
        self.phases_dict = {
            "explore": {
                "step": 0,
                "num_iters": 100,
                "patience": 1,
                "last_best": 0,
            },  # index 0 -> step number. index 1-> limit of iterations
            "contract": {
                "step": 0,
                "num_iters": 50,
            },
            "losses": [],
        }
        self.backup_last_loss = None

        self.contract_method = contract_method
        self.contract_group_size = contract_group_size
        self._generate_comm_groups(
            group_size=contract_group_size,
            init_group_ddp=self.contract_method == "ddp",
        )
        # ============================= explore params ====================================
        self.uniform_n1_p1 = torch.distributions.uniform.Uniform(-1, 1)
        # ============================= contract params - SGD ====================================
        if self.contract_method == "ddp":
            self.sgd_optimizer = optim.SGD(
                self.model.parameters(),
                lr=0.01,
                weight_decay=1e-5,
                momentum=0.9,
                nesterov=True,
            )
        # ============================= contract params - SVD ====================================
        elif self.contract_method == "svd-genetic":
            self.u_buffers, self.vh_buffers = dict(), dict()
            self.sigma_buffers = dict()
            # structure: {loss: sigma_dict} -> use rounded loss for simplicity
            self.population = dict()
            self.topk_sigma = topk_sigma
            self.steps_between_population_updates = 5

        self.current_phase = "contract1"
        # self._prep_contract_step()
        self.prep_contract_step_1()
        # self.prep_explore_phase()

    def step(self, loss: torch.Tensor) -> torch.nn.Module:
        # switch between the phases and call the step function for that phase
        logging.debug(f"Optimizer step: {self.current_phase}")

        self.phases_dict["losses"].append(loss)
        if self.current_phase == "explore":
            switch = self.explore_step(loss)
        else:  # self.current_phase == "exploit" / "contract" / "contract1"
            switch = self.contract_step(loss)

        if switch:
            self.backup_last_loss = self.phases_dict["losses"][-3:]
            self.phases_dict["losses"] = []
            if self.current_phase == "explore":
                log.debug("Switching to contract")
                self.current_phase = "contract"
                # rollback network to best performing version...unsure if good plan
                best_loss = self._rollback_network()
                # prep contract phase
                self._prep_contract_step(best_loss)
                if self.contract_method == "ddp":
                    # In this case, we need to have the group-local-ddp model
                    # log.debug("using DDP model")
                    return self.group_ddp_model
            elif self.current_phase == "contract1":
                log.debug("Switching to explore")
                self.current_phase = "explore"
                self.phases_dict["contract"]["num_iters"] = 10
                self.prep_explore_phase()
            else:  # elif self.current_phase == "contract": / "contract1"
                log.debug("Switching to explore")
                self.current_phase = "explore"
                self.prep_explore_phase()
            # print("using individual model")
        return self.model

    @torch.no_grad()
    def set_all_to_best(self):
        if self.current_phase.startswith("contract"):
            return 1
        # set all networks to the best performing network over the most recent phase
        # TODO: should the losses thing but removed??
        try:
            dev = self.phases_dict["losses"][0].device
            dtp = self.phases_dict["losses"][0].dtype
            # losslist = [ls.item() for ls in self.phases_dict["losses"][-5:]]
        except IndexError:
            dev = self.backup_last_loss.device
            dtp = self.backup_last_loss.dtype
        #     losslist = [ls.item() for ls in self.backup_last_loss]
        # local_avg_loss = sum(losslist) / len(losslist)

        avg_losses = torch.zeros(self.global_size, device=dev, dtype=dtp)
        avg_losses[self.global_rank] = self.best_local_loss  # local_avg_loss
        dist.all_reduce(avg_losses)
        src = torch.argmin(avg_losses).item()
        logging.debug(f"Sending from rank {src}")
        waits = []
        for p in self.model.parameters():
            # if self.global_rank != src:
            #     p.zero_()
            waits.append(dist.broadcast(p, src=src, async_op=True))
        for w in waits:
            w.wait()
        return src

    @torch.no_grad()
    def shuffle_positions(self, best_rank):
        # at the end of each epoch, the networks are all joined into the best one
        # need to reset all the networks, but we dont want to roll random
        # idea: keep Q the same, and generate a new R matrix for half, other half, shuffle Q and keep R the same
        # TODO: change the re-init mode to work from the
        # TODO: change blurring to be based on a parameter!

        blurring_factor = 0.1  # up to X% changes of weights up/down

        self.phases_dict["losses"] = []
        self.current_phase = "explore"
        if self.global_rank == best_rank:
            return
        keepq = torch.rand(1).item() >= 0.5
        if keepq:
            log.debug("Shuffling: keeping Q fixed, modifying R")
        else:
            log.debug("Shuffling: Shuffling Q, keeping R")
        self.model.train()
        for p in self.model.parameters():
            if p.requires_grad:
                if p.ndim == 1:  # if weights are 1D, blur them
                    p.mul_(1 + self.uniform_n1_p1.sample(p.shape) * blurring_factor)
                    continue
                shp = p.shape
                if p.ndim > 2:
                    hold = p.view(shp[0], -1)
                else:
                    hold = p

                trans = False
                if hold.shape[0] < hold.shape[1]:  # if the matrix is SF, transpose
                    trans = True
                    hold = hold.T
                q, r = torch.linalg.qr(hold, mode="reduced")
                if keepq:
                    r *= 1 + self.uniform_n1_p1.sample(r.shape) * blurring_factor
                else:  # keep r the same, shuffle Q
                    q = utils.roll_orthogonal_values(
                        q.shape,
                        dtype=q.dtype,
                        device=q.device,
                        scale_factor=1.0,
                    )
                    # q = q[:, torch.randperm(q.shape[1], device=q.device)]
                if trans:
                    p.set_((q @ r).T.reshape(shp))
                else:
                    p.set_((q @ r).reshape(shp))

    def _generate_comm_groups(self, group_size=2, init_group_ddp=False):
        # create comm groups for the contractions
        # if desired, init the group's DDP instance
        self.local_comm_group = comm.create_sub_groups(group_size)
        self.local_rank = dist.get_rank(self.local_comm_group)
        self.local_size = dist.get_world_size(self.local_comm_group)

        if init_group_ddp:
            self.group_ddp_model = nn.parallel.DistributedDataParallel(
                module=self.model,
                process_group=self.local_comm_group,
                # find_unused_parameters=False,
            )
        log.debug("Finished Group Comm init")

    # ============================== initialization functions ======================================
    @torch.no_grad()
    def init_models(self):
        # skipping chaotic init: bugs in this...
        for c in self.model.children():
            if hasattr(c, "track_running_stats"):
                c.track_running_stats = False
        self.model.train()
        utils.log0(msg=f"Model: \n {self.model}", level="debug")
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                # if p.data.ndim > 1:
                #     nn.init.kaiming_uniform_(p)
                # else:
                #     nn.init.uniform_(p)
                p.set_(torch.rand_like(p) * torch.sign(torch.randn_like(p)))
                # print(f"start {n}: {p.mean():.5f} {p.min():.5f} {p.max():.5f} {p.std():.5f}, {p.shape}")
        utils.log0(msg="Finished Model init", level="info")

    # ============================== Exploration functions =========================================
    def explore_step(self, current_loss):
        # continue to explore the local area
        # record the loss values and track them
        if self.best_local_loss is None or self.best_local_loss > current_loss:  # assume minimize
            log.info(
                f"New local best: {current_loss:.5f} v {self.best_local_loss:.5f}\t\t"
                f"step: {self.phases_dict['explore']['step']}",
            )
            self._store_best_network_in_buffers(current_loss)

        # if we are not improving, bail on a direction and test again
        if current_loss > self.best_local_loss:
            self._rollback_network()
            self._set_explore_grads()

        # self._test_explore_stability()
        # self._set_explore_grads()
        self._apply_gradients()

        # change to the contract step after a set number
        # TODO: 'smart' version of changing between steps
        self.phases_dict["explore"]["step"] += 1
        if self.phases_dict["explore"]["step"] == self.phases_dict["explore"]["num_iters"]:
            return True
        return False

    @torch.no_grad()
    def _apply_gradients(self):
        # apply the gradients to the network,
        # TODO: should the step size decrease?
        # This could do something like a binary reduction
        # just doing normal stepping here
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                # gradients are only on the parameters to be trained
                hold = p.data + p.grad
                p.set_(hold)
                # p.data = nn.functional.normalize(p.data, dim=None, p=2.0)

    def _store_best_network_in_buffers(self, loss):
        self.best_local_loss = loss
        self.phases_dict["explore"]["last_best"] = self.phases_dict["explore"]["step"] + 1
        for n, p in self.model.named_parameters():
            # if p.requires_grad:
            self.model_buffers[n].zero_()
            self.model_buffers[n].add_(p)

    @torch.no_grad()
    def _rollback_network(self):
        log.debug("Rolling back network")
        # NOT SURE IF NEEDED
        # roll back the network to the known best
        #   use the model buffers as a temporary place to store the good params
        #   (will be used later to send the good networks around)
        for n, p in self.model.named_parameters():
            # if p.requires_grad:
            p.zero_()
            p.add_(self.model_buffers[n])
        return self.best_local_loss  # the best local loss which is associated with this model

    @torch.no_grad()
    def _set_explore_grads(self):
        # scale gradients to be X% of the data at maximum
        scale_factor = 0.005  # scale factor of above
        for p in self.model.parameters():
            if p.requires_grad:
                # only work on the parameters which are to be trained...

                # get orthogonal 'grads' FIXME?
                # grads = utils.roll_orthogonal_values(
                #     p.data.shape,
                #     dtype=p.dtype,
                #     device=p.device,
                #     scale_factor=1.,  # p.data.std() / self.phases_dict["explore"]['num_iters'],
                # )

                # objective: the gradient change should be only 0.1% of a given weight
                # update will be p + grad (can also be minus...)
                # grad = 0.1% of data * rand[-1, 1]
                grads = p.data * self.uniform_n1_p1.sample(p.shape) * scale_factor
                if p.grad is None:
                    p.grad = grads
                else:
                    p.grad.zero_()
                    p.grad.add_(grads)

    def prep_explore_phase(self):
        # prepare for the exploration phase
        # roll new orthogonal matrices for the gradients
        self.phases_dict["explore"]["step"] = 0
        self._set_explore_grads()

    def _test_explore_stability(self):
        # test the stability of the direction of the network
        # TODO: how long until we bail on a direction if it isnt working?
        lb = self.phases_dict["explore"]["last_best"]
        sn = self.phases_dict["explore"]["step"]
        if sn - lb >= self.phases_dict["explore"]["patience"] and sn > 0:
            log.info(f"moving into a new direction, step num: {sn}, last_best: {lb}")
            self._rollback_network()
            self._set_explore_grads()
            self.phases_dict["explore"]["last_best"] = sn
        else:
            log.debug(
                f"moving in same direction step num: {sn}, "
                f"last_best: {lb} -> {self.phases_dict['losses'][-3:]}",
            )

    # ====================== Exploitation / Contraction functions ==================================
    def contract_step(self, loss):
        # "Contract" around a selected point
        # this is the exploitation phase -> should improve on the results
        # TODO: deciding between 2 options still: local SGD and Genetic SVD (only sigma)
        self.phases_dict["losses"].append(loss)
        if self.contract_method == "ddp":
            self.contract_ddp_step(loss)  # call backwards and then call the local optimizer
        # elif self.contract_method == "svd-genetic":
        #     self.contract_genetic_step(loss)

        self.phases_dict["contract"]["step"] += 1
        if self.phases_dict["contract"]["step"] == self.phases_dict["contract"]["num_iters"]:
            return True
        return False

    # ----------------------------- general contract step functions --------------------------------

    @torch.no_grad()
    def prep_contract_step_1(self):
        # both methods: set up the groups
        # get the best networks + communicate the best networks to the groups
        for p in self.model.parameters():
            if not p.is_contiguous():
                p.set_(p.contiguous())
        # self.sort_and_distributed_best_to_groups(current_loss)
        self.phases_dict["contract"]["step"] = 0
        # SGD optimization: do nothing else?? maybe find a way to change the LR??
        if self.contract_method == "ddp":
            self.sgd_optimizer.zero_grad(set_to_none=True)
            return
        # genetic optimization: generate populations within the groups
        # if self.contract_method == "svd-genetic":
        self.svd_generate_population()

    @torch.no_grad()
    def _prep_contract_step(self, current_loss=None):
        # both methods: set up the groups
        # get the best networks + communicate the best networks to the groups
        for p in self.model.parameters():
            if not p.is_contiguous():
                p.set_(p.contiguous())
        if current_loss is not None:
            self.sort_and_distributed_best_to_groups(current_loss)
        self.phases_dict["contract"]["step"] = 0
        # SGD optimization: do nothing else?? maybe find a way to change the LR??
        if self.contract_method == "ddp":
            self.sgd_optimizer.zero_grad(set_to_none=True)
            return
        # genetic optimization: generate populations within the groups
        # if self.contract_method == "svd-genetic":
        self.svd_generate_population()

    def sort_and_distributed_best_to_groups(self, loss):
        # get the best performing ranks (best networks)
        _, sorted_ranks = self.get_sorted_losses(loss, group=None)
        self._distributed_best_params_to_groups(sorted_ranks)
        # for n, p in self.model.named_parameters():
        #     print(f"\t{n}: {p.mean():.4f}, {p.min():.4f}, {p.max():.4f}, {p.std():.4f}")
        # raise ValueError

    @staticmethod
    def get_sorted_losses(loss, group=None, use_average_loss: bool = False):
        ws = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        losses = torch.zeros(ws, dtype=loss.dtype, device=loss.device)
        losses[rank] = loss
        dist.all_reduce(losses, group=group)  # sum op, NOT async operation
        # TODO: this should be topk, not sort! sort is much slower...change later
        sorted_losses, sorted_ranks = torch.sort(losses, descending=False)
        return sorted_losses, sorted_ranks

    def _distributed_best_params_to_groups(self, best_ranks):
        # send best params to all workers in the groups
        # NOTE: only need to send the params within the groups
        # TODO: group sizes? send + bcast?
        # get the number of groups from the group size
        num_groups = self.global_size // self.contract_group_size

        # send the network from the best ranks to each group's 0 rank
        group_rank0s = torch.arange(
            start=0,
            end=self.global_size,
            step=self.contract_group_size,
            dtype=best_ranks.dtype,
            device=best_ranks.device,
        ).tolist()
        # print(group_rank0s)
        for send, dest in zip(best_ranks[:num_groups], group_rank0s):  # , strict=True):
            if send == dest:
                continue
            if self.global_rank == send:
                log.debug(f"sending to rank: {dest}")
                self._send_network(dest)
            if self.global_rank == dest:
                log.debug(f"receiving from rank: {send}")
                self._recv_network(send)
        # now, the best parameters are on the group_rank0 in the best_parameter buffers
        # dist.barrier()
        self.local_comm_group.barrier()
        self._bcast_network_to_group()

    def _send_network(self, dest):
        tag = 0
        send_waits = []
        for p in self.model.parameters():
            # if p.requires_grad:
            send_waits.append(dist.isend(p.data, dst=dest, tag=tag))
        for w in send_waits:
            w.wait()

    def _recv_network(self, src):
        tag = 0
        for n, p in self.model.named_parameters():
            # if p.requires_grad:
            self.best_parameters_waits[n] = dist.irecv(
                self.best_parameters[n],
                src=src,
                tag=tag,
            )

    @torch.no_grad()
    def _bcast_network_to_group(self):
        for n, p in self.model.named_parameters():
            # if p.requires_grad:
            try:
                # if self.best_parameters_waits[n] is not None:
                # wait for the group rank0 to receive the model to optimize
                self.best_parameters_waits[n].wait()
            except (AttributeError, KeyError):
                pass

            # send the network to the other networks
            self.best_parameters_waits[n] = dist.broadcast(
                self.best_parameters[n],
                src=0,
                group=self.local_comm_group,
                async_op=True,
            )
        for n, p in self.model.named_parameters():
            # if p.requires_grad:
            try:
                self.best_parameters_waits[n].wait()
            except AttributeError:  # op was blocking
                pass
            p.set_(self.best_parameters[n])

    # ------------------------------ SGD functions -------------------------------------------------

    def contract_ddp_step(self, loss):
        # things are simple for this case. it should only be between the models within this group
        # self.local_comm_group.barrier()
        loss.backward()
        self.sgd_optimizer.step()

    # ------------------------------ SVD functions -------------------------------------------------

    def svd_contract(self):
        # do svd contraction
        pass

    def svd_generate_population(self):
        # find the U, S, Vh for each layer
        # freeze U and Vh, create population for training from S
        # TODO: should this be changed keep sigma in a single vector???
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # TODO: take the transpose if not TS??
            u, s, vh = torch.linalg.svd(p.data, full_matrices=False)
            if self.topk_sigma > 0:
                u = u[:, : self.topk_sigma]
                s = s[: self.topk_sigma]
                vh = vh[: self.topk_sigma, :]

            try:
                self.sigma_buffers[n].zero_()
                self.sigma_buffers[n].add_(s)
                self.u_buffers[n].zero_()
                self.u_buffers[n].add_(u)
                self.vh_buffers[n].zero_()
                self.vh_buffers[n].add_(vh)
            except (KeyError, RuntimeError):  # if no key, or the shape has changed, reset
                self.sigma_buffers[n] = s
                self.u_buffers[n] = u
                self.vh_buffers[n] = vh
            # generate the population in sigma_buffers
            # TODO:debate: should the population be random, or should it be blurred?
            #   -> going with blurred for now....
            # this should be different on each process
            # TODO: normal distribution or uniform?? add or multiply??
            self.sigma_buffers[n] += torch.randn_like(self.sigma_buffers[n])

    def svd_update_population(self, recent_losses):
        # since this will be blocking - easier to implement - this should be done after X steps (??)
        # get all the losses on all the ranks (rank x losses)
        # flatten + compare with existing population
        # get rank of methods to communicate around
        # TODO: the sigma values could be communicated around more easily with a vector,
        #   which would then be sliced to get the local values
        #   not sure if its worth it right now, maybe optimize later

        # for now, update population blocking after each forward step (multiple forwards?)
        # get all the losses
        sorted_losses, sorted_ranks = self.get_sorted_losses(
            recent_losses,
            group=self.local_comm_group,
        )
        # sort the losses into the population
        if len(self.population) > 0:
            updated_population = torch.cat([self.population.keys(), sorted_losses])
            # not growing the population by more than 2, throw out the worst stuff
            updated_population, _ = torch.sort(updated_population)
            updated_population = updated_population[: 2 * len(self.population)]
            # new_posisions = torch.nonzero(updated_population == self.population.keys())  # CHECKME
            # new_ranks =
            # update the new positions
        else:
            updated_population = sorted_losses

        # find positions which are within the population
        # broadcast the positions to the other ranks to update the

        if recent_losses.shape[0] == -1:  # FIXME
            # this stuff it to make the evolutionary algo work with multiple forward steps
            # between each population update.
            # not sure if needed or necessary. Maybe remove later if performance is good
            ws = dist.get_world_size(group=self.local_comm_group)
            rank = dist.get_rank(group=self.local_comm_group)
            losses = torch.zeros(
                (ws, *recent_losses.shape),
                dtype=recent_losses.dtype,
                device=recent_losses.device,
            )
            losses[rank] = recent_losses
            dist.all_reduce(losses, group=self.local_comm_group)  # sum op, NOT async operation
            flat_losses = losses.flatten()

            # if self.population
            sorted_losses, sorted_ranks = torch.sort(flat_losses, descending=False)
            return sorted_losses, sorted_ranks
        pass

    def svd_contract_step(self, loss):
        # set up the svd contract step
        # rank results in population
        sorted_losses, sorted_ranks = self.get_sorted_losses(loss, group=self.local_comm_group)
        # generate next sample to test
