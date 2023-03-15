import time

import torch
import torch.distributed as dist
import torch.nn as nn
from mpi4py import MPI


class TorchSMA(object):
    """
    torch slime mold optimizer method

    This is a work in progress and will be changed greatly later
    """

    def __init__(
        self,
        model,
        upper_bound=2.0,
        lower_bound=-2.0,
        z=0.03,
        individual=False,
        chaotic_init=True,
        num_steps=-1,
    ):
        self.model = model
        self.ub = upper_bound
        self.lb = lower_bound
        self.z = z
        self.individual = individual
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
        self.chaotic_init = chaotic_init
        self.chaotic_factor = 0.5
        self.model_buffers = {}
        self.model_buffers_waits = {}
        self.best_parameters = {}
        self.best_parameters_waits = {}
        self.model_buffers_waits = {}
        for nc, c in self.model.named_children():
            if hasattr(c, "reset_parameters"):
                c.reset_parameters()
            for np, p in c.named_parameters():
                self.model_buffers[f"{nc}-{np}"] = torch.zeros_like(p.data)
                self.model_buffers_waits[f"{nc}-{np}"] = None
                if p.requires_grad:
                    self.best_parameters[f"{nc}-{np}"] = torch.zeros_like(p.data)
                    self.best_parameters_waits[f"{nc}-{np}"] = None
        for np, p in self.model.named_parameters():
            if p.requires_grad:
                self.best_parameters[f"{np}"] = torch.zeros_like(p.data)
                self.best_parameters_waits[f"{np}"] = None
        self.uniform01 = None
        self.uniform_lbub = None
        self.step_count = 0
        self.rank_selector = torch.arange(self.size)
        if num_steps <= 0:
            self.num_steps = 100000

    def set_model_buffers_to_params(self):
        self.model_buffers = {}
        self.model_buffers_waits = {}
        for np, p in self.model.named_parameters():
            if p.requires_grad:
                self.model_buffers[f"{np}"] = torch.zeros_like(p.data)
                self.model_buffers_waits[f"{np}"] = None

    @torch.no_grad()
    def init_model(self, keep_rank0=False):
        # print(self.chaotic_init)
        if not self.chaotic_init:
            for c in self.model.children():
                if hasattr(c, "reset_parameters"):
                    c.reset_parameters()
            self.set_model_buffers_to_params()
            return
        if self.rank == 0:
            tag = 0
            # TODO: what to do about the requires grad stuff? that just means its training?
            for c in self.model.children():
                # print(c)
                if hasattr(c, "reset_parameters") and not keep_rank0:
                    c.reset_parameters()
                for _, p in c.named_parameters():
                    # print(f"sending {n} to rank 1")
                    dist.send(p.data, dst=1, tag=tag)
                    tag += 1
            self.set_model_buffers_to_params()
            return
        # rest of the ranks
        tag = 0
        # TODO: what to do about the requires grad stuff? that just means its training?
        for nc, c in self.model.named_children():
            for np, p in c.named_parameters():
                self.model_buffers[f"{nc}-{np}"] += p
                # print(f"recv {np} from rank {self.rank - 1}, sending to {self.rank + 1}")
                dist.recv(self.model_buffers[f"{nc}-{np}"], src=self.rank - 1, tag=tag)

                hold = self.model_buffers[f"{nc}-{np}"]
                p.set_(self.chaotic_factor * hold * (1 - hold))
                if self.rank < self.size - 1:
                    dist.send(p.data, dst=self.rank + 1, tag=tag)
                tag += 1
        self.set_model_buffers_to_params()

    def init_rngs(self, tensor):
        fact = {"dtype": tensor.dtype, "device": tensor.device}
        self.uniform01 = torch.distributions.uniform.Uniform(
            torch.tensor([0.0], **fact),
            torch.tensor([1.0], **fact),
        )
        # self.step_count = self.step_count.to(**fact)
        self.uniform_lbub = torch.distributions.uniform.Uniform(
            torch.tensor([self.lb], **fact),
            torch.tensor([self.ub], **fact),
        )
        self.rank_selector = self.rank_selector.to(**fact)

    def get_slime_masses(self, sorted_fitnesses, fitnesses):
        # 2. get the spread of the fitness values
        eps = torch.finfo(fitnesses.dtype).eps
        # NOTE: in the original, this is defined to be negative, but that is dumb
        fitness_spread = sorted_fitnesses[-1] - sorted_fitnesses[0] + eps
        # Eq.(2.5)
        masses = torch.zeros_like(fitnesses)
        # for first half of the sorted population use 1 +
        half = self.size // 2
        # print(1 + self.uniform01.sample(masses[:half].shape), torch.log10(
        #    (sorted_fitnesses[:half] - sorted_fitnesses[0]) / fitness_spread + 1))
        masses[:half] = 1 + self.uniform01.sample(masses[:half].shape).flatten() * torch.log10(
            (sorted_fitnesses[:half] - sorted_fitnesses[0]) / fitness_spread + 1,  # +1 in ()?
        )
        masses[half:] = 1 - self.uniform01.sample(masses[half:].shape).flatten() * torch.log10(
            (sorted_fitnesses[half:] - sorted_fitnesses[0]) / fitness_spread + 1,  # +1 in ()?
        )
        return masses

    def bcast_best_position(self, sorted_inds):
        for np, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # self.best_parameters[f"{np}"].set_(p.data)
            self.best_parameters[f"{np}"].zero_()
            if self.rank == sorted_inds[0]:
                self.best_parameters[f"{np}"].add_(p)

            self.best_parameters_waits[f"{np}"] = dist.broadcast(
                self.best_parameters[f"{np}"],
                src=sorted_inds[0],
                async_op=True,
            )

    def send_model_to_partner(self, partner):
        # send_waits = []
        # print(f"sending whole model to partner: {partner}")
        c = 0
        # to keep the tags together, need to do the smaller partner first???
        send_first = self.rank < partner
        for np, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if send_first:
                self.model_buffers_waits[f"{np}"] = dist.isend(
                    p,
                    dst=partner,
                    tag=c,
                )
                c += 1

            self.model_buffers_waits[f"{np}"] = dist.irecv(
                self.model_buffers[f"{np}"],
                src=partner,
                tag=c,
            )
            c += 1

            if not send_first:
                self.model_buffers_waits[f"{np}"] = dist.isend(
                    p,
                    dst=partner,
                    tag=c,
                )
                c += 1

        # for w in send_waits:
        #     w.wait()
        # for w in self.model_buffers_waits:
        #    self.model_buffers_waits[w].wait()

    @torch.no_grad()
    def step(self, fitness):
        # randomly roll parameter combinations -> init_model
        # for loop over number of epochs: (assume this is run at the end of a training step
        #   sort them by their fitness (get loss value)
        #   get the spread of the fitness values (losses)
        #   find the weights of each slime mold
        #   eq 2.4
        #   update the position of search agents - update network parameters - work on each pop
        #           member
        #       roll random value, if below self.z, roll new random params
        #       eq 2.2 -> eq 2.3
        #       two positions randomly selected from population
        #       randomly combine the two positions
        #       Check bounds (roll random if values are out of bounds)
        pass
        fact = {"dtype": fitness.dtype, "device": fitness.device}
        if fitness.isnan():
            fitness = torch.tensor(10000, **fact)
            for _, p in self.model.named_parameters():
                p.zero_()
                # self.best_parameters_waits[n].wait()
                # hold = self.best_parameters[n]
                p.set_(self.chaotic_factor * p.data * (1 - p.data))
        # NOTE: fitness must be a torch tensor
        # 1. get fitness of all ranks + sort
        fitnesses = torch.zeros((self.size, *tuple(fitness.shape)), **fact)
        fitnesses[self.rank] = fitness
        wait = dist.all_reduce(fitnesses, async_op=True)  # defaults are SUM and blocking
        if self.step_count == 0:
            self.init_rngs(fitness)
        wait.wait()

        # TODO: what do when fitness is not a single value
        # assuming ascending sort order (min at index 0)
        sorted_fitnesses, sorted_inds = torch.sort(fitnesses, dim=-1)
        self.bcast_best_position(sorted_inds)

        # get pair

        # 3. get the weights of each slime mold
        masses = self.get_slime_masses(sorted_fitnesses=sorted_fitnesses, fitnesses=fitnesses)
        # 4. Update the Position of search agents
        # do a local RNG roll to determine this
        # 4a. if the random roll is below a cutoff, reroll it
        reroll = self.uniform01.sample() < self.z
        rerolls = torch.zeros_like(fitnesses, dtype=torch.int32)
        rerolls[self.rank] = reroll.to(torch.int32)
        dist.all_reduce(rerolls)
        # print(f"rerolls {rerolls}")
        # pairs should be where the rerolls is False, other ranks are removed
        pairs = self.rank_selector[rerolls == 0]
        if self.rank == 0:
            pairs = pairs[torch.randperm(pairs.shape[0]).to(pairs.device)]
        dist.broadcast(pairs, src=0, async_op=False)
        if reroll:
            # if the random roll is below the cutoff need to REROLL
            # TODO: write new way to reroll these values (or at least how to better move forward)
            # # need to tell other ranks which ones are not syncing
            # for nc, c in self.model.named_children():
            # for _, p in self.model.named_parameters():
            #     p.zero_()
            #     # self.best_parameters_waits[n].wait()
            #     # hold = self.best_parameters[n]
            #     p.set_(self.chaotic_factor * p.data * (1 - p.data))
            # print("h")
            return

        if pairs.shape[0] % 2 == 1:
            # remrank = pairs[-1]  # what to do with me??? ignore for now.....
            pairs = pairs[:-1]
        pairs = pairs.view(pairs.shape[0] // 2, 2)
        # print("pairs", pairs)
        my_pair = pairs[torch.any(pairs == self.rank, 1)]
        partner = my_pair[my_pair != self.rank]
        # 4b. normal combination
        # Eq 2.4
        # TODO: make step_count an int
        a = torch.arctanh(torch.tensor(-((self.step_count + 1.0) / self.num_steps) + 1.0, **fact))  # Eq.(2.4)
        b = 1 - (self.step_count + 1) / self.num_steps
        cutoff = 0.1  # torch.tanh(torch.abs(fitness - sorted_fitnesses[0]))
        # print(a, b, cutoff)
        # vb and vc are the size of the parameters to replace
        # select 2 random ranks from the population..... start with the neighbor?
        # time.sleep(self.rank / 10)
        # print(pairs, reroll, partner)
        try:
            partner = int(partner.item())
        except ValueError:
            # # this is the case partner has nothing in it! (faster to fail then to check shape)
            return

        self.send_model_to_partner(partner)

        for n, p in self.model.named_parameters():
            # print("working on layer:", n)
            if not p.requires_grad:
                continue
            # todo: rand or randn?
            # r1 = torch.rand_like(p.data)
            r1 = self.uniform01.sample(p.data.shape).squeeze()
            vb = torch.rand_like(p.data) * 2 * a - a  # rescale vb to -a to a
            # vb = ((torch.rand_like(p.data) + a) / (2*a)) - a  # rescale vb to -a to a
            vc = torch.rand_like(p.data) * 2 * b - b  # rescale vb to -b to b
            # vc = ((torch.rand_like(p.data) + b) / (2*b)) - b  # rescale vb to -b to b
            # best_pos -> best values, need to send immediately
            # TODO: change uniform01 to be somethign different? just use torch rand?
            # merging_rand = self.uniform01.sample(p.shape).squeeze()

            self.best_parameters_waits[n].wait()
            self.model_buffers_waits[n].wait()
            partner_data = self.model_buffers[n]
            # pos_1 -> my values
            # pos_2 -> from partner values
            pos1 = self.best_parameters[n] + vb * (masses[self.rank] * p.data - partner_data)
            pos2 = vc * partner_data
            print(f"pos1, min: {pos1.min()}, max: {pos1.max()}, mean: {pos1.mean()}")
            print(f"pos2, min: {pos2.min()}, max: {pos2.max()}, mean: {pos2.mean()}")
            # merging_rand = r1
            # print(pos1.shape, pos2.shape, r1.shape, p.shape)
            # mask = merging_rand
            # print(mask)
            new_p = torch.where(r1 < cutoff, pos1, pos2)
            print(f"{n}, min: {new_p.min()}, max: {new_p.max()}, mean: {new_p.mean()}")
            # print("after torch where")
            if torch.any(new_p.isnan()):
                print(f"nans in layer: {n}")
            # scale new weights to same range?
            # new_p = new_p * ()
            p.zero_()
            p.add_(new_p)
            # print("after setting new p")
            # p.set_(new_p)
        # print("done with step")
        # dist.barrier()

        # NOTE: keep at end:
        self.step_count += 1
        pass


# class BaseSMA(object):
#     """
#         Modified version of: Slime Mould Algorithm (SMA)
#             (Slime Mould Algorithm: A New Method for Stochastic Optimization)
#         Notes:
#             + Selected 2 unique and random solution to create new solution
#             (not to create variable) --> remove third loop in original version
#             + Check bound and update fitness after each individual move instead
#             of after the whole population move in the original version
#     """
#
#     ID_MIN_PROB = 0  # min problem -> used in getting the best solution
#     ID_POS = 0  # Position - index of gbest to show where the best position is
#     # NOTE: the position refers to the network parameters
#     ID_FIT = 1  # Fitness - index of gbest with the best fitness (lowest loss)
#     ID_WEI = 2  # weight - indes of gbest with the weight of the slime mold at this position
#
#     def __init__(
#         self,
#         obj_func=None,    # function to train over
#         lb=None,          # lower bound for XXXXX
#         ub=None,          # upper bound for XXXXX
#         problem_size=50,  # size of problem?
#         verbose=True,     # ...
#         epoch=750,        # number of epochs to train for
#         pop_size=100,     # number of population members
#         z=0.03            #
#     ):
#         self.obj_func = obj_func
#         if (lb is None) or (ub is None):
#             if problem_size <= 0:
#                 raise ValueError(f"Problem size must > 0: given: {problem_size}")
#             else:
#                 self.problem_size = int(ceil(problem_size))
#                 self.lb = -1 * ones(problem_size)
#                 self.ub = 1 * ones(problem_size)
#         else:
#             if isinstance(lb, list) and isinstance(ub, list) and not (problem_size is None):
#                 if (len(lb) == len(ub)) and (problem_size > 0):
#                     if len(lb) == 1:
#                         self.problem_size = problem_size
#                         self.lb = lb[0] * ones(problem_size)
#                         self.ub = ub[0] * ones(problem_size)
#                     else:
#                         self.problem_size = len(lb)
#                         self.lb = array(lb)
#                         self.ub = array(ub)
#                 else:
#                     print(
#                         "Lower bound and Upper bound need to be same length. Problem size must > 0
#                     )
#                     exit(0)
#             else:
#                 print(
#                     "Lower bound and Upper bound need to be a list. Problem size is an int number"
#                 )
#                 exit(0)
#         self.verbose = verbose
#         self.epoch, self.pop_size = None, None
#         self.loss_train = []
#         self.epoch = epoch
#         self.pop_size = pop_size
#         self.z = z
#         self.eps = torch.finfo(torch.float).min
#
#     def create_solution(self):
#         """
#         create a new position (parameters) and test its fitness
#         """
#         position = uniform(self.lb, self.ub)
#         fitness = self.obj_func(position)
#         return [position, fitness]
#
#     def get_sorted_pop_and_global_best_solution(self, pop=None, id_fit=None, id_best=0):
#         """ Sort population and return the sorted population and the best position """
#         # this assumes an ascending sort order,
#         sorted_pop = sorted(pop, key=lambda temp: temp[id_fit])
#         return sorted_pop, deepcopy(sorted_pop[id_best])
#
#     def amend_position(self, position=None):
#         return clip(position, self.lb, self.ub)
#
#     def amend_position_random(self, position=None):
#         # gets the parameters which are within the lower and upper bounds
#         # parameters outside the bounds are replaced with random values
#         return where(
#           logical_and(self.lb <= position, position <= self.ub),
#           position,
#           uniform(self.lb, self.ub)
#           )
#
#     def update_sorted_population_and_global_best_solution(
#       self, pop=None, id_best=None, g_best=None
#       ):
#         """ Sort the population and update the current best position.
#         Return the sorted population and the new current best position """
#         sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
#         current_best = sorted_pop[id_best]
#         g_best = deepcopy(current_best) if current_best[self.ID_FIT] < g_best[self.ID_FIT] \
#               else deepcopy(g_best)
#         return sorted_pop, g_best
#
#     def create_solution(self):
#         pos = uniform(self.lb, self.ub)
#         fit = self.obj_func(pos)
#         weight = zeros(self.problem_size)
#         return [pos, fit, weight]
#
#     def train(self):
#         # randomly roll parameter combinations
#         # for loop over number of epochs:
#         #   sort them by their fitness (get loss value)
#         #   get the spread of the fitness values (losses)
#         #   find the weights of each slime mold
#         #   eq 2.4
#         #   update the position of search agents - update network parameters - work on each pop
#         #           member
#         #       roll random value, if below self.z, roll new random params
#         #       eq 2.2 -> eq 2.3
#         #       two positions randomly selected from population
#         #       randomly combine the two positions
#         #       Check bounds (roll random if values are out of bounds)
#
#         pop = [self.create_solution() for _ in range(self.pop_size)]
#         pop, g_best = self.get_sorted_pop_and_global_best_solution(
#             pop,
#             id_fit=self.ID_FIT,  # which dimension to sort on i.e. which dimension has the fitness
#             # id_best=self.ID_MIN_PROB,  # which element is the best -> 0 (minimization problem)
#         )      # Eq.(2.6) - sort smell index
#
#         for epoch in range(self.epoch):
#             # get the spread of the fitness values (losses)
#             s = pop[0][self.ID_FIT] - pop[-1][self.ID_FIT] + self.eps  # avoid zero denom
#
#             # calculate the fitness weight of each slime mold
#             for i in range(0, self.pop_size):
#                 # Eq.(2.5)
#                 if i <= int(self.pop_size / 2):
#                     pop[i][self.ID_WEI] = 1 + uniform(0, 1, self.problem_size) * \ log10((pop[
#                     0][self.ID_FIT] - pop[i][self.ID_FIT]) / s + 1)
#                 else:
#                     pop[i][self.ID_WEI] = 1 - uniform(0, 1, self.problem_size) * \ log10((pop[
#                     0][self.ID_FIT] - pop[i][self.ID_FIT]) / s + 1)
#
#             a = arctanh(-((epoch + 1) / self.epoch) + 1)                        # Eq.(2.4)
#             b = 1 - (epoch + 1) / self.epoch
#
#             # Update the Position of search agents
#             for i in range(0, self.pop_size):
#                 if uniform() < self.z:  # Eq.(2.7)
#                     pos_new = uniform(self.lb, self.ub)
#                 else:
#                     p = tanh(abs(pop[i][self.ID_FIT] - g_best[self.ID_FIT]))    # Eq.(2.2)
#                     vb = uniform(-a, a, self.problem_size)                      # Eq.(2.3)
#                     vc = uniform(-b, b, self.problem_size)
#
#                     # two positions randomly selected from population,
#                     #   apply for the whole problem size instead of 1 variable
#                     id_a, id_b = choice(
#                        list(set(range(0, self.pop_size)) - {i}), 2, replace=False
#                        )
#
#                     pos_1 = g_best[self.ID_POS] + vb * \
#                             (pop[i][self.ID_WEI] * pop[id_a][self.ID_POS] - pop[id_b][self.ID_POS])
#                     pos_2 = vc * pop[i][self.ID_POS]
#                     # combine positions
#                     pos_new = where(uniform(0, 1, self.problem_size) < p, pos_1, pos_2)  #
#
#                 # Check bound and re-calculate fitness after each individual move
#                 pos_new = self.amend_position(pos_new)
#                 fit_new = self.obj_func(pos_new)
#                 pop[i][self.ID_POS] = pos_new
#                 pop[i][self.ID_FIT] = fit_new
#
#             # Sorted population and update the global best
#             pop, g_best = self.update_sorted_population_and_global_best_solution(pop,
#                 self.ID_MIN_PROB, g_best)  # Eq. 2.6
#             self.loss_train.append(g_best[self.ID_FIT])
#             if self.verbose:
#                 print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
#         return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
