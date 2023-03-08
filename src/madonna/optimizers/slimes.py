import torch
import torch.distributed as dist
import torch.nn as nn


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
        for nc, c in self.model.named_children():
            if hasattr(c, "reset_parameters"):
                c.reset_parameters()
            for np, p in c.named_parameters():
                self.model_buffers[f"{nc}-{np}"] = torch.zeros_like(p.data)
            # self.model_buffers[c] = torch.zeros_like(p)
            # self.model_buffers_waits[c] = None

    @torch.no_grad()
    def init_model(self):
        print(self.chaotic_init)
        if not self.chaotic_init:
            for c in self.model.children():
                if hasattr(c, "reset_parameters"):
                    c.reset_parameters()
            return
        if self.rank == 0:
            tag = 0
            # TODO: what to do about the requires grad stuff? that just means its training?
            for c in self.model.children():
                print(c)
                if hasattr(c, "reset_parameters"):
                    c.reset_parameters()
                for n, p in c.named_parameters():
                    print(f"sending {n} to rank 1")
                    dist.send(p.data, dst=1, tag=tag)
                    tag += 1
            return
        # rest of the ranks
        tag = 0
        # TODO: what to do about the requires grad stuff? that just means its training?
        for nc, c in self.model.named_children():
            #if hasattr(c, "reset_parameters"):
                # c.reset_parameters()
            for np, p in c.named_parameters():
                self.model_buffers[f"{nc}-{np}"] += p
                print(f"recv {np} from rank {self.rank - 1}, sending to {self.rank + 1}")
                dist.recv(self.model_buffers[f"{nc}-{np}"], src=self.rank - 1, tag=tag)

                hold = self.model_buffers[f"{nc}-{np}"]
                p.set_(self.chaotic_factor * hold * (1 - hold))
                if self.rank < self.size - 1:
                    dist.send(p.data, dst=self.rank + 1, tag=tag)
                tag += 1
        #print(self.model)

    def step(self, fitness):
        # randomly roll parameter combinations
        # for loop over number of epochs:
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
