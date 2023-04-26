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


class ShuffleVhROpt(object):
    def __init__(
        self,
        model: nn.Module,
        sync_frequency: int = 100,
    ):
        """
        based of the idea of shuffling the R (of QR) or Vh (of SVD) to slightly shift the network in
        the loss space to find better gradients.

        TODO:
            - [ ] determine how often to shuffle
            - [ ] should the networks be trained together? or apart?
            - [ ] if apart, should the networks be synchronized
            - [ ] since the networks are now in different places, should they have have averaged
            gradients? would be similar to blurring the network
            - [ ] should the network be overwritten to enable fixing the Q/U and not training it

        Parameters
        ----------
        model
        sync_frequency
        """
        self.model = model
        # self.config = config
        self.model_buffers = dict()
        self.model_buffers_waits = dict()
        self.best_parameters = dict()
        self.best_parameters_waits = dict()

        for np, p in self.model.named_parameters():
            # if p.requires_grad:
            self.model_buffers[f"{np}"] = torch.zeros_like(p.data)
            self.best_parameters[f"{np}"] = torch.zeros_like(p.data)

        self.global_rank = dist.get_rank()
        self.global_size = dist.get_world_size()

        # ============================= init params ====================================

        # ============================= control params ====================================
        self.sync_frequency = sync_frequency
        # ================================ explore params ========================================
        self.uniform_n1_p1 = torch.distributions.uniform.Uniform(-1, 1)
        self.gaussian = torch.distributions.normal.Normal(0, 1)

    def step(self, loss: torch.Tensor) -> torch.nn.Module:
        pass

    # ------------------------------ SVD functions -------------------------------------------------
    def _svd_of_blocks(self):
        pass

    def _svd_col_slices(self):
        pass

    def _svd_merge_slices(self):
        pass

    def _svd_block_merge(self):
        pass

    # ------------------------------ Comm + tree functions -----------------------------------------
