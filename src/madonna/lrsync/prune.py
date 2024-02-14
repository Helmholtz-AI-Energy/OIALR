import logging
import time
from typing import Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


from torch.nn.utils import prune


class FeatureAveragePruning(prune.BasePruningMethod):
    PRUNING_TYPE = "structured"

    def __init__(self, threshold=None, mdim=1, perc=None, existing_mask=None):
        self.threshold = threshold
        self.mdim = mdim
        self.dim = 0
        self.existing_mask = existing_mask
        self.perc = perc
        self.feature_mask = None

    def compute_mask(self, tensor, default_mask):
        if self.existing_mask is not None:
            # early out for cases where we want to give the mask as an input
            # used for bias params which are pruned based on the weight features
            return self.existing_mask

        # Calculate the average value of features in the weight tensor
        # the abs be before the mean, see Sobel operators
        if self.perc:
            threshold = torch.quantile(tensor.abs().mean(dim=self.mdim), self.perc)
        else:
            threshold = self.threshold
        average_features = tensor.abs().mean(dim=self.mdim)

        # Create a mask that preserves the weights and biases whose average features are above the threshold
        # i.e. True -> keep, False -> prune
        m1 = average_features >= threshold
        self.feature_mask = m1
        mask = torch.zeros_like(tensor)
        print(
            f"{average_features.mean()}, {average_features.min()}, {average_features.max()}, features cut: {(~m1).sum()}",
        )
        mask[m1] = 1
        return mask


def _prune_module(module, threshold, last_layer_name, perc=None):
    for name, child in module.named_children():
        if name == last_layer_name:
            continue
        if isinstance(child, nn.Linear):
            # pruning_method = FeatureAveragePruning(threshold, dim=1)
            pruning_method = FeatureAveragePruning.apply(
                child,
                name="weight",
                threshold=threshold,
                mdim=1,
                perc=perc,
            )
            # prune.custom_from_mask(child, name="weight", mask=pruning_method)
            # Use the weight mask to prune the bias
            if hasattr(child, "bias") and child.bias is not None:
                # bias_mask = pruning_method.compute_mask(child.weight, None)[0]
                try:
                    bias_mask = pruning_method.feature_mask
                except AttributeError:
                    bias_mask = pruning_method._pruning_methods[-1].feature_mask
                # print(f"bias_mask shape: {bias_mask.shape}")
                # prune.custom(child, name="bias", mask=bias_mask, param_name="bias")
                pruning_method = FeatureAveragePruning.apply(
                    child,
                    name="bias",
                    threshold=threshold,
                    existing_mask=bias_mask,
                )

        elif isinstance(child, nn.Conv2d):
            # pruning_method = FeatureAveragePruning(threshold, dim=(1, 2, 3))
            pruning_method = FeatureAveragePruning.apply(
                child,
                name="weight",
                threshold=threshold,
                mdim=(1, 2, 3),
                perc=perc,
            )
            # prune.custom_from_mask(child, name="weight", mask=pruning_method)
            if hasattr(child, "bias") and child.bias is not None:
                # bias_mask = pruning_method.compute_mask(child.weight, None)[0]
                try:
                    bias_mask = pruning_method.feature_mask
                except AttributeError:
                    bias_mask = pruning_method._pruning_methods[-1].feature_mask
                # print(f"bias_mask shape: {bias_mask.shape} {child.bias.shape}")
                pruning_method = FeatureAveragePruning.apply(
                    child,
                    name="bias",
                    threshold=threshold,
                    existing_mask=bias_mask,
                )
                # prune.custom(child, name="bias", mask=bias_mask[0], param_name="bias")

        elif isinstance(child, nn.MultiheadAttention):
            # Prune the weight matrices in the multi-head attention layer
            for name, param in child.named_parameters():
                if "weight" in name:
                    # pruning_method = FeatureAveragePruning(threshold)
                    FeatureAveragePruning.apply(child, name="weight", threshold=threshold, mdim=1, perc=perc)
                    # prune.custom_from_mask(child, name=name, mask=pruning_method)

        # Recursively apply pruning to child modules
        _prune_module(child, threshold, last_layer_name, perc)


def prune_model(model, threshold, last_layer_name, perc=None):
    _prune_module(model, threshold, last_layer_name, perc=perc)
