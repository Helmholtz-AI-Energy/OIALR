import logging
import random
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

from . import mixing, utils

log = logging.getLogger(__name__)


class SVDSyncMultiheadAttention(nn.Module):
    r"""
    Almost all of this class is based on the torch standard MultiheadAttention class
    I have changed things to work with the SVD abstractions, but that is it.

    Daniel Coquelin - coquelin77

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
        start_q=None,
        start_k=None,
        start_v=None,
        start_in_proj=None,
        start_k_bias=None,
        start_v_bias=None,
        start_in_proj_bias=None,
        update_from_simga: bool = True,
        reinit_shapes=False,
        distributed_updates=True,
        inner_dim_init_ratio=1.0,
        random_sigma: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.update_from_simga = update_from_simga
        self.embed_dim = embed_dim

        self.inner_dim_init_ratio = inner_dim_init_ratio
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.reinit_shapes = reinit_shapes
        self.distributed_updates = distributed_updates
        assert self.head_dim * num_heads == self.embed_dim, "num_heads must be factor of embed_dim"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.randn((embed_dim, embed_dim), **factory_kwargs))
            self.q_inner_dim = torch.tensor(embed_dim * self.inner_dim_init_ratio, dtype=torch.int)

            self.q_u = Parameter(torch.empty((embed_dim, self.q_inner_dim), **factory_kwargs), requires_grad=False)
            self.q_s = Parameter(
                torch.empty((self.q_inner_dim, self.q_inner_dim), **factory_kwargs),
                requires_grad=True,
            )
            self.q_vh = Parameter(torch.empty((self.q_inner_dim, embed_dim), **factory_kwargs), requires_grad=False)
            self.q_trans = False

            self.k_proj_weight = Parameter(torch.randn((embed_dim, self.kdim), **factory_kwargs))
            if self.kdim > embed_dim:
                # u - kdim x embed, s - embed x embed, vh - embed x embed -> after trans is embed x kdim
                self.k_trans = True
                self.k_inner_dim = torch.tensor(embed_dim * self.inner_dim_init_ratio, dtype=torch.int)

                self.k_u = Parameter(
                    torch.empty((self.kdim, self.k_inner_dim), **factory_kwargs),
                    requires_grad=False,
                )
                self.k_s = Parameter(
                    torch.empty((self.k_inner_dim, self.k_inner_dim), **factory_kwargs),
                    requires_grad=True,
                )
                self.k_vh = Parameter(
                    torch.empty((self.k_inner_dim, embed_dim), **factory_kwargs),
                    requires_grad=False,
                )
            else:
                # u - embed x kdim, s - kdim x kdim, vh - kdim x kdim
                self.k_trans = False
                self.k_inner_dim = torch.tensor(self.kdim * self.inner_dim_init_ratio, dtype=torch.int)
                self.k_u = Parameter(
                    torch.empty((embed_dim, self.k_inner_dim), **factory_kwargs),
                    requires_grad=False,
                )
                self.k_s = Parameter(
                    torch.empty((self.k_inner_dim, self.k_inner_dim), **factory_kwargs),
                    requires_grad=True,
                )
                self.k_vh = Parameter(
                    torch.empty((self.k_inner_dim, self.kdim), **factory_kwargs),
                    requires_grad=False,
                )

            self.v_proj_weight = Parameter(torch.randn((embed_dim, self.vdim), **factory_kwargs))
            if self.vdim > embed_dim:
                # u - vdim x embed, s - embed x embed, vh - embed x embed -> after trans is embed x vdim
                self.v_trans = True
                self.v_inner_dim = torch.tensor(embed_dim * self.inner_dim_init_ratio, dtype=torch.int)
                self.v_u = Parameter(
                    torch.empty((self.vdim, self.v_inner_dim), **factory_kwargs),
                    requires_grad=False,
                )
                self.v_s = Parameter(
                    torch.empty((self.v_inner_dim, self.v_inner_dim), **factory_kwargs),
                    requires_grad=True,
                )
                self.v_vh = Parameter(
                    torch.empty((self.v_inner_dim, embed_dim), **factory_kwargs),
                    requires_grad=False,
                )
            else:
                # u - embed x vdim, s - vdim x vdim, vh - vdim x vdim
                self.v_trans = False
                self.v_inner_dim = torch.tensor(self.vdim * self.inner_dim_init_ratio, dtype=torch.int)
                self.v_u = Parameter(
                    torch.empty((embed_dim, self.v_inner_dim), **factory_kwargs),
                    requires_grad=False,
                )
                self.v_s = Parameter(
                    torch.empty((self.v_inner_dim, self.v_inner_dim), **factory_kwargs),
                    requires_grad=True,
                )
                self.v_vh = Parameter(
                    torch.empty((self.v_inner_dim, self.vdim), **factory_kwargs),
                    requires_grad=False,
                )
            self.q_p = Parameter(torch.empty(self.q_inner_dim, **factory_kwargs), requires_grad=True)
            self.k_p = Parameter(torch.empty(self.k_inner_dim, **factory_kwargs), requires_grad=True)
            self.v_p = Parameter(torch.empty(self.v_inner_dim, **factory_kwargs), requires_grad=True)
            self.register_parameter("in_proj_u", None)
            self.register_parameter("in_proj_s", None)
            self.register_parameter("in_proj_vh", None)
        else:
            self.in_proj_weight = Parameter(torch.randn((3 * embed_dim, embed_dim), **factory_kwargs))
            self.in_proj_inner_dim = torch.tensor(embed_dim * self.inner_dim_init_ratio, dtype=torch.int)
            # in_proj is always TS
            self.in_proj_u = Parameter(
                torch.empty((3 * embed_dim, self.in_proj_inner_dim), **factory_kwargs),
                requires_grad=False,
            )
            self.in_proj_s = Parameter(
                torch.empty((self.in_proj_inner_dim, self.in_proj_inner_dim), **factory_kwargs),
                requires_grad=True,
            )
            self.in_proj_vh = Parameter(
                torch.empty((self.in_proj_inner_dim, embed_dim), **factory_kwargs),
                requires_grad=False,
            )
            self.in_proj_p = Parameter(torch.empty(self.in_proj_inner_dim, **factory_kwargs), requires_grad=True)
            self.in_proj_trans = False
            self.register_parameter("q_u", None)
            self.register_parameter("q_s", None)
            self.register_parameter("q_vh", None)
            self.register_parameter("k_u", None)
            self.register_parameter("k_s", None)
            self.register_parameter("k_vh", None)
            self.register_parameter("v_u", None)
            self.register_parameter("v_s", None)
            self.register_parameter("v_vh", None)

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
        self.train_p = False

        # self._reset_parameters()
        self.sigma_cutoff_fraction = sigma_cutoff_fraction

        if not self._qkv_same_embed_dim:
            self.prev_uvh_q = None
            self.prev_uvh_k = None
            self.prev_uvh_v = None
            # self.q_p = nn.Parameter(torch.)
        else:
            self.prev_uvh_in_proj = None

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

            self.gen = torch.Generator(device=factory_kwargs["device"])
            self.gen.manual_seed(random.randint(0, 68719476735))
            self.sigma_generator = torch.Generator(device=factory_kwargs["device"])
            self.sigma_generator.manual_seed(random.randint(0, 68719476735))
            self.first_distribute_workload = True
            if random_sigma:
                log.debug(f"layer-local random seed: {self.gen.initial_seed()}")

            # reset the USVh vales for qkv/in
            if self._qkv_same_embed_dim:  # `in` case
                u, s, vh = torch.linalg.svd(self.in_proj_weight, full_matrices=False)
                if random_sigma:
                    s = torch.rand(s.shape[0], device=s.device, dtype=s.dtype, generator=self.gen) * s.max()

                self.in_proj_u = Parameter(u[:, : self.in_proj_inner_dim].contiguous(), requires_grad=False)
                self.in_proj_s = Parameter(torch.diag(s[: self.in_proj_inner_dim]).contiguous(), requires_grad=True)
                self.in_proj_vh = Parameter(vh[: self.in_proj_inner_dim].contiguous(), requires_grad=False)

                self.in_proj_weight = property(lambda self: self._get_in_proj())
                # del self.in_proj_weight
            else:
                for qkv in "qkv":
                    target = getattr(self, f"{qkv}_proj_weight")
                    if getattr(self, f"{qkv}_trans"):
                        target = target.T
                    u, s, vh = torch.linalg.svd(getattr(self, f"{qkv}_proj_weight"), full_matrices=False)
                    if random_sigma:
                        s = torch.rand(s.shape[0], device=s.device, dtype=s.dtype, generator=self.gen) * s.max()
                    # u, _ = torch.linalg.qr(torch.randn_like(u), mode="reduced")
                    # vh, _ = torch.linalg.qr(torch.randn_like(vh), mode="reduced")
                    setattr(
                        self,
                        f"{qkv}_u",
                        Parameter(u[:, : self.in_proj_inner_dim].contiguous(), requires_grad=False),
                    )
                    setattr(
                        self,
                        f"{qkv}_s",
                        Parameter(torch.diag(s[: self.in_proj_inner_dim]).contiguous(), requires_grad=True),
                    )
                    setattr(
                        self,
                        f"{qkv}_vh",
                        Parameter(vh[: self.in_proj_inner_dim].contiguous(), requires_grad=False),
                    )
                self.q_proj_weight = property(lambda self: self._get_q())
                self.k_proj_weight = property(lambda self: self._get_k())
                self.v_proj_weight = property(lambda self: self._get_v())
                # del self.q_proj_weight, self.k_proj_weight, self.v_proj_weight

        # internal random generator
        seed = torch.seed()
        if dist.is_initialized():
            seed = seed + dist.get_rank() + 1000
        # log.info(f"Internal seed for linear layer: {seed}")
        self.gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        self.gen.manual_seed(seed)

    def _reset_parameters(self):
        # if self._qkv_same_embed_dim:
        #     nn.init.xavier_uniform_(self.in_proj_weight)
        # else:
        #     nn.init.xavier_uniform_(self.q_proj_weight)
        #     nn.init.xavier_uniform_(self.k_proj_weight)
        #     nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def _get_device(self):
        if self._qkv_same_embed_dim:
            return self.in_proj_s.device
        else:
            return self.q_s.device

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super().__setstate__(state)

    # @torch.compile()
    def _get_q(self):
        if self.training:
            self.q_u.requires_grad = False
            self.q_s.requires_grad = True
            self.q_vh.requires_grad = False
        u, vh = self.q_u, self.q_vh
        s = self.q_s
        if self.train_p:
            self.q_p.requires_grad = True
            self.q_s.requires_grad = False
            ret = torch.linalg.multi_dot([u, self.q_p.view(-1, 1), s, vh])
        else:
            ret = torch.linalg.multi_dot([u, s, vh])
        return ret

    # @torch.compile()
    def _get_k(self):
        if self.training:
            self.k_u.requires_grad = False
            self.k_s.requires_grad = True
            self.k_vh.requires_grad = False
        u, vh = self.k_u, self.k_vh
        s = self.k_s
        if self.train_p:
            self.k_p.requires_grad = True
            self.k_s.requires_grad = False
            ret = torch.linalg.multi_dot([u, self.k_p.view(-1, 1), s, vh])
        else:
            w = torch.linalg.multi_dot([u, s, vh])
        ret = w.T if self.k_trans else w
        return ret

    # @torch.compile()
    def _get_v(self):
        if self.training:
            self.v_u.requires_grad = False
            self.v_s.requires_grad = True
            self.v_vh.requires_grad = False
        u, vh = self.v_u, self.v_vh
        s = self.v_s
        if self.train_p:
            self.v_p.requires_grad = True
            self.v_s.requires_grad = False
            ret = torch.linalg.multi_dot([u, self.v_p.view(-1, 1), s, vh])
        else:
            w = torch.linalg.multi_dot([u, s, vh])
        ret = w.T if self.v_trans else w
        return ret

    # @torch.compile()
    def _get_in_proj(self) -> Tensor:
        if not self._qkv_same_embed_dim:
            return None
        if self.training:
            self.in_proj_u.requires_grad = False
            self.in_proj_vh.requires_grad = False

        self.in_proj_s.requires_grad = True
        u = self.in_proj_u  # .detach()
        vh = self.in_proj_vh  # .detach()
        s = self.in_proj_s
        if self.train_p:
            self.in_proj_p.requires_grad = True
            self.in_proj_s.requires_grad = False
            ret = torch.linalg.multi_dot([u, self.in_proj_p.view(-1, 1), s, vh])
        else:
            ret = torch.linalg.multi_dot([u, s, vh])
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
        elif self.in_proj_s is not None and query.dtype != self.in_proj_s.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = (
                f"dtypes of query ({query.dtype}) and self.in_proj_s ({self.in_proj_s.dtype}) don't match"
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
        self.last_send_ranks[qkvin] = working_rank
        if working_rank != self.rank:
            # move on for now, coming back later
            # make sure to save the rank which did this layer
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
            # receive the info from the working process
            if self.distributed_updates:
                self.waits[qkvin]["inner_dim"] = dist.broadcast(
                    self.inner_dim_buffers[qkvin],
                    src=working_rank,
                    async_op=nonblocking,
                )
            # print('for waits')
            return

        # case 3: update the stable U and Vh from the full rank sigma matrix
        status = self._full_rank_sigma_update_usv_abs(qkvin, working_rank)
        # shapes are updated within above (as is slice update)
        perc, _, _ = self.get_perc_params()
        u, s, vh, sl = self._get_usvh_from_qkvin(qkvin)
        log.info(
            f"{name[-30:]}: Full rank update, csmean: {status['csmean']:.3f}, params: {perc * 100:.2f}, "
            f"\t[{u.shape[0]} {s.shape[0]} {vh.shape[1]}]",
        )

        # send K if it has changed
        # [in_proj, q, k, v]_inner_dim
        u, s, vh, sl = self._get_usvh_from_qkvin(qkvin)
        self.inner_dim_buffers[qkvin][0] = sl.to(torch.float)
        self.inner_dim_buffers[qkvin][1] = status["csmean"]
        self.inner_dim_buffers[qkvin][2] = status["change_k"]
        if not dist.is_initialized():
            return
        dev = self._get_device()
        for qkv in self.inner_dim_buffers:
            self.inner_dim_buffers[qkv] = self.inner_dim_buffers[qkv].to(dev)

        if self.distributed_updates:
            self.waits[qkvin]["inner_dim"] = dist.broadcast(
                self.inner_dim_buffers[qkvin],
                src=working_rank,
                async_op=nonblocking,
            )

    @torch.no_grad()
    def _full_rank_sigma_update_usv_abs(self, qkvin, working_rank=0):
        if self.rank == working_rank:
            log.debug("in full rank sigma update of usvh")
        # NOTE: no slicing because need the shapes to line up. self.s[self.k:, self.k:] should be 0?
        u, s, vh, _ = self._get_usvh_from_qkvin(qkvin)
        dtp = s.dtype
        s = s.to(torch.float32)
        usig, sig, vhsig = torch.linalg.svd(s)  # square mat, full mat arg doesnt matter
        usig = usig.to(dtp)
        sig = sig.to(dtp)
        vhsig = vhsig.to(dtp)

        holdu = u @ usig
        u.zero_()
        u.add_(holdu)
        holdvh = vhsig @ vh
        vh.zero_()
        vh.add_(holdvh)
        s.zero_()
        s.add_(torch.diag(sig))

        change_k = self._update_inner_dim_and_shapes_abs(qkvin)
        return {"csmean": 1.0, "change_k": change_k}

    @torch.no_grad()
    def _wait_inner_dim_reshape_bcast_usvh_abs(self, qkvin, nonblocking=True):
        # if wait_k is None -> K is the same -> optimizer is fine (shapes are the same)
        reset_optimizer = bool(self.inner_dim_buffers[qkvin][2].item())
        if self.last_send_ranks[qkvin] is None or not self.distributed_updates:
            return reset_optimizer, True
        if self.waits[qkvin]["inner_dim"] is not None:
            self.waits[qkvin]["inner_dim"].wait()
            self.waits[qkvin]["inner_dim"] = None

        if self.rank != self.last_send_ranks[qkvin]:
            # [in_proj, q, k, v]_inner_dim
            setattr(self, f"{qkvin}_inner_dim", self.inner_dim_buffers[qkvin][0].to(torch.int))
            self._update_usvh_shapes(qkvin)

        reset_optimizer = bool(self.inner_dim_buffers[qkvin][2].item())
        stable = self.inner_dim_buffers[qkvin][1] >= self.uvhthreshold
        self.inner_dim_buffers[qkvin] *= 0
        self._bcast_usvh_abs(qkvin=qkvin, src=self.last_send_ranks[qkvin], nonblocking=nonblocking)
        return reset_optimizer, stable

    def _bcast_usvh_abs(self, qkvin, src, nonblocking):
        if not dist.is_initialized() or not self.distributed_updates:
            return
        u, s, vh, _ = self._get_usvh_from_qkvin(qkvin)
        if not u.is_contiguous():
            u.set_(u.contiguous())
        if not s.is_contiguous():
            s.set_(s.contiguous())
        if not vh.is_contiguous():
            vh.set_(vh.contiguous())
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
        if self._qkv_same_embed_dim:
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
        if self._qkv_same_embed_dim:
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
        sdiag = s
        prevsl = sl.clone()
        # if getattr(self, f"uvh_stable_{qkvin}"):
        min_dim = int(vh.shape[-1] * 0.01)  # always TS
        # cutoff = sdiag[min_dim] * self.sigma_cutoff_fraction
        cutoff = s[0] * self.sigma_cutoff_fraction
        nz = torch.nonzero(sdiag < cutoff)

        if len(nz) == 0:
            # In this case ALL of the basis vectors are useful
            newsl = s.shape[0]
        elif nz[0] < prevsl * 0.5 and prevsl * 0.5 > min_dim:
            # add breaking, dont let it cut off more than 50% of the values
            newsl = int(prevsl * 0.5)
        elif nz[0].item() < min_dim:
            newsl = min_dim
        else:
            newsl = nz[0].item()

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
            s.set_(s[:sl, :sl].contiguous())
        else:
            u[:, sl:] *= 0
            vh[sl:] *= 0
            s[sl:, sl:].mul_(0)

    @torch.no_grad()
    def _get_usvh_from_qkvin(self, qkvin):
        u = getattr(self, f"{qkvin}_u")
        s = getattr(self, f"{qkvin}_s")
        vh = getattr(self, f"{qkvin}_vh")
        sl = getattr(self, f"{qkvin}_inner_dim")
        return u, s, vh, sl

    # @torch.compile()
    def get_perc_params_select(self, qkvin):
        u, vh = getattr(self, f"{qkvin}_u"), getattr(self, f"{qkvin}_vh")
        normal_params = u.shape[0] * vh.shape[1]
        # active_params = (self.u.shape[0] * self.k) + (self.k ** 2) + (self.k + self.vh.shape[1])
        trainable_params = getattr(self, f"{qkvin}_s").numel()
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

    def gather_distributed_factorizations(self, sigma_cutoff_fraction=0.2, name=None):
        if not self._qkv_same_embed_dim:
            for qkv in "qkv":
                u, s, vh, sl = self._get_usvh_from_qkvin(qkv)
                utils.collect_lr_reps_from_all(
                    local_u=u,
                    local_s=s,
                    local_vh=vh,
                    sigma_cutoff_fraction=sigma_cutoff_fraction,
                    set_usvh=True,
                    name=name + f".{qkv}",
                )
        else:
            u, s, vh, sl = self._get_usvh_from_qkvin("in_proj")
            utils.collect_lr_reps_from_all(
                local_u=u,
                local_s=s,
                local_vh=vh,
                sigma_cutoff_fraction=sigma_cutoff_fraction,
                set_usvh=True,
                name=name,
            )

    def distribute_workload(self, min_size_fraction=0.05, name=None):
        if not self._qkv_same_embed_dim:
            for qkv in "qkv":
                u, s, vh, sl = self._get_usvh_from_qkvin(qkv)
                utils.split_sigma_workload(
                    collected_u=u,
                    collected_s=s,
                    collected_vh=vh,
                    sigma_generator=self.sigma_generator,
                    min_size_fraction=min_size_fraction,
                    set_usvh=True,
                    name=name + f".{qkv}",
                )
        else:
            u, s, vh, sl = self._get_usvh_from_qkvin("in_proj")
            utils.split_sigma_workload(
                collected_u=u,
                collected_s=s,
                collected_vh=vh,
                sigma_generator=self.sigma_generator,
                min_size_fraction=min_size_fraction,
                set_usvh=True,
                name=name,
            )
        self.first_distribute_workload = False

    @torch.no_grad()
    def mix_sigma(self, method, *args, **kwargs):
        # NOTE: this will also update U/Vh as well
        # blur sigma with a specified method, only RBF available right now
        if not self._qkv_same_embed_dim:
            for qkv in "qkv":
                u, s, vh, sl = self._get_usvh_from_qkvin(qkv)
                mixing.mix_sigma(*args, u=u, s=s, vh=vh, method=method, generator=self.gen, **kwargs)
        else:
            u, s, vh, sl = self._get_usvh_from_qkvin("in_proj")
            mixing.mix_sigma(*args, u=u, s=s, vh=vh, method=method, generator=self.gen, **kwargs)

    @torch.no_grad()
    def _start_p_training_qkvin(self, qkv):
        # set P to be 1/sz to emulate an average, will let the network learn how to use this
        s = getattr(self, f"{qkv}_s")
        p = getattr(self, f"{qkv}_p")
        p.zero_()
        sz = dist.get_world_size() if dist.is_initialized() else 1
        p.add_(1 / sz)
        p.requires_grad = True
        self.train_p = True
        s.requires_grad = False

    @torch.no_grad()
    def _update_sp_train_normally_qkvin(self, qkv):
        # revert to normal training for the layer
        # update sigma from P and start training normally
        s = getattr(self, f"{qkv}_s")
        p = getattr(self, f"{qkv}_p")
        hold = p.view(-1, 1) * s
        s.zero_()
        s.add_(hold)
        p.zero_()
        p.add_(1)
        p.requires_grad = False
        self.train_p = False
        s.requires_grad = True

    @torch.no_grad()
    def start_p_training(self):
        if not self._qkv_same_embed_dim:
            for qkv in "qkv":
                self._start_p_training_qkvin(qkv)
        else:
            self._start_p_training_qkvin("in_proj")

    @torch.no_grad()
    def update_sp_train_normally(self):
        if not self._qkv_same_embed_dim:
            for qkv in "qkv":
                self._update_sp_train_normally_qkvin(qkv)
        else:
            self._update_sp_train_normally_qkvin("in_proj")
