import logging
import time
from functools import reduce

import torch
import torch.distributed as dist
import torch.nn as nn

log = logging.getLogger(__name__)


# _fib = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]


@torch.no_grad()
def sync_model_in_low_rank(model: torch.nn.Module, config, vecs_to_send=10):
    if not dist.is_initialized():
        raise RuntimeError("torch dist is not initialized and it must be for this to do anything.")

    rank = dist.get_rank()
    ws = dist.get_world_size()
    if ws > 2:  # TODO: full tree reduction after small scale tests
        raise NotImplementedError("do it properly Daniel")

    # TODO: async for speed
    partner = (rank + 1) % 2
    tag = rank * 1000  # todo: make this with an offset for the size of the network and the world size
    partner_tag = partner * 1000
    t0 = time.perf_counter()
    torch.cuda.set_device(rank % 4)
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # what to do with biases? they are 1D, easiest solution is to average them...
        if p.ndim == 1:
            # dist.all_reduce(p, dist.ReduceOp.AVG, async_op=False)
            continue
        # pruning should be done only for conv/lin/atten
        if p.ndim != 2:
            og_shape = p.shape
            w = p.view(og_shape[0], -1)
        else:  # p.ndim == 2:
            # DOING PRUNING HERE!!
            # these are linear layers (assumed...)
            # print(p.sum(dim=0), p.shape)
            og_shape = p.shape
            w = p
            # raise ValueError
        w, trans = (w, False) if w.shape[0] >= w.shape[1] else (w.T, True)
        # TODO: make this better then just saying < 10
        if w.shape[1] < 10:
            # dist.all_reduce(p, dist.ReduceOp.AVG, async_op=False)
            continue
        # print(f'{n} before: {p.mean()} {p.min()} {p.max()} {p.std()}')
        # print(f"start {n} {p.shape} {og_shape}")
        loc_u, loc_s, loc_vh = torch.linalg.svd(w, full_matrices=False)
        # cut an amount of the vals (from config, need to be changed later!!!)
        # TODO: remove this logging and set up cutoff to be based on sigma distribution
        # cutoff = int(loc_s.shape[0] * config.training.sync.cutoff_fraction)
        cutoff = vecs_to_send
        # log.info(f"Dist of sigma: -1/0 -> {n[-40:]}: {loc_s[-1] / loc_s[0]}")
        loc_u1, loc_s1, loc_vh1 = (
            loc_u[:, :cutoff].contiguous(),
            loc_s[:cutoff].contiguous(),
            loc_vh[:cutoff].contiguous(),
        )

        # trading USVh to get from partner ------------
        part_u = torch.zeros_like(loc_u1)
        part_s = torch.zeros_like(loc_s1)
        part_vh = torch.zeros_like(loc_vh1)
        usend = dist.P2POp(dist.isend, loc_u1, peer=partner, tag=tag)
        ssend = dist.P2POp(dist.isend, loc_s1, peer=partner, tag=tag + 1)
        vhsend = dist.P2POp(dist.isend, loc_vh1, peer=partner, tag=tag + 2)
        urecv = dist.P2POp(dist.irecv, part_u, peer=partner, tag=partner_tag)
        srecv = dist.P2POp(dist.irecv, part_s, peer=partner, tag=partner_tag + 1)
        vhrecv = dist.P2POp(dist.irecv, part_vh, peer=partner, tag=partner_tag + 2)
        reqs = dist.batch_isend_irecv([usend, urecv, ssend, srecv, vhsend, vhrecv])
        for req in reqs:
            req.wait()
        tag += 3
        partner_tag += 3

        # do merging -----------------------------------
        cat_u = torch.cat([loc_u, part_u], dim=1)
        cat_s = torch.cat([loc_s, part_s], dim=0)
        cat_vh = torch.cat([loc_vh, part_vh], dim=0)
        # print(f'before multidot {cat_u.shape} {cat_s.shape} {cat_vh.shape}')
        # print(cat_s)
        new_w = torch.linalg.multi_dot([cat_u, torch.diag(cat_s), cat_vh])
        # TODO: why does multi dot fail here??
        # hold = torch.diag(cat_s) @ cat_vh
        # print('h')
        # new_w = cat_u @ torch.diag(cat_s) @ cat_vh

        # print(f"after multidot")
        if trans:
            new_w = new_w.T
        # TODO: fixme to work with view -> size and stride is wrong
        try:
            w = new_w.view(og_shape)
        except RuntimeError:
            w = new_w.reshape(og_shape)
        p.mul_(0)
        p.add_(w)
        # print(f'{n} after: {p.mean()} {p.min()} {p.max()} {p.std()}')
        # print(f'end of {n}')

    log.info(f"Time to sync: {time.perf_counter() - t0}")


def get_param_dict_with_svds(model: nn.Module) -> dict:
    param_dict = {}
    for n, p in model.named_parameters():
        if n.endswith(".u"):
            names = n.split(".")
            lay = reduce(getattr, names[:-1], model.model)
            s = lay.s
            vh = lay.vh
            # print(p.shape, s.shape, vh.shape)
            param_dict[".".join(names[:-1])] = [p, s, vh]  # p is the
        elif n.endswith((".s", ".vh")):
            continue
        else:
            param_dict[n] = p
    return param_dict


@torch.no_grad()
def sync_topn_singulars_oialr(svd_model_dict, vecs_to_send=1, method="random"):
    # method options: topn, random
    if method not in ["topn", "random"]:
        raise ValueError(f"method ({method}) not in options [topn, random]")
    # svd_model_dict -> name/layer name = parameter OR [u, s, vh]
    if not dist.is_initialized():
        raise RuntimeError("torch dist is not initialized and it must be for this to do anything.")

    rank = dist.get_rank()
    ws = dist.get_world_size()
    if ws > 2:  # TODO: full tree reduction after small scale tests
        raise NotImplementedError("do it properly Daniel")

    # TODO: async for speed
    partner = (rank + 1) % 2
    tag = rank * 10000  # todo: make this with an offset for the size of the network and the world size
    partner_tag = partner * 10000
    t0 = time.perf_counter()
    torch.cuda.set_device(rank % 4)

    for n in svd_model_dict:
        params = svd_model_dict[n]
        if len(params) != 3:
            if not params.requires_grad:
                continue
            dist.all_reduce(params, dist.ReduceOp.AVG, async_op=False)
            continue
        loc_u, loc_s, loc_vh = params  # split the params
        # using the OIALR formulation, S might be dense...but it should be (mostly) ordered still

        if method == "topn":
            loc_u1 = loc_u[:, :vecs_to_send].contiguous()
            loc_s1 = loc_s[:vecs_to_send, :vecs_to_send].contiguous()
            loc_vh1 = loc_vh[:vecs_to_send].contiguous()
        else:  # method == 'random'
            rndms = torch.randint(0, loc_s.shape[1], (vecs_to_send,), device=loc_s.device)
            loc_u1 = loc_u[:, rndms].contiguous()
            loc_s1 = loc_s[:, rndms][vecs_to_send].contiguous()
            loc_vh1 = loc_vh[rndms].contiguous()
        # loc_u2, loc_s2, loc_vh2 = loc_u[:, topn:].contiguous(), loc_s[topn:, topn:].contiguous(), loc_vh[topn:].contiguous()

        # trading USVh to get from partner ------------
        part_u = torch.zeros_like(loc_u1)
        part_s = torch.zeros_like(loc_s1)
        part_vh = torch.zeros_like(loc_vh1)
        usend = dist.P2POp(dist.isend, loc_u1, peer=partner, tag=tag)
        ssend = dist.P2POp(dist.isend, loc_s1, peer=partner, tag=tag + 1)
        vhsend = dist.P2POp(dist.isend, loc_vh1, peer=partner, tag=tag + 2)
        urecv = dist.P2POp(dist.irecv, part_u, peer=partner, tag=partner_tag)
        srecv = dist.P2POp(dist.irecv, part_s, peer=partner, tag=partner_tag + 1)
        vhrecv = dist.P2POp(dist.irecv, part_vh, peer=partner, tag=partner_tag + 2)
        reqs = dist.batch_isend_irecv([usend, urecv, ssend, srecv, vhsend, vhrecv])
        for req in reqs:
            req.wait()
        tag += 3
        partner_tag += 3

        # do merging -----------------------------------
        # print(loc_u.shape, part_u.shape)
        cat_u = torch.cat([loc_u, part_u], dim=1)
        # print(loc_s.shape, part_s.shape)
        cat_s = torch.eye(loc_s.shape[0] + part_s.shape[0], device=loc_s.device, dtype=loc_s.dtype)
        cat_s[: loc_s.shape[0], : loc_s.shape[1]] = loc_s
        cat_s[loc_s.shape[0] :, loc_s.shape[1] :] = part_s
        # cat_s = torch.cat([loc_s, part_s], dim=0)
        # print(loc_vh.shape, part_vh.shape)
        cat_vh = torch.cat([loc_vh, part_vh], dim=0)
        loc_u.data.set_(cat_u)
        loc_u.grad = None
        loc_s.data.set_(cat_s)
        loc_s.grad = None
        loc_vh.data.set_(cat_vh)
        loc_vh.grad = None
    log.info(f"Time to sync: {time.perf_counter() - t0}")
