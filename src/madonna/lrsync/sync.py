import logging
import time
from functools import reduce

import torch
import torch.distributed as dist
import torch.nn as nn

log = logging.getLogger(__name__)


_fib = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]


@torch.no_grad()
def sync_full_rank_model_in_low_rank(model: torch.nn.Module, config, vecs_to_send=10):
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
def sync_low_rank_model(svd_model_dict, vecs_to_send=1, method="random", sort=False):
    # method options: topn, random
    if method not in ["all", "topn", "random", "fib"]:
        raise ValueError(f"method ({method}) not in options [topn, random, fib]")
    # svd_model_dict -> name/layer name = parameter OR [u, s, vh]
    if not dist.is_initialized():
        raise RuntimeError("torch dist is not initialized and it must be for this to do anything.")

    rank = dist.get_rank()
    ws = dist.get_world_size()
    # if ws > 2:  # TODO: full tree reduction after small scale tests
    #     raise NotImplementedError("do it properly Daniel")

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

        # TODO: make method for when there are less vecs to send then selected by arg
        #       i.e. vecs_to_send > inner dim
        if method == "topn":
            sel_vecs = torch.arange(vecs_to_send, device=loc_s.device)[:vecs_to_send]
        if method == "all":
            sel_vecs = slice(None)
        elif method == "fib":
            valid_fibs = torch.tensor(_fib, dtype=torch.int, device=loc_s.device)
            sel_vecs = valid_fibs[valid_fibs < loc_s.shape[0]][:vecs_to_send]
        else:  # method == 'random'
            sel_vecs = torch.randint(0, loc_s.shape[0], (vecs_to_send,), device=loc_s.device)
        loc_u1 = loc_u[:, sel_vecs].contiguous()
        loc_s1 = loc_s[:, sel_vecs][sel_vecs].contiguous()
        loc_vh1 = loc_vh[sel_vecs].contiguous()
        # inner dim may change between ranks, but outer dims are the same
        #   number of vec/val groups to send should be maximum,
        #   adding zeros will only result in extra comp, wont hurt algo
        # shapes : u1 x s, s x s, s x vh2
        #   we can be lazy, and do an allreduce where each mat (u,s,vh) is [rank, *shape]
        #       with high-speed networks this is invisible, but with slower networks, it has tons of overhead
        # if we want efficiency on all networks, we need a series of bcasts
        #   in this case, we should collapse the matrices into a single matrix of size [u1 x (s + s**2 + s * vh2)]
        # however, all the send matrices need to be the same size,
        # TODO: optimizer bcast sending if its a bottleneck in the future
        for r in range(ws):  # loop over ranks to bcast from each
            buff_u = torch.zeros_like(loc_u1)
            buff_s = torch.zeros_like(loc_s1)
            buff_vh = torch.zeros_like(loc_vh1)
            if r == rank:
                buff_u.add_(loc_u1)
                buff_s.add_(loc_s1)
                buff_vh.add_(loc_vh1)
            wait_u = dist.broadcast(buff_u, src=r, async_op=True)
            wait_s = dist.broadcast(buff_s, src=r, async_op=True)
            wait_vh = dist.broadcast(buff_vh, src=r, async_op=True)

            if r != rank:
                wait_u.wait()
                cat_u = torch.cat([loc_u, buff_u], dim=1)

                wait_s.wait()
                cat_s = torch.eye(loc_s.shape[0] + buff_s.shape[0], device=loc_s.device, dtype=loc_s.dtype)
                cat_s[: loc_s.shape[0], : loc_s.shape[1]] = loc_s
                cat_s[loc_s.shape[0] :, loc_s.shape[1] :] = buff_s

                wait_vh.wait()
                cat_vh = torch.cat([loc_vh, buff_vh], dim=0)
            else:
                wait_u.wait()
                wait_s.wait()
                wait_vh.wait()

        if sort:
            # sort the vecs based on the values of sigma
            # sorting does not change what the math says, the order of the vectors is chosen be default anyway
            vals, inds = torch.sort(cat_s.diag(), descending=True)
            cat_s = torch.diag(vals)
            cat_u = cat_u[:, inds]
            cat_vh = cat_vh[inds]
        loc_u.data.set_(cat_u)
        loc_u.grad = None
        loc_s.data.set_(cat_s)
        loc_s.grad = None
        loc_vh.data.set_(cat_vh)
        loc_vh.grad = None

        # # trading USVh to get from partner ------------
        # part_u = torch.zeros_like(loc_u1)
        # part_s = torch.zeros_like(loc_s1)
        # part_vh = torch.zeros_like(loc_vh1)
        # usend = dist.P2POp(dist.isend, loc_u1, peer=partner, tag=tag)
        # ssend = dist.P2POp(dist.isend, loc_s1, peer=partner, tag=tag + 1)
        # vhsend = dist.P2POp(dist.isend, loc_vh1, peer=partner, tag=tag + 2)
        # urecv = dist.P2POp(dist.irecv, part_u, peer=partner, tag=partner_tag)
        # srecv = dist.P2POp(dist.irecv, part_s, peer=partner, tag=partner_tag + 1)
        # vhrecv = dist.P2POp(dist.irecv, part_vh, peer=partner, tag=partner_tag + 2)
        # reqs = dist.batch_isend_irecv([usend, urecv, ssend, srecv, vhsend, vhrecv])
        # for req in reqs:
        #     req.wait()
        # tag += 3
        # partner_tag += 3

        # # do merging -----------------------------------
        # # print(loc_u.shape, part_u.shape)
        # cat_u = torch.cat([loc_u, part_u], dim=1)
        # # print(loc_s.shape, part_s.shape)
        # cat_s = torch.eye(loc_s.shape[0] + part_s.shape[0], device=loc_s.device, dtype=loc_s.dtype)
        # cat_s[: loc_s.shape[0], : loc_s.shape[1]] = loc_s
        # cat_s[loc_s.shape[0] :, loc_s.shape[1] :] = part_s
        # # cat_s = torch.cat([loc_s, part_s], dim=0)
        # # print(loc_vh.shape, part_vh.shape)
        # cat_vh = torch.cat([loc_vh, part_vh], dim=0)
        # loc_u.data.set_(cat_u)
        # loc_u.grad = None
        # loc_s.data.set_(cat_s)
        # loc_s.grad = None
        # loc_vh.data.set_(cat_vh)
        # loc_vh.grad = None
    log.info(f"Time to sync: {time.perf_counter() - t0}")
