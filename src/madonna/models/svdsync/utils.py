import logging
import time

import torch
import torch.distributed as dist
import torch.nn as nn

log = logging.getLogger(__name__)


def split_sigma_workload(
    collected_u: torch.Tensor,
    collected_s: torch.Tensor,
    collected_vh: torch.Tensor,
    sigma_generator: torch.Generator,
    min_size_fraction: float,
    set_usvh: bool = True,
):
    # assumptions: - sigma is diagonal and full-rank
    #              - collection of all LR reps has already been done
    # TODO: what happens when there are not enough vectors in
    # -1: collect sigmas from all ranks
    #   are the shapes known??
    # 0: join all sigmas together
    # ----- previous 2 are done with the collection function -----
    # 1: deterine number of vecs to keep on each rank
    # 2: shuffle/sort sigma
    # 3: select new vec to keep on each rank
    # 4: update u/s/vh with the selected vectors and reset their shapes
    # NOTE: need to reset optimizer states after this
    # NOTE: sigma is diagonal here, need to apply mixing after this if desired

    if not dist.is_initialized():
        return
    collected_s_diag = collected_s.diag()
    fact = {"device": collected_s_diag.device, "dtype": collected_s_diag.dtype}
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # unify the generator for the random permutation in step 4
    genstate = sigma_generator.get_state().to(device=collected_s_diag.device, dtype=torch.float)
    dist.broadcast(genstate, src=0)
    sigma_generator.set_state(genstate.type(torch.ByteTensor))

    min_num_vecs = int(collected_vh.shape[1] * min_size_fraction)
    # upper limit on number of vecs to send is the number of vecs across the process space
    #   in the frist iteration through this, that would be the number of diag elements in sigma
    total_vecs = collected_s_diag.shape[0]

    # 3 determine number of vecs to keep locally
    num_vecs_to_get = total_vecs // world_size
    vecs_to_get_all = torch.zeros(world_size, device=collected_s_diag.device, dtype=torch.int)
    vecs_to_get_all += num_vecs_to_get
    vecs_to_get_all[: total_vecs % world_size] += 1  # deal with remainer on lower ranks

    # 4: shuffle sigma -> no sort
    #       need to know where everything is. random works too (hopefully)
    # NOTE: will be effected if the seeds are different!!
    inds = torch.randperm(total_vecs, device=collected_s_diag.device, dtype=torch.int, generator=sigma_generator)

    # 5: select new elements to keep locally
    rank_inds_list = []
    rank_inds_list.append(inds[: vecs_to_get_all[0]])
    for r in range(1, world_size):
        rank_inds_list.append(inds[vecs_to_get_all[:r].sum() : vecs_to_get_all[: r + 1].sum()])

    new_sigma = collected_s_diag[rank_inds_list[rank]].to(**fact)
    new_vh = collected_vh[rank_inds_list[rank]].to(**fact)
    new_u = collected_u[:, rank_inds_list[rank]].to(**fact)

    if new_sigma.shape[0] < min_num_vecs:
        to_gen = min_num_vecs - new_sigma.shape[0]
        log.info(f"Generating {to_gen} extra orthogonal vectors to fill in extra space")
        # in this case, there are not enough vectors (hyperparam)
        #   if there are < 10 vectors or <5% of posibilities, then issues can arrise (maybe)
        # TODO: test me / determine if worth it
        # need to add orthogonal vectors to u and vh, random values for sigma (use QR)
        holdu = torch.randn((collected_u.shape[0], to_gen), **fact)
        new_u_additional, _ = torch.linalg.qr(holdu)  # mode=reduced by default
        holdvh = torch.randn((collected_vh.shape[1], to_gen), **fact)
        new_vh_additional, _ = torch.linalg.qr(holdvh)  # mode=reduced by default
        new_vh_additional = new_vh_additional.T
        sigma_additional = torch.rand(to_gen, **fact)
        new_vh = torch.cat([new_vh, new_vh_additional], dim=0)
        new_sigma = torch.cat([new_sigma, sigma_additional], dim=0)
        new_u = torch.cat([new_u, new_u_additional], dim=1)
    if set_usvh:
        collected_u.set_(new_u)
        collected_s.set_(torch.diag(new_sigma))
        collected_vh.set_(new_vh)
    else:
        return new_u, new_sigma, new_vh
    # TODO: return u/s/vh or just set it here?
    # return new_u, new_sigma, new_vh


def collect_lr_reps_from_all(
    local_u: torch.Tensor,
    local_s: torch.Tensor,
    local_vh: torch.Tensor,
    sigma_cutoff_fraction: float,
    set_usvh: bool = True,
):
    if not dist.is_initialized():
        return
    updateu, new_s, updatevh = torch.linalg.svd(local_s, full_matrices=False)
    new_u = local_u @ updateu
    new_vh = updatevh @ local_vh
    if set_usvh:
        local_s.zero_()
        local_u.zero_()
        local_vh.zero_()
    fact = {"device": local_s.device, "dtype": local_s.dtype}

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # self.u.add_(new_u)
    # self.s.add_(torch.diag(new_s))
    # self.vh.add_(new_vh)

    # 2. create a holding U/S/Vh for everything, then allreduce into it
    shapes = torch.zeros(world_size, dtype=torch.int, device=local_s.device)
    shapes[rank] = new_s.shape[0]
    dist.all_reduce(shapes)  # sum everything up
    shapes = [0] + shapes.cumsum(dim=0).tolist()
    total_vecs = shapes[-1]

    # 2. join all sigmas together
    full_sigma = torch.zeros(total_vecs, **fact)
    full_sigma[shapes[rank] : shapes[rank + 1]] = new_s
    wait_sigma = dist.all_reduce(full_sigma, async_op=True)  # sum op is default
    # send vh -> smaller
    full_vh = torch.zeros((total_vecs, new_vh.shape[1]), **fact)
    full_vh[shapes[rank] : shapes[rank + 1]] = new_vh
    wait_vh = dist.all_reduce(full_vh, async_op=True)  # smaller than U -> send first
    # send U
    full_u = torch.zeros((new_u.shape[0], total_vecs), **fact)
    full_u[:, shapes[rank] : shapes[rank + 1]] = new_u
    wait_u = dist.all_reduce(full_u, async_op=True)
    # 3. sort the singular vecs
    wait_sigma.wait()
    sort_sigma, inds = torch.sort(full_sigma, descending=True)
    # 4. throw out the small values (cutoff?)
    if sigma_cutoff_fraction is not None and sigma_cutoff_fraction < 1.0:
        cutoff = sort_sigma[0] * sigma_cutoff_fraction
        # TODO: should there be the min_dim here??
        # min_dim = int(full_vh.shape[-1] * 0.01)  # always TS
        # cutoff = s[min_dim] * self.sigma_cutoff_fraction
        nz = torch.nonzero(sort_sigma < cutoff)
        if len(nz) == 0:
            # In this case ALL of the basis vectors are useful
            k = sort_sigma.shape[0]
        # elif nz[0].item() < min_dim:
        #     newk = min_dim
        else:
            k = nz[0].item()
        sort_sigma = sort_sigma[:k]
        inds = inds[:k]

    # 5. wait for Vh/U and then re-order those to match sigma
    #   remember: cols of U are the vecs, while rows of Vh are vecs
    wait_vh.wait()
    full_vh = full_vh[inds]
    wait_u.wait()
    full_u = full_u[:, inds]
    if set_usvh:
        local_u.set_(full_u)
        local_s.set_(full_sigma.diag())
        local_vh.set_(full_vh)
    else:
        return full_u, full_sigma, full_vh
