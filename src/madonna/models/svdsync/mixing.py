import torch


def mix_sigma(u, s, vh, method: str = "exp", generator=None, *args, **kwargs):
    if method == "rbf":
        return rbf_update_usvh_mix_sigma(*args, u=u, s=s, vh=vh, generator=generator, **kwargs)
    elif method == "exp":
        return exp_update_usvh_mix_sigma(u=u, s=s, vh=vh, **kwargs)
    elif method is None:
        pass
    else:
        return NotImplementedError(f"blurring method ({method}) not implemented, write it and add switch here")


def rbf_update_usvh_mix_sigma(u, s, vh, sigma=1.0, neigh=0.05, generator=None):
    # NOTE: this will update U/S/Vh IN PLACE!!!
    # TODO: fix me!! this is not showing great things
    #       ISSUE:
    # """
    # this will not make the same mixing for all vars along the whole matrix. see below
    #
    # ```python
    #     r = torch.rand(sz, sz, device="cpu", dtype=torch.float)
    #     # using RBF formulation
    #     r = (r * 1) / (sz * 0.1)  # torch.arange(sz, device=r.device)).T / sz
    #     sigma = 1.0  # 707106781  # hyperparam
    #     neigh = 1.0
    #     m = (sigma**2 * torch.exp(-((torch.cdist(r, r, p=2) ** 2) / (2 * neigh**2)))).triu()
    #     print(f"{m[:5, :5]}")
    #     print(f"{m[sz//2:sz//2 + 5, sz//2:sz//2 + 15]}")
    #     print(f"{m[-5:, -5:]}")
    # ```
    # """
    # blur sigma with a specified method, only RBF available right now

    usig, sig, vhsig = torch.linalg.svd(s)
    holdu = u @ usig
    u.zero_()
    u.add_(holdu)
    holdvh = vhsig @ vh
    vh.zero_()
    vh.add_(holdvh)

    r = torch.rand(
        sig.shape[0],
        sig.shape[0],
        device=sig.device,
        dtype=sig.dtype,
        generator=generator,
    )  # sig is 1D vec
    # r = torch.randn(sig.shape[0], sig.shape[0], device=sig.device, dtype=sig.dtype)  # sig is 1D vec
    r = (r * torch.arange(sig.shape[0], device=sig.device)).T / sig.shape[0]
    # sigma_dist = 1.0  # hyperparam
    # neighbor_influ = 0.05
    m = (sigma**2 * torch.exp(-((torch.cdist(r, r, p=2) ** 2) / (2 * neigh**2)))).triu()
    # print(f"{m[:5, :5]}")
    s = sig * m
    s.zero_()
    s.add_(s)


def exp_update_usvh_mix_sigma(u, s, vh, scaling=100):
    """
    NOTE: scaling will scale the array by sigma.shape[0] / scaling
            - this will make it invariant of the size of sigma
    """
    # NOTE: this will update U/S/Vh IN PLACE!!!
    # mix sigma with a specified method, only RBF available right now

    usig, sig, vhsig = torch.linalg.svd(s, full_matrices=False)
    holdu = u @ usig
    u.zero_()
    u.add_(holdu)
    holdvh = vhsig @ vh
    vh.zero_()
    vh.add_(holdvh)

    def _generate_diag_decreasing_like(sigma):
        n = sigma.shape[0]
        ar = torch.arange(sigma.shape[0], dtype=sigma.dtype, device=sigma.device)
        matrix = (n - torch.abs(ar.view(-1, 1) - ar.view(1, -1))) / n
        return matrix

    scaling = sig.shape[0] / scaling
    m = torch.exp(scaling * _generate_diag_decreasing_like(sig) - scaling).triu()
    s = sig * m
    s.zero_()
    s.add_(s)
