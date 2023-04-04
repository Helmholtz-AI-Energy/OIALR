from __future__ import annotations

import datetime as dt
import os
import socket
import time

import torch
import torch.distributed as dist
from mpi4py import MPI
from torch._C._distributed_c10d import _DEFAULT_PG_TIMEOUT
from torch.distributed.distributed_c10d import (
    _new_process_group_helper,
    _pg_group_ranks,
    _store_based_barrier,
)

_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_ROOT = 0


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        size = dist.get_world_size()
    else:
        size = 1
    return size


def get_world_rank():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 1
    return rank


def get_data_parallel_size():
    """
    Gets size of DP communicator
    """
    if dist.is_available() and dist.is_initialized():
        size = dist.get_world_size(group=_DATA_PARALLEL_GROUP)
    else:
        size = 1
    return size


def get_data_parallel_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group=_DATA_PARALLEL_GROUP)
    else:
        rank = 0
    return rank


def get_data_parallel_root(global_rank=False):
    if dist.is_available() and dist.is_initialized():
        if global_rank:
            root = _DATA_PARALLEL_ROOT
        else:
            root = 0
    else:
        root = 0
    return root


def get_local_rank():
    """
    Gets node local rank or returns zero if distributed is not initialized.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return 0

    # number of GPUs per node
    if torch.cuda.is_available():
        local_rank = dist.get_rank(group=_DATA_PARALLEL_GROUP) % torch.cuda.device_count()
    else:
        local_rank = 0

    return local_rank


def get_data_parallel_group():
    if dist.is_available() and dist.is_initialized():
        grp = _DATA_PARALLEL_GROUP
    else:
        grp = None
    return grp


def get_local_size():
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    if torch.cuda.is_available():
        local_size = torch.cuda.device_count()
        # be sure to not return something bigger than world size
        local_size = min([local_size, get_world_size()])
    else:
        local_size = 1

    return local_size


def init_local_group(batchnorm_group_size, batchnorm_group_stride=1):
    # get comm stats
    my_rank = get_world_rank()
    world_size = get_world_size()

    # create local group
    num_groups = world_size // batchnorm_group_size
    assert (
        get_data_parallel_size() % batchnorm_group_size == 0
    ), "Error, make sure that the batchnorm group size is evenly divides the data parallel size"
    assert (
        get_data_parallel_size() >= batchnorm_group_size
    ), "Error, make sure the batchnorm groups do not extend beyond data parallel groups"
    local_group = None
    if world_size > 1 and batchnorm_group_size > 1:
        num_stride_groups = num_groups // batchnorm_group_stride
        local_groups = []
        for i in range(num_stride_groups):
            for j in range(batchnorm_group_stride):
                start = j + i * (batchnorm_group_size * batchnorm_group_stride)
                end = start + batchnorm_group_size * batchnorm_group_stride
                ranks = list(range(start, end, batchnorm_group_stride))
                local_groups.append(ranks)
                tmp_group = dist.new_group(ranks=ranks)
                if my_rank in ranks:
                    local_group = tmp_group

        # print(local_groups)

        # for i in range(num_groups):
        #    start = i * batchnorm_group_size
        #    end = start + batchnorm_group_size
        #    ranks = list(range(start, end))
        #    tmp_group = dist.new_group(ranks = ranks)
        #    if my_rank in ranks:
        #        local_group = tmp_group

    return local_group


# split comms using MPI
def init_split(
    method,
    instance_size,
    split_groups=True,
    batchnorm_group_size=1,
    batchnorm_group_stride=1,
    verbose=False,
    directory=None,
):
    # import MPI here:
    from mpi4py import MPI

    # data parallel group
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_ROOT

    # get MPI stuff
    mpi_comm = MPI.COMM_WORLD.Dup()
    comm_size = mpi_comm.Get_size()
    comm_rank = mpi_comm.Get_rank()

    # determine the number of instances
    num_instances = comm_size // instance_size
    # determine color dependent on instance id:
    # comm_rank = instance_rank +  instance_id * instance_size
    instance_id = comm_rank // instance_size
    instance_rank = comm_rank % instance_size

    # split the communicator
    mpi_instance_comm = mpi_comm.Split(color=instance_id, key=instance_rank)

    # for a successful scaffolding, we need to retrieve the IP addresses
    port = 29500
    master_address = socket.gethostname()
    if split_groups:
        master_address = mpi_instance_comm.bcast(master_address, root=0)
    else:
        master_address = mpi_comm.bcast(master_address, root=0)

    # save env vars
    os.environ["MASTER_ADDR"] = master_address
    os.environ["MASTER_PORT"] = str(port)

    # special stuff for file wireup method
    if method == "nccl-file":
        master_filename = os.path.join(directory, f"instance{instance_id}.store")
        if comm_rank == 0:
            os.makedirs(directory, exist_ok=True)
        mpi_comm.Barrier()

        # delete the wireup file if it exists
        if (instance_rank == 0) and os.path.isfile(master_filename):
            os.remove(master_filename)
        mpi_instance_comm.Barrier()

    # set the parameters depending on whether we want to split or not
    if split_groups:
        nccl_world_size = instance_size
        nccl_world_rank = instance_rank
    else:
        nccl_world_size = comm_size
        nccl_world_rank = comm_rank

    # do the dist init (if we have non trivial instances)
    if instance_size > 1:
        if verbose and instance_rank == 0:
            print(
                f"Starting NCCL wireup for instance {instance_id} with method {method}",
                flush=True,
            )
        # dangerous but necessary: done in run.sub now
        # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        if method == "nccl-slurm":
            # get TCP Store
            wireup_store = dist.TCPStore(
                host_name=master_address,
                port=port,
                world_size=nccl_world_size,
                is_master=(nccl_world_rank == 0),
                timeout=dt.timedelta(seconds=3600),
            )
        else:
            raise NotImplementedError(
                f"Error, unknown wireup method {method}, supported are [nccl-slurm, nccl-file]",
            )

        # initialize group
        dist.init_process_group(
            backend="nccl",
            store=wireup_store,
            world_size=nccl_world_size,
            rank=nccl_world_rank,
        )

        if split_groups:
            _DATA_PARALLEL_GROUP = None
            _DATA_PARALLEL_ROOT = 0
        else:
            # create data parallel group:
            for inst_id in range(num_instances):
                start = inst_id * instance_size
                end = start + instance_size
                ranks = list(range(start, end))
                tmp_group = dist.new_group(ranks=ranks)
                if inst_id == instance_id:
                    _DATA_PARALLEL_GROUP = tmp_group
                    _DATA_PARALLEL_ROOT = ranks[0]

        # make sure to call a barrier here in order for sharp to use the default comm:
        dist.barrier(device_ids=[get_local_rank()], group=_DATA_PARALLEL_GROUP)
        # the nccl wireup call could be non blocking, so we wait for the first barrier
        # to complete before printing this message
        if verbose and instance_rank == 0:
            print(f"Completed NCCL wireup for instance {instance_id}", flush=True)

    # get the local process group for batchnorm
    batchnorm_group = init_local_group(batchnorm_group_size, batchnorm_group_stride)

    return mpi_comm, mpi_instance_comm, instance_id, batchnorm_group


# do regular init
def init(method, ranks_per_gpu=1, batchnorm_group_size=1, batchnorm_group_stride=1):
    # get master address and port
    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_ROOT

    mpi_comm = MPI.COMM_WORLD
    port = 29500
    master_address = socket.gethostname()
    master_address = mpi_comm.bcast(master_address, root=0)

    # save env vars
    os.environ["MASTER_ADDR"] = master_address
    os.environ["MASTER_PORT"] = str(port)

    comm_size = mpi_comm.Get_size()
    comm_rank = mpi_comm.Get_rank()

    nccl_world_size = comm_size
    nccl_world_rank = comm_rank

    if method == "nccl-openmpi":
        # addrport = os.getenv("PMIX_SERVER_URI2").split("//")[1]
        # use that URI
        # address = addrport.split(":")[0]
        # use the default pytorch port
        # port = "29500"
        # os.environ["MASTER_ADDR"] = address
        # os.environ["MASTER_PORT"] = port
        rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 0))

        # init DDP
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )

    elif method == "nccl-slurm":
        # rank = int(os.getenv("SLURM_PROCID"))
        # world_size = int(os.getenv("SLURM_NTASKS"))
        # address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        # port = "29500"
        # os.environ["MASTER_ADDR"] = address
        # os.environ["MASTER_PORT"] = port
        # print(f"device count: {torch.cuda.device_count()}")
        torch.cuda.set_device(comm_rank % 4)
        time.sleep(0.01 * comm_rank)

        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=nccl_world_size,
            is_master=(nccl_world_rank == 0),
            timeout=dt.timedelta(seconds=3600),
        )
        dist.init_process_group(
            backend="nccl",
            store=wireup_store,
            world_size=nccl_world_size,
            rank=nccl_world_rank,
        )

        # init DDP
        # dist.init_process_group(
        #     backend="nccl",
        #     rank=rank,
        #     world_size=world_size
        # )
    elif method == "gloo":
        time.sleep(0.001 * comm_rank)

        wireup_store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=nccl_world_size,
            is_master=(nccl_world_rank == 0),
            timeout=dt.timedelta(seconds=3600),
        )
        dist.init_process_group(
            backend="gloo",
            store=wireup_store,
            world_size=nccl_world_size,
            rank=nccl_world_rank,
        )
    else:
        raise NotImplementedError()

    # make sure to call a barrier here in order for sharp to use the default comm:
    if dist.is_initialized():
        if ranks_per_gpu > 1 and method != "gloo":
            torch.cuda.set_device(get_local_rank() // ranks_per_gpu)
        elif method == "gloo":
            pass
        else:
            torch.cuda.set_device(get_local_rank())
        dist.barrier()
        # dist.barrier(device_ids=[get_local_rank()], group=_DATA_PARALLEL_GROUP)
        disttest = torch.ones(1)
        if method != "gloo":
            disttest = disttest.cuda()
        # print(disttest)

        dist.all_reduce(disttest)
        assert disttest[0] == nccl_world_size, "failed test of dist!"
    else:
        disttest = None

    # get the local process group for batchnorm
    batchnorm_group = init_local_group(batchnorm_group_size, batchnorm_group_stride)

    print(f"finished dist init - rank: {dist.get_rank()} ws: {dist.get_world_size()}, test: {disttest}")
    return batchnorm_group


def create_sub_groups(group_size: int) -> dist.ProcessGroup:
    """
    Create local sub-groups in the communicator.
    NOTE: only the local group will be returned on each process, all procs will be in a local group

    Parameters
    ----------
    group_size: int
        size of groups to create

    Returns
    -------
    torch.distributed.ProcessGroup
    """
    global_size = dist.get_world_size()
    global_rank = dist.get_rank()

    assert global_size % group_size == 0, f"global_size % group_size != 0 ({global_size}, {group_size})"

    global _pg_group_ranks

    group_id = global_rank // group_size
    group_rank = global_rank % group_size
    time.sleep(global_rank * 0.01)

    mpi_comm = MPI.COMM_WORLD
    gp_ranks = [i for i in range(group_id * group_size, (group_id + 1) * group_size)]
    # my_groups_rank0 =

    group = mpi_comm.group.Incl(gp_ranks)
    mpi_group = mpi_comm.Create_group(group)
    master_address = socket.gethostname()
    # if mpi_group.Get_rank() != 0:
    # master_address = None
    master_address = mpi_group.bcast(master_address, root=0)
    # print(master_address)

    # save env vars
    os.environ["MASTER_ADDR"] = master_address
    port = 29510 + group_id
    os.environ["MASTER_PORT"] = str(port)
    # print(master_address, port)

    ranks = torch.arange(global_size).tolist()
    grp_st, grp_sp = group_id * group_size, (group_id + 1) * group_size
    local_ranks = ranks[grp_st:grp_sp]

    # ------- from torch.distributed --------------------------------
    wireup_store = dist.TCPStore(
        host_name=master_address,
        port=port,
        world_size=group_size,
        is_master=(group_rank == 0),
        timeout=dt.timedelta(seconds=3600),
    )
    # pg = dist.new_group(ranks=local_ranks)
    pg = _new_process_group_helper(
        group_size,
        group_rank,
        local_ranks,
        backend="gloo",
        store=wireup_store,
        pg_options=None,
        timeout=_DEFAULT_PG_TIMEOUT,
    )

    # Create the global rank to group rank mapping
    _pg_group_ranks[pg] = {global_rank: group_rank for group_rank, global_rank in enumerate(local_ranks)}
    pg._set_sequence_number_for_group()
    return pg


def init_and_set_config_rank_size(config):
    size = 1
    rank = 0
    if "comm_method" in config and config.comm_method == "gloo":
        init(method="gloo")
        rank = dist.get_rank()
        size = dist.get_world_size()
        return rank, size
    try:
        if int(os.environ["SLURM_NTASKS"]) > 1 or int(os.environ["OMPI_COMM_WORLD_SIZE"]) > 1:
            init(method="nccl-slurm")
            rank = dist.get_rank()
            size = dist.get_world_size()
    except KeyError:
        try:
            if int(os.environ["OMPI_COMM_WORLD_SIZE"]) > 1:
                init(method="nccl-slurm")
                rank = dist.get_rank()
                size = dist.get_world_size()
        except KeyError:
            pass

    return rank, size
