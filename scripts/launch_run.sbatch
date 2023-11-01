#!/usr/bin/env bash

# Slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH --job-name=madonna
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-madonna
#SBATCH --output="/hkfs/work/workspace/scratch/qv2382-madonna-ddp/madonna/logs/slurm/slurm-%j"
#####SBATCH --output="/pfs/work7/workspace/scratch/qv2382-madonna-ddp/qv2382-madonna-ddp/madonna/logs/slurm/slurm-%j"

ml purge

# pmi2 cray_shasta
BASE_DIR="/hkfs/work/workspace/scratch/qv2382-madonna-ddp/"
# export EXT_DATA_PREFIX="/hkfs/home/dataset/datasets/"
# BASE_DIR="/pfs/work7/workspace/scratch/qv2382-madonna-ddp/qv2382-madonna-ddp/madonna/"

BASE_DIR="/hkfs/work/workspace/scratch/qv2382-madonna-ddp/"
export EXT_DATA_PREFIX="/hkfs/home/dataset/datasets/"
TOMOUNT='/etc/slurm/task_prolog:/etc/slurm/task_prolog,'
TOMOUNT+="${EXT_DATA_PREFIX},"
TOMOUNT+="${BASE_DIR},"
TOMOUNT+="/scratch,/tmp"
export TOMOUNT="${TOMOUNT}"
export WANDB_API_KEY="4a4a69b3f101858c816995e6dfa553718fdf0dbe"

# TOMOUNT="/pfs/work7/workspace/scratch/qv2382-madonna-ddp/qv2382-madonna-ddp/,/scratch,"
# TOMOUNT+='/etc/slurm/:/etc/slurm/,'
# # TOMOUNT+="${EXT_DATA_PREFIX},"
# # TOMOUNT+="${BASE_DIR},"
# # TOMOUNT+="/sys,/tmp,"
# export TOMOUNT+="/home/kit/scc/qv2382/"

SRUN_PARAMS=(
  --mpi="pmi2"
#  --ntasks-per-node=4
  # --gpus-per-task=4
  --cpus-per-task=8
  #--cpu-bind="ldoms"
  --gpu-bind="closest"
  --label
  --container-name=torch2.1.2
  --container-writable
  --container-mount-home
  --container-mounts="${TOMOUNT}"
)

export CUDA_VISIBLE_DEVICES="0,1,2,3"
# SCRIPT_DIR="/hkfs/work/workspace/scratch/qv2382-madonna-ddp/"
# TODO: set up singularity as well?

export UCX_MEMTYPE_CACHE=0
export NCCL_IB_TIMEOUT=100
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0

#export CONFIGS="${SCRIPT_DIR}DLRT/configs/"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CONFIG_NAME="ortho_train.yaml"
srun "${SRUN_PARAMS[@]}" bash -c "export CONFIG_NAME=${CONFIG_NAME}; python -u ${BASE_DIR}madonna/scripts/train.py name='svd-untuned-final' baseline=False enable_tracking=True"
# srun "${SRUN_PARAMS[@]}" bash -c "export CONFIG_NAME=${CONFIG_NAME}; wandb agent hai-energy/madonna2/mv00rpa5"

# srun "${SRUN_PARAMS[@]}" bash -c "cd ${BASE_DIR}madonna/; python -u ${BASE_DIR}madonna/scripts/propulate_train.py"
