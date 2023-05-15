#!/usr/bin/env bash

# Slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=madonna-test
#SBATCH --partition=sdil
####SBATCH --account=hk-project-test-mlperf
#SBATCH --output="/pfs/work7/workspace/scratch/qv2382-madonna/qv2382-madonna/madonna/logs/slurm/slurm-%j"

####SBATCH --output="/hkfs/work/workspace/scratch/qv2382-madonna/madonna/logs/slurm/slurm-%j"

ml purge

# pmi2 cray_shasta
# BASE_DIR="/hkfs/work/workspace/scratch/qv2382-madonna/"
# export EXT_DATA_PREFIX="/hkfs/home/dataset/datasets/"
BASE_DIR="/pfs/work7/workspace/scratch/qv2382-madonna/qv2382-madonna/madonna/"

# TOMOUNT='/etc/slurm/task_prolog.hk:/etc/slurm/task_prolog.hk,'
# TOMOUNT+="${EXT_DATA_PREFIX},"
# TOMOUNT+="${BASE_DIR},"
# TOMOUNT+="/scratch,/tmp,"
# TOMOUNT+="/hkfs/work/workspace/scratch/qv2382-dlrt/datasets"
# export TOMOUNT="${TOMOUNT}"

TOMOUNT="/pfs/work7/workspace/scratch/qv2382-madonna/qv2382-madonna/,/scratch,"
TOMOUNT+='/etc/slurm/:/etc/slurm/,'
# TOMOUNT+="${EXT_DATA_PREFIX},"
# TOMOUNT+="${BASE_DIR},"
# TOMOUNT+="/sys,/tmp,"
export TOMOUNT+="/home/kit/scc/qv2382/"

SRUN_PARAMS=(
  --mpi="pmi2"
#  --ntasks-per-node=4
  # --gpus-per-task=4
  --cpus-per-task=8
  #--cpu-bind="ldoms"
  --gpu-bind="closest"
  --label
  --container-name=torch
  --container-writable
  --container-mount-home
  --container-mounts="${TOMOUNT}"
)

export CUDA_VISIBLE_DEVICES="0,1,2,3"
# SCRIPT_DIR="/hkfs/work/workspace/scratch/qv2382-madonna/"
# TODO: set up singularity as well?

export UCX_MEMTYPE_CACHE=0
export NCCL_IB_TIMEOUT=100
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0

#export CONFIGS="${SCRIPT_DIR}DLRT/configs/"
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CONFIG_NAME="ortho_train.yaml"

srun "${SRUN_PARAMS[@]}" bash -c "export CONFIG_NAME=${CONFIG_NAME}; python -u ${BASE_DIR}scripts/train.py"
