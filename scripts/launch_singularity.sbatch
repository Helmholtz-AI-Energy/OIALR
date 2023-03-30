#!/usr/bin/env bash

# Slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=madonna-test
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-test-mlperf
#SBATCH --output="/hkfs/work/workspace/scratch/qv2382-madonna/madonna/logs/slurm/slurm-%j"

ml purge

# pmi2 cray_shasta
BASE_DIR="/hkfs/work/workspace/scratch/qv2382-madonna/"

TOMOUNT='/etc/slurm/task_prolog.hk:/etc/slurm/task_prolog.hk,'
TOMOUNT+="${BASE_DIR},"
TOMOUNT+="/scratch,/tmp,"
TOMOUNT+="/hkfs/work/workspace/scratch/qv2382-dlrt/datasets"
export TOMOUNT="${TOMOUNT}"

SRUN_PARAMS=(
  --mpi="pmi2"
#  --ntasks-per-node=4
  --gpus-per-task=1
  --cpus-per-task=8
  #--cpu-bind="ldoms"
  --gpu-bind="closest"
  --label
)

#export CUDA_AVAILABLE_DEVICES="0,1,2,3"
SCRIPT_DIR="/hkfs/work/workspace/scratch/qv2382-madonna/"

export UCX_MEMTYPE_CACHE=0
export NCCL_IB_TIMEOUT=100
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/lib/intel64

TOMOUNT='/etc/slurm/task_prolog.hk:/etc/slurm/task_prolog.hk,'
TOMOUNT+="${EXT_DATA_PREFIX},"
TOMOUNT+="${BASE_DIR},"
TOMOUNT+="/scratch,/tmp,"
TOMOUNT+="/hkfs/work/workspace/scratch/qv2382-dlrt/datasets,"
TOMOUNT+="${SCRIPT_DIR}"

#export CONFIGS="${SCRIPT_DIR}DLRT/configs/"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
#echo "Loading data from ${DATA_PREFIX}"
#echo "${SCRIPT_DIR}"
export CONFIG_NAME="slime_train.yaml"
srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${TOMOUNT}" \
  "${SINGULARITY_FILE}" \
  bash -c "CONFIG_NAME=${CONFIG_NAME} python -u ${SCRIPT_DIR}madonna/scripts/train.py"
