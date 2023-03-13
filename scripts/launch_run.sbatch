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
export EXT_DATA_PREFIX="/hkfs/home/dataset/datasets/"

TOMOUNT='/etc/slurm/task_prolog.hk:/etc/slurm/task_prolog.hk,'
TOMOUNT+="${EXT_DATA_PREFIX},"
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
  --container-name=torch
  --container-writable
  --container-mount-home
  --container-mounts="${TOMOUNT}"
)

#export CUDA_AVAILABLE_DEVICES="0,1,2,3"
SCRIPT_DIR="/hkfs/work/workspace/scratch/qv2382-madonna/"
# TODO: set up singularity as well?

export UCX_MEMTYPE_CACHE=0
export NCCL_IB_TIMEOUT=100
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0

#export CONFIGS="${SCRIPT_DIR}DLRT/configs/"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
#echo "Loading data from ${DATA_PREFIX}"
#echo "${SCRIPT_DIR}"
#srun "${SRUN_PARAMS[@]}" singularity exec --nv \
#  --bind "${DATA_PREFIX}","${SCRIPT_DIR}","/scratch","/tmp","/hkfs/work/workspace/scratch/qv2382-dlrt/DLRT/dlrt/":"/opt/conda/lib/python3.8/site-packages/dlrt/" "${SINGULARITY_FILE}" \
#    bash -c "TORCH_DISTRIBUTED_DEBUG=DETAIL python -u baseline-resnet.py --config ${CONFIGS}resnet18.yaml"

export CONFIG_NAME="slime_train.yaml"

srun "${SRUN_PARAMS[@]}" bash -c "export CONFIG_NAME=${CONFIG_NAME}; python -u ${SCRIPT_DIR}madonna/scripts/train.py"