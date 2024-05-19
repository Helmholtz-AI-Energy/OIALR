#!/bin/bash

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Launcher for training + timing for DeepCam on either HoreKa or Juwels Booster"
      echo " "
      echo "[options] application [arguments]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-N, --nodes               number of nodes to compute on"
      echo "-G, --gpus                gpus per node to use, i.e. the gres value"
      echo "-c, --config              config file to use"
      echo "--reservation             name of reservation"
      exit 0
      ;;
    -s|--system) shift; export TRAINING_SYSTEM=$1; shift; ;;
    -N|--nodes) shift; export SLURM_NNODES=$1; shift; ;;
    -G|--gpus) shift; export GPUS_PER_NODE=$1; shift; ;;
    -t|--time) shift; export TIMELIMIT=$1; shift; ;;
    *) break; ;;
  esac
done

if [ -z "${TIMELIMIT}" ]; then TIMELIMIT="8:00:00"; fi
if [ -z "${GPUS_PER_NODE}" ]; then GPUS_PER_NODE="4"; fi
if [ -z "${SLURM_NNODES}" ]; then SLURM_NNODES="1"; fi


export CUDA_VISIBLE_DEVICES="0,1,2,3"

export UCX_MEMTYPE_CACHE=0
export NCCL_IB_TIMEOUT=100
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0
export WANDB_API_KEY="NONE"

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/lib/intel64
  # -A hk-project-test-mlperf \
system=${HOSTNAME:0:3}

if [ $system == "hkn" ]
then
  BASE_DIR="/CHANGE/ME/"
  export EXT_DATA_PREFIX="/hkfs/home/dataset/datasets/"
  TOMOUNT='/etc/slurm/task_prolog:/etc/slurm/task_prolog,'
  TOMOUNT+="${EXT_DATA_PREFIX},"
  TOMOUNT+="${BASE_DIR},"
  TOMOUNT+="/scratch,/tmp"
  # TOMOUNT+="/hkfs/work/workspace/scratch/CHANGE/ME-dlrt2/datasets"

  salloc --partition=accelerated \
    -N "${SLURM_NNODES}" \
    --time "${TIMELIMIT}" \
    --gres gpu:"${GPUS_PER_NODE}" \
    --container-name=torch \
    --container-mounts="${TOMOUNT}" \
    --container-mount-home \
    --container-writable \
    -A hk-project-madonna
elif [ $system == "uc2" ]
then
  TOMOUNT="/pfs/work7/workspace/scratch/CHANGE/ME-madonna2/CHANGE/ME-madonna2/,/scratch,"
  TOMOUNT+='/etc/slurm/:/etc/slurm/,'
  # TOMOUNT+="${EXT_DATA_PREFIX},"
  # TOMOUNT+="${BASE_DIR},"
  # TOMOUNT+="/sys,/tmp,"
  export TOMOUNT+="/home/kit/scc/CHANGE/ME/"
  salloc --partition=sdil \
    -N "${SLURM_NNODES}" \
    --time "${TIMELIMIT}" \
    --gres gpu:"${GPUS_PER_NODE}" \
    --container-name=torch \
    --container-mounts="${TOMOUNT}" \
    --container-mount-home \
    --container-writable
fi
