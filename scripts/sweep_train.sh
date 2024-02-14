#!/bin/bash

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Catching sweep params from WandB and forwarding them to hyrda because i like pain"
      echo " "
      echo "[options] application [arguments]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-N, --nodes               number of nodes to compute on"
      echo "-G, --gpus                gpus per node to use, i.e. the gres value"
      echo "-c, --config              config file to use"
      echo "--reservation             name of reservation"
      echo "-p, --partition           partition to work on"
      exit 0
      ;;
    --training.fixing_method.delay) shift; export HYDRADELAY=$1; shift; ;;
    --training.fixing_method.keep_first_layer) shift; export keep_first_layer=$1; shift; ;;
    --training.fixing_method.keep_last_layer) shift; export keep_last_layer=$1; shift; ;;
    --training.fixing_method.sigma_cutoff_fraction) shift; export sigma_cutoff_fraction=$1; shift; ;;
    --training.fixing_method.stability_frequency) shift; export stability_frequency=$1; shift; ;;
    --training.lr) shift; export lr=$1; shift; ;;
    --training.lr_schedule.min_lr) shift; export min_lr=$1; shift; ;;
    --training.lr_schedule.warmup_epochs) shift; export warmup_epochs=$1; shift; ;;
    --training.lr_schedule.warmup_lr) shift; export warmup_lr=$1; shift; ;;
    --training.max_grad_norm) shift; export max_grad_norm=$1; shift; ;;
    --training.fixing_method.stable_update_delay) shift; export stable_update_delay=$1; shift; ;;
    --training.optimizer.weight_decay) shift; export weight_decay=$1; shift; ;;
    --training.optimizer.beta1) shift; export beat1=$1; shift; ;;
    --training.optimizer.beta2) shift; export beta2=$1; shift; ;;
    *) break; ;;
  esac
done


if [ ${SLURM_PROCID} -eq 0 ];
then
    python scripts/train.py
else
    python scripts/train.py \
        training.fixing_method.delay=${HYDRADELAY} \
        training.fixing_method.keep_first_layer=${keep_first_layer} \
        training.fixing_method.keep_last_layer=${keep_last_layer} \
        training.fixing_method.sigma_cutoff_fraction=${sigma_cutoff_fraction} \
        training.fixing_method.stability_frequency=${stability_frequency} \
        training.fixing_method.stable_update_delay=${stable_update_delay} \
        training.optimizer.weight_decay=${weight_decay} \
        training.optimizer.beta1=${beta1} \
        training.optimizer.beta1=${beta2} \
        training.lr=${lr} \
        training.lr_schedule.min_lr=${min_lr} \
        training.lr_schedule.warmup_epochs=${warmup_epochs} \
        training.lr_schedule.warmup_lr=${warmup_lr} \
        training.max_grad_norm=${max_grad_norm} \
        baseline=False enable_tracking=True name='svd-sweep'

        # training.lr_schedule.lr_cycle_decay=${lr_cycle_decay} \
        # training.lr_schedule.lr_cycle_mul=${lr_cycle_mul} \
        # training.lr_schedule.lr_k_decay=${lr_k_decay} \
fi
