#!/bin/bash

source /hkfs/work/workspace/scratch/qv2382-madonna/venv_madonna/bin/activate

mlflow server \
    --backend-store-uri sqlite:////hkfs/work/workspace/scratch/qv2382-madonna/mlflowsql/runsdb.sqlite \
    --default-artifact-root  \
    --port 5000
