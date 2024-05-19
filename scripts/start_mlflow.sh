#!/bin/bash

source /hkfs/work/workspace/scratch/CHANGE/ME-madonna/venv_madonna/bin/activate

mlflow server \
    --backend-store-uri sqlite:////hkfs/work/workspace/scratch/CHANGE/ME-madonna/mlflowsql/runsdb.sqlite \
    --default-artifact-root  \
    --port 5000
