#!/bin/bash

# Example evaluation script for MuJoCo environments

ENV_NAME="Hopper-v4"
RUN_NAME="storm_mujoco_hopper_mps"

echo "🎯 Evaluating STORM model on $ENV_NAME"
echo "📂 Loading checkpoints from: ckpt/$RUN_NAME"
echo ""

python eval.py \
    -config_path config_files/STORM_mujoco.yaml \
    -env_name $ENV_NAME \
    -run_name $RUN_NAME

echo ""
echo "✅ Evaluation completed!"
