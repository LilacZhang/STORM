#!/bin/bash
set -euo pipefail

# Example training script for MuJoCo environments on macOS with MPS support
# This script demonstrates how to train the STORM model on MuJoCo environments

# Environment options:
# - Hopper-v4: 2D hopping robot (good for testing)
# - HalfCheetah-v4: Fast 2D running robot
# - Walker2d-v4: 2D walking robot
# - Ant-v4: 4-legged robot (more complex)
# - Humanoid-v4: 3D humanoid (very complex)

ENV_NAME="Hopper-v4"
RUN_NAME="storm_mujoco_hopper_mps"
SEED=42

echo "🚀 Starting STORM training on $ENV_NAME"
echo "📱 Device: Auto-detect (MPS on macOS, CUDA on Linux/Windows, CPU fallback)"
echo "🎲 Random seed: $SEED"
echo ""

python train.py \
    -n $RUN_NAME \
    -seed $SEED \
    -config_path config_files/STORM_mujoco.yaml \
    -env_name $ENV_NAME \
    -trajectory_path ""  # No demonstration trajectory for MuJoCo

echo ""
echo "✅ Training completed! Check tensorboard logs:"
echo "   tensorboard --logdir runs/$RUN_NAME"
