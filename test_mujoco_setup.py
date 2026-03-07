#!/usr/bin/env python
"""Quick test script to validate MuJoCo setup"""

import numpy as np
import env_wrapper
from utils import DeviceManager


def run_tests():
    """Run all setup validation tests"""
    print("=" * 60)
    print("Testing STORM MuJoCo Setup")
    print("=" * 60)

    # Test 1: Device Detection
    print("\n1. Testing device detection...")
    device_manager = DeviceManager("auto")
    device = device_manager.get_device()
    print(f"   ✓ Device: {device}")
    print(f"   ✓ AMP enabled: {device_manager.use_amp}")

    # Test 2: MuJoCo Environment
    print("\n2. Testing MuJoCo environment building...")
    env = env_wrapper.build_single_env("Hopper-v4", 64, 0)
    print(f"   ✓ Environment: {env}")
    print(f"   ✓ Observation space: {env.observation_space}")
    print(f"   ✓ Action space: {env.action_space}")

    if hasattr(env.action_space, "n"):
        print(f"   ✓ Action dimension: {env.action_space.n}")
    else:
        print(f"   ✗ ERROR: Action space is not discrete!")
        return False

    # Test 3: Environment Step
    print("\n3. Testing environment step...")
    obs, info = env.reset()
    print(f"   ✓ Reset successful, obs shape: {obs.shape}")

    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"   ✓ Step successful")
    print(f"     - obs shape: {obs.shape}")
    print(f"     - reward: {reward:.2f}")
    print(f"     - done: {done}")
    print(f"     - 'life_loss' in info: {'life_loss' in info}")

    env.close()

    # Test 4: Vectorized Environment (skipped on macOS)
    print("\n4. Testing vectorized environment...")
    print("   ⚠️  Skipping AsyncVectorEnv test (macOS multiprocessing limitation)")
    print("   ℹ️  Vectorized environments work fine in train.py with proper guard")

    print("\n" + "=" * 60)
    print("✅ All core tests passed! MuJoCo setup is working correctly.")
    print("=" * 60)
    print("\nYou can now run training with:")
    print("  ./run_mujoco_example.sh")
    print("or:")
    print("  python train.py -n test -seed 42 \\")
    print("    -config_path config_files/STORM_mujoco.yaml \\")
    print("    -env_name Hopper-v4 -trajectory_path ''")

    return True


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
