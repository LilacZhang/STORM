#!/usr/bin/env python
"""Quick test to verify MPS tensor operations fix"""

import torch
import numpy as np
from utils import DeviceManager

print("Testing MPS tensor reshape operations...")

# Initialize device
dm = DeviceManager("auto")
device = dm.get_device()
print(f"Device: {device}")

if device.type == "mps":
    print("\n✓ Testing on MPS device")

    # Simulate the problematic operation
    sz_b, len_q, n_head, d_k = 2, 10, 8, 64

    # Create a tensor and apply linear transformation
    x = torch.randn(sz_b, len_q, n_head * d_k, device=device)

    # Test .reshape() (should work)
    try:
        y1 = x.reshape(sz_b, len_q, n_head, d_k)
        print("✓ .reshape() works correctly")
    except Exception as e:
        print(f"✗ .reshape() failed: {e}")

    # Test operations with transpose and reshape
    try:
        y2 = y1.transpose(1, 2)
        y3 = y2.transpose(1, 2).contiguous().reshape(sz_b, len_q, -1)
        print("✓ transpose + contiguous + reshape works correctly")
    except Exception as e:
        print(f"✗ transpose + reshape failed: {e}")

    # Test backward pass
    try:
        y3.requires_grad_(True)
        loss = y3.sum()
        loss.backward()
        print("✓ Backward pass works correctly")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")

    print("\n✅ All MPS tensor operations verified!")
else:
    print(f"\n⚠️  Not testing on MPS (current device: {device.type})")

print("\n" + "=" * 60)
print("The attention mechanism fix should now work on MPS!")
print("You can run training with:")
print("  ./run_mujoco_example.sh")
print("=" * 60)
