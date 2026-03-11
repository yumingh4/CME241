#!/usr/bin/env python3
"""
Quick test to verify Phase 3 environment is working.
Run this after installing requirements.
"""

import sys

print("="*60)
print("Phase 3 Installation Test")
print("="*60)

# Test 1: Core packages
print("\n1. Testing core packages...")
try:
    import numpy as np
    print("   ✓ numpy")
except ImportError as e:
    print(f"   ✗ numpy: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("   ✓ pandas")
except ImportError as e:
    print(f"   ✗ pandas: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("   ✓ matplotlib")
except ImportError as e:
    print(f"   ✗ matplotlib: {e}")
    sys.exit(1)

# Test 2: PyTorch
print("\n2. Testing PyTorch...")
try:
    import torch
    print(f"   ✓ torch (version {torch.__version__})")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"   ✗ torch: {e}")
    print("   Run: pip install torch")
    sys.exit(1)

# Test 3: Gym
print("\n3. Testing Gym...")
try:
    import gym
    print(f"   ✓ gym (version {gym.__version__})")
except ImportError as e:
    print(f"   ✗ gym: {e}")
    print("   Run: pip install gym")
    sys.exit(1)

# Test 4: Import environment
print("\n4. Testing Buy vs. Rent environment...")
try:
    from buy_rent_environment import BuyRentEnv, BuyRentParams
    print("   ✓ buy_rent_environment imports successfully")
    
    # Try to create environment
    params = BuyRentParams()
    env = BuyRentEnv(params)
    print("   ✓ Environment created successfully")
    
    # Try reset
    obs, info = env.reset()
    print(f"   ✓ Environment reset works (obs shape: {obs.shape})")
    
except Exception as e:
    print(f"   ✗ Environment error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Import DQN agent
print("\n5. Testing DQN agent...")
try:
    from modified_dqn import ActionMaskedDQNAgent
    print("   ✓ modified_dqn imports successfully")
    
    # Try to create agent
    agent = ActionMaskedDQNAgent()
    print("   ✓ Agent created successfully")
    
except Exception as e:
    print(f"   ✗ Agent error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nYou're ready to run:")
print("  - python buy_rent_environment.py")
print("  - python modified_dqn.py")
print("  - python evaluate_policies.py")
print("  - jupyter notebook phase3_demo.ipynb")
print("="*60)
