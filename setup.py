#!/usr/bin/env python3
"""
🚀 Quick Setup Guide for Deepfake Detection System

This script helps users quickly set up the environment and test the system.
Run this script to check dependencies and create necessary configuration files.

Author: Advanced Deepfake Detection Team
License: MIT
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'torch_geometric',
        'opencv-python', 'numpy', 'scikit-learn',
        'mediapipe', 'optuna', 'matplotlib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True

def create_env_file():
    """Create .env file from example if it doesn't exist"""
    if not os.path.exists('.env') and os.path.exists('.env.example'):
        with open('.env.example', 'r') as f:
            content = f.read()
        with open('.env', 'w') as f:
            f.write(content)
        print("✅ Created .env file from example")
    else:
        print("✅ .env file already exists")

def main():
    print("🔍 Deepfake Detection System - Quick Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return
    print(f"✅ Python {sys.version}")
    
    # Check dependencies
    print("\n📦 Checking dependencies:")
    if not check_dependencies():
        return
    
    # Create environment file
    print("\n🔧 Setting up environment:")
    create_env_file()
    
    # Check model files
    print("\n🤖 Checking models:")
    model_files = ["model.pth", "best_deepfake_detector_full.pth"]
    model_found = False
    for model in model_files:
        if os.path.exists(model):
            print(f"✅ Found: {model}")
            model_found = True
        else:
            print(f"❌ Missing: {model}")
    
    if not model_found:
        print("⚠️ No model files found. Train a model first with: python train_model.py")
    
    print("\n🚀 Setup complete! Try running:")
    print("   python test_model.py    # For inference")
    print("   python train_model.py   # For training")
    print("   python preprocess.py    # For data preprocessing")

if __name__ == "__main__":
    main()
