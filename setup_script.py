#!/usr/bin/env python3
"""
SETUP SCRIPT - Wine Pouring Robot

Run this to check dependencies and set up your environment

Usage:
    python setup.py
"""

import sys
import subprocess
import os
from pathlib import Path


def print_header(text):
    """Print fancy header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"   ❌ Python {version.major}.{version.minor} detected")
        print(f"   ⚠️  Python 3.8+ required")
        print(f"   📥 Download from: https://www.python.org/downloads/")
        return False
    
    print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_pip():
    """Check if pip is available"""
    print("📦 Checking pip...")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            capture_output=True
        )
        print("   ✅ pip available")
        return True
    except subprocess.CalledProcessError:
        print("   ❌ pip not found")
        print("   📥 Install with: python -m ensurepip --upgrade")
        return False


def install_dependencies():
    """Install Python dependencies"""
    print("📥 Installing dependencies...")
    
    requirements = [
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "opencv-python>=4.5.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pybullet>=3.2.0",
        "huggingface-hub>=0.16.0",
        "datasets>=2.14.0",
        "opencv-contrib-python>=4.5.0",
        "tqdm>=4.65.0",
    ]
    
    print("   This may take a few minutes...")
    
    try:
        for package in requirements:
            print(f"   Installing {package.split('>=')[0]}...", end=" ")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package, "-q"],
                check=True,
                capture_output=True
            )
            print("✅")
        
        print("\n   ✅ All dependencies installed!")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n   ❌ Failed to install: {e}")
        return False


def create_directory_structure():
    """Create project directories"""
    print("📁 Creating directory structure...")
    
    dirs = [
        "config",
        "calibration",
        "simulation",
        "physics",
        "training",
        "deployment",
        "huggingface",
        "demonstrations",
        "models",
        "docs",
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✅ {dir_name}/")
    
    return True


def create_gitignore():
    """Create .gitignore file"""
    print("📝 Creating .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/

# PyTorch
*.pth
*.pt
models/*.pth
checkpoints/

# Data
demonstrations/*/observations/*.jpg
demonstrations/*/observations/*.png
*.npy
*.pkl

# Large files
*.mp4
*.avi

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Config
robot_config_calibrated.json
"""
    
    Path(".gitignore").write_text(gitignore_content)
    print("   ✅ .gitignore created")
    return True


def check_gpu():
    """Check for NVIDIA GPU"""
    print("🎮 Checking for GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ GPU found: {gpu_name}")
            print(f"   ✅ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("   ⚠️  No GPU detected (CPU will be used)")
            print("   💡 GPU recommended for training (10x faster)")
            return False
    except ImportError:
        print("   ⚠️  PyTorch not yet installed")
        return False


def test_imports():
    """Test that all imports work"""
    print("🧪 Testing imports...")
    
    imports = [
        ("numpy", "np"),
        ("cv2", "cv2"),
        ("torch", "torch"),
        ("pybullet", "p"),
        ("matplotlib.pyplot", "plt"),
    ]
    
    all_good = True
    for module, alias in imports:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} - failed to import")
            all_good = False
    
    return all_good


def run_quick_test():
    """Run a quick test to make sure everything works"""
    print("🧪 Running quick test...")
    
    try:
        import numpy as np
        import pybullet as p
        
        # Test PyBullet
        print("   Testing PyBullet...")
        client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.disconnect()
        print("   ✅ PyBullet works!")
        
        # Test fluid dynamics
        print("   Testing fluid dynamics...")
        from physics.fluid_dynamics import LiquidStreamSimulator, BottleState
        
        bottle = BottleState(
            position=np.array([0.5, 0.0, 0.7]),
            orientation=np.array([0, 0.785, 0]),
            tilt_angle=0.785
        )
        
        sim = LiquidStreamSimulator()
        flow_rate = sim.calculate_flow_rate(bottle)
        print(f"   ✅ Flow rate calculation: {flow_rate*1e6:.1f} ml/s")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def print_next_steps():
    """Print what to do next"""
    print_header("🎉 SETUP COMPLETE!")
    
    print("""
✅ Your environment is ready!

🚀 QUICK START:

1. Run the visualization:
   python run_visualization.py

2. Collect training data:
   python training/data_collection_system.py

3. Configure your robot:
   Edit config/robot_config_master.py with YOUR measurements

4. Calibrate (optional but recommended):
   python calibration/auto_calibration.py

5. Train models:
   python training/diffusion_training.py

6. Deploy to robot:
   python deployment/realtime_deployment.py


📚 DOCUMENTATION:

- README.md - Start here!
- docs/CALIBRATION.md - How to measure/calibrate
- docs/TRAINING.md - How to train models
- docs/DEPLOYMENT.md - How to deploy to hardware


💬 NEED HELP?

- GitHub Issues: https://github.com/YOUR_USERNAME/wine-pouring-robot/issues
- Documentation: See docs/ folder


🌟 If this helps you, star the repo!

Happy pouring! 🍷🤖
""")


def main():
    """Main setup script"""
    print_header("🍷 WINE POURING ROBOT SETUP 🤖")
    
    print("This script will:")
    print("  1. Check Python version")
    print("  2. Install dependencies")
    print("  3. Create directory structure")
    print("  4. Test everything works")
    print()
    
    input("Press Enter to continue (Ctrl+C to cancel)...")
    
    # Check Python
    if not check_python_version():
        sys.exit(1)
    
    # Check pip
    if not check_pip():
        sys.exit(1)
    
    # Install dependencies
    install = input("\n📥 Install dependencies? (y/n): ").strip().lower()
    if install == 'y':
        if not install_dependencies():
            print("\n⚠️  Some packages failed to install")
            print("Try manually: pip install -r requirements.txt")
    
    # Create directories
    create_directory_structure()
    
    # Create .gitignore
    create_gitignore()
    
    # Check GPU
    check_gpu()
    
    # Test imports
    if not test_imports():
        print("\n⚠️  Some imports failed")
        print("Try reinstalling dependencies")
    
    # Run quick test
    test = input("\n🧪 Run quick test? (y/n): ").strip().lower()
    if test == 'y':
        run_quick_test()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        sys.exit(1)