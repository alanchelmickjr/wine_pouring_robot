#!/usr/bin/env python3
"""
AUTO-PUSH TO GITHUB

This script:
1. Creates all project files
2. Initializes git
3. Creates GitHub repo
4. Pushes everything

Usage:
    python github_push_script.py YOUR_GITHUB_USERNAME
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, check=True):
    """Run shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True

def check_requirements():
    """Check if git and gh are installed"""
    print("Checking requirements...")
    
    # Check git
    if not run_command("git --version", check=False):
        print("❌ Git not installed!")
        print("Install: https://git-scm.com/downloads")
        return False
    print("✅ Git installed")
    
    # Check GitHub CLI
    if not run_command("gh --version", check=False):
        print("❌ GitHub CLI not installed!")
        print("Install: https://cli.github.com/")
        return False
    print("✅ GitHub CLI installed")
    
    # Check if authenticated
    result = subprocess.run("gh auth status", shell=True, capture_output=True)
    if result.returncode != 0:
        print("⚠️  Not authenticated with GitHub")
        print("Running: gh auth login")
        subprocess.run("gh auth login", shell=True)
    else:
        print("✅ Authenticated with GitHub")
    
    return True

def create_project_files():
    """Create all project files"""
    print("\nCreating project files...")
    
    # Create directories
    dirs = ['config', 'simulation', 'physics', 'training', 'docs']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    
    # Copy matplotlib_viz.py (you'll need to save this manually)
    viz_file = Path("matplotlib_viz.py")
    if not viz_file.exists():
        print("\n⚠️  matplotlib_viz.py not found!")
        print("Please save the artifact as 'matplotlib_viz.py' first")
        return False
    
    # Create README
    readme = """# Wine Pouring Robot 🍷🤖

**Autonomous wine pouring with real fluid dynamics and auto-aim**

![Demo](demo.gif)

## ⚡ Quick Start

```bash
pip install matplotlib numpy
python matplotlib_viz.py
```

**Controls:**
- Press **SPACE** to start pouring
- Auto-aim is ON by default (watch it hit perfectly!)
- Press **A** to toggle auto-aim
- Press **ESC** to quit

## 🎯 Features

- ✅ **Real fluid dynamics** (Torricelli's law + projectile motion)
- ✅ **Auto-aim** (calculates perfect tilt angle)
- ✅ **IoU tracking** (circle overlap detection)
- ✅ **Interactive visualization** (3D + top-down views)
- ✅ **No PyBullet needed** (just matplotlib!)

## 🧠 How It Works

The robot calculates the optimal bottle tilt to hit the cup:

```python
# Physics-based auto-aim
distance = cup_position - bottle_position
optimal_tilt = calculate_projectile_angle(distance, gravity, velocity)
# Stream lands perfectly in cup!
```

## 📊 What You'll See

- **3D View**: Wine stream trajectory from bottle to cup
- **Top-Down View**: IoU visualization (yellow landing circle overlaps green cup)
- **Metrics Panel**: Real-time tilt angle, flow rate, IoU, fill level
- **Auto-Aim Status**: 🎯 Shows when active

## 🔧 For Real Robots

This visualization is the foundation for:
1. Training diffusion policies
2. Collecting demonstrations  
3. Testing control algorithms
4. Deploying to 6DOF arms

**Coming soon:**
- PyBullet 3D particle simulation
- Diffusion policy training
- Real robot deployment code
- Camera-based cup tracking

## 🤝 Contributing

Pull requests welcome! This is just the beginning.

## 📜 License

MIT

## 🙏 Acknowledgments

- Inspired by RL-100 and LeRobot
- Built during an epic Claude conversation about robot pouring
- No calipers were harmed in the making of this robot 😄

---

**Made with 🍷 and 🤖**
"""
    
    Path("README.md").write_text(readme)
    print("✅ README.md created")
    
    # Create requirements.txt
    requirements = """matplotlib>=3.4.0
numpy>=1.21.0
scipy>=1.7.0
"""
    Path("requirements.txt").write_text(requirements)
    print("✅ requirements.txt created")
    
    # Create .gitignore
    gitignore = """__pycache__/
*.pyc
*.pyo
.DS_Store
*.swp
venv/
env/
"""
    Path(".gitignore").write_text(gitignore)
    print("✅ .gitignore created")
    
    return True

def initialize_git():
    """Initialize git repository"""
    print("\nInitializing git repository...")
    
    if not run_command("git init"):
        return False
    print("✅ Git initialized")
    
    if not run_command("git add ."):
        return False
    print("✅ Files staged")
    
    if not run_command('git commit -m "Initial commit: Wine pouring robot with auto-aim"'):
        return False
    print("✅ Initial commit created")
    
    return True

def create_and_push_to_github(username):
    """Create GitHub repo and push"""
    print(f"\nCreating GitHub repository for user: {username}")
    
    repo_name = "wine-pouring-robot"
    
    # Create repo
    cmd = f'gh repo create {repo_name} --public --description "Autonomous wine pouring robot with fluid dynamics and auto-aim" --source=. --push'
    
    if not run_command(cmd):
        print("\n⚠️  Failed to create repo")
        print("You can create it manually:")
        print(f"  1. Go to https://github.com/new")
        print(f"  2. Name: {repo_name}")
        print(f"  3. Public")
        print(f"  4. Then run:")
        print(f"     git remote add origin https://github.com/{username}/{repo_name}.git")
        print(f"     git branch -M main")
        print(f"     git push -u origin main")
        return False
    
    print(f"\n✅ Repository created and pushed!")
    print(f"\n🎉 Your repo is live at:")
    print(f"   https://github.com/{username}/{repo_name}")
    
    return True

def main():
    print("="*70)
    print("WINE POURING ROBOT - AUTO PUSH TO GITHUB")
    print("="*70)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: python github_push_script.py YOUR_GITHUB_USERNAME")
        print("\nExample:")
        print("  python github_push_script.py coolrobotbuilder")
        sys.exit(1)
    
    username = sys.argv[1]
    print(f"\nGitHub Username: {username}")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create files
    if not create_project_files():
        print("\n⚠️  Missing files. Make sure you have:")
        print("  - matplotlib_viz.py (save from artifact)")
        sys.exit(1)
    
    # Initialize git
    if not initialize_git():
        sys.exit(1)
    
    # Push to GitHub
    if not create_and_push_to_github(username):
        sys.exit(1)
    
    print("\n" + "="*70)
    print("SUCCESS! 🎉")
    print("="*70)
    print(f"\nYour robot is now on GitHub!")
    print(f"Share it: https://github.com/{username}/wine-pouring-robot")
    print("\nNext steps:")
    print("  1. Add a demo video/GIF")
    print("  2. Star your own repo 😄")
    print("  3. Share on Twitter/Reddit")
    print("  4. Build the actual robot!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)
