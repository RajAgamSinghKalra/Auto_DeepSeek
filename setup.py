#!/usr/bin/env python3
"""
AutoDeepSeek Setup Script
Automated setup for AutoDeepSeek on Linux with ROCm support
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and return success status"""
    print(f"🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def check_system():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check OS
    if platform.system() != "Linux":
        print("❌ This setup is designed for Linux systems")
        return False
    
    # Check Python version
    # Compare only major and minor numbers to avoid TypeError
    if sys.version_info[:2] < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version}")
    print(f"✅ Linux {platform.release()}")
    
    return True

def setup_rocm():
    """Setup ROCm for AMD GPU support"""
    print("\n🚀 Setting up ROCm for AMD GPU...")
    
    # Check if ROCm is already installed
    try:
        result = subprocess.run("rocm-smi", shell=True, capture_output=True)
        if result.returncode == 0:
            print("✅ ROCm already installed")
            return True
    except:
        pass
    
    print("Installing ROCm...")
    
    # Add ROCm repository
    commands = [
        "wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -",
        "echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list",
        "sudo apt update",
        "sudo apt install -y rocm-dkms rocm-libs rocm-dev rocm-utils",
        "sudo usermod -a -G render,video $USER"
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"ROCm setup step"):
            print("❌ ROCm installation failed")
            return False
    
    print("✅ ROCm installed successfully")
    print("⚠️  Please reboot your system and run this script again")
    return True

def setup_firefox():
    """Setup Firefox and geckodriver"""
    print("\n🦊 Setting up Firefox and geckodriver...")
    
    # Install Firefox if not present
    try:
        subprocess.run("firefox --version", shell=True, check=True, capture_output=True)
        print("✅ Firefox already installed")
    except:
        if not run_command("sudo apt install -y firefox", "Installing Firefox"):
            return False
    
    # Download and setup geckodriver
    geckodriver_path = Path("/usr/local/bin/geckodriver")
    if not geckodriver_path.exists():
        try:
            import json
            import urllib.request

            # Fetch latest geckodriver release
            with urllib.request.urlopen(
                "https://api.github.com/repos/mozilla/geckodriver/releases/latest"
            ) as resp:
                data = json.load(resp)
                version = data.get("tag_name", "")

            tarball = f"geckodriver-{version}-linux64.tar.gz"
            url = (
                "https://github.com/mozilla/geckodriver/releases/download/"
                f"{version}/{tarball}"
            )
        except Exception as e:
            print(f"❌ Failed to determine geckodriver version: {e}")
            return False

        commands = [
            f"wget -O /tmp/geckodriver.tar.gz {url}",
            "cd /tmp && tar -xzf geckodriver.tar.gz",
            "sudo mv /tmp/geckodriver /usr/local/bin/",
            "sudo chmod +x /usr/local/bin/geckodriver"
        ]
        
        for cmd in commands:
            if not run_command(cmd, "Setting up geckodriver"):
                return False
    
    print("✅ Firefox and geckodriver ready")
    return True

def setup_python_environment():
    """Setup Python virtual environment and dependencies"""
    print("\n🐍 Setting up Python environment...")
    
    # Create virtual environment
    venv_path = Path("venv")
    if not venv_path.exists():
        if not run_command("python3 -m venv venv", "Creating virtual environment"):
            return False
    
    # Activate and install dependencies
    activate_cmd = "source venv/bin/activate"
    
    # Install PyTorch with ROCm support
    torch_cmd = f"{activate_cmd} && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6"
    if not run_command(torch_cmd, "Installing PyTorch with ROCm"):
        print("⚠️  ROCm PyTorch failed, trying CPU version...")
        torch_cmd = f"{activate_cmd} && pip install torch torchvision torchaudio"
        if not run_command(torch_cmd, "Installing PyTorch (CPU)"):
            return False
    
    # Install other requirements
    req_cmd = f"{activate_cmd} && pip install -r requirements.txt"
    if not run_command(req_cmd, "Installing Python dependencies"):
        return False
    
    print("✅ Python environment ready")
    return True

def create_config():
    """Create configuration file"""
    print("\n⚙️  Creating configuration...")
    
    config = """# AutoDeepSeek Configuration
# Model settings
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
DEVICE=auto  # auto, cuda, cpu
MAX_ITERATIONS=10

# Browser settings
BROWSER_HEADLESS=false
BROWSER_TIMEOUT=30

# Workspace settings
WORKSPACE_DIR=./autodeepseek_workspace
LOG_LEVEL=INFO

# Security settings
ALLOW_SYSTEM_COMMANDS=true
BLOCKED_COMMANDS=rm -rf,format,del /f,shutdown,reboot
"""
    
    with open(".env", "w") as f:
        f.write(config)
    
    print("✅ Configuration created")
    return True

def create_launcher():
    """Create launcher script"""
    print("\n🚀 Creating launcher script...")
    
    launcher = """#!/bin/bash
# AutoDeepSeek Launcher

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.py first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Set environment variables for ROCm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH

# Launch AutoDeepSeek
python3 autodeepseek.py "$@"
"""
    
    with open("run_autodeepseek.sh", "w") as f:
        f.write(launcher)
    
    os.chmod("run_autodeepseek.sh", 0o755)
    
    print("✅ Launcher script created")
    return True

def main():
    """Main setup function"""
    print("🚀 AutoDeepSeek Setup")
    print("====================")
    
    if not check_system():
        return 1
    
    # Setup components
    setup_steps = [
        ("ROCm", setup_rocm),
        ("Firefox", setup_firefox),
        ("Python Environment", setup_python_environment),
        ("Configuration", create_config),
        ("Launcher", create_launcher)
    ]
    
    for step_name, step_func in setup_steps:
        print(f"\n{'='*50}")
        print(f"Setting up {step_name}")
        print('='*50)
        
        if not step_func():
            print(f"❌ Failed to setup {step_name}")
            return 1
    
    print(f"\n{'='*50}")
    print("🎉 Setup Complete!")
    print('='*50)
    print("""
Next steps:
1. Reboot your system (if ROCm was installed)
2. Run: ./run_autodeepseek.sh
3. Start giving tasks to your AI agent!

Example tasks:
- "Create a Python web scraper for news articles"
- "Write a calculator app with GUI"
- "Analyze this data file and create visualizations"
- "Build a simple REST API server"

Files created:
- autodeepseek.py (main agent)
- requirements.txt (dependencies)
- .env (configuration)
- run_autodeepseek.sh (launcher)
- venv/ (Python environment)
""")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
