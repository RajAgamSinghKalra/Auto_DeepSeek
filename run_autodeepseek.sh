#!/bin/bash
# AutoDeepSeek Launcher Script
# Launches AutoDeepSeek with proper environment setup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if we're in the right directory
if [ ! -f "autodeepseek.py" ]; then
    print_error "autodeepseek.py not found in current directory"
    print_error "Please run this script from the AutoDeepSeek root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found"
    print_error "Please run 'python3 setup.py' first to set up the environment"
    exit 1
fi

print_header "🚀 Starting AutoDeepSeek..."

# Load environment variables if .env exists
if [ -f ".env" ]; then
    print_status "Loading configuration from .env"
    export $(grep -v '^#' .env | xargs)
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Set up ROCm environment variables for AMD GPU
print_status "Setting up GPU environment..."
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-10.3.0}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Set PyTorch/HuggingFace cache directories
export HF_HOME=${HF_HOME:-./models_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-./models_cache}
export TORCH_HOME=${TORCH_HOME:-./models_cache}

# Create cache directory if it doesn't exist
mkdir -p $HF_HOME

# Set memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# Check system requirements
print_status "Checking system requirements..."

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.8"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_status "Python version: $python_version ✓"
else
    print_error "Python 3.8+ required, found: $python_version"
    exit 1
fi

# Check GPU availability
if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    print_status "GPU check completed"
else
    print_warning "GPU availability check failed, will use CPU"
fi

# Check ROCm if available
if command -v rocm-smi >/dev/null 2>&1; then
    print_status "ROCm detected ✓"
    rocm-smi --showproductname 2>/dev/null | head -5 || true
else
    print_warning "ROCm not detected, GPU acceleration may not work optimally"
fi

# Check available memory  
total_mem=$(free -g | awk '/^Mem:/{print $2}')
if [ "$total_mem" -lt 8 ]; then
    print_warning "Low system memory detected: ${total_mem}GB (16GB+ recommended)"
else
    print_status "System memory: ${total_mem}GB ✓"
fi

# Check disk space
workspace_dir=${WORKSPACE_DIR:-./autodeepseek_workspace}
available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$available_space" -lt 10 ]; then
    print_warning "Low disk space: ${available_space}GB (20GB+ recommended)"
else
    print_status "Available disk space: ${available_space}GB ✓"
fi

# Create workspace directory
mkdir -p "$workspace_dir"
mkdir -p logs

# Function to handle cleanup on exit
cleanup() {
    print_status "Cleaning up..."
    # Kill any background processes if needed
    jobs -p | xargs -r kill 2>/dev/null || true
    print_status "Goodbye! 👋"
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Check for updates (optional)
if [ "${CHECK_UPDATES:-true}" = "true" ]; then
    print_status "Checking for dependency updates..."
    pip list --outdated | grep -E "(torch|transformers|selenium)" || print_status "Dependencies up to date"
fi

# Launch AutoDeepSeek
print_header "🤖 Launching AutoDeepSeek Agent..."
print_status "Workspace: $workspace_dir"
print_status "Logs: ./logs/"
print_status "Model cache: $HF_HOME"

# Add some helpful tips
echo
print_header "💡 Quick Tips:"
echo "  • Type 'help' for available commands"
echo "  • Type 'workspace' to see workspace location"  
echo "  • Type 'logs' to view recent activity"
echo "  • Type 'exit' to quit safely"
echo "  • Use Ctrl+C to interrupt current task"
echo

# Launch with proper error handling
if python3 autodeepseek.py "$@"; then
    print_status "AutoDeepSeek exited normally"
else
    exit_code=$?
    print_error "AutoDeepSeek exited with error code: $exit_code"
    print_error "Check the logs for more details: ./logs/"
    exit $exit_code
fi
