#!/bin/bash
# ============================================
# SCOPE environment setup script
# ============================================
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh [cuda_version]
#
# Args:
#   cuda_version: cu118, cu121, cpu (default: cu118)
# ============================================

set -e  # Exit immediately on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[OK]   $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERR]  $1${NC}"
}

# Get CUDA version argument
CUDA_VERSION=${1:-cu118}

print_info "============================================"
print_info "SCOPE project environment setup"
print_info "============================================"
echo ""

# Check Python version
print_info "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
print_success "Python version: $PYTHON_VERSION"
echo ""

# Check whether running in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]] && [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    print_warning "It is recommended to install in a virtual environment."
    echo ""
    echo "Create a new environment:"
    echo "  conda create -n scope python=3.12 -y"
    echo "  conda activate scope"
    echo ""
    read -p "Continue installing in the current environment? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Installation cancelled."
        exit 1
    fi
else
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        print_success "Current Conda env: $CONDA_DEFAULT_ENV"
    elif [[ -n "$VIRTUAL_ENV" ]]; then
        print_success "Current venv: $VIRTUAL_ENV"
    fi
fi
echo ""

# Select CUDA version
print_info "CUDA version: $CUDA_VERSION"
case $CUDA_VERSION in
    cu118)
        TORCH_URL="https://download.pytorch.org/whl/cu118"
        print_info "Installing PyTorch for CUDA 11.8"
        ;;
    cu121)
        TORCH_URL="https://download.pytorch.org/whl/cu121"
        print_info "Installing PyTorch for CUDA 12.1"
        ;;
    cpu)
        TORCH_URL="https://download.pytorch.org/whl/cpu"
        print_warning "Installing CPU-only PyTorch"
        ;;
    *)
        print_error "Unsupported CUDA version: $CUDA_VERSION"
        echo "Supported versions: cu118, cu121, cpu"
        exit 1
        ;;
esac
echo ""

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip
print_success "pip upgrade completed"
echo ""

# Install PyTorch
print_info "Installing PyTorch..."
pip install torch torchvision --index-url $TORCH_URL
print_success "PyTorch installation completed"
echo ""

# Verify PyTorch installation
print_info "Verifying PyTorch..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Install other dependencies
print_info "Installing project dependencies..."
pip install -r requirements_minimal.txt
print_success "Project dependencies installed"
echo ""

# Optional: segmentation / detection
print_info "============================================"
echo ""
read -p "Install segmentation/detection dependencies? (y/N): " -n 1 -r
echo ""
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Installing MMSegmentation and MMDetection..."

    # Install mmcv
    print_info "Installing mmcv-full..."
    if [[ $CUDA_VERSION == "cu118" ]]; then
        pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
    elif [[ $CUDA_VERSION == "cu121" ]]; then
        pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.0.0/index.html
    else
        pip install mmcv-full
    fi

    # Install mmseg and mmdet
    pip install mmsegmentation mmdet

    print_success "Segmentation/detection dependencies installed"
else
    print_info "Skipped segmentation/detection dependencies"
fi
echo ""

# Verify installation
print_info "============================================"
print_info "Verifying installation..."
print_info "============================================"
echo ""

python -c "
import sys
print('Python:', sys.version.split()[0])

try:
    import torch
    print('OK  torch:', torch.__version__)
    print('    CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('    CUDA version:', torch.version.cuda)
except Exception:
    print('ERR torch not installed')

try:
    import torchvision
    print('OK  torchvision:', torchvision.__version__)
except Exception:
    print('ERR torchvision not installed')

try:
    import timm
    print('OK  timm:', timm.__version__)
except Exception:
    print('ERR timm not installed')

try:
    import einops
    print('OK  einops:', einops.__version__)
except Exception:
    print('ERR einops not installed')

try:
    import PIL
    print('OK  Pillow:', PIL.__version__)
except Exception:
    print('ERR Pillow not installed')

try:
    import cv2
    print('OK  opencv-python:', cv2.__version__)
except Exception:
    print('ERR opencv-python not installed')

try:
    import numpy
    print('OK  numpy:', numpy.__version__)
except Exception:
    print('ERR numpy not installed')

try:
    import matplotlib
    print('OK  matplotlib:', matplotlib.__version__)
except Exception:
    print('ERR matplotlib not installed')

try:
    import wandb
    print('OK  wandb:', wandb.__version__)
except Exception:
    print('ERR wandb not installed')

try:
    import thop
    print('OK  thop installed')
except Exception:
    print('WARN thop not installed (optional)')

try:
    import mmcv
    print('OK  mmcv:', mmcv.__version__)
except Exception:
    print('WARN mmcv not installed (optional)')
"

echo ""
print_info "============================================"
print_success "Environment setup completed"
print_info "============================================"
echo ""
