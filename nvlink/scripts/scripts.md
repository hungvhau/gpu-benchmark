# ⚡ PyTorch Dual-GPU Benchmark

This project provides a simple script to benchmark the training speed of a deep learning model on a single GPU versus a dual-GPU setup. It is configured to run on a Windows 11 machine with two NVIDIA RTX Titan GPUs, utilizing WSL2 and an Ubuntu 22.04 environment.

## 🎯 Project Goal

The primary goal is to provide a clear, quantitative comparison of training time between using one GPU and two GPUs with PyTorch's `nn.DataParallel`. This helps in understanding the real-world performance gains for deep learning tasks when scaling hardware.

## 🧰 Prerequisites

Before running the benchmark, ensure your system is properly configured:

- **System**: Windows 11 with WSL2 enabled  
- **Linux Distribution**: Ubuntu 22.04 under WSL2  
- **Hardware**: Two NVIDIA TITAN RTX GPUs  
- **Host Driver**: Latest NVIDIA drivers installed on Windows
- **WSL2 Environment**:
  - NVIDIA CUDA Toolkit (e.g., v12.6) installed
  - `nvidia-smi` shows both GPUs inside Ubuntu terminal
  - `uv` Python package manager installed

## 🚀 Setup and Installation

Run the following steps inside your Ubuntu 22.04 terminal:

```bash
# Clone or create project directory
git clone git@github.com:hungvhau/gpu-benchmark.git

# Create and activate virtual environment
cd nvlink/scripts
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install tqdm

# Run the python scrips
python 01_nvlink_and_p2p_status_check.py
python 02_nvlink_bandwidth_test.py
python 03_benchmark_resnet18_cifar10.py
python 04_benchmark_resnet50_heavy_cifar10.py
```