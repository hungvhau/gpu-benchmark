
import torch
import subprocess

print("PyTorch CUDA version:", torch.version.cuda)
try:
    print("cuDNN version:", torch.backends.cudnn.version())
except Exception as e:
    print("cuDNN version not available:", e)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
    print(f"Total CUDA devices: {torch.cuda.device_count()}")
    print(f"can_device_access_peer(0, 1): {torch.cuda.can_device_access_peer(0, 1)}")

# Check if CUDA is available
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    # Check NVLink status using nvidia-smi
    try:
        result = subprocess.run([
            'nvidia-smi', 'nvlink', '--status'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        if result.returncode != 0:
            print("nvidia-smi nvlink command failed:", result.stderr)
        else:
            print("NVLink status:")
            print(output)
    except Exception as e:
        print(f"Error running nvidia-smi nvlink: {e}")

    # Check P2P status using PyTorch and try to enable peer access
    for i in range(num_gpus):
        for j in range(i + 1, num_gpus):
            p2p_available = torch.cuda.can_device_access_peer(i, j)
            print(f"P2P between GPU {i} and GPU {j}: {p2p_available}")
            if not p2p_available:
                try:
                    torch.cuda.set_device(i)
                    torch.cuda.enable_peer_access(j)
                    print(f"Enabled peer access from GPU {i} to GPU {j}.")
                except Exception as e:
                    print(f"Failed to enable peer access from GPU {i} to GPU {j}: {e}")
else:
    print("CUDA is not available.")