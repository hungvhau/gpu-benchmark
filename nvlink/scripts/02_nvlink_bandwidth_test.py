import torch
import time

if torch.cuda.device_count() < 2:
    print("This test requires at least 2 GPUs.")
else:
    # A large tensor, ~3.2 GB
    tensor_size = (2048, 2048, 200) 
    x = torch.randn(tensor_size, device='cuda:0')
    
    print("Starting bandwidth test...")
    
    # Warm-up transfers
    for _ in range(5):
        _ = x.to('cuda:1')
        torch.cuda.synchronize()

    # Timed transfer
    start_time = time.time()
    for _ in range(10):
        # Copy from GPU 0 to GPU 1
        y = x.to('cuda:1') 
        # Crucial: wait for the operation to finish
        torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate bandwidth
    duration = end_time - start_time
    # Data size * number of transfers
    total_data_gb = (x.nelement() * x.element_size() * 10) / (1024**3)
    bandwidth = total_data_gb / duration
    
    print(f"Transfer from GPU 0 to GPU 1 took {duration:.4f} seconds.")
    print(f"Measured Bandwidth: {bandwidth:.4f} GB/s")