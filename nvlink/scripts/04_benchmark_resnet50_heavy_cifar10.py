import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from tqdm import tqdm

def run_heavy_benchmark():
    """
    Runs a very intensive benchmark designed for high-end GPUs.
    Uses ResNet-50 and a large batch size to ensure the computational load
    is significantly higher than the communication overhead.
    """
    # --- 1. Define Parameters ---
    BATCH_SIZE = 1024  # Maximize the batch size to fit in 24GB VRAM
    NUM_EPOCHS = 5     # Increase epochs to make the run longer and more stable
    LEARNING_RATE = 0.001

    # --- 2. Prepare the Dataset ---
    print("Preparing CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Using more workers can help with data loading on powerful systems
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    print("Dataset ready.")

    # --- 3. Check for GPUs ---
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs!")

    # --- 4. Define the Training Function ---
    def train(model, device):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        model.to(device)
        model.train()

        for epoch in range(NUM_EPOCHS):
            progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
            for i, data in enumerate(progress_bar):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    # --- 5. Single GPU Benchmark ---
    if num_gpus > 0:
        # ResNet-50 is a much larger and more complex model than ResNet-18
        model_single = torchvision.models.resnet50(weights=None, num_classes=10)
        device_single = torch.device("cuda:0")
        print(f"\nStarting HEAVY training on 1 GPU ({device_single})...")
        start_time_single = time.time()
        train(model_single, device_single)
        end_time_single = time.time()
        time_single = end_time_single - start_time_single
        print(f"\nTime for 1 GPU: {time_single:.2f} seconds\n")

    # --- 6. Multi-GPU Benchmark ---
    if num_gpus > 1:
        model_multi = torchvision.models.resnet50(weights=None, num_classes=10)
        print("Wrapping model with nn.DataParallel for multi-GPU training.")
        model_multi = nn.DataParallel(model_multi)
        device_multi = torch.device("cuda:0")
        print(f"Starting HEAVY training on device: {device_multi}")
        start_time_multi = time.time()
        train(model_multi, device_multi)
        end_time_multi = time.time()
        time_multi = end_time_multi - start_time_multi
        print(f"\nTime for {num_gpus} GPUs: {time_multi:.2f} seconds\n")

    # --- 7. Summary ---
    if num_gpus > 1:
        print("--- Benchmark Summary ---")
        print(f"Single GPU Time: {time_single:.2f} seconds")
        print(f"Multi-GPU Time:  {time_multi:.2f} seconds")
        if time_multi > 0:
            speedup = time_single / time_multi
            print(f"Speedup: {speedup:.2f}x")

if __name__ == '__main__':
    run_heavy_benchmark()