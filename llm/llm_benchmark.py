# llm_benchmark.py
# A script to benchmark the performance of local Hugging Face models.

import time
import threading
import argparse
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Hardware Monitoring ---
# We use separate monitoring for NVIDIA GPUs and general system stats (CPU/RAM).

try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_SMI = True
except (ImportError, pynvml.NVMLError):
    NVIDIA_SMI = False

class SystemMonitor:
    """Monitors CPU and RAM usage in a separate thread."""
    def __init__(self):
        self.stop_event = threading.Event()
        self.monitoring_thread = threading.Thread(target=self._monitor, daemon=True)
        self.peak_cpu = 0
        self.peak_ram = 0

    def _monitor(self):
        """The monitoring loop that runs in the background."""
        while not self.stop_event.is_set():
            # Get current CPU and RAM usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram_mb = psutil.virtual_memory().used / (1024 ** 2)

            # Update peak values
            self.peak_cpu = max(self.peak_cpu, cpu_percent)
            self.peak_ram = max(self.peak_ram, ram_mb)
            time.sleep(0.1) # Check every 100ms

    def start(self):
        """Starts the monitoring thread."""
        self.monitoring_thread.start()

    def stop(self):
        """Stops the monitoring thread and returns peak values."""
        self.stop_event.set()
        self.monitoring_thread.join()
        return self.peak_cpu, self.peak_ram

def get_gpu_memory_usage():
    """Gets peak GPU memory usage if an NVIDIA GPU is available."""
    if not NVIDIA_SMI:
        return 0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 2)
    except pynvml.NVMLError:
        return 0

# --- Benchmarking Logic ---

def benchmark_model(model_id: str, prompt: str, max_new_tokens: int):
    """
    Runs a benchmark for a given Hugging Face model ID.

    Args:
        model_id: The Hugging Face model identifier (e.g., "gpt2").
        prompt: The input text to feed to the model.
        max_new_tokens: The number of tokens to generate.
    """
    print("-" * 50)
    print(f"🚀 Benchmarking Model: {model_id}")
    print("-" * 50)

    # Check for GPU availability
    if torch.cuda.is_available():
        print("Using device: CUDA (single GPU)")
        device = "cuda"
    else:
        print("Using device: CPU")
        device = "cpu"

    # 1. Model Loading Benchmark
    start_time = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    except Exception as e:
        print(f"\n❌ Error loading model {model_id}. Skipping.")
        print(f"   Reason: {e}")
        return

    load_time = time.time() - start_time
    print(f"✅ Model loaded in: {load_time:.2f} seconds")

    initial_vram = get_gpu_memory_usage()

    # 2. Inference Benchmark
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Start system monitoring
    monitor = SystemMonitor()
    monitor.start()

    # Time to First Token (TTFT) and Tokens per Second (TPS)
    start_gen_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id # Suppress warning
    )
    generation_time = time.time() - start_gen_time

    # Stop monitoring to get peak values
    peak_cpu, peak_ram = monitor.stop()
    peak_vram = get_gpu_memory_usage()

    # Decode the output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_output_tokens = len(outputs[0]) - len(inputs.input_ids[0])

    # Calculate metrics
    # Note: A simple TTFT isn't easily measured with a single `generate` call.
    # We calculate average token generation speed instead.
    tokens_per_second = num_output_tokens / generation_time if generation_time > 0 else 0

    # --- Display Results ---
    print("\n--- Benchmark Results ---")
    print(f"📝 Prompt: '{prompt}'")
    print(f"📜 Generated Output ({num_output_tokens} tokens): '{output_text[len(prompt):]}...'")
    print("\n--- Performance ---")
    print(f"⏱️ Total Generation Time: {generation_time:.2f} s")
    print(f"⚡ Tokens per Second (TPS): {tokens_per_second:.2f} tokens/s")
    print("\n--- Hardware Utilization ---")
    print(f"🧠 Peak CPU Usage: {peak_cpu:.2f}%")
    print(f"💾 Peak System RAM Usage: {peak_ram:.2f} MB")
    if NVIDIA_SMI:
        vram_used_during_gen = peak_vram - initial_vram
        print(f"🎮 VRAM Used for Model: {initial_vram:.2f} MB")
        print(f"📈 Additional VRAM for Generation: {vram_used_during_gen:.2f} MB")
        print(f"🔝 Peak VRAM Usage: {peak_vram:.2f} MB")
    print("-" * 50 + "\n")


def main():
    """Main function to parse arguments and run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark local Hugging Face models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2", "distilgpt2"],
        help="A list of Hugging Face model IDs to benchmark."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, my name is",
        help="The prompt to use for generation."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="The number of new tokens to generate."
    )
    args = parser.parse_args()

    # Add a check for torch/cuda
    if torch.cuda.is_available() and not NVIDIA_SMI:
        print("⚠️ Warning: PyTorch detects a CUDA device, but the `pynvml` library is not installed.")
        print("   GPU memory usage will not be monitored. To fix, run: uv pip install pynvml")


    for model_id in args.models:
        benchmark_model(model_id, args.prompt, args.max_new_tokens)

if __name__ == "__main__":
    main()
