# Local LLM Benchmarking Script

This repository contains a simple Python script to benchmark the performance of local, open-source Large Language Models (LLMs) from the Hugging Face Hub. It measures token generation speed and hardware utilization (CPU, System RAM, and GPU VRAM) to give you an idea of how well your workstation can run different models.

## Features

- **Model Loading Time:** Measures how long it takes to load a model into memory.
- **Inference Speed:** Calculates Tokens per Second (TPS) to show how fast the model generates text.
- **Hardware Monitoring:** Tracks peak usage for:
  - CPU (%)
  - System RAM (MB)
  - GPU VRAM (MB) for NVIDIA GPUs.

---

## 1. Setup with uv

This project uses [uv](https://github.com/astral-sh/uv), a fast Python package manager.

**Step 1: Install uv**  
Follow the official uv installation instructions for your operating system:  
https://github.com/astral-sh/uv

**Step 2: Create a Virtual Environment**  
Open your terminal in the project directory and run:
```bash
uv venv
```
This will create a `.venv` folder for your project's dependencies.

**Step 3: Activate the Environment**  
On macOS and Linux:
```bash
source .venv/bin/activate
```

**Step 4: Install Dependencies**  
Install the required Python packages using uv pip:
```bash
uv pip install torch transformers psutil huggingface-hub
```

**For NVIDIA GPU Monitoring (Optional):**  
If you have an NVIDIA GPU and want to monitor VRAM usage, install the pynvml library:
```bash
uv pip install pynvml
```

---

## 2. How to Run the Benchmark

You can run the script from your terminal. Use the `--models` flag to specify which models from the Hugging Face Hub you want to test.

**Example (Default Models):**
```bash
python llm_benchmark.py
```
This will run the benchmark for `gpt2` and `distilgpt2`.

**Example (Custom Models):**  
You can test other models, like Microsoft's Phi-3 mini. Be aware that larger models will require more resources and download time.
```bash
python llm_benchmark.py --models gpt2 microsoft/Phi-3-mini-4k-instruct
```

**Customizing the Prompt and Length:**
```bash
python llm_benchmark.py --prompt "The future of AI is" --max-new-tokens 100
```

> **Note:** The first time you run the benchmark for a specific model, it will be downloaded from Hugging Face, which may take some time depending on the model size and your internet connection. Subsequent runs will use the cached version and will be much faster to start.

---

## 3. Understanding the Metrics

- **Model loaded in:** The time it took to load the model weights from your disk into RAM/VRAM.
- **Total Generation Time:** The total wall-clock time spent generating the new tokens.
- **Tokens per Second (TPS):** The key metric for inference speed. A higher number means faster text generation.
- **Peak CPU Usage:** The highest percentage of your CPU used during text generation.
- **Peak System RAM Usage:** The maximum amount of system RAM (in megabytes) used by the script during generation.
- **VRAM Used for Model:** The amount of GPU memory (in megabytes) occupied by just the model's weights.
- **Peak VRAM Usage:** The maximum GPU memory used during the entire generation process. The difference between this and "VRAM Used for Model" shows how much extra VRAM is needed for the actual computation (the "context").