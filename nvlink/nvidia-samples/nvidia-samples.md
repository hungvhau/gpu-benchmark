# Testing NVLink with NVIDIA CUDA Samples


## Basic NVLink Status and Topology Commands

Before running CUDA samples, you can use these commands to check NVLink status and topology:

1. **Check NVLink status:**
	```bash
	nvidia-smi nvlink --status
	```
2. **Show GPU topology:**
	```bash
	nvidia-smi topo -m
	```

These commands help verify NVLink connectivity and GPU topology on your system.

## Running CUDA Samples: p2pBandwidthLatencyTest

To test NVLink bandwidth/latency between GPUs using CUDA samples:

1. **Clone the CUDA Samples repository:**
	```bash
	git clone https://github.com/NVIDIA/cuda-samples.git
	```
2. **Checkout the desired version (e.g., v12.5):**
	```bash
	cd cuda-samples
	git checkout v12.5
	```
3. **Navigate to the p2pBandwidthLatencyTest sample directory:**
	```bash
	cd Samples/5_Domain_Specific/p2pBandwidthLatencyTest/
	```
4. **Build the sample:**
	```bash
	make
	```
5. **Run the test:**
	```bash
	./bin/x86_64/linux/release/p2pBandwidthLatencyTest
	```

This will output bandwidth and latency results for peer-to-peer GPU communication, useful for verifying NVLink performance.
