# Custom Flash Attention CUDA Kernel

Implementation of Flash Attention (forward pass) from scratch using CUDA and PyTorch C++ Extension.
Aligned with Flash Attention 2 interface.

## Prerequisites

- NVIDIA GPU (Ampere or newer recommended for BF16)
- PyTorch (CUDA version)
- `ninja` build system (`pip install ninja`)

## Project Structure

- `src/`: CUDA kernels and C++ interface
- `benchmark.py`: JIT compiler, correctness check, and speed benchmark

## Usage

Run the benchmark script to compile (JIT) and test:

```bash
python benchmark.py