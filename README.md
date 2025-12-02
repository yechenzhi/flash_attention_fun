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

---

### 4. 初始化本地 Git 仓库
打开终端，进入你的项目文件夹，执行以下命令：

```bash
# 1. 初始化 git
git init

# 2. 将文件添加到暂存区
# 注意：因为有 .gitignore，它会自动忽略垃圾文件
git add .

# 3. 提交第一次 commit
git commit -m "Initial commit: Naive implementation of Flash Attn forward pass with BF16 support"

# 4. (可选) 将默认分支重命名为 main
git branch -M main