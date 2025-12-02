import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import time

# -----------------------------------------------------------------------------
# 1. 编译所有 Kernel
# -----------------------------------------------------------------------------
print("Compiling Kernels...")
# 注意：sources 列表里要包含所有 .cu 文件
my_kernels = load(
    name="my_flash_attn_lib",
    sources=[
        "src/interface.cpp",
        "src/kernels/1_fa_baseline.cuh",
    ],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False
)
print("Compilation finished!\n")

# -----------------------------------------------------------------------------
# 2. 准备数据
# -----------------------------------------------------------------------------
BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM = 2, 128, 4, 64
DTYPE = torch.bfloat16
DEVICE = "cuda"

q = torch.randn((BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=DTYPE, device=DEVICE)
k = torch.randn((BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=DTYPE, device=DEVICE)
v = torch.randn((BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=DTYPE, device=DEVICE)

# -----------------------------------------------------------------------------
# 3. 定义测试函数集
# -----------------------------------------------------------------------------
# 官方/PyTorch实现
def run_official():
    # 优先尝试官方 flash-attn，没有则用 PyTorch SDPA
    try:
        from flash_attn import flash_attn_func
        return flash_attn_func(q, k, v, causal=False)
    except ImportError:
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t)
        return out.transpose(1, 2).contiguous()

# 收集你的不同版本
versions = [
    ("Official", run_official),
    ("My V1 (Naive)", lambda: my_kernels.forward_v1(q, k, v)),
]

# -----------------------------------------------------------------------------
# 4. 运行对比
# -----------------------------------------------------------------------------
# 获取基准结果
ref_output = run_official()

print(f"{'Version':<20} | {'Status':<10} | {'Max Diff':<10} | {'Latency (ms)':<10}")
print("-" * 60)

for name, func in versions:
    # --- 正确性 ---
    try:
        out = func()
        if name == "Official":
            status = "Ref"
            diff = 0.0
        else:
            # 允许一定误差 (FP16误差较大)
            is_close = torch.allclose(ref_output, out, atol=1e-2, rtol=1e-2)
            diff = (ref_output - out).abs().max().item()
            status = "✅ Pass" if is_close else "❌ Fail"
    except Exception as e:
        status = "Error"
        diff = -1.0
        print(f"\n[Error in {name}]: {e}")

    # --- 速度 ---
    # 预热
    for _ in range(5): func()
    torch.cuda.synchronize()
    
    # 计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(50):
        func()
    end.record()
    torch.cuda.synchronize()
    avg_time = start.elapsed_time(end) / 50

    print(f"{name:<20} | {status:<10} | {diff:<10.4f} | {avg_time:<10.3f}")