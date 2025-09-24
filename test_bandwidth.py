import torch
import time


def measure_bandwidth(device, data_size_gb=1, num_iterations=10):
    # 计算数据大小（以元素数量为单位）
    element_size = 4  # 假设数据类型为 float32，每个元素 4 字节
    num_elements = int(data_size_gb * (1024 ** 3) / element_size)

    # 创建张量
    a = torch.randn(num_elements, device=device)
    b = torch.empty_like(a, device=device)

    # 同步设备
    torch.cuda.synchronize(device)

    # 测量带宽
    start = time.time()
    for _ in range(num_iterations):
        b.copy_(a)
    torch.cuda.synchronize(device)
    elapsed = time.time() - start

    # 计算平均时间
    avg_time = elapsed / num_iterations

    # 计算带宽
    bw = (data_size_gb / avg_time)  # GB/s
    return bw


def print_gpu_bandwidth_info():
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPUs detected.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    for i in range(num_gpus):
        device = torch.device(f"cuda:{i}")
        print(f"\nGPU {i} details:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(
            f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024.0 ** 3):.2f} GB")
        print(
            f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")

        # 测量带宽
        bw = measure_bandwidth(device, data_size_gb=1, num_iterations=10)
        print(f"  Device-to-Device bandwidth: {bw:.2f} GB/s")


if __name__ == "__main__":
    print_gpu_bandwidth_info()
