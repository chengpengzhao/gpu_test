import torch
import time


def measure_floating_point_performance(device, matrix_size=1024, num_iterations=10, data_type=torch.float32):
    a = torch.randn(matrix_size, matrix_size, device=device, dtype=data_type)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=data_type)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    # 预热
    for _ in range(5):
        torch.matmul(a, b)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    # 测量
    start = time.time()
    for _ in range(num_iterations):
        torch.matmul(a, b)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.time() - start

    avg_time = elapsed / num_iterations
    flops = 2 * (matrix_size ** 3) / avg_time
    gflops = flops / 1e9
    return gflops


def print_gpu_floating_point_performance(matrix_size=1024, num_iterations=10):
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPUs detected.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    for i in range(num_gpus):
        device = torch.device(f"cuda:{i}")
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i} details:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {props.total_memory / (1024.0 ** 3):.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")

        gflops_single = measure_floating_point_performance(
            device, matrix_size, num_iterations, torch.float32)
        print(
            f"  Single Precision Floating Point Performance: {gflops_single:.2f} GFLOPS")

        gflops_double = measure_floating_point_performance(
            device, matrix_size, num_iterations, torch.float64)
        print(
            f"  Double Precision Floating Point Performance: {gflops_double:.2f} GFLOPS")


if __name__ == "__main__":
    print_gpu_floating_point_performance(matrix_size=2048, num_iterations=10)
