import torch
import time

# PCIE:
# sudo vi /etc/default/grub
# line change to ：GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=off"
# sudo update-grub
# sudo reboot


def measure_inter_gpu_bandwidth(src_device, dst_device, data_size_gb=1, num_iterations=10, dtype=torch.float32):
    """
    测量两个 GPU 之间的数据传输带宽（GB/s），并进行完整正确性检查。
    """
    # 计算元素数量
    element_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = int(data_size_gb * (1024 ** 3) / element_size)

    # 创建源张量
    a = torch.randn(num_elements, device=src_device, dtype=dtype)

    # 为目标设备分配内存
    b = torch.empty_like(a, device=dst_device)

    # 同步设备，预热
    torch.cuda.synchronize(src_device)
    torch.cuda.synchronize(dst_device)
    for _ in range(3):
        b.copy_(a)
        torch.cuda.synchronize(dst_device)

    # 测量带宽
    start = time.time()
    for _ in range(num_iterations):
        b.copy_(a)
        torch.cuda.synchronize(dst_device)
    elapsed = time.time() - start

    avg_time = elapsed / num_iterations
    bw = data_size_gb / avg_time  # GB/s

    # 完整正确性检查
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    if not torch.equal(a_cpu, b_cpu):
        raise ValueError(
            f"Data mismatch detected between GPU {src_device} and GPU {dst_device}!")
    else:
        print("  Data correctness check passed (all elements match).")

    return bw


def print_inter_gpu_bandwidth_info(data_size_gb=1, num_iterations=10):
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPUs detected.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    for src_i in range(num_gpus):
        for dst_i in range(num_gpus):
            if src_i != dst_i:
                src_device = torch.device(f"cuda:{src_i}")
                dst_device = torch.device(f"cuda:{dst_i}")
                print(f"\nTesting bandwidth from GPU {src_i} to GPU {dst_i}:")
                print(f"  Source GPU: {torch.cuda.get_device_name(src_i)}")
                print(
                    f"  Destination GPU: {torch.cuda.get_device_name(dst_i)}")

                bw = measure_inter_gpu_bandwidth(
                    src_device,
                    dst_device,
                    data_size_gb=data_size_gb,
                    num_iterations=num_iterations
                )
                print(f"  Inter-GPU bandwidth: {bw:.2f} GB/s")


if __name__ == "__main__":
    print_inter_gpu_bandwidth_info(data_size_gb=1, num_iterations=10)
