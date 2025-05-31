import cupy as cp
import numpy as np
import time

def benchmark_pcie(data_size):
    # 1. Tạo dữ liệu trên CPU (Host) sử dụng NumPy
    h_data = np.ones(data_size, dtype=np.float32)

    # 2. Chuyển dữ liệu từ Host (CPU) sang Device (GPU)
    start_h2d = time.time()
    d_data = cp.asarray(h_data)  # CuPy tự động chuyển từ NumPy (host) sang CuPy (device)
    end_h2d = time.time()

    # Thời gian truyền H2D
    time_h2d = end_h2d - start_h2d
    bandwidth_h2d = (data_size * h_data.itemsize) / (1024**3) / time_h2d  # GB/s
    print(f"Host to Device (H2D): {bandwidth_h2d:.2f} GB/s")

    # 3. Chuyển dữ liệu từ Device (GPU) về Host (CPU)
    start_d2h = time.time()
    h_result = cp.asnumpy(d_data)  # Chuyển từ CuPy (device) về NumPy (host)
    end_d2h = time.time()

    # Thời gian truyền D2H
    time_d2h = end_d2h - start_d2h
    bandwidth_d2h = (data_size * h_data.itemsize) / (1024**3) / time_d2h  # GB/s
    print(f"Device to Host (D2H): {bandwidth_d2h:.2f} GB/s")

if __name__ == "__main__":
    # Kích thước dữ liệu (số phần tử, mỗi phần tử 4 byte - float32)
    data_size = 1 << 30  # ~256 MiB (2^26 elements * 4 bytes/element)

    print("Benchmark PCIe Bandwidth:")
    benchmark_pcie(data_size)
