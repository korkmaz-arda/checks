import torch
import time

size = (5000, 5000)

a_cpu = torch.rand(size)
b_cpu = torch.rand(size)

start_cpu = time.time()
c_cpu = torch.matmul(a_cpu, b_cpu)
end_cpu = time.time()
print(f"CPU time: {end_cpu - start_cpu:.4f} seconds")

if torch.cuda.is_available():
    device = torch.device("cuda")
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)
    torch.cuda.synchronize()
    start_gpu = time.time()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    end_gpu = time.time()
    print(f"GPU time: {end_gpu - start_gpu:.4f} seconds")
else:
    print("GPU is not available")