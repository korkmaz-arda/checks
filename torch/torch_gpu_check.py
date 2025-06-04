import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU is not available")

a = torch.rand(10000, 10000, device=device)
b = torch.rand(10000, 10000, device=device)
c = torch.matmul(a, b)
print(f"Calculation completed on: {device}")