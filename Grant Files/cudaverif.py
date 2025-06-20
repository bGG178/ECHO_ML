import torch

cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}") # This should now be True

if cuda_available:
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    current_gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Current GPU name: {current_gpu_name}")
    pytorch_cuda_version = torch.version.cuda
    print(f"PyTorch was compiled with CUDA version: {pytorch_cuda_version}")
else:
    print("PyTorch still cannot access CUDA. There might be other issues.")

print(f"PyTorch version: {torch.__version__}")