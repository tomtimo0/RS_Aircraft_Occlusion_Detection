import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices count:", torch.cuda.device_count())