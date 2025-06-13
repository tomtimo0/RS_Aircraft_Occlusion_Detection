import torch
print(torch.__version__)
print(torch.cuda.is_available()) # 如果安装了GPU版本且驱动和CUDA配置正确，应输出 True
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0)) # 输出你的GPU型号