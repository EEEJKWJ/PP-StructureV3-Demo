"""Verify that PyTorch and Paddle can share one CUDA 12.9 process."""

import torch
import paddle

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"torch_cudnn={torch.backends.cudnn.version()}")
print(f"torch_available={torch.cuda.is_available()}")
print(f"torch_device={torch.cuda.get_device_name(0)}")
print(f"torch_capability={torch.cuda.get_device_capability(0)}")

print(f"paddle={paddle.__version__}")
print(f"paddle_cuda={paddle.version.cuda()}")
print(f"paddle_device={paddle.device.get_device()}")
print(f"paddle_gpu_count={paddle.device.cuda.device_count()}")

torch_value = (torch.ones((2, 2), device="cuda") * 2).sum().item()
paddle_value = (paddle.ones((2, 2)).cuda() * 3).sum().item()
print(f"torch_result={torch_value}")
print(f"paddle_result={paddle_value}")
