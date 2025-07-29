__version__ = '0.22.1+cu126'
git_version = '59a3e1f9f78cfe44cb989877cc6f4ea77c8a75ca'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
