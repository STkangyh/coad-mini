"""
CUDA fallback shim for running PyCIL on machines without NVIDIA GPUs.

PyCIL hardcodes `.cuda()` in many places (utils/inc_net.py, several models).
Rather than patching each call site, this shim redirects Tensor.cuda() /
Module.cuda() to MPS (Apple GPU) or CPU when CUDA is unavailable.

Injected by benchmarks/pycil/setup_pycil.sh as the first import of PyCIL's
main.py. No effect on CUDA machines.
"""
import torch

if not torch.cuda.is_available():
    _dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def _tensor_cuda(self, *args, **kwargs):
        return self.to(_dev)

    def _module_cuda(self, *args, **kwargs):
        return self.to(_dev)

    torch.Tensor.cuda = _tensor_cuda
    torch.nn.Module.cuda = _module_cuda
    print(f"[cuda_shim] CUDA unavailable -> .cuda() redirected to {_dev}")
