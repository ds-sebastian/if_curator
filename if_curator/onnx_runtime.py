"""Initialize ONNX Runtime independently of PyTorch import order."""


def preload_cuda(ort, providers):
    """Load the installed NVIDIA runtime libraries before creating CUDA sessions.

    Available providers describe the wheel's capabilities, not whether its shared
    libraries can be loaded. Use ORT's supported loader for the NVIDIA packages
    supplied by our PyTorch dependency, without requiring a system CUDA install.
    CPU-only and non-CUDA execution do not need this initialization.
    """
    if "CUDAExecutionProvider" in providers:
        ort.preload_dlls()
