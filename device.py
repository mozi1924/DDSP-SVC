import torch


def is_mps_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    if is_mps_available():
        return "mps"
    return "cpu"


def get_device_type(device):
    return torch.device(str(device)).type


def resolve_device(device=None):
    if device is None:
        return get_default_device()

    device_str = str(device).strip()
    if device_str == "" or device_str.lower() == "auto":
        return get_default_device()

    device_type = get_device_type(device_str)
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(" [x] CUDA device requested, but CUDA is not available")
        return device_str
    if device_type == "mps":
        if not is_mps_available():
            raise ValueError(" [x] MPS device requested, but MPS is not available")
        return device_str
    if device_type == "cpu":
        return "cpu"

    raise ValueError(f" [x] Unsupported device: {device_str}")


def clear_device_cache(device):
    device_type = get_device_type(device)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
