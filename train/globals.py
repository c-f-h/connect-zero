import torch

_DEVICE = None

def init_device(allow_cuda):
    global _DEVICE
    if allow_cuda:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _DEVICE = torch.device("cpu")        # cuda is just slower currently
    print(f"Using device: {_DEVICE}")
    return _DEVICE

def get_device():
    if _DEVICE is None:
        raise RuntimeError("Device not initialized. Call init_device first.")
    return _DEVICE


# Define board dimensions
ROWS = 6
COLS = 7

