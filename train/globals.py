import torch

DEVICE = None

def init_device(allow_cuda):
    global DEVICE
    if allow_cuda:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device("cpu")        # cuda is just slower currently
    print(f"Using device: {DEVICE}")
    return DEVICE


# Define board dimensions
ROWS = 6
COLS = 7

