"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    # Check if distributed is available
    if not dist.is_available():
        print("Warning: PyTorch distributed is not available. Running in single-process mode.")
        return
    
    # Check if distributed is already initialized (for newer PyTorch versions)
    if hasattr(dist, 'is_initialized') and dist.is_initialized():
        return
    
    # For single device training, we don't need to initialize distributed
    return


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.backends.mps.is_available():
        return th.device("mps")
    elif th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize parameters (no-op for single device).
    """
    pass


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
