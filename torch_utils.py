import os

import torch
import numpy as np

def transfer_batch_to_device(batch, device):
    """
        transfer a tuple, list, or dict of tensors to device
        device can be either int (denoting index of GPU) or torch.device instance
    """
    if isinstance(device, int):
        device = torch.device(device)
    if isinstance(batch, tuple):
        return tuple(transfer_batch_to_device(item, device) for item in batch)
    elif isinstance(batch, list):
        return [transfer_batch_to_device(item, device) for item in batch]
    elif isinstance(batch, dict):
        return {key: transfer_batch_to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        try:
            return batch.to(device)
        except:
            raise TypeError(
                f"Cannot send batch to device, got {type(batch)}."
                f" Expect one of: `tuple`, `list`, `torch.Tensor`."
                f" Otherwise, implement batch.to(device) for your custom type."
            )

def load_model_tensors_from_numpys(model, path):
    for name, par in model.named_parameters():
        file_name = f"{path}/{name}.npy"
        if os.path.exists(file_name):
            nparray = np.load(file_name)
            par.data[...] = torch.from_numpy(nparray)
