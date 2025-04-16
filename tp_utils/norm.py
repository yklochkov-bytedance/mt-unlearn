import torch
from torch.distributed._tensor import (
    DTensor,
    Replicate,
    Shard
)

def dist_norm(dtensor):
    """
        This function can be used to calculate norm of a distributed tensor.

        Input:
            tensor: Union[torch.Tensor, DTensor]
        Output:
            value: torch.Tensor (scalar)
    """

    if type(dtensor) is torch.Tensor:
        local_tensor = dtensor
    elif type(dtensor) is DTensor:
        local_tensor = dtensor._local_tensor
    else:
        print("Unkown tensor")
        print(dtensor)
        raise ValueError

    norm = local_tensor.norm().unsqueeze(0)
    if (type(dtensor) is DTensor and
        type(dtensor.placements[0]) is Shard):
            norm = DTensor.from_local(norm, dtensor.device_mesh, [Shard(0)])
            norm = norm.redistribute(dtensor.device_mesh, [Replicate()])
            norm = norm.to_local()

    max_norm = norm.max()
    norm = norm / max_norm # avoid large numbers before taking squares
    norm = max_norm * norm.square().sum().sqrt()
    return norm
