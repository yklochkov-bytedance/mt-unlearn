"""
Implementation of Megatron style parallelizm for Huggingface models. See https://arxiv.org/pdf/1909.08053.pdf

Parallelizes attention and MLP layers. Does not affect embeddings, dropouts and layer normalizations.

Supported models:
    - GPT2
    - OPT
    - GPT-J
    - Mistral-7b
    - Llama 1 and 2

Remark:
This implementation uses Torch v2.0.1 and tolerates versions up to torch v.2.1.2.
Some of the classes and functions have to be imported from torch.distributed._tensor,
which in the latest version is depricated, and API for distribute_module function has changed.
"""

__all__ = [
    "IS_DISTRIBUTED",
    "parallelize_language_model",
    "setup",
    "cleanup",
    "print_rank",
    "print_memory",
    "make_device_mesh",
    "share_params_and_buffers",
    "get_rank",
    "BroadcastingDataLoader",
    "save_named_tensors_as_numpy",
    "save_model_as_numpy",
    "load_named_tensors_from_numpy",
    "dist_norm"
]

import torch
if torch.cuda.device_count() > 1:
    print("Distributed regime.")
    IS_DISTRIBUTED = True
    from tp_utils.hf_models import parallelize_language_model
    from tp_utils.utils import (
        setup,
        cleanup,
        print_rank,
        print_memory,
        make_device_mesh,
        share_params_and_buffers,
        get_rank,
        BroadcastingDataLoader,
        save_model_as_numpy,
        save_named_tensors_as_numpy,
        load_named_tensors_from_numpy
    )
else:
    print("Non-distributed regime.")
    IS_DISTRIBUTED = False
    def parallelize_language_model(model, device_mesh):
        assert(device_mesh is None)
        return model.to(0)
    from tp_utils.utils_dummy import (
        setup,
        cleanup,
        print_rank,
        print_memory,
        make_device_mesh,
        share_params_and_buffers,
        get_rank,
        BroadcastingDataLoader,
        save_model_as_numpy,
        save_named_tensors_as_numpy,
        load_named_tensors_from_numpy
    )

from tp_utils.norm import dist_norm
