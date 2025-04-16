import torch
from math import sqrt

import tp_utils

def save_optimizer_state(optimizer, reference_model, state_name, folder_path, main_rank=0):
    # save momentums
    states = {}
    for name, par in reference_model.named_parameters():
        if par.requires_grad:
            states[name] = optimizer.state[par][state_name]
    tp_utils.save_named_tensors_as_numpy(states, folder_path, main_rank=main_rank)

def load_optimizer_state(optimizer, reference_model, state_name, folder_path, main_rank=0, check_requires_grad=True):
    loaded_states = tp_utils.load_named_tensors_from_numpy(folder_path, reference_model, main_rank=main_rank)
    for name, par in reference_model.named_parameters():
        if (not check_requires_grad or
            par.requires_grad):
            optimizer.state[par][state_name] = loaded_states[name]

def set_optimizer_state(optimizer, model, state_name, value, check_requires_grad=True):
    for _, par in model.named_parameters():
        if (not check_requires_grad or
            par.requires_grad):
            optimizer.state[par][state_name] = torch.tensor(value)

def clip_gradients(model, value=1.0):
    # distributed clipping
    norm = 0
    for par in model.parameters():
        if par.requires_grad:
            norm += (tp_utils.dist_norm(par.grad).item() ** 2)
    norm = sqrt(norm)

    clip_coef = 1.0
    if norm > value:
        clip_coef = value/norm
        for par in model.parameters():
            if par.requires_grad:
                par.grad.multiply_(clip_coef)

    return norm, clip_coef

def get_optimizer(optim_args, model):
    if optim_args['optimizer'] == "sgd":
        optimizer = torch.optim.SGD(
            [
                par for par in model.parameters()
                if par.requires_grad
            ],
            lr=optim_args['lr'],
            momentum=optim_args['momentum']
        )
    elif optim_args['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            [
                par for par in model.parameters()
                if par.requires_grad
            ],
            lr=optim_args['lr'],
            betas=(optim_args['b1'], optim_args['b2'])
        )
    else:
        raise ValueError
    return optimizer

def get_warmup_lr(lr, warmup, global_step):
    if global_step > 2 * warmup:
        return lr
    if global_step <= warmup:
        return 0.01 * lr
    return ((global_step - warmup) / warmup) * lr
