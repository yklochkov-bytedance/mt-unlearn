import torch
import torch.multiprocessing as mp
from torch.utils.data import RandomSampler
from transformers import DataCollatorForLanguageModeling

import numpy as np
import argparse
import os
import time
from tqdm import trange

from utils import Logger, yaml_config_hook

import tp_utils

from torch_utils import (
    transfer_batch_to_device,
    load_model_tensors_from_numpys
)
import optim_utils

from networks import get_network, get_tokenizer
from dataset import get_dataset

import losses
import evaluations

from eval_and_log import do_eval


def set_deterministic_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_random_dataloader(dataset, batch_size, num_batches, collate_fn):
    sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size*num_batches)
    dataloader = tp_utils.BroadcastingDataLoader(
        dataset, 0,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
        use_tqdm=False
    )
    return dataloader

def get_full_dataloader(dataset, batch_size, collate_fn):
    dataloader = tp_utils.BroadcastingDataLoader(
        dataset, 0,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        use_tqdm=False
    )
    return dataloader

def run_epoch(
        rank,
        epoch,
        model,
        model_ref,
        model_base,
        dataset_pt,
        dataset,
        tokenizer,
        args,
        logger,
        world_size
    ):
    tp_utils.setup(rank, world_size)
    tp_utils.print_rank("Starting the worker...")
    set_deterministic_seed(args['seed'] + epoch * 100)

    # create a sharding plan based on the given world_size and shard.
    device_mesh = tp_utils.make_device_mesh(world_size)
    # is one of lanuage models
    model = tp_utils.parallelize_language_model(model, device_mesh)
    model_ref = tp_utils.parallelize_language_model(model_ref, device_mesh)
    if model_base is not None:
        model_base = tp_utils.parallelize_language_model(model_base, device_mesh)

    tp_utils.print_memory(rank)

    optimizer = optim_utils.get_optimizer(args, model)
    # loading the Optimizer States if epoch > 0
    if epoch > 0:
        if args['optimizer'] == 'sgd':
            if args['momentum'] > 0:
                optim_utils.load_optimizer_state(optimizer, model,
                                                 "momentum_buffer", "mom_buf")
        elif args['optimizer'] == 'adamw':
            optim_utils.load_optimizer_state(optimizer, model,
                                             "exp_avg", "exp_avg_buf")
            optim_utils.load_optimizer_state(optimizer, model,
                                             "exp_avg_sq", "exp_avg_sq_buf")
            optim_utils.set_optimizer_state(optimizer, model, "step",
                                            args['steps'] * epoch)

    print(f"Size of train data: {len(dataset)}")

    # define datasets
    eval_steps = max(1, int(epoch == 0) + args['steps'] // args['eval_every'])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = get_random_dataloader(
        dataset,
        args['micro_batch'],
        args['steps'] * args['forget_accum_steps'],
        collator
    )
    train_loader = iter(train_loader)

    eval_loader = get_random_dataloader(
        dataset,
        args['micro_batch'],
        eval_steps * args['eval_accum_steps'],
        collator
    )
    eval_loader = iter(eval_loader)

    pt_loader = get_random_dataloader(
        dataset_pt,
        args['micro_batch'],
        args['steps'] * args['kl_accum_steps'] + eval_steps * args['eval_accum_steps'],
        collator
    )
    pt_loader = iter(pt_loader)

    steps_range = (trange if rank == 0 else range)(args['steps'])

    for step in steps_range:
        # define global step variable
        global_step = epoch * args['steps'] + step

        # set learning rate
        if args['optimizer'] == "adamw":
            optimizer.param_groups[0]['lr'] = optim_utils.get_warmup_lr(
                args['lr'], args['warmup'], global_step
            )

        # CALCULATE ascent gradient
        time_tr = time.time()
        loss_tr = 0
        for _ in range(args['forget_accum_steps']):
            batch_tr = next(train_loader)
            batch_tr = transfer_batch_to_device(batch_tr, rank)

            if args['loss'] in ["kl", "qkl", "dpo"]:
                loss = {
                    "kl": losses.kl,
                    "qkl": losses.qkl,
                    "dpo": losses.dpo_loss,
                }[args['loss']](model, model_base, batch_tr, args)
                loss_val = loss.item()
            elif args['loss'] in ['npo']:
                loss, loss_val = {
                    "npo": losses.npo,
                }[args['loss']](model, model_base, batch_tr, args)
            else:
                loss, loss_val = {
                    "nll": losses.neg_loglikelihood,
                    "ll": losses.loglikelihood,
                    "nlul": losses.neg_log_unlikelihood,
                }[args['loss']](
                    model, batch_tr, args, return_nll_val=True
                )
            loss /= args['forget_accum_steps']
            loss_tr += (loss_val / args['forget_accum_steps'])
            (loss * args['alpha']).backward()

        time_tr = time.time() - time_tr

        # CALCULATE KL gradient
        time_kl = time.time()
        loss_kl = 0
        for _ in range(args['kl_accum_steps']):
            batch_pt = next(pt_loader)
            batch_pt = transfer_batch_to_device(batch_pt, rank)

            # get penalty according to the function specified in args
            penalty = {
                "qkl": losses.qkl,
                "kl": losses.kl,
            }[args["penalty"]](model, model_ref, batch_pt, args)
            penalty /= args['kl_accum_steps']
            (penalty * args['beta']).backward()
            loss_kl += penalty.item()

        time_kl = time.time() - time_kl
        grad_norm, clip_coef = optim_utils.clip_gradients(model, value=args['clip'])

        # ADD DAMPING TERM TO GRADIENT
        if args['damp'] != 0:
            for par, par_ref in zip(
                    model.parameters(),
                    model_ref.parameters()
                ):
                if par.requires_grad:
                    par.grad.add_(
                        par.data - par_ref.data,
                        alpha=args['damp'] * clip_coef
                    )

        # DO STEP
        optimizer.step()
        model.zero_grad()

        # DO EVALUATION OF PT LOSS
        loss_eval_pt = 0
        loss_eval = 0
        acc_eval = 0
        if (step == 0 and epoch == 0) or ((step + 1) % args['eval_every'] == 0):
            for _ in range(args['eval_accum_steps']):
                batch_eval_pt = next(pt_loader)
                batch_eval_pt = transfer_batch_to_device(batch_eval_pt, rank)
                inputs = batch_eval_pt['input_ids']
                mask = batch_eval_pt['attention_mask']
                with torch.no_grad():
                    loss_eval_pt += (
                        model(inputs, attention_mask=mask, labels=inputs)['loss'].item()
                        / args['eval_accum_steps']
                    )

            for _ in range(args['eval_accum_steps']):
                batch_eval = next(eval_loader)
                batch_eval = transfer_batch_to_device(batch_eval, rank)

                curr_loss, curr_acc = evaluations.nll_and_acc(model, batch_eval)
                loss_eval += (curr_loss / args['eval_accum_steps'])
                acc_eval += (curr_acc / args['eval_accum_steps'])

        # DO CONTRACTION
        for par, par_ref in zip(model.parameters(), model_ref.parameters()):
            delta = args['delta'] * clip_coef
            if par.requires_grad and args['delta'] != 0:
                par_ref.data.mul_(1 - delta)
                par_ref.data.add_(par.data, alpha=delta)

        # LOGGING
        if rank == 0:
            log = {
                "global_step": global_step, #epoch * args['steps'] + step,
                "loss_tr": loss_tr,
                "loss_kl": loss_kl,
                "eval_loss_pt": loss_eval_pt,
                "eval_loss": loss_eval,
                "eval_acc": acc_eval,
                "time_kl": time_kl,
                "time_tr": time_tr,
                "grad_norm": grad_norm,
            }
            logger.dump(log)

            steps_range.set_postfix(**{k: f"{v:.2g}" for k, v in log.items()})

    # save model and reference
    tp_utils.save_model_as_numpy(model, "tmp_model")
    tp_utils.save_model_as_numpy(model_ref, "tmp_ref")

    # save momentums
    if args['optimizer'] == "sgd":
        if args['momentum'] > 0:
            optim_utils.save_optimizer_state(optimizer, model, "momentum_buffer", "mom_buf")
    elif args['optimizer'] == 'adamw': # TODO: generalize
        optim_utils.save_optimizer_state(optimizer, model, "exp_avg", "exp_avg_buf")
        optim_utils.save_optimizer_state(optimizer, model, "exp_avg_sq", "exp_avg_sq_buf")

    tp_utils.print_memory(rank)
    tp_utils.cleanup()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()

    # The main entry point is called directly without using subprocess
    if n_gpus < 1:
        print("Requires at least 1 GPU to run.")
    else:
        parser = argparse.ArgumentParser()
        config = yaml_config_hook(os.environ.get("CONFIG_PATH", "./config.yaml"))
        for k, v in config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))
        args = parser.parse_args()
        args = vars(args)

        # create logger
        log_filename = os.path.join(
            args["log_dir"],
            'exp_{}_seed_{}.out'.format(args["exp_id"], args["seed"])
        )
        print(log_filename)
        logger = Logger(log_filename)
        logger.dump({"args": args})

        model_cpu = get_network(args["network"])
        model_ref = get_network(args["network"])
        model_cpu.eval()
        model_ref.eval()

        # load base model if required
        model_base = None
        if args["network_base"]:
            model_base = get_network(args["network_base"]).half()
            model_base.eval()

        # We are setting 'high' precision to make use of TensorFloat32 on A100/H100 GPUs
        torch.set_float32_matmul_precision('high')

        # load pt and task datasets
        tokenizer = get_tokenizer(args['tokenizer'])
        dataset = get_dataset(
            args['dataset'],
            tokenizer,
            split=args['split'],
            context=args['context'],
            chunking=args['use_chunking']
        )
        dataset_pt = get_dataset(
            args['dataset_pt'],
            tokenizer,
            split=args['split_pt'],
            context=512,
            chunking=True
        )

        out = do_eval(
            args['gen_eval_name'],
            args["network"],
            args["tokenizer"],
            tokenizer
        )
        if args['stop_metric'] is not None and args['stop_metric']:
            metric_best = out[args['stop_metric']]
        logger.dump({"epoch": -1, "metrics": out})

        for epoch in range(args['epochs']):

            tp_utils.share_params_and_buffers(model_cpu)
            tp_utils.share_params_and_buffers(model_ref)
            if model_base is not None:
                tp_utils.share_params_and_buffers(model_base)

            mp_args = (
                epoch,
                model_cpu,
                model_ref,
                model_base,
                dataset_pt,
                dataset,
                tokenizer,
                args,
                logger,
                n_gpus,
            )

            if n_gpus > 1:
                mp.spawn(run_epoch, args=mp_args, nprocs=n_gpus, join=True)
            else:
                run_epoch(0, *mp_args)

            # load models saved in MP
            load_model_tensors_from_numpys(model_cpu, "tmp_model")
            load_model_tensors_from_numpys(model_ref, "tmp_ref")

            # saving the model in Huggingface format,
            # so we can run generative evaluations with vLLM
            model_cpu.save_pretrained("checkpoints/temp")
            out = do_eval(
                args['gen_eval_name'],
                "checkpoints/temp",
                args['tokenizer'],
                tokenizer
            )

            # log the evaluations
            logger.dump({"epoch": epoch, "metrics": out})

            # (we are training until 'stop_metric' reaches below 'stop_val')
            # update best metric and save model; if required, stop the loop:
            if args['stop_metric'] is not None and args['stop_metric']:
                if (metric_best > out[args['stop_metric']] >= args['stop_val']):
                    metric_best = out[args['stop_metric']]
                    model_cpu.save_pretrained("checkpoints/best")

                if out[args['stop_metric']] < args['stop_val']:
                    break
