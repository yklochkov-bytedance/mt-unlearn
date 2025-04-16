from .logger import RougeEvalLogger

from vllm import SamplingParams, LLM

from tqdm.contrib import tzip
from typing import List


def eval(
    model, tokenizer,
    prompts: List[str], gts: List[str],
    max_new_tokens : int = 128
):
    if isinstance(model, LLM):
        return _eval_vllm(
            model, tokenizer,
            prompts, gts,
            max_new_tokens=max_new_tokens
        )
    else:
        return _eval(
            model, tokenizer,
            prompts, gts,
            max_new_tokens=max_new_tokens
        )

def _eval(
    model, tokenizer,
    prompts: List[str], gts: List[str],
    max_new_tokens : int = 128
):
    logger = RougeEvalLogger()
    for prompt, gt in tzip(prompts, gts):
        # Encode the `prompt` into `input_ids`
        input_ids = tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=True
        ).input_ids
        assert len(input_ids) == 1

        gt_ids = tokenizer(gt, return_tensors='pt', add_special_tokens=True).input_ids[:, :max_new_tokens]

        # Use the `model` to generate the continuation of the `input_ids`.

        # I'm not sure what's in model.device, presumably it returns 
        # next(model.parameters()).device, which would correpond to the embedding layer,
        # which is always replicated in our implementation
        #
        output_ids = model.generate(
            input_ids.to(model.device), 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
        output_ids = output_ids[:, len(input_ids[0]):]
        output = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]
        gt_short = tokenizer.batch_decode(
            gt_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]
        logger.log(prompt, gt_short, output)

    return logger.report()


def _eval_vllm(
    model, tokenizer,
    prompts: List[str], gts: List[str],
    max_new_tokens : int = 128
):
    logger = RougeEvalLogger()

    sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    outputs = model.generate(prompts, sp)
    outputs = [output.outputs[0].text for output in outputs]

    for prompt, output, gt in tzip(prompts, outputs, gts):
        # Encode the `prompt` into `input_ids`
        input_ids = tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=True
        ).input_ids
        assert len(input_ids) == 1

        gt_ids = tokenizer(gt, return_tensors='pt', add_special_tokens=True).input_ids[:, :max_new_tokens]

        # Use the `model` to generate the continuation of the `input_ids`.

        # I'm not sure what's in model.device, presumably it returns 
        # next(model.parameters()).device, which would correpond to the embedding layer,
        # which is always replicated in our implementation
        #
        gt_short = tokenizer.batch_decode(
            gt_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]
        logger.log(prompt, gt_short, output)

    return logger.report()
