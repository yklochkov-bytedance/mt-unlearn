import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from vllm import LLM
from vllm.model_executor.parallel_utils.parallel_state import (
    destroy_model_parallel as vllm_destroy_mp
)

from muse_bench.eval import eval_model

def do_eval(corpus,
            model_path,
            tokenizer_path,
            tokenizer,
            temp_dir=None,
            tensor_parallel_size=1,
            generative=True
            ):
    torch.cuda.empty_cache()

    # TODO when using TOFU make load an HF model instead?
    if generative:
        model = LLM(
            model_path, tokenizer=tokenizer_path, trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size
        )
        metrics = {
            "news": ["verbmem_f", "knowmem_f", "knowmem_r", "mmlu_val"],
            "books": ["verbmem_f", "knowmem_f", "knowmem_r", "mmlu_val"],
            "alice": ["verbmem_f", "knowmem_f", "mmlu_val"],
            "general": ["mmlu_val"],
        }[corpus]

    else: # non-generative measures (logit based)
        # use huggingface model instead
        model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
        metrics = {
            "news": ["privleak"],
            "books": ["privleak"],
        }[corpus]

    out = eval_model(
        model,
        tokenizer,
        corpus=corpus,
        metrics=metrics,
        temp_dir=temp_dir
    )

    vllm_destroy_mp()
    del model
    torch.cuda.empty_cache()
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str,
                        help="Path to HF model weights")
    parser.add_argument("--tokenizer", type=str,
                        help="Path to HF tokenizer")
    parser.add_argument("--corpus", type=str,
                        help="Evaluation corpuse: news, books, alice, general")
    parser.add_argument("--non_generative", action="store_true",
                        help=("Use only non-generative metrics, such as privleak."
                              "When not used, generative metrics are used, based "
                              "on completions and question answering"))
    parser.add_argument("--append_result_to", default=None, type=str,
                        help="Append output to a given file as a json.")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    out = do_eval(
        args.corpus,
        args.network,
        args.tokenizer,
        tokenizer,
        temp_dir="tmp_eval",
        tensor_parallel_size=torch.cuda.device_count(),
        generative=(not args.non_generative)
    )
    print(out)

    # add to an existing log file
    if args.append_result_to is not None:
        with open(args.append_result_to, "a") as file:
            file.write(json.dumps(out) + "\n")
