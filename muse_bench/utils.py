import torch
import json
import pandas as pd
import os
from typing import List, Dict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM


def read_json(fpath):
    with open(fpath, 'r') as f:
        return json.load(f)


def read_text(fpath: str) -> str:
    with open(fpath, 'r') as f:
        return f.read()


def write_json(obj: Union[Dict, List], fpath: str):
    dirpath = os.path.dirname(fpath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(fpath, 'w') as f:
        return json.dump(obj, f)


def write_text(obj: str, fpath: str):
    dirpath = os.path.dirname(fpath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(fpath, 'w') as f:
        return f.write(obj)


def write_csv(obj, fpath: str):
    dirpath = os.path.dirname(fpath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    pd.DataFrame(obj).to_csv(fpath, index=False)


def load_model(model_dir: str, **kwargs):
    return AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)

def load_vllm_model(model_dir: str, tokenizer_dir: str, **kwargs):
    return LLM(
        model_dir,
        tokenizer=tokenizer_dir,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(), 
        **kwargs
    )

def load_tokenizer(tokenizer_dir: str, **kwargs):
    return AutoTokenizer.from_pretrained(tokenizer_dir, **kwargs)
    