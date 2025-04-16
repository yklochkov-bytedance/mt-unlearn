import datasets
from transformers import AutoTokenizer

import argparse

import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str,
                        help="Please provide name for loading HF dataset from disk")
    parser.add_argument("--tokenizer", type=str,
                        help="Please provide name for loading HF tokenizer")
    parser.add_argument("--max_len", type=int, default=512,
                        help="Split sequences into chunks of size max_len. Defaults to 512.")
    parser.add_argument("--save_to", type=str,
                        help="Path to save tokenized dataset.")
    parser.add_argument("--subsample_docs", type=int, default=-1,
                        help="Subsample this number of documents before tokenization. Defaults to -1 (no subsampling).")
    args = parser.parse_args()

    tokenizer = args.tokenizer
    max_len = args.max_len

    cache_dir = ".cache"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_overflowing_tokens=True, max_length=max_len, truncation=True)

    dataset = datasets.load_from_disk(args.dataset_path)['train']
    
    if args.subsample_docs > 0:
        num = args.subsample_docs
        print("Taking subset of", num, f"documents (out of{len(dataset)})")
        dataset = dataset.select(random.sample(range(len(dataset)), num))
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, num_proc=24)
    print("After tokenization:", len(tokenized_dataset))

    def filter_func(examples):
        return [len(e) == max_len for e in examples['input_ids']]
    filtered_dataset = tokenized_dataset.filter(filter_func, batched=True, num_proc=24)

    print("After filtering:", len(filtered_dataset))
    
    
    filtered_dataset.save_to_disk(args.save_to)
    test_saved_dataset = datasets.load_from_disk(args.save_to)

    for i in range(3):
        print(f"\nCHUNK {i}")
        print(tokenizer.decode(test_saved_dataset[i]['input_ids']))
