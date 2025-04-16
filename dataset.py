from datasets import load_from_disk, DatasetDict

def get_dataset(
    dataset_path,
    tokenizer,
    context=None,
    split=None,
    chunking=False,
):
    """
        Loads dataset at dataset_path, taking 'split' if there is a DatasetDict.

        If the dataset is not already tokenized, tokenization will be applied.

        If chunking is True, each sequence will be chunked into multiple sequences of size 
        context. We apply this option for MUSE-Books (there is only 4 sequences) but not 
        for MUSE-News. The latter simply uses truncation.

        TODO: add preprocessing option for QA / insturction datasets
    """
    ds = load_from_disk(dataset_path)
    if split is not None:
        if isinstance(ds, DatasetDict):
            ds = ds[split]
        else:
            print(f"Not an instance of DatasetDict. Ignoring split='{split}'")

    # check if it requires tokenization
    if 'input_ids' in ds.features:
        print(f"Dataset is already tokenized. Ignoring context='{context}'")

    else:
        if context is None:
            raise ValueError(
                "Context lenght must be specified if dataset is not tokenized"
            )

        # Tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                return_overflowing_tokens=chunking,
                max_length=context
            )

        ds = ds.map(
            tokenize_function,
            batched=True,
            remove_columns=ds.column_names,
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=16
        )
        ds.set_format(type='torch', columns=['input_ids'])
    
    return ds
