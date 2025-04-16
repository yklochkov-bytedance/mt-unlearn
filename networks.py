from transformers import AutoModelForCausalLM, AutoTokenizer

def _freeze_embedding_and_lm_head(model):
    """
        Freezes embedding, head, and layer norms.
        Every parameter that would not be sharded when using TP.
    """

    def is_embedding_or_head_param_name(pname):

        names = [
            "wte",
            "wpe",
            "embed_tokens",
            "embed_positions",
            "rotary_emb",
            "lm_head",
            "norm",
            "ln_1",
            "ln_2"
        ]

        return (any(map(lambda x: pname.endswith(f"{x}.weight"), names))
                or any(map(lambda x: pname.endswith(f"{x}.bias"), names)))

    for name, par in model.named_parameters():
        if is_embedding_or_head_param_name(name):
            par.requires_grad = False
    return


def get_network(network_path):
    model = AutoModelForCausalLM.from_pretrained(network_path)
    # freeze token embedding layer
    _freeze_embedding_and_lm_head(model)

    return model


def get_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
