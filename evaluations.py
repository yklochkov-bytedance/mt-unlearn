import torch
import torch.nn.functional as F

def nll_and_acc(model, batch):
    inputs = batch["input_ids"]
    mask = batch["attention_mask"]
    with torch.no_grad():
        logits = model.forward(inputs)['logits'].detach()

    logits = logits[:, :-1, :].contiguous()
    pred = torch.argmax(logits, dim=-1)
    target = inputs[:, 1:].contiguous()
    mask = mask[:, 1:].contiguous().float()

    # Calculate correct predictions
    correct_predictions = (pred == target).float() * mask
    acc = correct_predictions.sum() / mask.sum()

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction="none")
    loss = (loss * mask.view(-1)).sum() / mask.sum()

    return loss.item(), acc.item()
