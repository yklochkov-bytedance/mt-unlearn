import torch
import torch.nn.functional as F

def get_logits_and_attention_mask(model, batch):
    # TODO: comment
    inputs = batch['input_ids']
    logits = model(inputs)['logits']
    mask = batch['attention_mask']
    return logits, mask

def get_shifted_logits_and_attention_mask_and_target(model, batch):
    """
        First calculate the logits and extract the mask

        Then, trim the logits and shift the mask to the right to use for prediction

        Also shift input ids to the left, to use as a target
    """
    logits, mask = get_logits_and_attention_mask(model, batch)
    logits = logits[:, :-1, :].contiguous()
    mask = mask[:, 1:].contiguous()
    target = batch['input_ids'][:, 1:].contiguous()
    return logits, mask, target

# one-model losses

def neg_loglikelihood(model, batch, args, return_nll_val=True):
    """
        Computes NLL $-\log p(x)$

        Return also the float value of the NLL on this batch for monitoring purposes (option return_nll_val=True)
    """
    logits, mask, target = get_shifted_logits_and_attention_mask_and_target(model, batch)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction="none")
    loss = (loss * mask.view(-1)).sum() / mask.sum()
    
    if return_nll_val:
        return loss, loss.item()
    else:
        return loss

def mismatch(model, batch, args, return_nll_val=True):
    """
        Computes mismatch loss $KL(p(y|x); 1/|Y|)$,
        see https://arxiv.org/pdf/2310.10683

        Return also the float value of the NLL on this batch for monitoring purposes (option return_nll_val=True)
    """
    logits, mask, target = get_shifted_logits_and_attention_mask_and_target(
        model, batch
    )
    loss_val = F.cross_entropy(
        logits.detach().view(-1, logits.size(-1)),
        target.view(-1),
        reduction="none"
    )
    loss_val = (loss_val * mask.view(-1)).sum() / mask.sum()
    loss_val = loss_val.item()

    logsf = F.log_softmax(logits, dim=-1)
    uniform = torch.ones_like(logsf) / logits.size(-1)
    loss = (F.kl_div(logsf, uniform, reduction='none', log_target=False).sum(-1) * mask).mean()

    if return_nll_val:
        return loss, loss_val
    else:
        return loss

def loglikelihood(model, batch, args, return_nll_val=True):
    """
        Computes log-likelihood $\log p(x)$

        Return also the float value of the NLL on this batch
          for monitoring purposes (option return_nll_val=True)
    """
    loss, val = neg_loglikelihood(model, batch, args, return_nll_val=True)
    if return_nll_val:
        return -loss, val
    else:
        return -loss


def neg_log_unlikelihood(model, batch, args, return_nll_val=True):
    """
        Computes negative log-unlikelihood - \log (1 - p(x))
        Instead of calculating it derectly, we calculate [p(x) / (1 - p(x))].detach() \times \log p(x),
        which has the same derivatives.

        We add a tolerance parameter 1 + tol - p(x) to avoid overflow.
        Option to threshold probabilities, so that we do not unlearn ones that are already small, see parameter 'nlul_threshold'.

        Final formula is
            
            [nn.Threshold(threshold, 0)(p(x)) / (1 + tol - p(x))].detach() \times \log p(x)

        We additionally return the float value of the NLL on this batch for monitoring purposes (option return_nll_val=True)
    """
    logits, mask, targets = get_shifted_logits_and_attention_mask_and_target(model, batch)

    log_sf = F.log_softmax(logits, dim=-1)
    sf_detach = torch.exp(log_sf.detach())
    sf_detach = -F.nll_loss(sf_detach.view(-1, sf_detach.size(-1)), targets.view(-1), reduction='none')
    if args.get('nlul_threshold', 0) != 0:
        sf_detach = torch.nn.Threshold(args['nlul_threshold'], 0.0, inplace=True)(sf_detach)
    weights = -sf_detach / (1 + args['tol'] - sf_detach)

    loss = F.nll_loss(log_sf.view(-1, log_sf.size(-1)), targets.view(-1), reduction='none')
    loss = loss * mask.view(-1)
    if return_nll_val:
        loss_val = loss.detach().sum() / mask.sum()
    loss = loss * weights # weights are negative, so we have negative negative equal positive...
    loss = loss.sum() / mask.sum()

    if return_nll_val:
        return loss, loss_val.item()
    else:
        return loss


def npo(model, model_base, batch, args):
    """
        Computes NPO - 2 / \beta \log \sigma(- \beta p(x) + \beta p_base(x))
        Instead of calculating it derectly, we calculate 
            [2 p(s)^beta / (p^beta(s) + p_base^beta(s))].detach() \times \log p(x),
        which has the same derivatives.

        We additionally return the float value of the NLL on this batch for monitoring purposes (option return_nll_val=True)
    """
    logits, mask, targets = get_shifted_logits_and_attention_mask_and_target(model, batch)
    with torch.no_grad():
        logits_base, _, _ = get_shifted_logits_and_attention_mask_and_target(model_base, batch)
        log_sf_base_detach = F.log_softmax(logits_base.detach(), dim=-1)

    log_sf = F.log_softmax(logits, dim=-1)
    log_sf_detach = log_sf.detach()

    log_diff = (
        F.nll_loss(
            log_sf_base_detach.view(-1, log_sf_detach.size(-1)),
            targets.view(-1), reduction='none'
        )
        - F.nll_loss(
            log_sf_detach.view(-1, log_sf_detach.size(-1)),
            targets.view(-1), reduction='none'
        )
    ).view(targets.shape)

    log_diff_sigmoid = F.sigmoid((log_diff * mask).mean(-1) * args['beta_npo'])
    weights = -2.0 * log_diff_sigmoid # negative NLL = LL

    loss = F.nll_loss(log_sf.view(-1, log_sf.size(-1)), targets.view(-1), reduction='none')
    loss = loss.view(targets.shape) * mask
    loss_val = loss.detach().sum() / mask.sum()
    # weights are negative, 
    # so we have negative negative equal positive
    loss = loss * weights[:, None]
    loss = loss.sum() / mask.sum()

    return loss, loss_val.item()


# two-model losses

def kl(model, model_ref, batch, args):
    logits = model(batch['input_ids'])['logits']
    with torch.no_grad():
        logits_ref = model_ref(batch['input_ids'])['logits'].detach()
    mask = batch['attention_mask']
    logsf = F.log_softmax(logits, dim=-1)
    sf_ref = F.softmax(logits_ref, dim=-1)
    return (F.kl_div(logsf, sf_ref, reduction='none', log_target=False).sum(-1) * mask).mean()

def _qkl(logits, logits_ref, mask):
    sf_ref = F.softmax(logits_ref, dim=-1)
    logits_diff = logits - logits_ref

    t = sf_ref * (logits_diff.detach())
    t -= t.sum(-1).unsqueeze(-1) * sf_ref

    return ((logits_diff * t).sum(-1) * mask).mean()

def qkl(model, model_ref, batch, args):
    logits = model(batch['input_ids'])['logits']
    with torch.no_grad():
        logits_ref = model_ref(batch['input_ids'])['logits'].detach()
    mask = batch['attention_mask']
    return _qkl(logits, logits_ref, mask)
