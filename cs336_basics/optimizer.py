import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

def cross_entropy_loss(
    inputs: Float[Tensor, "batch_size seq_len vocab_size"], 
    targets: Int[Tensor, " batch_size seq_len"]
) -> Float[Tensor, ""]:
    # Vectorized cross-entropy computation
    # inputs: (batch_size, seq_len, vocab_size)
    # targets: (batch_size, seq_len)
    log_probs = inputs - torch.amax(inputs, dim=-1, keepdim=True)
    log_sum_exp = torch.log(torch.sum(torch.exp(log_probs), dim=-1))
    # Gather the logits corresponding to the targets
    target_logits = torch.gather(inputs, -1, targets.unsqueeze(-1)).squeeze(-1)
    losses = - (target_logits - torch.amax(inputs, dim=-1)) + log_sum_exp
    return losses.mean()