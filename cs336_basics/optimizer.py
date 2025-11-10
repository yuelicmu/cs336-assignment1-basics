import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn.utils import parameters_to_vector
from collections.abc import Callable, Iterable
from typing import Optional
import math

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

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # look up current state 
                state = self.state[p]
                grad = p.grad.data
                m = state.get("m", torch.zeros_like(p.grad.data))
                v = state.get("v", torch.zeros_like(p.grad.data))
                t = state.get("t", 1)

                # update first and second moment
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad ** 2

                # adjust parameters and weight decay
                alpha_t = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                p.data *= (1-lr * weight_decay) 

                # update state
                state["m"] = m
                state["v"] = v
                state["t"] = t+1

        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (
            1 + math.cos(math.pi * progress)
        ) * (max_learning_rate - min_learning_rate)
    return min_learning_rate
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    flat_params = parameters_to_vector([p.grad for p in parameters if p.grad is not None])
    total_norm = torch.linalg.vector_norm(flat_params)
    if total_norm <= max_l2_norm:
        return
    for p in parameters:
        if p.grad is not None:
            p.grad *= max_l2_norm / (total_norm + 1e-6)
    return
