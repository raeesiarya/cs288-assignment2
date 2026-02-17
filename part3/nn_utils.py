"""
Neural network utilities for Transformer implementation.
Contains basic building blocks: softmax, cross-entropy, gradient clipping, token accuracy, perplexity.
"""

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute softmax along the specified dimension.

    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute softmax (default: -1)

    Returns:
        Tensor of same shape as input with softmax applied along dim
    """
    # TODO: Implement numerically stable softmax. You can re-use the same one
    # used in part 2. But for this problem, you need to implement a numerically stable version to pass harder tests.
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute cross-entropy loss.

    Args:
        logits: Unnormalized log probabilities of shape (N, C) where N is batch size
                and C is number of classes
        targets: Ground truth class indices of shape (N,)

    Returns:
        Scalar tensor containing the mean cross-entropy loss
    """
    # TODO: Implement cross-entropy loss
    x_max = logits.max(dim=-1, keepdim=True).values
    logsumexp = (
        torch.log(torch.exp(logits - x_max).sum(dim=-1, keepdim=True)) + x_max
    )  # (N, 1)
    log_probs = logits - logsumexp  # (N, C)

    # Gather log prob of the correct class
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (N,)
    return nll.mean()


def gradient_clipping(parameters, max_norm: float) -> Tensor:
    """
    Clip gradients of parameters by global norm.

    Args:
        parameters: Iterable of parameters with gradients
        max_norm: Maximum allowed gradient norm

    Returns:
        The total norm of the gradients before clipping
    """
    # TODO: Implement gradient clipping
    grads = []
    for p in parameters:
        if p is None or (not getattr(p, "requires_grad", True)):
            continue
        g = getattr(p, "grad", None)
        if g is None:
            continue
        grads.append(g)

    if not grads:
        return torch.tensor(0.0)

    # Compute total norm (L2)
    # Use float for stability, and keep device consistent
    device = grads[0].device
    total_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
    for g in grads:
        total_norm_sq = total_norm_sq + (g.detach().float().norm(2) ** 2)
    total_norm = torch.sqrt(total_norm_sq)

    # Clip in-place if needed
    if max_norm > 0:
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for g in grads:
                g.mul_(clip_coef.to(g.dtype))

    return total_norm


def token_accuracy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute token-level accuracy for language modeling.

    Computes the fraction of tokens where the predicted token (argmax of logits)
    matches the target token, ignoring positions where target equals ignore_index.

    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing accuracy (default: -100)

    Returns:
        Scalar tensor containing the accuracy (between 0 and 1)

    Example:
        >>> logits = torch.tensor([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2], [1.0, 0.5, 2.5]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> token_accuracy(logits, targets)
        tensor(1.)  # All predictions correct: argmax gives [0, 1, 2]

        >>> logits = torch.tensor([[2.0, 1.0], [0.1, 3.0], [1.0, 0.5]])
        >>> targets = torch.tensor([1, 1, 0])
        >>> token_accuracy(logits, targets)
        tensor(0.6667)  # 2 out of 3 correct
    """
    # TODO: Implement token accuracy
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        valid = targets != ignore_index
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        correct = (preds[valid] == targets[valid]).float().mean()
        return correct


def perplexity(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute perplexity for language modeling.

    Perplexity is defined as exp(cross_entropy_loss). It measures how well the
    probability distribution predicted by the model matches the actual distribution
    of the tokens. Lower perplexity indicates better prediction.

    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing perplexity (default: -100)

    Returns:
        Scalar tensor containing the perplexity (always >= 1)

    Example:
        >>> # Perfect predictions (one-hot logits matching targets)
        >>> logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(1.0001)  # Close to 1 (perfect)

        >>> # Uniform predictions (high uncertainty)
        >>> logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(3.)  # Equal to vocab_size (worst case for uniform)
    """
    # TODO: Implement perplexity
    x_max = logits.max(dim=-1, keepdim=True).values
    logsumexp = (
        torch.log(torch.exp(logits - x_max).sum(dim=-1, keepdim=True)) + x_max
    )  # (N, 1)
    log_probs = logits - logsumexp  # (N, C)
    nll = -log_probs.gather(dim=-1, index=targets.clamp_min(0).unsqueeze(-1)).squeeze(
        -1
    )  # (N,)

    valid = targets != ignore_index
    if valid.sum() == 0:
        return torch.tensor(1.0, device=logits.device)

    ce = nll[valid].mean()
    return torch.exp(ce)
