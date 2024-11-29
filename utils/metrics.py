from torch.nn import functional as F
import torch


def cosine_similarity(pred, target):
    """
    Computes the mean cosine similarity between two tensors.

    Args:
        pred (torch.Tensor): The predicted tensor of shape (B, N, 3).
        target (torch.Tensor): The target tensor of shape (B, N, 3).

    Returns:
        torch.Tensor: The mean cosine similarity between the input tensors.
    """
    pred_flat = pred.view(-1, 3)
    target_flat = target.view(-1, 3)
    return F.cosine_similarity(pred_flat, target_flat).mean().item()


def mse_error(pred, target):
    """
    Computes the mean squared error between two tensors.

    Args:
        pred (torch.Tensor): The predicted tensor of shape (B, N, 3).
        target (torch.Tensor): The target tensor of shape (B, N, 3).

    Returns:
        torch.Tensor: The mean squared error between the input tensors.
    """
    pred_flat = pred.view(-1, 3)
    target_flat = target.view(-1, 3)
    return F.mse_loss(pred_flat, target_flat).item()


def mae_error(pred, target):
    """
    Computes the mean absolute error between two tensors.

    Args:
        pred (torch.Tensor): The predicted tensor of shape (B, N, 3).
        target (torch.Tensor): The target tensor of shape (B, N, 3).

    Returns:
        torch.Tensor: The mean absolute error between the input tensors.
    """
    pred_flat = pred.view(-1, 3)
    target_flat = target.view(-1, 3)
    return F.l1_loss(pred_flat, target_flat).item()

    