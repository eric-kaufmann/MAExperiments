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

    
def calculate_angle_difference(tensor1, tensor2, chunk_size=10000, degrees=True, skip_zero_vectors=True):
    """
    Calculate the angle difference between two tensors of shape (N, 3) in chunks, skipping zero-vectors.

    Args:
    tensor1 (torch.Tensor): First tensor of shape (N, 3).
    tensor2 (torch.Tensor): Second tensor of shape (N, 3).
    chunk_size (int): Size of chunks to split the tensors into for processing.
    degrees (bool): If True, return the angle difference in degrees. Otherwise, return in radians.
    skip_zero_vectors (bool): If True, skip zero-vectors. Otherwise, set angle to 360 degrees for zero vectors.

    Returns:
    torch.Tensor: Angle difference between the two tensors in radians or degrees.
    """
    angle_differences = []
    for i in range(0, tensor1.shape[0], chunk_size):
        t1_chunk = tensor1[i:i+chunk_size]
        t2_chunk = tensor2[i:i+chunk_size]
        
        # Calculate norms
        norm_tensor1 = torch.norm(t1_chunk, dim=1)
        norm_tensor2 = torch.norm(t2_chunk, dim=1)
        
        # Skip zero-vectors
        non_zero_mask = (norm_tensor1 != 0) & (norm_tensor2 != 0)
        if skip_zero_vectors:
            t1_chunk = t1_chunk[non_zero_mask]
            t2_chunk = t2_chunk[non_zero_mask]
        else:
            zero_mask = ~non_zero_mask
            zero_angles = torch.full((zero_mask.sum(),), 360 if degrees else 2 * torch.pi)
        
        # Normalize vectors
        t1_chunk = t1_chunk / norm_tensor1[non_zero_mask].unsqueeze(1)
        t2_chunk = t2_chunk / norm_tensor2[non_zero_mask].unsqueeze(1)
        
        # Calculate dot product
        dot_product = torch.sum(t1_chunk * t2_chunk, dim=1)
        
        # Clamp dot product to avoid numerical issues with arccos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Calculate angle difference
        angle_diff = torch.acos(dot_product)
        if degrees:
            angle_diff = torch.rad2deg(angle_diff)
        angle_differences.append(angle_diff)
        
        if not skip_zero_vectors:
            angle_differences.append(zero_angles)
    
    return torch.cat(angle_differences).mean().item()