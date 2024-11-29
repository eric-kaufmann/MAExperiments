from torch import nn
import torch

def _get_k_closest_points(tensor, k=5):
    distances = torch.cdist(tensor, tensor, p=2)  # Shape (N, N)
    _, indices = torch.topk(distances, k=k+1, largest=False)  # Shape (N, k+1)
    closest_neighbors_indices = indices[:, :]  # Shape (N, k)
    return closest_neighbors_indices

def get_k_closest_points(tensor, k=5):
    distances = torch.cdist(tensor, tensor, p=2)  # Shape (B, N, N)
    _, indices = torch.topk(distances, k=k+1, largest=False)  # Shape (B, N, k+1)
    closest_neighbors_indices = indices[:, :, :]  # Shape (B, N, k)
    return closest_neighbors_indices

def _norm_to_closest_points(tensor, closest_neighbors_indices):
    return torch.norm(tensor[closest_neighbors_indices][:,0].unsqueeze(1) - tensor[closest_neighbors_indices][:,1:], dim=2)

def norm_to_closest_points(tensor, closest_neighbors_indices):
    B, N, _ = tensor.shape
    expanded_indices = closest_neighbors_indices.unsqueeze(-1).expand(-1, -1, -1, tensor.size(-1))
    neighbors = torch.gather(tensor.unsqueeze(1).expand(-1, N, -1, -1), 2, expanded_indices)
    return torch.norm(neighbors[:, :, 0].unsqueeze(2) - neighbors[:, :, 1:], dim=3)

def _dist_to_closest_points(tensor, closest_neighbors_indices):
    return tensor[closest_neighbors_indices][:,0].unsqueeze(1) - tensor[closest_neighbors_indices][:,1:]

def dist_to_closest_points(tensor, closest_neighbors_indices):
    expanded_indices = closest_neighbors_indices.unsqueeze(-1).expand(-1, -1, -1, tensor.size(-1))
    neighbors = torch.gather(tensor.unsqueeze(1).expand(-1, tensor.shape[1], -1, -1), 2, expanded_indices)
    return neighbors[:, :, 0].unsqueeze(2) - neighbors[:, :, 1:]

def calculate_continuity_loss(xyz, vel, k=5):
    print(xyz.shape, vel.shape)
    idx = get_k_closest_points(xyz, k=k)
    xyz_norm = norm_to_closest_points(xyz, idx).unsqueeze(3)
    xyz_dist = dist_to_closest_points(xyz, idx)
    vel_dist = dist_to_closest_points(vel, idx)
    continuity_eq = 1/k *((xyz_dist / xyz_norm) * (vel_dist / xyz_norm)).sum(dim=2)
    return continuity_eq.abs().mean()

def _calculate_continuity_loss(xyz, vel, k=5):
    print(xyz.shape, vel.shape)
    idx = get_k_closest_points(xyz, k=k)
    xyz_norm = norm_to_closest_points(xyz, idx).unsqueeze(2)
    xyz_dist = dist_to_closest_points(xyz, idx)
    vel_dist = dist_to_closest_points(vel, idx)
    continuity_eq = 1/k *((xyz_dist / xyz_norm) * (vel_dist / xyz_norm)).sum(dim=1)
    return continuity_eq.abs().mean()
    
    
def _calculate_jacobian(xyz, vel, k=5):
    idx = get_k_closest_points(xyz, k=k)
    xyz_norm = norm_to_closest_points(xyz, idx).unsqueeze(2)
    xyz_dist = dist_to_closest_points(xyz, idx)
    vel_dist = dist_to_closest_points(vel, idx)
    vel_dist_exp = vel_dist.unsqueeze(3)  # Form: [32768, 5, 3, 1]
    xyz_exp = xyz_dist.unsqueeze(2)            # Form: [32768, 5, 1, 3]
    outer_product = vel_dist_exp @ xyz_exp
    weighted_outer_product = outer_product / xyz_norm.unsqueeze(2)
    sum_weighted_outer_product = torch.sum(weighted_outer_product, dim=1)
    return sum_weighted_outer_product / k

def _calculate_momentum_loss(xyz, vel, k=5, density=1, viscosity=1):
    idx = get_k_closest_points(xyz, k=k)
    xyz_norm = norm_to_closest_points(xyz, idx).unsqueeze(2)
    xyz_dist = dist_to_closest_points(xyz, idx)
    jacobian = calculate_jacobian(xyz, vel, k=k)
    jacobian_dist = dist_to_closest_points(jacobian, idx)

    xyz_dist_exp = xyz_dist.unsqueeze(3)

    frac_term = (jacobian_dist @ xyz_dist_exp).squeeze(3) / xyz_norm
    lap_u = frac_term.sum(dim=1) / k
    momentum_term = density * (jacobian @ vel.unsqueeze(2)).squeeze(2) - viscosity * lap_u
    return momentum_term.norm(dim=1).mean()

# both functions can only work when xyz is of shape (N,3) and vel of shape (N,3). I want to include the batch_size so that xyz.shape = (B,N,3) and vel.shape = (B,N,3). I already adapted the functions get_k_closest_points, norm_to_closest_points and dist_to_closest_points so that they can handle the included batch size. Can you adapt this 2 functions aswell?

def calculate_jacobian(xyz, vel, k=5):
    B, N, _ = xyz.shape
    idx = get_k_closest_points(xyz, k=k)
    xyz_norm = norm_to_closest_points(xyz, idx).unsqueeze(3)
    xyz_dist = dist_to_closest_points(xyz, idx)
    vel_dist = dist_to_closest_points(vel, idx)
    vel_dist_exp = vel_dist.unsqueeze(4)  # Shape: (B, N, k, 3, 1)
    xyz_exp = xyz_dist.unsqueeze(3)       # Shape: (B, N, k, 1, 3)
    outer_product = vel_dist_exp @ xyz_exp
    weighted_outer_product = outer_product / xyz_norm.unsqueeze(3)
    sum_weighted_outer_product = torch.sum(weighted_outer_product, dim=2)
    return sum_weighted_outer_product / k

def calculate_momentum_loss(xyz, vel, k=5, density=1, viscosity=1):
    B, N, _ = xyz.shape
    idx = get_k_closest_points(xyz, k=k)
    xyz_norm = norm_to_closest_points(xyz, idx).unsqueeze(-1)
    xyz_dist = dist_to_closest_points(xyz, idx)
    jacobian = calculate_jacobian(xyz, vel, k=k)
    jacobian_dist = dist_to_closest_points(jacobian, idx)
    

    xyz_dist_exp = xyz_dist.unsqueeze(-1)

    frac_term = (jacobian_dist @ xyz_dist_exp).squeeze(-1) / xyz_norm
    lap_u = frac_term.sum(dim=2) / k
    momentum_term = density * (jacobian @ vel.unsqueeze(3)).squeeze(3) - viscosity * lap_u
    return momentum_term.norm(dim=2).mean()


class NSLoss(nn.Module):
    def __init__(self, continuity_loss_weight=0.5, momentum_loss_weight=0.5, num_neighbors=5) -> None:
        super(NSLoss, self).__init__()
        self.continuity_loss_weight = continuity_loss_weight
        self.momentum_loss_weight = momentum_loss_weight
        self.num_neighbors = num_neighbors
        
    def forward(self, target_tensor, pred_tensor):
        continuity_loss = calculate_continuity_loss(target_tensor, pred_tensor, k=self.num_neighbors)
        momentum_loss = calculate_momentum_loss(target_tensor, pred_tensor, k=self.num_neighbors)
        
        loss = self.continuity_loss_weight * continuity_loss + self.momentum_loss_weight * momentum_loss
        return loss

    
if __name__ == '__main__':
    batch_size = 4
    num_points = 100
    num_features = 3

    # Generate random sample data
    target_tensor = torch.randn(batch_size, num_points, num_features)
    pred_tensor = torch.randn(batch_size, num_points, num_features)
    
    # Generate random sample data
    #target_tensor = torch.randn(num_points, num_features)
    #pred_tensor = torch.randn(num_points, num_features)

    # Initialize the loss function
    loss_fn = NSLoss()

    # Calculate the loss
    loss = loss_fn(target_tensor, pred_tensor)
    print(f'Loss: {loss.item()}')