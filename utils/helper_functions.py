import torch

import numpy as np

from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import griddata

import os
import fnmatch

            
##########################
# General Helper Functions
##########################

TEST_VESSELS = ["case_k_003_left", "case_k_008_right", "case_m_006_left", "case_w_027_right", "case_w_038_left"]


def create_model_name_infix(dict):
    """
    Generates a model name infix string based on the key-value pairs in the given dictionary.

    Args:
        dict (dict): A dictionary containing key-value pairs to be included in the infix.

    Returns:
        str: A string that represents the model name infix, with each key-value pair formatted as '_key_value'.
    """
    infix = "_"
    for key, value in dict.items():
        if '.' in str(value):
            value = str(value).split('.')[1]
        infix += f"_{key}_{value}"
    return infix

def decode_model_name(model_name):
    """
    Decodes a model name string into a dictionary of model parameters.

    The model name string is expected to be in the format:
    'prefix__lr_<learning_rate>__ep_<epochs>__transf_<transform>__rot_<rotation>__loss_<loss_function>'

    Parameters:
    model_name (str): The encoded model name string.

    Returns:
    dict: A dictionary containing the following keys:
        - 'lr' (float): The learning rate divided by 10.
        - 'ep' (int): The number of epochs.
        - 'transf' (str): The transformation type.
        - 'rot' (bool): The rotation flag.
        - 'loss' (str): The loss function.
    """
    model_split = model_name.split('_')
    # Remove the model name (wait for double underscore)
    model_split = model_split[model_split.index('')+1:]
    model_dict = {
        'lr': int(model_split[1])/10,
        'ep': int(model_split[3]),
        'transf': model_split[5],
        'rot': bool(model_split[7]),
        'loss': model_split[8]
    }
    return model_dict

def get_vessel_files(data_path):
    """
    Retrieves a list of vessel file paths from the specified directory, excluding test vessels.

    Args:
        data_path (str): The path to the directory containing vessel files.

    Returns:
        list: A list of file paths for the vessels, excluding those that match any pattern in TEST_VESSELS.
    """
    vessel_files = []
    for fluid_filename in os.listdir(data_path):
        # Filter out test vessels
        if any(fnmatch.fnmatch(fluid_filename, f"*{vessel}*") for vessel in TEST_VESSELS):
            continue
        fluid_file_path = os.path.join(data_path, fluid_filename)
        vessel_files.append(fluid_file_path)
    return vessel_files

def split_dataset(dataset, ratio=0.8):
    """
    Splits a dataset into training and validation sets based on the given ratio.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be split.
        ratio (float, optional): The ratio of the training set size to the total dataset size. 
                                 Default is 0.8.

    Returns:
        tuple: A tuple containing the training dataset and the validation dataset.
    """
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def print_hyperparameters(dict):
    """
    Prints the hyperparameters in a formatted table.
    Args:
        dict (dict): A dictionary containing hyperparameter names as keys and their values.
    The function calculates the maximum length of the keys and values to format the output
    in a tabular form. It then prints the hyperparameters with their corresponding values.
    """
    max_key_length = max(len(key) for key in dict.keys())
    max_value_length = max(len(str(value)) for value in dict.values())
    
    print("Hyperparameters:")
    print(f"{'Key'.ljust(max_key_length)} | {'Value'.ljust(max_value_length)}")
    print("-" * (max_key_length + max_value_length + 3))
    
    for key, value in dict.items():
        print(f"{key.ljust(max_key_length)} | {str(value).ljust(max_value_length)}")
    print()
    
def get_model_size(model):
    """
    Calculate the size of a PyTorch model in megabytes (MB).

    This function computes the total size of the model's parameters and buffers
    in memory and returns the size in megabytes.

    Args:
        model (torch.nn.Module): The PyTorch model whose size is to be calculated.

    Returns:
        float: The size of the model in megabytes (MB).
    """
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
    #print('Size: {:.3f} MB'.format(size_all_mb))    
    
#####################
# Grid Data Functions
#####################    
    
    
def check_points_in_hull(hull_tensor, check_tensor):
    """
    Check which points from `check_tensor` are inside the convex hull defined by `hull_tensor`.
    Args:
        hull_tensor (torch.Tensor): A tensor containing points that define the convex hull.
        check_tensor (torch.Tensor): A tensor containing points to be checked against the convex hull.
    Returns:
        tuple: A tuple containing two tensors:
            - inside_tensor (torch.Tensor): Points from `check_tensor` that are inside the convex hull.
            - outside_tensor (torch.Tensor): Points from `check_tensor` that are outside the convex hull.
    """
    hull_points_np = hull_tensor.numpy()
    check_points_np = check_tensor.numpy()
    
    hull = ConvexHull(hull_points_np)
    
    delaunay = Delaunay(hull_points_np[hull.vertices])
    
    inside = delaunay.find_simplex(check_points_np) >= 0
    
    inside_points = check_points_np[inside]
    outside_points = check_points_np[~inside]
    
    inside_tensor = torch.from_numpy(inside_points)
    outside_tensor = torch.from_numpy(outside_points)
    
    return inside_tensor, outside_tensor

def interpolate_vectors_to_grid(points, velocities, grid_shape, method='linear'):
    """
    Interpolates velocity vectors onto a 3D grid.
    Parameters:
    points (ndarray): An array of shape (N, 3) containing the coordinates of the points.
    velocities (ndarray): An array of shape (N, 3) containing the velocity vectors at the given points.
    grid_shape (tuple): A tuple of three integers defining the shape of the output grid (nx, ny, nz).
    method (str, optional): The interpolation method to use. Options are 'linear', 'nearest', and 'cubic'. Default is 'linear'.
    Returns:
    ndarray: A 4D array of shape (nx, ny, nz, 3) containing the interpolated velocity vectors on the grid.
    """
    
    # Generate linearly spaced points for each axis based on the bounds and the desired shape
    x = np.linspace(np.min(points[:,0]), np.max(points[:,0]), grid_shape[0])
    y = np.linspace(np.min(points[:,1]), np.max(points[:,1]), grid_shape[1])
    z = np.linspace(np.min(points[:,2]), np.max(points[:,2]), grid_shape[2])

    # Create a 3D grid from the 1D arrays
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    
    # Interpolate the velocity vectors onto the grid
    grid_vx = griddata(points, velocities[:,0], (grid_x, grid_y, grid_z), method=method, fill_value=0)
    grid_vy = griddata(points, velocities[:,1], (grid_x, grid_y, grid_z), method=method, fill_value=0)
    grid_vz = griddata(points, velocities[:,2], (grid_x, grid_y, grid_z), method=method, fill_value=0)
    
    # Combine the interpolated velocities into a single array
    grid_velocities = np.stack((grid_vx, grid_vy, grid_vz), axis=-1)
    
    return grid_velocities

# def get_vessel_grid_data(batch, size=(64, 64, 64), method='linear', threashold=0.1):
#     points, velocities = batch
#     interpolated_velocities = interpolate_vectors_to_grid(
#         np.array(points), 
#         np.array(velocities), 
#         size, 
#         method=method
#     )
#     vessel_mask = np.sum(interpolated_velocities**2, axis=-1) > threashold
#     interpolated_velocities[vessel_mask == False] = 0
#     return torch.Tensor(vessel_mask), torch.Tensor(interpolated_velocities)


def reshape_vel(vel):
    """
    Reshapes a 4-dimensional velocity tensor into a 2-dimensional tensor.

    Args:
        vel (numpy.ndarray or torch.Tensor): A 4-dimensional tensor with shape (d1, d2, d3, p).

    Returns:
        torch.Tensor: A 2-dimensional tensor with shape (d1 * d2 * d3, p).
    """
    d1, d2, d3, p = vel.shape
    new_shape = (d1 * d2 * d3, p)
    return torch.Tensor(vel).view(new_shape)

def create_grid_points(cube_size):
    """
    Generates a grid of points within a cube of given size.
    Args:
        cube_size (int): The size of the cube along each dimension.
    Returns:
        torch.Tensor: A tensor of shape (cube_size^3, 3) containing the coordinates of the grid points.
    """
    x = torch.linspace(0, 1, cube_size)
    y = torch.linspace(0, 1, cube_size)
    z = torch.linspace(0, 1, cube_size)
    
    grid = torch.stack(torch.meshgrid(x, y, z), dim=-1)
    return grid.view(-1, 3)

def grid_to_point_cloud(mask, vel):
    """
    Converts a 3D grid representation to a point cloud representation.
    Args:
        mask (numpy.ndarray): A 3D binary mask array indicating the presence of points.
        vel (numpy.ndarray): A 3D array representing velocity values at each grid point.
    Returns:
        tuple: A tuple containing:
            - grid (numpy.ndarray): A 2D array of shape (N, 3) where N is the number of points, 
              representing the coordinates of the points in the grid.
            - point_vel (numpy.ndarray): A 2D array of shape (N, 3) representing the velocity 
              values at each point in the grid.
    """
    cube_size = mask.shape[0]
    grid = create_grid_points(cube_size)
    point_vel = reshape_vel(vel)
    
    return grid, point_vel
    
#####################
# Transform Functions
#####################

def transform_linear(input_tensor, target_tensor, mesh_points=None):
    """
    Transforms the input and target tensors linearly. If mesh points are provided, they are included in the output.

    Args:
        input_tensor (Tensor): The input tensor to be transformed.
        target_tensor (Tensor): The target tensor to be transformed.
        mesh_points (Tensor, optional): Additional mesh points to be included in the output. Defaults to None.

    Returns:
        tuple: A tuple containing the input tensor, target tensor, and optionally the mesh points.
    """
    if mesh_points is None:
        return input_tensor, target_tensor
    else:
        return input_tensor, target_tensor, mesh_points

def transform_unit_cube(input_tensor, target_tensor, mesh_points=None):
    """
    Transforms the input tensor to fit within a unit cube centered at (0.5, 0.5, 0.5).
    This function normalizes the input tensor such that its values are scaled to fit within a unit cube.
    Optionally, it can also transform mesh points in the same manner.
    Args:
        input_tensor (torch.Tensor): The input tensor to be transformed.
        target_tensor (torch.Tensor): The target tensor, which is returned unchanged.
        mesh_points (torch.Tensor, optional): Additional points to be transformed. Defaults to None.
    Returns:
        tuple: A tuple containing:
            - input_tensor (torch.Tensor): The transformed input tensor.
            - target_tensor (torch.Tensor): The unchanged target tensor.
            - mesh_points (torch.Tensor, optional): The transformed mesh points, if provided.
    """
    min_vals = torch.min(input_tensor, dim=0)[0]
    max_vals = torch.max(input_tensor, dim=0)[0]
    middle_point = (min_vals + max_vals) / 2
    
    translated_point_cloud = input_tensor - middle_point
    
    scale_factor = torch.max(max_vals - min_vals)
    
    normalized_point_cloud = translated_point_cloud / scale_factor
    input_tensor = normalized_point_cloud + 0.5

    if mesh_points is not None:
        translated_mesh_points = mesh_points - middle_point
        normalized_mesh_points = translated_mesh_points / scale_factor
        mesh_points = normalized_mesh_points + 0.5
        
        return input_tensor, target_tensor, mesh_points
    else:    
        return input_tensor, target_tensor

def transform_zero_center(input_tensor, target_tensor, mesh_points=None):
    """
    Transforms the input tensor to be zero-centered by subtracting the middle point of its min and max values.
    Args:
        input_tensor (torch.Tensor): The input tensor to be transformed.
        target_tensor (torch.Tensor): The target tensor, which is returned unchanged.
        mesh_points (torch.Tensor, optional): Additional tensor of mesh points to be transformed. Defaults to None.
    Returns:
        tuple: A tuple containing the transformed input tensor, the target tensor, and optionally the transformed mesh points.
    """
    min_vals = torch.min(input_tensor, dim=0)[0]
    max_vals = torch.max(input_tensor, dim=0)[0]
    middle_point = (min_vals + max_vals) / 2
    
    input_tensor = input_tensor - middle_point
    
    if mesh_points is not None:
        mesh_points = mesh_points - middle_point
        return input_tensor, target_tensor, mesh_points
    else:
        return input_tensor, target_tensor

transform_function_mapping = {
    'linear': transform_linear,
    'unitcube': transform_unit_cube,
    'zerocenter': transform_zero_center,
    'grid64': transform_unit_cube,
    'grid128': transform_unit_cube,
    'rel': transform_linear
}

data_path_mapping = {
    'linear': '/home/ne34gux/workspace/experiments/data/vessel_point_data',
    'unitcube': '/home/ne34gux/workspace/experiments/data/vessel_point_data',
    'zerocenter': '/home/ne34gux/workspace/experiments/data/vessel_point_data',
    'grid64': '/home/ne34gux/workspace/experiments/data/vessel_grid64_data',
    'grid128': '/home/ne34gux/workspace/experiments/data/vessel_grid128_data',
    'rel': '/home/ne34gux/workspace/experiments/data/relative_inputs'
}

if __name__ == '__main__':
    DATA_PATH = "/home/ne34gux/workspace/experiments/data/vessel_point_data"
    vessel_files = get_vessel_files(DATA_PATH)
    print(vessel_files)