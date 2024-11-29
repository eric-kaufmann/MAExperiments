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
    infix = "_"
    for key, value in dict.items():
        if '.' in str(value):
            value = str(value).split('.')[1]
        infix += f"_{key}_{value}"
    return infix

def decode_model_name(model_name):
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
    vessel_files = []
    for fluid_filename in os.listdir(data_path):
        if any(fnmatch.fnmatch(fluid_filename, f"*{vessel}*") for vessel in TEST_VESSELS):
            continue
        fluid_file_path = os.path.join(data_path, fluid_filename)
        vessel_files.append(fluid_file_path)
    return vessel_files

def split_dataset(dataset, ratio=0.8):
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def print_hyperparameters(dict):
    max_key_length = max(len(key) for key in dict.keys())
    max_value_length = max(len(str(value)) for value in dict.values())
    
    print("Hyperparameters:")
    print(f"{'Key'.ljust(max_key_length)} | {'Value'.ljust(max_value_length)}")
    print("-" * (max_key_length + max_value_length + 3))
    
    for key, value in dict.items():
        print(f"{key.ljust(max_key_length)} | {str(value).ljust(max_value_length)}")
    print()
    
def get_model_size(model):
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

def get_vessel_grid_data(batch, size=(64, 64, 64), method='linear', threashold=0.1):
    points, velocities = batch
    interpolated_velocities = interpolate_vectors_to_grid(
        np.array(points), 
        np.array(velocities), 
        size, 
        method=method
    )
    vessel_mask = np.sum(interpolated_velocities**2, axis=-1) > threashold
    interpolated_velocities[vessel_mask == False] = 0
    return torch.Tensor(vessel_mask), torch.Tensor(interpolated_velocities)


def reshape_vel(vel):
    d1, d2, d3, p = vel.shape
    new_shape = (d1 * d2 * d3, p)
    return torch.Tensor(vel).view(new_shape)

def create_grid_points(cube_size):
    x = torch.linspace(0, 1, cube_size)
    y = torch.linspace(0, 1, cube_size)
    z = torch.linspace(0, 1, cube_size)
    
    grid = torch.stack(torch.meshgrid(x, y, z), dim=-1)
    return grid.view(-1, 3)

def grid_to_point_cloud(mask, vel):
    cube_size = mask.shape[0]
    grid = create_grid_points(cube_size)
    point_vel = reshape_vel(vel)
    
    return grid, point_vel
    
#####################
# Transform Functions
#####################

def transform_linear(input_tensor, target_tensor, mesh_points=None):
    if mesh_points is None:
        return input_tensor, target_tensor
    else:
        return input_tensor, target_tensor, mesh_points

def transform_unit_cube(input_tensor, target_tensor, mesh_points=None):
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
    'grid64': transform_linear,
    'grid128': transform_linear,
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