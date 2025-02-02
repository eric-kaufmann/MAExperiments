import os
import numpy as np
import torch
import meshio
import time

from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import griddata

GRID_SIZE = 128
THREASHOLD = 0.1

INPUT_DIR = "/home/ne34gux/workspace/experiments/data/carotid_flow_database"
OUTPUT_DIR = f"/home/ne34gux/workspace/experiments/data/vessel_grid{GRID_SIZE}_data"

def get_vessel_files(directory):
    vessel_files = []
    for fluid_filename in os.listdir(directory):
        if fluid_filename.endswith('fluid.vtu'):
            fluid_file_path = os.path.join(directory, fluid_filename)
            mesh_file_path = os.path.join(directory, fluid_file_path.replace('fluid.vtu', 'wss.vtu'))
            vessel_files.append({'fluid_file': fluid_file_path, 'mesh_file': mesh_file_path, 'case': "_".join(fluid_filename.split('_')[:4])})
    return vessel_files

def unit_cube(input_array):
    min_vals = np.min(input_array, axis=0)
    max_vals = np.max(input_array, axis=0)
    middle_point = (min_vals + max_vals) / 2
    translated_point_cloud = input_array - middle_point
    scale_factor = np.max(max_vals - min_vals)
    normalized_point_cloud = translated_point_cloud / scale_factor
    input_array = normalized_point_cloud + 0.5
    
    return input_array

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
    """
    Interpolates velocity vectors at 3D points into a 3D grid of a predefined shape.

    Parameters:
    - points: A numpy array of shape (N, 3), where N is the number of points, representing the 3D coordinates.
    - velocities: A numpy array of shape (N, 3), where N is the number of points, representing the velocity vectors at these points.
    - grid_shape: A tuple of 3 integers defining the shape of the 3D grid (depth, height, width).
    - method: Interpolation method. Options include 'linear', 'nearest', and 'cubic'.

    Returns:
    - A numpy array of shape (grid_shape[0], grid_shape[1], grid_shape[2], 3) representing the interpolated velocity vectors on the 3D grid.
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

if __name__ == '__main__':
    start_time = time.time()
    vessel_files = get_vessel_files(INPUT_DIR)
    for v_idx, v in enumerate(vessel_files):
        print(f"Processing vessel {v_idx + 1}/{len(vessel_files)}")
        
        fluid_data = meshio.read(v['fluid_file'])
        mesh_data = meshio.read(v['mesh_file'])
        
        fluid_points = np.array(fluid_data.points)
        mesh_points = np.array(mesh_data.points)
        sys_vel = np.array(fluid_data.point_data['velocity_systolic'])
        
        # transform the fluid points to the unit cube
        fluid_points = unit_cube(fluid_points)
        
        # add points outside of the hull for better interpolation
        _, out_points = check_points_in_hull(
            torch.Tensor(fluid_points), 
            torch.rand(300_000, 3)
        )
        
        fluid_points = np.concatenate([fluid_points, out_points.numpy()])
        sys_vel = np.concatenate([sys_vel, np.zeros(out_points.shape)])
        
        # calculate the grid data
        vessel_mask, interp_vel = get_vessel_grid_data(
            (fluid_points, sys_vel), 
            size=(GRID_SIZE, GRID_SIZE, GRID_SIZE), 
            method='linear', 
            threashold=THREASHOLD
        )
        
        # save the grid data
        
        filename = f"{v['case']}.npz"
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, filename), 
            vessel_mask=vessel_mask, 
            interp_vel=interp_vel
        )
        print(f"  current duration: {time.time() - start_time} seconds")
    
    end_time = time.time()
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("Processing complete.")
    print(f"Loading and processing took {int(hours)} hours, {int(minutes)} minutes, and {seconds} seconds.")
