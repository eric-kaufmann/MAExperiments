


# 
# 
# 
# 
#           THIS FILE IS OUTDATED!!!
# 
# 
# 
# 
# 
# 






import os
import numpy as np
import pandas as pd
import meshio

"""
Generates vessel point data from the carotid flow database. 
The data is saved in a compressed numpy file with the following keys:
- fluid_points: The points of the fluid mesh.
- mesh_points: The points of the mesh mesh.
- sys_vel: The systolic velocity of the fluid mesh.
"""

INPUT_DIR = "/home/ne34gux/workspace/experiments/data/pinn_blood_flow_geometries"
OUTPUT_DIR = "/home/ne34gux/workspace/experiments/data/geom_point_data"

if __name__ == '__main__':
    test_geometry_files = [g for g in os.listdir(INPUT_DIR) if "velocities" in g]
    for g_idx, g in enumerate(test_geometry_files):
        g_file_path = os.path.join(INPUT_DIR, g)
        
        print(f"Processing vessel {g_idx + 1}/{len(test_geometry_files)}")
        
        g_object = pd.read_csv(g_file_path).to_numpy()
        
        fluid_points = g_object[:,4:7]
        sys_vel = g_object[:,0:3]
        mesh_points = meshio.read(g_file_path.replace('velocities.csv', 'wall.stl')).points
        
        filename = f"{g.split('_')[0]}.npz"
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, filename), 
            fluid_points=fluid_points, 
            mesh_points=mesh_points, 
            sys_vel=sys_vel
        )
