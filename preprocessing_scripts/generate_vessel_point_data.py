import os
import numpy as np
import meshio

"""
Generates vessel point data from the carotid flow database. 
The data is saved in a compressed numpy file with the following keys:
- fluid_points: The points of the fluid mesh.
- mesh_points: The points of the mesh mesh.
- sys_vel: The systolic velocity of the fluid mesh.
"""

INPUT_DIR = "/home/ne34gux/workspace/experiments/data/carotid_flow_database"
OUTPUT_DIR = "/home/ne34gux/workspace/experiments/data/vessel_point_data"

def get_vessel_files(directory):
    vessel_files = []
    for fluid_filename in os.listdir(directory):
        if fluid_filename.endswith('fluid.vtu'):
            fluid_file_path = os.path.join(directory, fluid_filename)
            mesh_file_path = os.path.join(directory, fluid_file_path.replace('fluid.vtu', 'wss.vtu'))
            vessel_files.append({'fluid_file': fluid_file_path, 'mesh_file': mesh_file_path, 'case': "_".join(fluid_filename.split('_')[:4])})
    return vessel_files


if __name__ == '__main__':
    vessel_files = get_vessel_files(INPUT_DIR)
    for v_idx, v in enumerate(vessel_files):
        print(f"Processing vessel {v_idx + 1}/{len(vessel_files)}")
        
        fluid_data = meshio.read(v['fluid_file'])
        mesh_data = meshio.read(v['mesh_file'])
        
        fluid_points = np.array(fluid_data.points)
        mesh_points = np.array(mesh_data.points)
        sys_vel = np.array(fluid_data.point_data['velocity_systolic'])
        
        filename = f"{v['case']}.npz"
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        np.savez_compressed(os.path.join(OUTPUT_DIR, filename), fluid_points=fluid_points, mesh_points=mesh_points, sys_vel=sys_vel)
