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
OUTPUT_DIR = "/home/ne34gux/workspace/experiments/data/rotated_vessel_point_data"

def get_vessel_files(directory):
    vessel_files = []
    for fluid_filename in os.listdir(directory):
        if fluid_filename.endswith('fluid.vtu'):
            fluid_file_path = os.path.join(directory, fluid_filename)
            mesh_file_path = os.path.join(directory, fluid_file_path.replace('fluid.vtu', 'wss.vtu'))
            vessel_files.append({'fluid_file': fluid_file_path, 'mesh_file': mesh_file_path, 'case': "_".join(fluid_filename.split('_')[:4])})
    return vessel_files


def rotate_vector_field(vector_field, angle, center=[0,0,0]):
    """
    Rotiert ein 3D-Vektorfeld um die Z-Achse um einen gegebenen Punkt.
    
    :param vector_field: Ein Array von Vektoren (x, y, z).
    :param center: Der Punkt (x, y, z), um den rotiert wird.
    :param angle: Der Rotationswinkel in Grad.
    :return: Das rotierte Vektorfeld.
    """
    # Umrechnung des Winkels von Grad in Radiant
    angle_rad = np.radians(angle)
    
    # Rotationsmatrix für die Z-Achse
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                [np.sin(angle_rad), np.cos(angle_rad), 0],
                                [0, 0, 1]])
    
    # Rotiertes Vektorfeld
    rotated_field = np.zeros_like(vector_field)
    
    for i, vector in enumerate(vector_field):

        # Verschiebung zum Ursprung
        shifted_vector = vector - center
        
        # Anwendung der Rotation
        rotated_vector = np.dot(rotation_matrix, shifted_vector)
        
        # Zurückverschiebung
        rotated_field[i] = rotated_vector + center
    
    return rotated_field

if __name__ == '__main__':
    vessel_files = get_vessel_files(INPUT_DIR)
    for v_idx, v in enumerate(vessel_files):
        for angle in np.linspace(0, 360, 8):
            print(f"Processing vessel {v_idx + 1}/{len(vessel_files)} - Angle {int(angle)}")
            fluid_data = meshio.read(v['fluid_file'])
            mesh_data = meshio.read(v['mesh_file'])
            
            fluid_points = np.array(fluid_data.points)
            mesh_points = np.array(mesh_data.points)
            sys_vel = np.array(fluid_data.point_data['velocity_systolic'])
            
            center = np.mean(fluid_points, axis=0)
            fluid_points = rotate_vector_field(fluid_points, angle, center=center)
            mesh_points = rotate_vector_field(mesh_points, angle, center=center)
            sys_vel = rotate_vector_field(sys_vel, angle)
            
            filename = f"{v['case']}_angle_{int(angle)}.npz"
            
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            
            np.savez_compressed(os.path.join(OUTPUT_DIR, filename), fluid_points=fluid_points, mesh_points=mesh_points, sys_vel=sys_vel)
        
            filename = f"{v['case']}_{int(angle)}deg.npz"
            
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            
            np.savez_compressed(os.path.join(OUTPUT_DIR, filename), fluid_points=fluid_points, mesh_points=mesh_points, sys_vel=sys_vel)
