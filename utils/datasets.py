import torch
from torch.utils.data import Dataset

from utils.helper_functions import *


# Data Sample Classes

class VesselPointCloudData(Dataset):
    def __init__(self, data_path, num_samples=None, transform_function=transform_linear):
        self.data_path = data_path
        self.num_samples = num_samples
        self.vessel_files = get_vessel_files(data_path)
        self.num_vessels = len(self.vessel_files)
        self.transform_function = transform_function
    
    def __len__(self): 
        return self.num_vessels
    
    def __getitem__(self, index):
        vessel_file = self.vessel_files[index]
        fluid_data = np.load(vessel_file)
        if self.num_samples is not None:
            sample_indices = np.random.choice(fluid_data['fluid_points'].shape[0], self.num_samples, replace=False)
            input_tensor = torch.Tensor(fluid_data['fluid_points'][sample_indices])
            target_tensor = torch.Tensor(fluid_data['sys_vel'][sample_indices])
        else:
            input_tensor = torch.Tensor(fluid_data['fluid_points'])
            target_tensor = torch.Tensor(fluid_data['sys_vel'])
            
        return self.transform_function(input_tensor, target_tensor)
    

class VesselRelativePointCloudData(Dataset):
    def __init__(self, data_path, num_samples=None, transform_function=transform_linear):
        self.data_path = data_path
        self.num_samples = num_samples
        self.vessel_files = get_vessel_files(data_path)
        self.num_vessels = len(self.vessel_files)
        self.transform_function = transform_function
    
    def __len__(self): 
        return self.num_vessels
    
    def __getitem__(self, index):
        vessel_file = self.vessel_files[index]
        fluid_data = np.load(vessel_file)
        if self.num_samples is not None:
            sample_indices = np.random.choice(fluid_data['input'].shape[0], self.num_samples, replace=False)
            input_tensor = torch.Tensor(fluid_data['input'][sample_indices])
            target_tensor = torch.Tensor(fluid_data['target'][sample_indices])
        else:
            input_tensor = torch.Tensor(fluid_data['input'])
            target_tensor = torch.Tensor(fluid_data['target'])
            
        return self.transform_function(input_tensor, target_tensor)
    
    
class VesselGridData(Dataset):
    def __init__(self, data_path, filter_zero_velocities=False, sample_size=2**12):
        self.data_path = data_path
        self.vessel_files = get_vessel_files(data_path)
        self.num_vessels = len(self.vessel_files)
        self.filter_zero_velocities = filter_zero_velocities
        self.sample_size = sample_size
    
    def __len__(self): 
        return self.num_vessels
    
    def __getitem__(self, index):
        vessel_file = self.vessel_files[index]
        fluid_data = np.load(vessel_file)
       
        grid_xyz = fluid_data['vessel_mask']
        grid_vel = fluid_data['interp_vel']
        
        grid, point_vel = grid_to_point_cloud(grid_xyz, grid_vel)
        
        if self.filter_zero_velocities:
            mask = torch.all(point_vel == 0, dim=1)
            valid_indices = np.where(~mask)[0]
            if len(valid_indices) < self.sample_size:
                additional_indices = np.random.choice(np.where(mask)[0], self.sample_size - len(valid_indices), replace=False)
                mask_indices = np.concatenate([valid_indices, additional_indices])
            else:
                mask_indices = np.random.choice(valid_indices, self.sample_size, replace=False)
            grid = grid[mask_indices]
            point_vel = point_vel[mask_indices]
        
        return grid, point_vel
    
    
# Single Point Classes


class VesselDatasetSinglePoint(Dataset):
    def __init__(self, data_path, sample_size=256, transform_function=transform_linear, sample_cap=None):
        self.data_path = data_path
        self.sample_size = sample_size
        self.transform_function = transform_function
        self.sample_cap = sample_cap
        self.input_data = []
        self.target_data = []
        self.ref_idx = []
        self.ref_data = []
        self.load_data()
        if sample_cap is not None and sample_cap < len(self.input_data):
            self.apply_cap()

    def load_data(self):
        for file_idx, file_name in enumerate(os.listdir(self.data_path)):
            vessel_array = np.load(os.path.join(self.data_path, file_name))
            input_array = vessel_array['fluid_points']
            target_array = vessel_array['sys_vel']
            mesh_points = vessel_array['mesh_points']
            
            input_array, target_array, mesh_points = self.transform_function(
                torch.Tensor(input_array), 
                torch.Tensor(target_array), 
                mesh_points=torch.Tensor(mesh_points)
            )
            input_array = input_array.numpy()
            target_array = target_array.numpy()
            mesh_points = mesh_points.numpy()

            self.input_data.append(input_array)
            self.target_data.append(target_array)
            self.ref_idx.append(np.repeat(file_idx, input_array.shape[0]))
            sample_idx = np.random.choice(mesh_points.shape[0], self.sample_size, replace=False)
            self.ref_data.append(mesh_points[sample_idx])

        self.input_data = np.concatenate(self.input_data, axis=0)
        self.target_data = np.concatenate(self.target_data, axis=0)
        self.ref_idx = np.concatenate(self.ref_idx, axis=0)
        self.ref_data = np.array(self.ref_data)
        
    def apply_cap(self):
        sample_idx = np.random.choice(len(self.input_data), self.sample_cap, replace=False)
        self.input_data = self.input_data[sample_idx]
        self.target_data = self.target_data[sample_idx]
        self.ref_idx = self.ref_idx[sample_idx]
        
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return torch.Tensor(self.input_data[idx]), torch.Tensor(self.target_data[idx]), torch.Tensor(self.ref_data[self.ref_idx[idx]])

class VesselDatasetRelativeSinglePoint(Dataset):
    def __init__(self, data_path, sample_size=256, transform_function=transform_linear):
        self.data_path = data_path
        self.sample_size = sample_size
        self.transform_function = transform_function
        self.input_data = []
        self.target_data = []
        self.ref_idx = []
        self.ref_data = []
        self.load_data()

    def load_data(self):
        for file_idx, file_name in enumerate(os.listdir(self.data_path)):
            vessel_array = np.load(os.path.join(self.data_path, file_name))
            input_array = vessel_array['input']
            target_array = vessel_array['target']
            mesh_points = vessel_array['input'][:, :3]
            
            self.input_data.append(input_array)
            self.target_data.append(target_array)
            self.ref_idx.append(np.repeat(file_idx, input_array.shape[0]))
            sample_idx = np.random.choice(mesh_points.shape[0], self.sample_size, replace=False)
            self.ref_data.append(mesh_points[sample_idx])

        self.input_data = np.concatenate(self.input_data, axis=0)
        self.target_data = np.concatenate(self.target_data, axis=0)
        self.ref_idx = np.concatenate(self.ref_idx, axis=0)
        self.ref_data = np.array(self.ref_data)
        
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return torch.Tensor(self.input_data[idx]), torch.Tensor(self.target_data[idx]), torch.Tensor(self.ref_data[self.ref_idx[idx]])
    
    
class VesselGridSinglePointData(Dataset):
    def __init__(self, data_path, sample_size=256, transform_function=transform_linear, sample_cap=None):
        self.data_path = data_path
        self.transform_function = transform_function
        self.sample_size = sample_size
        self.sample_cap = sample_cap
        self.vessel_files = get_vessel_files(data_path)
        self.num_vessels = len(self.vessel_files)
        self.input_data = []
        self.target_data = []
        self.ref_idx = []
        self.ref_data = []
        self.load_data()
    
    def __len__(self):
        return len(self.input_data)

    def load_data(self):
        for file_idx, file_name in enumerate(os.listdir(self.data_path)):
            fluid_data = np.load(os.path.join(self.data_path, file_name))

            grid_xyz = fluid_data['vessel_mask']
            grid_vel = fluid_data['interp_vel']
            
            input_array, target_array = grid_to_point_cloud(grid_xyz, grid_vel)
            geom_array = input_array[torch.all(target_array != 0, dim=1)]
            
            input_array, target_array, geom_array = self.transform_function(input_array, target_array, geom_array)
            
            if self.sample_cap:
                mask = torch.all(target_array == 0, dim=1)
                valid_indices = np.where(~mask)[0]
                if target_array.shape[0] < self.sample_cap:
                    mask_indices = np.arange(target_array.shape[0])
                elif len(valid_indices) < self.sample_cap:
                    additional_indices = np.random.choice(np.where(mask)[0], self.sample_cap - len(valid_indices), replace=False)
                    mask_indices = np.concatenate([valid_indices, additional_indices])
                else:
                    mask_indices = np.random.choice(valid_indices, self.sample_cap, replace=False)
                input_array = input_array[mask_indices]
                target_array = target_array[mask_indices]

            self.input_data.append(input_array)
            self.target_data.append(target_array)
            self.ref_idx.append(np.repeat(file_idx, input_array.shape[0]))
            sample_idx = np.random.choice(geom_array.shape[0], self.sample_size, replace=False)
            self.ref_data.append(input_array[sample_idx])

        self.input_data = np.concatenate(self.input_data, axis=0)
        self.target_data = np.concatenate(self.target_data, axis=0)
        self.ref_idx = np.concatenate(self.ref_idx, axis=0)
        self.ref_data = np.array(self.ref_data)
    
    def __getitem__(self, idx):
        return torch.Tensor(self.input_data[idx]), torch.Tensor(self.target_data[idx]), torch.Tensor(self.ref_data[self.ref_idx[idx]])


# Specialized Datasets

class PointnetVesselData(Dataset):
    def __init__(self, data_path, num_samples=None, enable_transform=True):
        self.data_path = data_path
        self.num_samples = num_samples
        self.vessel_files = self._get_vessel_files()
        self.num_vessels = len(self.vessel_files)
        self.enable_transform = enable_transform
        
    def _get_vessel_files(self):
        vessel_files = []
        for fluid_filename in os.listdir(self.data_path):
            if fluid_filename in TEST_VESSELS:
                continue
            if fluid_filename.endswith('.npz'):
                fluid_file_path = os.path.join(self.data_path, fluid_filename)
                vessel_files.append(fluid_file_path)
        return vessel_files
    
    def transform(self, input_tensor, target_tensor):
        if self.enable_transform:
            centroid = input_tensor.mean(dim=0)
            translated_points = input_tensor - centroid
            
            max_abs_val = torch.max(torch.abs(translated_points))
            
            scaled_points = translated_points / (2 * max_abs_val)
            
            input_tensor = scaled_points + 0.5
        return input_tensor, target_tensor
    
    def __len__(self): 
        return self.num_vessels
    
    def __getitem__(self, index):
        vessel_file = self.vessel_files[index]
        fluid_data = np.load(vessel_file)
        if self.num_samples is not None:
            sample_indices = np.random.choice(len(fluid_data['fluid_points']), self.num_samples, replace=False)
            input_tensor = torch.Tensor(fluid_data['fluid_points'][sample_indices])
            target_tensor = torch.Tensor(fluid_data['sys_vel'][sample_indices])
        else:
            input_tensor = torch.Tensor(fluid_data['fluid_points'])
            target_tensor = torch.Tensor(fluid_data['sys_vel'])
        return self.transform(input_tensor, target_tensor)