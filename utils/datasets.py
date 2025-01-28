import torch
from torch.utils.data import Dataset

from utils.helper_functions import *

#######################
# Data Sample Classes #
#######################

class VesselPointCloudData(Dataset):
    """
    A PyTorch Dataset class for loading and transforming vessel point cloud data.
    Args:
        data_path (str): Path to the directory containing vessel data files.
        num_samples (int, optional): Number of samples to randomly select from each vessel file. 
                                     If None, all samples are used. Default is None.
        transform_function (callable, optional): A function to transform the input and target tensors. 
                                                 Default is transform_linear.
    Attributes:
        data_path (str): Path to the directory containing vessel data files.
        num_samples (int or None): Number of samples to randomly select from each vessel file.
        vessel_files (list): List of file paths to the vessel data files.
        num_vessels (int): Number of vessel data files.
        transform_function (callable): Function to transform the input and target tensors.
    Methods:
        __len__(): Returns the number of vessel data files.
        __getitem__(index): Loads the vessel data file at the given index, applies sampling if specified,
                            and transforms the input and target tensors using the transform_function.
                            Returns the transformed input and target tensors.
    """
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
    

class VesselRelativePointCloudData(Dataset): # Dataset for Relative Inputs
    """
    Dataset class for loading and transforming vessel relative point cloud data.
    Args:
        data_path (str): Path to the directory containing vessel data files.
        num_samples (int, optional): Number of samples to randomly select from each vessel file. 
                                        If None, all samples are used. Default is None.
        transform_function (callable, optional): Function to apply transformations to the input and target tensors.
                                                    Default is transform_linear.
    Attributes:
        data_path (str): Path to the directory containing vessel data files.
        num_samples (int, optional): Number of samples to randomly select from each vessel file.
        vessel_files (list): List of file paths to the vessel data files.
        num_vessels (int): Number of vessel data files.
        transform_function (callable): Function to apply transformations to the input and target tensors.
    Methods:
        __len__(): Returns the number of vessel data files.
        __getitem__(index): Loads and returns the transformed input and target tensors for the specified index.
    """
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
    
class VesselGridData(Dataset): # Dataset for Grid Inputs
    """
    Dataset class for loading and processing vessel grid data.
    Args:
        data_path (str): Path to the directory containing vessel data files.
        filter_zero_velocities (bool, optional): If True, filters out points with zero velocities. Default is False.
        sample_size (int, optional): Number of samples to return. Default is 4096 (2**12).
    Attributes:
        data_path (str): Path to the directory containing vessel data files.
        vessel_files (list): List of vessel data file paths.
        num_vessels (int): Number of vessel data files.
        filter_zero_velocities (bool): If True, filters out points with zero velocities.
        sample_size (int): Number of samples to return.
    Methods:
        __len__(): Returns the number of vessel data files.
        __getitem__(index): Loads and processes the vessel data file at the given index. Returns the grid and point velocities.
    """
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

class VesselGridCondData(Dataset): # Dataset for Grid Inputs with Condition Array
    """
    Dataset for Grid Inputs with Condition Array.
    This dataset class loads vessel grid data and corresponding condition arrays from specified file paths.
    Attributes:
        data_path (str): Path to the directory containing vessel data files.
        vessel_files (list): List of file paths for vessel data.
        condition_files (list): List of file paths for condition data.
        num_vessels (int): Number of vessel data files.
    Methods:
        __len__(): Returns the number of vessel data files.
        __getitem__(index): Returns the grid data and condition array for the specified index.
    Args:
        data_path (str): Path to the directory containing vessel data files.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.vessel_files = get_vessel_files(data_path)
        self.condition_files = get_vessel_files("/home/ne34gux/workspace/experiments/data/vessel_cond_data")
        self.num_vessels = len(self.vessel_files)
    
    def __len__(self): 
        return self.num_vessels
    
    def __getitem__(self, index):
        vessel_file = self.vessel_files[index]
        cond_file = self.condition_files[index]
        fluid_data = np.load(vessel_file)
        cond_data = np.load(cond_file)
       
        grid_xyz = fluid_data['vessel_mask']
        grid_vel = fluid_data['interp_vel']
        cond_array = cond_data['condition_array']
        
        return grid_xyz, grid_vel, cond_array


########################
# Single Point Classes #
########################

class VesselDatasetSinglePoint(Dataset):
    """
    A PyTorch Dataset class for loading and processing vessel data from numpy files.
    Args:
        data_path (str): Path to the directory containing the numpy files.
        sample_size (int, optional): Number of samples to draw from each file. Default is 256.
        transform_function (callable, optional): Function to transform the input and target arrays. Default is transform_linear.
        sample_cap (int, optional): Maximum number of samples to retain after loading data. If None, no cap is applied. Default is None.
    Attributes:
        data_path (str): Path to the directory containing the numpy files.
        sample_size (int): Number of samples to draw from each file.
        transform_function (callable): Function to transform the input and target arrays.
        sample_cap (int or None): Maximum number of samples to retain after loading data.
        input_data (list): List of input data arrays.
        target_data (list): List of target data arrays.
        ref_idx (list): List of reference indices for each sample.
        ref_data (list): List of reference data arrays.
    Methods:
        load_data(): Loads and processes the data from the numpy files.
        apply_cap(): Applies the sample cap to the loaded data.
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the input, target, and reference data for the given index.
    """
    
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

class VesselDatasetRelativeSinglePoint(Dataset): # Dataset for Relative Inputs
    """
    A PyTorch Dataset class for loading vessel data with relative inputs.
    Args:
        data_path (str): Path to the directory containing the dataset files.
        sample_size (int, optional): Number of sample points to select from each file. Default is 256.
        transform_function (callable, optional): A function to transform the data. Default is transform_linear.
    Attributes:
        data_path (str): Path to the directory containing the dataset files.
        sample_size (int): Number of sample points to select from each file.
        transform_function (callable): A function to transform the data.
        input_data (list): List to store input data from all files.
        target_data (list): List to store target data from all files.
        ref_idx (list): List to store reference indices for each input data point.
        ref_data (list): List to store reference data points.
    Methods:
        load_data():
            Loads and processes data from the specified directory.
        __len__():
            Returns the total number of samples in the dataset.
        __getitem__(idx):
            Returns the input, target, and reference data for the given index.
    """
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
    
    
class VesselGridSinglePointData(Dataset): # Dataset for Grid Inputs
    """
    Dataset class for loading and processing vessel grid data for single point inputs.
    Args:
        data_path (str): Path to the directory containing the vessel data files.
        sample_size (int, optional): Number of samples to draw from each file for reference data. Default is 256.
        transform_function (callable, optional): Function to transform the input, target, and geometry arrays. Default is transform_linear.
        sample_cap (int, optional): Maximum number of samples to draw from each file. If None, all samples are used. Default is None.
    Attributes:
        data_path (str): Path to the directory containing the vessel data files.
        transform_function (callable): Function to transform the input, target, and geometry arrays.
        sample_size (int): Number of samples to draw from each file for reference data.
        sample_cap (int or None): Maximum number of samples to draw from each file.
        vessel_files (list): List of vessel data files in the data_path directory.
        num_vessels (int): Number of vessel data files.
        input_data (list): List of input data arrays.
        target_data (list): List of target data arrays.
        ref_idx (list): List of reference indices.
        ref_data (list): List of reference data arrays.
    Methods:
        __len__(): Returns the total number of samples in the dataset.
        load_data(): Loads and processes the vessel data files.
        __getitem__(idx): Returns the input, target, and reference data for the given index.
    """

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