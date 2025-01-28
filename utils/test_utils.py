import torch
import os
import pandas as pd

from utils.metrics import *
from utils.helper_functions import *
from utils.models import *
from utils.pointnet2_utils import PointNet2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_POINT_RESULTS = False

def load_model_weights(model, model_path):
    """
    Loads the weights of the specified model and moves it to the correct device.

    Parameters:
        model (torch.nn.Module): The PyTorch model instance.
        model_path (str): Path to the model weights file.

    Returns:
        torch.nn.Module: The model with loaded weights on the configured device.
    """
    state_dict = torch.load(model_path, map_location=torch.device(DEVICE))
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    return model

def decode_model_name(model_name):
    """
    Splits the model name to extract relevant parts, removing extra markers.

    Parameters:
        model_name (str): The raw model name string, including its file extension.

    Returns:
        list: A list of strings representing the processed parts of the model name.
    """
    model_split = model_name.split('.')[0].split('_')
    # Remove the model name (wait for double underscore)
    model_split = model_split[model_split.index('')+1:]
    model_dict = {
        'lr': float('0.'+model_split[1]),
        'ep': int(model_split[3]),
        'transf': model_split[5],
        'rot': True if model_split[7]=="True" else False,
        'loss': model_split[9]
    }
    return model_dict

def save_results(results, name, result_path=None):
    """
    Save the results to a CSV file.
    Parameters:
    results (DataFrame): The results to be saved.
    name (str): The name of the file (without extension).
    result_path (str, optional): The directory path where the file will be saved. 
                                 Defaults to "/home/ne34gux/workspace/experiments/results".
    Returns:
    None
    """
    if result_path is None:
        result_path = "/home/ne34gux/workspace/experiments/results"
    results.to_csv(
        os.path.join(result_path, name+".csv"), 
        index=False
    )
    print(f"Results saved to {os.path.join(result_path, name.replace('.pth', '.csv'))}")
        
def get_test_files(transform_function_str):
    """
    Get the test file paths based on the transformation function string.
    Parameters:
    transform_function_str (str): The transformation function identifier. 
                                  It can be 'grid64', 'grid128', 'rel', or any other string for default paths.
    Returns:
    zip: A zip object containing tuples of test file names and their corresponding paths.
    """
    
    if transform_function_str == 'grid64':
        vessel_data_path = "/home/ne34gux/workspace/experiments/data/vessel_grid64_data"
        geometry_data_path = "/home/ne34gux/workspace/experiments/data/geom_grid64_data"
    elif transform_function_str == 'grid128':
        vessel_data_path = "/home/ne34gux/workspace/experiments/data/vessel_grid128_data"
        geometry_data_path = "/home/ne34gux/workspace/experiments/data/geom_grid128_data"
    elif transform_function_str == 'rel':
        vessel_data_path = "/home/ne34gux/workspace/experiments/data/vessel_relative_data"
        geometry_data_path = "/home/ne34gux/workspace/experiments/data/geom_rel_data"
    else:
        vessel_data_path = "/home/ne34gux/workspace/experiments/data/vessel_point_data"
        geometry_data_path = "/home/ne34gux/workspace/experiments/data/geom_point_data"
        
    test_vessel_filenames = TEST_VESSELS
    test_geometry_filenames = [f.replace(".npz", "") for f in os.listdir(geometry_data_path)]

    test_vessel_paths = [os.path.join(vessel_data_path, f) for f in test_vessel_filenames]
    test_geometry_paths = [os.path.join(geometry_data_path, f) for f in test_geometry_filenames]
    
    file_information = zip(test_vessel_filenames + test_geometry_filenames, test_vessel_paths + test_geometry_paths)
    
    return file_information

def get_model(param_dict, model_dict):
    """
    Constructs and returns a model based on the provided parameters.
    Args:
        param_dict (dict): Dictionary containing parameters for model construction.
            Expected keys:
                - 'sample_size': Size of the sample (int).
                - 'model_name': Name of the model to construct (str).
                - 'input_size': Size of the input (int).
                - 'hidden_size': List of hidden layer sizes (list of int).
                - 'output_size': Size of the output (int).
                - 'encoder_input_size': Size of the encoder input (int).
                - 'encoder_hidden_size': List of encoder hidden layer sizes (list of int).
                - 'encoder_output_size': Size of the encoder output (int).
        model_dict (dict): Dictionary containing model-specific parameters.
            Expected keys:
                - 'transf': Transformation type (str).
    Returns:
        model: An instance of the constructed model.
    """
    
    if model_dict['transf'] == 'rel':
        input_size = 8 * param_dict['sample_size']
    elif (param_dict['model_name'] != 'pointnet') and (param_dict['model_name'] != 'pointnetpp'):
        input_size = param_dict['input_size']


    if ('linear_mlp' in param_dict['model_name']) or ('linear_autoencoder' in param_dict['model_name']):    
        layers = [input_size] + param_dict['hidden_size'] + [param_dict['output_size']]
        model = MLP(layers)
    elif 'linear_encoder' in param_dict['model_name']:
        encoder = MLP([param_dict['encoder_input_size']] + param_dict['encoder_hidden_size'] + [param_dict['encoder_output_size']])
        mlp = MLP([input_size + param_dict['encoder_output_size']] + param_dict['hidden_size'] + [param_dict['output_size']])
        model = EncoderMLP(encoder=encoder, mlp=mlp)
    elif param_dict['model_name'] == 'pointnet':
        encoder = PointNetEncoder(in_channels=3 if model_dict['transf'] != 'rel' else 8, z_size=64)
        decoder = PointNetDecoder(z_size=64, out_dim=param_dict['sample_size'])
        model = PointNet(encoder=encoder, decoder=decoder)
    elif param_dict['model_name'] == 'pointnetpp':
        model = PointNet2(c_in=3 if model_dict['transf'] != 'rel' else 8, c_out=3)
    
    return model
        
def evaluate_model(model_path, param_dict):
    """
    Evaluates a machine learning model on a set of test files and returns the results as a DataFrame.
    Parameters:
    -----------
    model_path : str
        Path to the model file.
    param_dict : dict
        Dictionary containing model parameters and configurations.
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame containing evaluation results with the following columns:
        - "Model_Name": Name of the model.
        - "Case": Test case filename.
        - "Learning_Rate": Learning rate used for training the model.
        - "Epochs": Number of epochs the model was trained for.
        - "Transformation": Type of transformation applied to the data.
        - "Rotation": Rotation applied to the data.
        - "Loss_Function": Loss function used for training the model.
        - "MSE": Mean Squared Error of the model predictions.
        - "MAE": Mean Absolute Error of the model predictions.
        - "Cosine_Similarity": Cosine similarity between the model predictions and the target values.
        - "Angle_Difference": Angle difference between the model predictions and the target values.
    """
    
    model_dict = decode_model_name(model_path)
    
    transform_function = transform_function_mapping[model_dict['transf']]

    test_files = get_test_files(model_dict['transf'])
    
    model = get_model(param_dict, model_dict)
    model = load_model_weights(model, model_path)
    
    print("   model_size:", get_model_size(model))

    results = []
    
    for i, (filename, file_path) in enumerate(test_files):
        print(f" - {i}: {filename}")
        test_object = np.load(file_path+".npz")

        vessel_dict = {
            'fluid_points': 'fluid_points',
            'sys_vel': 'sys_vel',
            'mesh_points': 'mesh_points'
        }
        if model_dict['transf'] == 'grid64' or model_dict['transf'] == 'grid128':
            vessel_dict = {
                'fluid_points': 'vessel_mask',
                'sys_vel': 'interp_vel',
                'mesh_points': 'vessel_mask'
            }
        elif model_dict['transf'] == 'rel':
            vessel_dict = {
                'fluid_points': 'input',
                'sys_vel': 'target',
                'mesh_points': 'mesh_points'
            }
            
        
        # Load the data
        if 'linear_encoder' in param_dict['model_name']:
            xyz = torch.Tensor(test_object[vessel_dict['fluid_points']])
            vel = torch.Tensor(test_object[vessel_dict['sys_vel']])
            if model_dict['transf'] == 'rel':
                geom = xyz[:,:3]
            elif model_dict['transf'] == 'grid64' or model_dict['transf'] == 'grid128':
                xyz, vel = grid_to_point_cloud(xyz, vel)              
                # get only the points with non-zero velocity
                geom = xyz[torch.all(vel != 0, dim=1)]
            else:
                geom = torch.Tensor(test_object[vessel_dict['mesh_points']])
            
            if geom.shape[0] < param_dict['encoder_sample_size']:
                sample_idx = np.random.choice(geom.shape[0], param_dict['encoder_sample_size'], replace=True)
            else:
                sample_idx = np.random.choice(geom.shape[0], param_dict['encoder_sample_size'], replace=False)
            geom = geom[sample_idx]
        else:
            xyz = torch.Tensor(test_object[vessel_dict['fluid_points']])
            vel = torch.Tensor(test_object[vessel_dict['sys_vel']])
            if model_dict['transf'] == 'grid64' or model_dict['transf'] == 'grid128':
                xyz, vel = grid_to_point_cloud(xyz, vel)
        
        
        # Transform the data
        if 'linear_encoder' in param_dict['model_name']:
            xyz, vel, geom = transform_function(xyz, vel, mesh_points=geom)
        else:
            xyz, vel = transform_function(xyz, vel)
        
        # Batch the data
        in_channels = 3 if model_dict['transf'] != 'rel' else 8
        out_channels = 3

        B = xyz.shape[0] // param_dict['sample_size']
        if 'linear_encoder' in param_dict['model_name']:
            batched_input_tensor = xyz[:B*param_dict['sample_size']].view(B, in_channels)
            batched_target_tensor = vel[:B*param_dict['sample_size']].view(B, out_channels)
            batched_geom_tensor = geom.repeat(B, 1, 1)
        else:
            batched_input_tensor = xyz[:B*param_dict['sample_size']].view(B, param_dict['sample_size'], in_channels)
            batched_target_tensor = vel[:B*param_dict['sample_size']].view(B, param_dict['sample_size'], out_channels)
            
        batched_input_tensor = batched_input_tensor.to(DEVICE)
        batched_target_tensor = batched_target_tensor.to(DEVICE)
            
        # Run through the model
        if 'linear_encoder' in param_dict['model_name']:
            batched_pred_tensor = model(batched_input_tensor, batched_geom_tensor).reshape(B, out_channels)
        elif 'linear_autoencoder' in param_dict['model_name']:
            batched_pred_tensor = model(batched_input_tensor.flatten(start_dim=1)).reshape(B, param_dict['sample_size'], out_channels)
        elif 'pointnet_' in param_dict['model_name']:
            batched_pred_tensor = model(batched_input_tensor).reshape(B, param_dict['sample_size'], out_channels)
        elif 'pointnetpp' in param_dict['model_name']:
            sub_batch_size = 4  # Define a smaller batch size to avoid memory issues
            sub_batches = torch.split(batched_input_tensor, sub_batch_size, dim=0)
            batched_pred_tensor = torch.zeros_like(batched_target_tensor, dtype=torch.float32).to(DEVICE)
            for s_idx, sub_batch in enumerate(sub_batches):
                sub_batch = sub_batch.to(DEVICE)
                batched_pred_tensor[s_idx*sub_batch_size:(s_idx+1)*sub_batch_size] = model(sub_batch).reshape(sub_batch.shape[0], param_dict['sample_size'], out_channels).detach()
        elif 'linear_mlp' in param_dict['model_name']:
            batched_pred_tensor = model(batched_input_tensor).reshape(B, out_channels)
        else:
            batched_pred_tensor = model(batched_input_tensor)
        
        batched_input_tensor = batched_input_tensor.view(-1, in_channels)
        batched_target_tensor = batched_target_tensor.view(-1, 3)
        batched_pred_tensor = batched_pred_tensor.view(-1, 3)  
            
        if model_dict['transf'] == 'grid64' or model_dict['transf'] == 'grid128':
            non_zero_vel_idx = torch.all(batched_target_tensor != 0, dim=1)
            
            batched_input_tensor = batched_input_tensor[non_zero_vel_idx]
            batched_pred_tensor = batched_pred_tensor[non_zero_vel_idx]
            batched_target_tensor = batched_target_tensor[non_zero_vel_idx]
        
        
        mse = mse_error(batched_pred_tensor, batched_target_tensor)
        mae = mae_error(batched_pred_tensor, batched_target_tensor)
        cs = cosine_similarity(batched_pred_tensor, batched_target_tensor)
        angle_diff = calculate_angle_difference(batched_pred_tensor, batched_target_tensor)
        
        
        if SAVE_POINT_RESULTS:
            save_path = "/home/ne34gux/workspace/experiments/results/point_results"
            save_path = os.path.join(save_path, param_dict['model_name'])
            save_path = os.path.join(save_path, model_dict['transf'])
            
            save_filename = str(filename.replace(".npz", "")) + "__" + model_path.split("/")[-1].replace(".pth", "") + ".npz"
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            np.savez(
                os.path.join(save_path, save_filename),
                input=batched_input_tensor.cpu().detach().numpy(),
                target=batched_target_tensor.cpu().detach().numpy(),
                pred=batched_pred_tensor.cpu().detach().numpy(),
                model_path=model_path
            )

        results.append([
            param_dict['model_name'], 
            filename, 
            model_dict['lr'], 
            model_dict['ep'], 
            model_dict['transf'], 
            model_dict['rot'], 
            model_dict['loss'], 
            mse, 
            mae, 
            cs,
            angle_diff
        ])

    results_df = pd.DataFrame(
        data=results, 
        columns=[
            "Model_Name", 
            "Case", 
            "Learning_Rate", 
            "Epochs", 
            "Transformation",
            "Rotation",
            "Loss_Function", 
            "MSE", 
            "MAE", 
            "Cosine_Similarity",
            "Angle_Difference"
        ]
    )
    return results_df