import torch
import os
import pandas as pd

from utils.metrics import *
from utils.helper_functions import *
from utils.models import *
from utils.pointnet2_utils import PointNet2_2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_weights(model, model_path):
    state_dict = torch.load(model_path, map_location=torch.device(DEVICE))
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    return model

def decode_model_name(model_name):
    model_split = model_name.split('_')
    # Remove the model name (wait for double underscore)
    model_split = model_split[model_split.index('')+1:]
    model_dict = {
        'lr': float('0.'+model_split[1]),
        'ep': int(model_split[3]),
        'transf': model_split[5],
        'rot': bool(model_split[7]),
        'loss': model_split[8]
    }
    return model_dict

def save_results(results, name):
    result_path = "/home/ne34gux/workspace/experiments/results"
    results.to_csv(
        os.path.join(result_path, name+".csv"), 
        index=False
    )
    print(f"Results saved to {os.path.join(result_path, name.replace('.pth', '.csv'))}")
        
def get_test_files(transform_function_str):
    if transform_function_str == 'grid64':
        vessel_data_path = "/home/ne34gux/workspace/experiments/data/vessel_grid64_data"
        geometry_data_path = "/home/ne34gux/workspace/experiments/data/geom_grid64_data"
    elif transform_function_str == 'grid128':
        vessel_data_path = "/home/ne34gux/workspace/experiments/data/vessel_grid128_data"
        geometry_data_path = "/home/ne34gux/workspace/experiments/data/geom_grid128_data"
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
    if ('linear_mlp' in param_dict['model_name']) or ('linear_autoencoder' in param_dict['model_name']):    
        layers = [param_dict['input_size']] + param_dict['hidden_size'] + [param_dict['output_size']]
        model = MLP(layers)
    elif 'linear_encoder' in param_dict['model_name']:
        encoder = MLP([param_dict['encoder_input_size']] + param_dict['encoder_hidden_size'] + [param_dict['encoder_output_size']])
        mlp = MLP([param_dict['input_size']] + param_dict['hidden_size'] + [param_dict['output_size']])
        model = EncoderMLP(encoder=encoder, mlp=mlp)
    elif param_dict['model_name'] == 'pointnet':
        encoder = PointNetEncoder(in_channels=3 if model_dict['transf'] != 'rel' else 8, z_size=64)
        decoder = PointNetDecoder(z_size=64, out_dim=param_dict['sample_size'])
        model = PointNet(encoder=encoder, decoder=decoder)
    elif param_dict['model_name'] == 'pointnetpp':
        model = PointNet2_2(c_in=3 if model_dict['transf'] != 'rel' else 8, c_out=3)
    
    return model
        
def evaluate_model(model_path, param_dict):
    model_dict = decode_model_name(model_path)
    
    if model_dict['transf'] == 'rel':
        print("   >> Skipped")
        return pd.DataFrame()
    
    transform_function = transform_function_mapping[model_dict['transf']]

    test_files = get_test_files(model_dict['transf'])
    
    model = get_model(param_dict, model_dict)
    model = load_model_weights(model, model_path)
    
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
                'fluid_points': 'input_tensor',
                'sys_vel': 'target_tensor',
                'mesh_points': 'mesh_points'
            }
            param_dict['input_size'] = 8 * param_dict['sample_size']
        
        # Load the data
        if 'linear_encoder' in param_dict['model_name']:
            xyz = torch.Tensor(test_object[vessel_dict['fluid_points']])
            vel = torch.Tensor(test_object[vessel_dict['sys_vel']])
            if model_dict['transf'] == 'rel':
                geom = xyz[torch.all(vel != 0, dim=1)]
            else:
                geom = torch.Tensor(test_object[vessel_dict['mesh_points']])
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
        if 'linear_encoder' in param_dict['model_name']:
            B = xyz.shape[0] // param_dict['sample_size']
            batched_input_tensor = xyz[:B*param_dict['sample_size']].view(B, 3)
            batched_target_tensor = vel[:B*param_dict['sample_size']].view(B, 3)
            batched_geom_tensor = geom.repeat(B, 1, 1)
        else:
            B = xyz.shape[0] // param_dict['sample_size']
            batched_input_tensor = xyz[:B*param_dict['sample_size']].view(B, param_dict['sample_size'], 3)
            batched_target_tensor = vel[:B*param_dict['sample_size']].view(B, param_dict['sample_size'], 3)
            
        # Run through the model
        if 'linear_encoder' in param_dict['model_name']:
            batched_pred_tensor = model(batched_input_tensor, batched_geom_tensor).reshape(B, 3)
        elif 'linear_autoencoder' in param_dict['model_name']:
            batched_pred_tensor = model(batched_input_tensor.flatten(start_dim=1)).reshape(B, param_dict['sample_size'], 3)
        elif 'pointnet' in param_dict['model_name']:
            batched_pred_tensor = model(batched_input_tensor).reshape(B, param_dict['sample_size'], 3)
        elif 'linear_mlp' in param_dict['model_name']:
            batched_pred_tensor = model(batched_input_tensor).reshape(B, 3)
        else:
            batched_pred_tensor = model(batched_input_tensor)
        
        mse = mse_error(batched_pred_tensor, batched_target_tensor)
        mae = mae_error(batched_pred_tensor, batched_target_tensor)
        cs = cosine_similarity(batched_pred_tensor, batched_target_tensor)

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
            cs
        ])

    results_df = pd.DataFrame(data=results, columns=["Model_Name", "Case", "Learning_Rate", "Epochs", "Transformation", "Rotation", "Loss_Function", "MSE", "MAE", "Cosine_Similarity"])
    return results_df