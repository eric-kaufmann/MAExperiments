import os
import pandas as pd
import torch

from train_convolutional_vae import MODEL_PATH, MODEL_NAME
from utils.test_utils import save_results, get_test_files, load_model_weights
from utils.metrics import *
from utils.helper_functions import *
from utils.models import *

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
failed_model_list = []

def decode_model_name(model_name):
    model_split = model_name.split('.')[0].split('_')# model_name.split('_')
    # Remove the model name (wait for double underscore)
    model_split = model_split[model_split.index('')+1:]
    if len(model_split) <= 12:
        model_split += ['cond', 'False']
    model_dict = {
        'lr': float('0.'+model_split[1]),
        'ep': int(model_split[3]),
        'transf': model_split[5],
        'rot': True if model_split[7]=="True" else False,
        'loss': model_split[9],
        'cond': True if model_split[11]=="True" else False,
    }
    return model_dict

def evaluate_model(model_path, param_dict):
    #decode model
    model_dict = decode_model_name(os.path.basename(model_path))
    
    test_files = get_test_files(model_dict['transf'])
    
    model = ConvVAE(
        batch_size=1,
        grid=64 if model_dict['transf'] == 'grid64' else 128,
        cond_dim=21 if model_dict['cond'] else 0
    )
    try:
        model = load_model_weights(model, model_path)
    except:
        print(f"Failed to load model: {model_path}")
        failed_model_list.append(model_path)
        return None
    
    model.to(DEVICE)
    model.eval()
    
    print("   model_size:", get_model_size(model))
    
    results = []
    for i, (filename, file_path) in enumerate(test_files):
        print(f" - {i}: {filename}")
        test_object = np.load(file_path+".npz")

        vessel_dict = {
            'fluid_points': 'vessel_mask',
            'sys_vel': 'interp_vel',
            'mesh_points': 'vessel_mask'
        }
            
        xyz = torch.Tensor(test_object[vessel_dict['fluid_points']])
        vel = torch.Tensor(test_object[vessel_dict['sys_vel']])
        if model_dict['cond']:
            if 'grid64' in model_dict['transf']:
                cond_filepath = file_path.replace('grid64_data', 'cond_data')
            else: #'grid128' in model_dict['transf']:
                cond_filepath = file_path.replace('grid128_data', 'cond_data')
            cond_object = np.load(cond_filepath+".npz")
            cond_array = torch.Tensor(cond_object['condition_array']).to(DEVICE, dtype=torch.float32)
        
        # Todo - Do batching
        batched_input_tensor = xyz.unsqueeze(0)
        batched_target_tensor = vel.permute(3, 0, 1, 2)
            
        batched_input_tensor = batched_input_tensor.to(DEVICE)
        batched_target_tensor = batched_target_tensor.to(DEVICE)
        
        # Run through the model
        if model_dict['cond']:
            batched_pred_tensor, mu, log_var = model(batched_input_tensor, cond_array.unsqueeze(0))
        else:
            batched_pred_tensor, mu, log_var = model(batched_input_tensor)
        
        # batched_pred_tensor to points
        # batched_target_tensor to points
        # batched_input_tensor to points
        
        
        
        _, batched_target_tensor = grid_to_point_cloud(
            batched_input_tensor.permute(1,2,3,0), 
            batched_target_tensor.permute(1,2,3,0)
        )
        
        batched_input_tensor, batched_pred_tensor = grid_to_point_cloud(
            batched_input_tensor.permute(1,2,3,0), 
            batched_pred_tensor.detach().squeeze().permute(1,2,3,0)
        )
        
        # filter out non-zero velocities
        non_zero_vel_idx = torch.all(batched_target_tensor != 0, dim=1)
        batched_input_tensor = batched_input_tensor[non_zero_vel_idx]
        batched_pred_tensor = batched_pred_tensor[non_zero_vel_idx]
        batched_target_tensor = batched_target_tensor[non_zero_vel_idx]
        
        mse = mse_error(batched_pred_tensor, batched_target_tensor)
        mae = mae_error(batched_pred_tensor, batched_target_tensor)
        cs = cosine_similarity(batched_pred_tensor, batched_target_tensor)
        angle_diff = calculate_angle_difference(batched_pred_tensor, batched_target_tensor)

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
            angle_diff,
            model_dict['cond']
        ])

    results_df = pd.DataFrame(data=results, columns=["Model_Name", "Case", "Learning_Rate", "Epochs", "Transformation", "Rotation", "Loss_Function", "MSE", "MAE", "Cosine_Similarity", "Angle_Difference", "Conditionals"])
    
    return results_df

if __name__ == "__main__":
    
    param_dict = {
        "model_name": MODEL_NAME,
        #"sample_size": SAMPLE_SIZE,
    }
    
    model_list = [model for model in os.listdir(MODEL_PATH) if model.startswith(MODEL_NAME+'_')]
    
    result = pd.DataFrame()
    for i, model_file in enumerate(model_list):
        print(f"{i}: {model_file}")
        model_results = evaluate_model(
            os.path.join(MODEL_PATH, model_file), 
            param_dict
        )
        result = pd.concat([result, model_results]).reset_index(drop=True)
        
    save_results(result, "results_"+MODEL_NAME)
    
    print(f"Failed to load model: {failed_model_list}")