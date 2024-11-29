from utils.pointnet2_utils import PointNet2_2
import torch
import os
import pandas as pd

from utils.metrics import *
from utils.helper_functions import *
from train_pointnetpp import MODEL_PATH, MODEL_NAME, SAMPLE_SIZE, SAVE_MODEL_NAME

PINN_PATH = "/home/ne34gux/workspace/experiments/data/pinn_blood_flow_geometries"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import warnings
warnings.filterwarnings("ignore")


def load_model(model, model_path):
    state_dict = torch.load(model_path, map_location=torch.device(DEVICE))
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    return model

def iterate_over_models():
    model_list = [model for model in os.listdir(MODEL_PATH) if model.startswith(MODEL_NAME+'_')]
    for i, model_file in enumerate(model_list):
        print(f"{i}: {model_file}")
        model_dict = decode_model_name(model_file)

        transform_function = transform_function_mapping[model_dict['transf']]
        data_path = data_path_mapping[model_dict['transf']]

        test_vessel_files = [os.path.join(data_path, f) for f in TEST_VESSELS]
        test_geometry_files = [os.path.join(PINN_PATH, f) for f in os.listdir(PINN_PATH) if "velocities" in f]

        if model_dict['transf'] == 'grid64':
            sample_size = 64**3
        elif model_dict['transf'] == 'grid128':
            sample_size = 128**3
        else:
            sample_size = SAMPLE_SIZE
 
        model = PointNet2_2(c_in=3, c_out=3)

        model = load_model(model, os.path.join(MODEL_PATH, model_file))

        test_vessel_results = evaluate_test_vessels(model, test_vessel_files, transform_function, sample_size)
        test_geometry_results = evaluate_test_geometries(model, test_geometry_files, transform_function, sample_size)

        results = pd.concat([test_vessel_results, test_geometry_results])
        
        save_results(results, model_file)

def save_results(results, name):
    result_path = "/home/ne34gux/workspace/experiments/results"
    results.to_csv(
        os.path.join(result_path, name.replace(".pth", ".csv")), 
        index=False
    )
        
def evaluate_test_vessels(model, test_vessel_files, transform_function, sample_size):
    results = []
    for g in test_vessel_files:
        g_object = np.load(g+".npz")
        xyz = torch.Tensor(g_object['fluid_points'])
        vel = torch.Tensor(g_object['sys_vel'])
    
        xyz, vel = transform_function(xyz, vel)
        
        B = xyz.shape[0] // sample_size
        batched_input_tensor = xyz[:B*sample_size].view(B, sample_size, 3)
        batched_target_tensor = vel[:B*sample_size].view(B, sample_size, 3)
        batched_pred_tensor = model(batched_input_tensor).reshape(B, sample_size, 3)
        batched_input_tensor.shape, batched_target_tensor.shape, batched_pred_tensor.shape  

        mse = mse_error(batched_pred_tensor, batched_target_tensor)
        mae = mae_error(batched_pred_tensor, batched_target_tensor)
        cs = cosine_similarity(batched_pred_tensor, batched_target_tensor)

        print(g)
        print(f"   MSE: {mse}")
        print(f"   MAE: {mae}")
        print(f"   Cosine Similarity: {cs}")
        
        results.append([os.path.basename(g), mse, mae, cs])
    results_df = pd.DataFrame(data=results, columns=["Vessel", "MSE", "MAE", "Cosine_Similarity"])
    return results_df  


def evaluate_test_geometries(model, test_geometry_files, transform_function, sample_size):
    results = []
    for g in test_geometry_files:
        print(g)
        g_object = pd.read_csv(g).to_numpy()
        xyz = torch.Tensor(g_object[:,4:7])
        vel = torch.Tensor(g_object[:,0:3])

        xyz, vel = transform_function(xyz, vel)

        B = xyz.shape[0] // sample_size
        batched_input_tensor = xyz[:B*sample_size].view(B, sample_size, 3)
        batched_target_tensor = vel[:B*sample_size].view(B, sample_size, 3)
        batched_pred_tensor = model(batched_input_tensor).reshape(B, sample_size, 3)
        batched_input_tensor.shape, batched_target_tensor.shape, batched_pred_tensor.shape  

        mse = mse_error(batched_pred_tensor, batched_target_tensor)
        mae = mae_error(batched_pred_tensor, batched_target_tensor)
        cs = cosine_similarity(batched_pred_tensor, batched_target_tensor)

        print(f"   MSE: {mse}")
        print(f"   MAE: {mae}")
        print(f"   Cosine Similarity: {cs}")
    
        results.append([os.path.basename(g), mse, mae, cs])    
    results_df = pd.DataFrame(data=results, columns=["Vessel", "MSE", "MAE", "Cosine_Similarity"])
    return results_df    

if __name__ == "__main__":
    iterate_over_models()