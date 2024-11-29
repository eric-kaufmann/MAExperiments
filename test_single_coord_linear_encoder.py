# import torch
# import os
# import pandas as pd

# from utils.metrics import *
# from utils.helper_functions import *
# from utils.models import MLP, Encoder
# from train_single_coord_linear_encoder import MODEL_PATH, MODEL_NAME, SAMPLE_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SAVE_MODEL_NAME, ENCODER_SAMPLE_SIZE, ENCODER_INPUT_SIZE, ENCODER_HIDDEN_SIZE, ENCODER_OUTPUT_SIZE, Model

# PINN_PATH = "/home/ne34gux/workspace/experiments/data/pinn_blood_flow_geometries"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# import warnings
# warnings.filterwarnings("ignore")

# def load_model(model, model_path):
#     state_dict = torch.load(model_path, map_location=torch.device(DEVICE))
#     model.load_state_dict(state_dict)
#     model = model.to(DEVICE)
#     return model

# def iterate_over_models():
#     model_list = [model for model in os.listdir(MODEL_PATH) if model.startswith(MODEL_NAME)]
#     for i, model_file in enumerate(model_list):
#         print(f"{i}: {model_file}")
#         model_dict = decode_model_name(model_file)

#         transform_function = transform_function_mapping[model_dict['transf']]
#         data_path = data_path_mapping[model_dict['transf']]

#         test_vessel_files = [os.path.join(data_path, f) for f in TEST_VESSELS]
#         test_geometry_files = [os.path.join(PINN_PATH, f) for f in os.listdir(PINN_PATH) if "velocities" in f]
        
#         if model_dict['transf'] == 'grid64':
#             encoder_sample_size = 64**3
#         elif model_dict['transf'] == 'grid128':
#             encoder_sample_size = 128**3
#         else:
#             encoder_sample_size = ENCODER_SAMPLE_SIZE

#         encoder = Encoder([ENCODER_INPUT_SIZE] + ENCODER_HIDDEN_SIZE + [ENCODER_OUTPUT_SIZE])
#         mlp = MLP([INPUT_SIZE] + HIDDEN_SIZE + [OUTPUT_SIZE])
#         model = Model(encoder=encoder, mlp=mlp)
#         model = load_model(model, os.path.join(MODEL_PATH, model_file))

#         test_vessel_results = evaluate_test_vessels(model, test_vessel_files, transform_function, encoder_sample_size)
#         test_geometry_results = evaluate_test_geometries(model, test_geometry_files, transform_function, encoder_sample_size)

#         results = pd.concat([test_vessel_results, test_geometry_results])
        
#         save_results(results, model_file.replace(".pth", ".csv"))

# def save_results(results, name):
#     result_path = "/home/ne34gux/workspace/experiments/results"
#     results.to_csv(
#         os.path.join(result_path, name.replace(".pth", ".csv")), 
#         index=False
#     )
#     print(f"Results saved to {os.path.join(result_path, SAVE_MODEL_NAME.replace('.pth', '.csv'))}")
        
# def evaluate_test_vessels(model, test_vessel_files, transform_function, sample_size):
#     results = []
#     for g in test_vessel_files:
#         g_object = np.load(g+".npz")
#         if model_dict['transf'] == 'grid64' or model_dict['transf'] == 'grid128':
#             grid_xyz = torch.Tensor(g_object['vessel_mask'])
#             grid_vel = torch.Tensor(g_object['interp_vel'])
#             xyz, vel = grid_to_point_cloud(grid_xyz, grid_vel)
#         else:
#             xyz = torch.Tensor(g_object['fluid_points'])
#             vel = torch.Tensor(g_object['sys_vel'])
#         xyz = torch.Tensor(g_object['fluid_points'])
#         vel = torch.Tensor(g_object['sys_vel'])
#         geom = torch.Tensor(g_object['mesh_points'])
        
#         sample_idx = np.random.choice(geom.shape[0], sample_size, replace=False)
#         geom = geom[sample_idx]
    
#         xyz, vel, geom = transform_function(xyz, vel, mesh_points=geom)
        
#         B = xyz.shape[0] // SAMPLE_SIZE
#         batched_input_tensor = xyz[:B*SAMPLE_SIZE].view(B, 3)
#         batched_target_tensor = vel[:B*SAMPLE_SIZE].view(B, 3)
#         batched_geom_tensor = geom.repeat(B, 1, 1)
#         batched_pred_tensor = model(batched_input_tensor, batched_geom_tensor).reshape(B, 3)

#         mse = mse_error(batched_pred_tensor, batched_target_tensor)
#         mae = mae_error(batched_pred_tensor, batched_target_tensor)
#         cs = cosine_similarity(batched_pred_tensor, batched_target_tensor)

#         # print(g)
#         # print(f"   MSE: {mse}")
#         # print(f"   MAE: {mae}")
#         # print(f"   Cosine Similarity: {cs}")
        
#         results.append([g.split(".")[0], mse, mae, cs])
#     results_df = pd.DataFrame(data=results, columns=["Vessel", "MSE", "MAE", "Cosine_Similarity"])
#     return results_df  


# def evaluate_test_geometries(model, test_geometry_files, transform_function, sample_size):
#     results = []
#     for g in test_geometry_files:
#         g_object = pd.read_csv(g).to_numpy()
#         xyz = torch.Tensor(g_object[:,4:7])
#         vel = torch.Tensor(g_object[:,0:3])
        

#         xyz, vel = transform_function(xyz, vel)

#         B = xyz.shape[0] // SAMPLE_SIZE
#         batched_input_tensor = xyz[:B*SAMPLE_SIZE].view(B, SAMPLE_SIZE, 3)
#         batched_target_tensor = vel[:B*SAMPLE_SIZE].view(B, SAMPLE_SIZE, 3)
#         batched_pred_tensor = model(batched_input_tensor.flatten(start_dim=1)).reshape(B, SAMPLE_SIZE, 3)
#         batched_input_tensor.shape, batched_target_tensor.shape, batched_pred_tensor.shape  
        
#         sample_idx = np.random.choice(geom.shape[0], sample_size, replace=False)
#         geom = geom[sample_idx]
    
#         xyz, vel, geom = transform_function(xyz, vel, mesh_points=geom)
        
#         B = xyz.shape[0] // SAMPLE_SIZE
#         batched_input_tensor = xyz[:B*SAMPLE_SIZE].view(B, 3)
#         batched_target_tensor = vel[:B*SAMPLE_SIZE].view(B, 3)
#         batched_geom_tensor = geom.repeat(B, 1, 1)
#         batched_pred_tensor = model(batched_input_tensor, batched_geom_tensor).reshape(B, 3)

#         mse = mse_error(batched_pred_tensor, batched_target_tensor)
#         mae = mae_error(batched_pred_tensor, batched_target_tensor)
#         cs = cosine_similarity(batched_pred_tensor, batched_target_tensor)

#         # print(g)
#         # print(f"   MSE: {mse}")
#         # print(f"   MAE: {mae}")
#         # print(f"   Cosine Similarity: {cs}")
    
#         results.append([g.split(".")[0], mse, mae, cs])    
#     results_df = pd.DataFrame(data=results, columns=["Vessel", "MSE", "MAE", "Cosine_Similarity"])
#     return results_df    

# if __name__ == "__main__":
#     iterate_over_models()

import os
import pandas as pd

from train_single_coord_linear_encoder import MODEL_PATH, MODEL_NAME, SAMPLE_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, ENCODER_INPUT_SIZE, ENCODER_HIDDEN_SIZE, ENCODER_OUTPUT_SIZE

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    from utils.test_utils import evaluate_model, save_results
    
    param_dict = {
        "model_name": MODEL_NAME,
        "sample_size": SAMPLE_SIZE,
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "output_size": OUTPUT_SIZE,
        "encoder_input_size": ENCODER_INPUT_SIZE,
        "encoder_hidden_size": ENCODER_HIDDEN_SIZE,
        "encoder_output_size": ENCODER_OUTPUT_SIZE
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