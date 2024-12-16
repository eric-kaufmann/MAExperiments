import os
import pandas as pd

from train_single_coord_linear_encoder import MODEL_PATH, MODEL_NAME, SAMPLE_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, ENCODER_SAMPLE_SIZE, ENCODER_INPUT_SIZE, ENCODER_HIDDEN_SIZE, ENCODER_OUTPUT_SIZE
from utils.test_utils import evaluate_model, save_results

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    
    param_dict = {
        "model_name": MODEL_NAME,
        "sample_size": SAMPLE_SIZE,
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "output_size": OUTPUT_SIZE,
        "encoder_sample_size": ENCODER_SAMPLE_SIZE,
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