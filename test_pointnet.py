import os
import pandas as pd

from train_pointnet import MODEL_PATH, MODEL_NAME, SAMPLE_SIZE
from utils.test_utils import evaluate_model, save_results

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    
    param_dict = {
        "model_name": MODEL_NAME,
        "sample_size": SAMPLE_SIZE,
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
        
    save_results(result, "results2_"+MODEL_NAME)