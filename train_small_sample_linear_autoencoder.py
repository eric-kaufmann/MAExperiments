import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import *
from utils.helper_functions import *
from utils.models import MLP

DATA_PATH = "/home/ne34gux/workspace/experiments/data/vessel_point_data"
MODEL_PATH = "/home/ne34gux/workspace/experiments/trained_models"

MODEL_NAME = "small_sample_linear_autoencoder"

# Model Architecture Parameters
SAMPLE_SIZE = 256
INPUT_SIZE = 3*SAMPLE_SIZE
HIDDEN_SIZE = [128, 64, 64, 128]
OUTPUT_SIZE = 3*SAMPLE_SIZE

# Experiment Parameters
UNIT_CUBE = True
ZERO_CENTER = False
GRID64 = False
GRID128 = False
RELATIVE_INPUT = False

ROTATE = False
NSLOSS = False

# Training Parameters
NUM_EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 0.001
SPLIT_RATIO = 0.8
LOG_EVERY = 5
NUM_WORKERS = 2
SAVE_MODEL = True

# Adapt Logic based on Parameters

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--job_id', type=str, required=False, default='0', help='Job ID for the training run')
parser.add_argument('--num_epochs', type=int, required=False, default=None, help='Number of epochs for training')
parser.add_argument('--lr', type=float, required=False, default=None, help='Learning rate for training')
parser.add_argument('--transform', type=str, required=False, default=None, choices=['unit_cube', 'zero_center', 'grid64', 'grid128', 'relative_input'], help='Transformation to apply to the data')
parser.add_argument('--num_workers', type=int, required=False, default=None, help='Number of workers for data loading')
parser.add_argument('--rotate', type=str, required=False, default=None, choices=['rot', 'no_rot'], help='Apply rotation to the data')
args = parser.parse_args()


if args.num_epochs is not None:
    NUM_EPOCHS = args.num_epochs
if args.lr is not None:
    LEARNING_RATE = args.lr
if args.transform is not None:
    UNIT_CUBE = False
    ZERO_CENTER = False
    GRID64 = False
    GRID128 = False
    RELATIVE_INPUT = False
    if args.transform == 'unit_cube':
        UNIT_CUBE = True
    elif args.transform == 'zero_center':
        ZERO_CENTER = True
    elif args.transform == 'grid64':
        GRID64 = True
    elif args.transform == 'grid128':
        GRID128 = True
    elif args.transform == 'relative_input':
        RELATIVE_INPUT = True
    else:
        raise ValueError("Invalid transformation type.")
if args.num_workers is not None:
    NUM_WORKERS = args.num_workers
if args.rotate is not None:
    if args.rotate == 'rot':
        ROTATE = True
    elif args.rotate == 'no_rot':
        ROTATE = False
    else:
        raise ValueError("Invalid rotation type.")

if ROTATE:
    DATA_PATH = "/home/ne34gux/workspace/experiments/data/rotated_vessel_point_data"
    LOG_EVERY *= 10

assert sum([UNIT_CUBE, ZERO_CENTER, GRID64, GRID128, RELATIVE_INPUT]) == 1, "Exactly one transformation should be enabled."
assert sum([GRID64, GRID128, RELATIVE_INPUT, ROTATE]) <= 1, "Grid / Relative Input and rotation cannot be enabled at the same time."

if UNIT_CUBE:
    TRANSF = 'unitcube'
    transform_function = transform_unit_cube
elif ZERO_CENTER:
    TRANSF = 'zerocenter'
    transform_function = transform_zero_center
elif GRID64:
    TRANSF = 'grid64'
    DATA_PATH = "/home/ne34gux/workspace/experiments/data/vessel_grid64_data"
    # No adaptation of sample size because of memory constraints
    transform_function = transform_unit_cube
elif GRID128:
    TRANSF = 'grid128'
    DATA_PATH = "/home/ne34gux/workspace/experiments/data/vessel_grid128_data"
    # No adaptation of sample size because of memory constraints
    transform_function = transform_unit_cube
elif RELATIVE_INPUT:
    TRANSF = 'rel'
    DATA_PATH = "/home/ne34gux/workspace/experiments/data/vessel_relative_data"
    INPUT_SIZE = 8*SAMPLE_SIZE
    transform_function = transform_linear

if NSLOSS:
    LOSSF = 'nsloss'
else:
    LOSSF = 'mse'

model_name_dict = {
    'lr': LEARNING_RATE,
    'ep': NUM_EPOCHS,
    'transf': TRANSF,
    'rot': ROTATE,
    'loss': LOSSF
}

SAVE_MODEL_NAME = MODEL_NAME + create_model_name_infix(model_name_dict) + '.pth'

# Dictionary of all parameters
params_dict = {
    'SAMPLE_SIZE': str(SAMPLE_SIZE),
    'INPUT_SIZE': str(INPUT_SIZE),
    'HIDDEN_SIZE': str(HIDDEN_SIZE),
    'OUTPUT_SIZE': str(OUTPUT_SIZE),
    'UNIT_CUBE': str(UNIT_CUBE),
    'ZERO_CENTER': str(ZERO_CENTER),
    'ROTATE': str(ROTATE),
    'GRID64': str(GRID64),
    'GRID128': str(GRID128),
    'NSLOSS': str(NSLOSS),
    'RELATIVE_INPUT': str(RELATIVE_INPUT),
    'NUM_EPOCHS': str(NUM_EPOCHS),
    'BATCH_SIZE': str(BATCH_SIZE),
    'LEARNING_RATE': str(LEARNING_RATE),
    'SPLIT_RATIO': str(SPLIT_RATIO),
    'LOG_EVERY': str(LOG_EVERY),
    'DEVICE': str(DEVICE),
    'NUM_WORKERS': str(NUM_WORKERS),
    'SAVE_MODEL': str(SAVE_MODEL),
    'DATA_PATH': str(DATA_PATH),
    'MODEL_PATH': str(MODEL_PATH),
    'SAVE_MODEL_NAME': str(SAVE_MODEL_NAME),
    'JOB_ID': str(args.job_id)
}

np.random.seed(42)

def train(args):
    print_hyperparameters(params_dict)
    
    tensorboard_log_dir = os.path.join("tensorboard_runs", MODEL_NAME + create_model_name_infix(model_name_dict) + "_" + args.job_id)
    writer = SummaryWriter(tensorboard_log_dir)
    
    writer.add_hparams(params_dict, {})
    
    print("Loading data...")
    if GRID64 or GRID128:
        vessel_dataset = VesselGridData(DATA_PATH, filter_zero_velocities=True, sample_size=SAMPLE_SIZE)
    elif RELATIVE_INPUT:
        vessel_dataset = VesselRelativePointCloudData(DATA_PATH, transform_function=transform_function, num_samples=SAMPLE_SIZE)
    else:
        vessel_dataset = VesselPointCloudData(DATA_PATH, num_samples=SAMPLE_SIZE, transform_function=transform_function)

    train_dataset, val_dataset = split_dataset(vessel_dataset, ratio=SPLIT_RATIO)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print("Data loaded.")

    # Model, loss function, optimizer
    print("Initializing model...")
    layers = [INPUT_SIZE] + HIDDEN_SIZE + [OUTPUT_SIZE]
    model = MLP(layers)
    print(f"Model size: {get_model_size(model):.2f} MB")
    model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model initialized.")

    # Training loop
    print("Training model...")
    start_time = time.time()
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        epoch_mse_list = []
        for i, (input_tensor, target_tensor) in enumerate(train_loader):
            input_tensor = input_tensor.to(DEVICE)
            target_tensor = target_tensor.to(DEVICE)

            pred_tensor = model(input_tensor.flatten(start_dim=1))

            loss = criterion(pred_tensor, target_tensor.flatten(start_dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            writer.add_scalar('Train/MSE/step', loss.item(), epoch*len(train_loader)+i)
            epoch_mse_list.append(loss.item())
            
            step_id = i+1
            if (step_id) % LOG_EVERY == 0:
                now = time.time()
                epoch_duration = now - epoch_start_time
                elapsed_time = now - start_time
                remaining_time = (epoch_duration / (step_id + 1)) * (len(train_loader) - step_id - 1)
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{step_id}/{len(train_loader)}], Loss: {loss.item():.4f}, Elapsed Time: {elapsed_time:.2f}s, Epoch Duration: {epoch_duration:.2f}s, Remaining Epoch Time: {remaining_time:.2f}s')
        epoch_mse = np.mean(epoch_mse_list)
        writer.add_scalar('Train/MSE/epoch', epoch_mse, epoch)

        model.eval()
        val_loss = 0
        with torch.no_grad():  # Disable gradient computation
            for i, (input_tensor, target_tensor) in enumerate(val_loader):
                input_tensor = input_tensor.to(DEVICE)
                target_tensor = target_tensor.to(DEVICE)

                pred_tensor = model(input_tensor.flatten(start_dim=1))

                loss = criterion(pred_tensor, target_tensor.flatten(start_dim=1))
                val_loss += loss.item()
                
                writer.add_scalar('Validation/MSE/Step', loss.item(), epoch*len(val_loader)+i)

        val_loss /= len(val_loader)  # Average validation loss
        writer.add_scalar('Validation/MSE/Epoch', val_loss, epoch)
        epoch_duration = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (epoch + 1)) * (NUM_EPOCHS - epoch - 1)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss:.4f}, Epoch Duration: {epoch_duration:.2f}s, Remaining Time: {remaining_time:.2f}s')
        if SAVE_MODEL and (val_loss < best_val_loss):
            best_val_loss = val_loss
            model_filepath = os.path.join(MODEL_PATH, SAVE_MODEL_NAME)
            torch.save(model.state_dict(), model_filepath)
            print(f'Model saved with validation loss: {val_loss:.4f}')
            
    print("Training complete.")
    torch.cuda.empty_cache()
    
    
if __name__ == "__main__":
    train(args)