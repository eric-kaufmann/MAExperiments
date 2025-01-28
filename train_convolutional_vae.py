import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import *
from utils.helper_functions import *
from utils.models import ConvVAE

DATA_PATH = "/home/ne34gux/workspace/experiments/data/vessel_point_data"
MODEL_PATH = "/home/ne34gux/workspace/experiments/trained_models"

MODEL_NAME = "convolutional_vae"

# Model Architecture Parameters
CONDITIONING = True

# Experiment Parameters
UNIT_CUBE = False
ZERO_CENTER = False
GRID64 = True
GRID128 = False
RELATIVE_INPUT = False

ROTATE = False
NSLOSS = False

# Training Parameters
NUM_EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 0.001
SPLIT_RATIO = 0.8
LOG_EVERY = 10
NUM_WORKERS = 1
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
parser.add_argument('--cond', type=str, required=False, default=None, choices=['cond', 'no_cond'], help='Apply condition to the data')
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
if args.cond is not None:
    if args.cond == 'cond':
        CONDITIONING = True
    elif args.cond == 'no_cond':
        CONDITIONING = False
    else:
        raise ValueError("Invalid conditioning type.")

if ROTATE:
    DATA_PATH = "/home/ne34gux/workspace/experiments/data/rotated_vessel_point_data"
    LOG_EVERY *= 10
 
assert not (GRID64 and GRID128), "Cannot enable both grid64 and grid128."
assert not ROTATE, "Rotation not implemented yet."

if UNIT_CUBE:
    TRANSF = 'unitcube'
    transform_function = transform_unit_cube
elif ZERO_CENTER:
    TRANSF = 'zerocenter'
    transform_function = transform_zero_center
elif GRID64:
    TRANSF = 'grid64'
    DATA_PATH = "/home/ne34gux/workspace/experiments/data/vessel_grid64_data"
    transform_function = transform_unit_cube
elif GRID128:
    TRANSF = 'grid128'
    DATA_PATH = "/home/ne34gux/workspace/experiments/data/vessel_grid128_data"
    transform_function = transform_unit_cube
elif RELATIVE_INPUT:
    TRANSF = 'rel'
    DATA_PATH = "/home/ne34gux/workspace/experiments/data/vessel_relative_data"
    transform_function = transform_linear
    IN_CHANNELS = 8

if NSLOSS:
    LOSSF = 'nsloss'
else:
    LOSSF = 'mse'

model_name_dict = {
    'lr': LEARNING_RATE,
    'ep': NUM_EPOCHS,
    'transf': TRANSF,
    'rot': ROTATE,
    'loss': LOSSF,
    'cond': CONDITIONING
}

SAVE_MODEL_NAME = MODEL_NAME + create_model_name_infix(model_name_dict) + '.pth'

# Dictionary of all parameters
params_dict = {
    'SAMPLE_SIZE': str(1),
    'IN_CHANNELS': str(1),
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
    vessel_dataset = VesselGridCondData(DATA_PATH)
    
    train_dataset, val_dataset = split_dataset(vessel_dataset, ratio=SPLIT_RATIO)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print("Data loaded.")

    # Model, loss function, optimizer
    print("Initializing model...")

    model = ConvVAE(
        batch_size=BATCH_SIZE,
        grid=64 if GRID64 else 128,
        cond_dim=21 if CONDITIONING else 0
    )
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
        epoch_mse_list = []
        
        # Training
        model.train()
        for i, (input_tensor, target_tensor, cond_tensor) in enumerate(train_loader):
            input_tensor = input_tensor.to(DEVICE, dtype=torch.float32)
            target_tensor = target_tensor.to(DEVICE, dtype=torch.float32)
            cond_tensor = torch.Tensor(cond_tensor).to(DEVICE, dtype=torch.float32)
            
            if CONDITIONING:
                x_hat, mu, log_var = model(input_tensor, cond_tensor)
            else:
                x_hat, mu, log_var = model(input_tensor)

            recon_loss = criterion(x_hat, target_tensor.permute(0,4,1,2,3))
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            writer.add_scalar('Train/MSE/step', loss.item(), epoch*len(train_loader)+i)
            epoch_mse_list.append(loss.item())
            
            # Log statistics + print step progress
            step_id = i+1
            if (step_id) % LOG_EVERY == 0:
                now = time.time()
                epoch_duration = now - epoch_start_time
                elapsed_time = now - start_time
                remaining_time = (epoch_duration / (step_id + 1)) * (len(train_loader) - step_id - 1)
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{step_id}/{len(train_loader)}], Loss: {loss.item():.4f}, Elapsed Time: {elapsed_time:.2f}s, Epoch Duration: {epoch_duration:.2f}s, Remaining Epoch Time: {remaining_time:.2f}s')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (input_tensor, target_tensor, cond_tensor) in enumerate(val_loader):
                input_tensor = input_tensor.to(DEVICE, dtype=torch.float32)
                target_tensor = target_tensor.to(DEVICE, dtype=torch.float32)
                cond_tensor = torch.Tensor(cond_tensor).to(DEVICE, dtype=torch.float32)
        
                if CONDITIONING:
                    x_hat, mu, log_var = model(input_tensor, cond_tensor)
                else:
                    x_hat, mu, log_var = model(input_tensor)

                recon_loss = criterion(x_hat, target_tensor.permute(0,4,1,2,3))
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kl_loss

                val_loss += loss.item()
                
                writer.add_scalar('Validation/MSE/Step', loss.item(), epoch*len(val_loader)+i)

        val_loss /= len(val_loader)  # Average validation loss
        writer.add_scalar('Validation/MSE/Epoch', val_loss, epoch)
        
        # Print epoch statistics
        epoch_duration = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (epoch + 1)) * (NUM_EPOCHS - epoch - 1)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss:.4f}, Epoch Duration: {epoch_duration:.2f}s, Remaining Time: {remaining_time:.2f}s')
        
        # Save model if validation loss is the best so far
        if SAVE_MODEL and (val_loss < best_val_loss):
            best_val_loss = val_loss
            model_filepath = os.path.join(MODEL_PATH, SAVE_MODEL_NAME)
            torch.save(model.state_dict(), model_filepath)
            print(f'Model saved with validation loss: {val_loss:.4f}')
            
    print("Training complete.")
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    train(args)