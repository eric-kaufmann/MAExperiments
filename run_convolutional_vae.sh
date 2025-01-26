#!/bin/bash
#SBATCH --job-name=convolutional_vae
#SBATCH --output=./training_logs/convolutional_vae_%j.out
#SBATCH --error=./training_logs/convolutional_vae_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=vascunet
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

PROJECT_DIR=$HOME/workspace/experiments

source $PROJECT_DIR/.venv/bin/activate

# Use LR and TRANSFORM environment variables, default if not set
EP=${EP:-1}
LR=${LR:-0.01}
TRANSFORM=${TRANSFORM:-'grid64'}
ROTATION=${ROTATION:-'no_rot'}
COND=${COND:-'no_cond'}

python -W ignore $PROJECT_DIR/train_convolutional_vae.py \
  --job_id $SLURM_JOB_ID \
  --num_epochs $EP \
  --lr $LR \
  --transform $TRANSFORM \
  --num_workers 2 \
  --rot $ROTATION \
  --cond $COND

# ['unit_cube', 'zero_center', 'grid64', 'grid128', 'relative_input']

#[
# '/home/ne34gux/workspace/experiments/trained_models/convolutional_vae__lr_01_ep_1_transf_grid64_rot_False_loss_mse.pth', 
# '/home/ne34gux/workspace/experiments/trained_models/convolutional_vae__lr_1_ep_10_transf_grid64_rot_False_loss_mse.pth', 
# '/home/ne34gux/workspace/experiments/trained_models/convolutional_vae__lr_1_ep_30_transf_grid64_rot_False_loss_mse.pth'
#]