#!/bin/bash
#SBATCH --job-name=small_sample_linear_autoencoder
#SBATCH --output=./training_logs/small_sample_linear_autoencoder_%j.out
#SBATCH --error=./training_logs/small_sample_linear_autoencoder_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=vascunet
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

PROJECT_DIR=$HOME/workspace/experiments

source $PROJECT_DIR/.venv/bin/activate

# Use LR and TRANSFORM environment variables, default if not set
LR=${LR:-0.0001}
TRANSFORM=${TRANSFORM:-'relative_input'}
ROTATION=${ROTATION:-'no_rot'}

python -W ignore $PROJECT_DIR/train_small_sample_linear_autoencoder.py \
  --job_id $SLURM_JOB_ID \
  --num_epochs 30 \
  --lr $LR \
  --transform $TRANSFORM \
  --num_workers 2 \
  --rot $ROTATION

# ['unit_cube', 'zero_center', 'grid64', 'grid128', 'relative_input']