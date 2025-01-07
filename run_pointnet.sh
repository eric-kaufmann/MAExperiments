#!/bin/bash
#SBATCH --job-name=pointnet
#SBATCH --output=./training_logs/pointnet_%j.out
#SBATCH --error=./training_logs/pointnet_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=vascunet
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

PROJECT_DIR=$HOME/workspace/experiments

source $PROJECT_DIR/.venv/bin/activate

# Use LR and TRANSFORM environment variables, default if not set
EP=${EP:-30}
LR=${LR:-0.1}
TRANSFORM=${TRANSFORM:-'unit_cube'}
ROTATION=${ROTATION:-'no_rot'}

python -W ignore $PROJECT_DIR/train_pointnet.py \
  --job_id $SLURM_JOB_ID \
  --num_epochs $EP \
  --lr $LR \
  --transform $TRANSFORM \
  --num_workers 2 \
  --rot $ROTATION

# ['unit_cube', 'zero_center', 'grid64', 'grid128', 'relative_input']