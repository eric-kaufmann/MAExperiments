#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --output=./training_logs/preprocessing_%j.out
#SBATCH --error=./training_logs/preprocessing_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=vascunet
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

PROJECT_DIR=$HOME/workspace/experiments

source $PROJECT_DIR/.venv/bin/activate

python $PROJECT_DIR/preprocessing_scripts/generate_grid_data.py