#!/bin/bash

# Define the model name
model_name='pointnetpp'

# Define arrays of transforms and learning rates
transforms=('unit_cube' 'zero_center' 'grid64' 'grid128' 'relative_input')
learning_rates=(0.1 0.01 0.001 0.0001)

# Create a directory for logs if it doesn't exist
mkdir -p ./training_logs

for lr in "${learning_rates[@]}"; do
  for transform in "${transforms[@]}"; do

    # Determine rotation options based on the transform
    if [[ "$transform" == "unit_cube" || "$transform" == "zero_center" ]]; then
      rotations=('rot' 'no_rot')
    else
      rotations=('no_rot')
    fi

    for rotation in "${rotations[@]}"; do
      # Define a unique job name
      job_name="${model_name}_${transform}_lr${lr}_${rotation}"

      # Submit the job using sbatch
      sbatch \
        --job-name=${job_name} \
        --output=./training_logs/${job_name}_%j.out \
        --error=./training_logs/${job_name}_%j.err \
        --ntasks=1 \
        --partition=vascunet \
        --gres=gpu:1 \
        --mem=32G \
        --export=ALL,LR=${lr},TRANSFORM=${transform},ROTATION=${rotation} \
        run_${model_name}.sh
    done
  done
done