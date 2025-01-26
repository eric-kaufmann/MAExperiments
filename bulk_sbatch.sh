#!/bin/bash

# Define the model name
#models=('single_coord_linear_mlp' 'single_coord_linear_encoder' 'small_sample_linear_autoencoder' 'large_sample_linear_autoencoder' 'pointnet' 'pointnetpp' 'convolutional_vae')
models=('convolutional_vae')

# Define arrays of transforms and learning rates
# transforms=('unit_cube' 'zero_center' 'grid64' 'grid128' 'relative_input')
transforms=('grid64' 'grid128')

learning_rates=(0.1 0.01 0.001 0.0001)
#learning_rates=(0.0001)

#epochs=(10)
epochs=(1 10 30 50)

# define gpu partition
partition='vascunet'
#partition='gpu'


# Create a directory for logs if it doesn't exist
mkdir -p ./training_logs

for model in "${models[@]}"; do
  for eps in "${epochs[@]}"; do
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
          job_name="${model}_${transform}_lr${lr}_${rotation}"

          # Submit the job using sbatch
          sbatch \
            --job-name=${job_name} \
            --output=./training_logs/${job_name}_%j.out \
            --error=./training_logs/${job_name}_%j.err \
            --ntasks=1 \
            --partition=${partition} \
            --gres=gpu:1 \
            --mem=32G \
            --export=ALL,LR=${lr},TRANSFORM=${transform},ROTATION=${rotation},EP=${eps} \
            run_${model}.sh
        done
      done
    done
  done
done