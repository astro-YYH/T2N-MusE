#!/bin/bash
#SBATCH --job-name=z04
#SBATCH --time=0-48:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gh
#SBATCH -A AST21005
#SBATCH --output=%x-%j.out

set -euo pipefail

source ~/.bashrc
conda activate pytorch-env
which python

export PYTHONUNBUFFERED=1

echo "🔹 Running on: $(hostname)"
echo "🔹 Start time: $(date)"

i_z="4"

# Build suffix like _z00
suffix="_z04"

COMMON_ARGS="--trials=80 --save_kfold --save_best --lr=0.01 --zero_centering --min_pca=.99999"

TEST_FOLDS="144,145,146,168,169,170,195,196,197,204,205,206,336,337,338"

declare -A jobs
jobs=(
  ["N_L1A"]="--data_x=./data/N_L1A/train_input.txt --data_y=./data/N_L1A/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --lgk=./data/N_L1A/kf.txt --kfolds=564 --test_folds=$TEST_FOLDS --k2r"
  ["N_L1HAr"]="--data_x=./data/N_L1HAr/train_input.txt --data_y=./data/N_L1HAr/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --lgk=./data/N_L1HAr/kf.txt --kfolds=15 --trials_train=5"
  ["N_L2"]="--data_x=./data/N_L2/train_input.txt --data_y=./data/N_L2/train_output.txt --bound_x=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/input_limits.txt --lgk=./data/N_L2/kf.txt --kfolds=564 --test_folds=$TEST_FOLDS --k2r"
  ["N_L2Hr"]="--data_x=./data/N_L2Hr/train_input.txt --data_y=./data/N_L2Hr/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --lgk=./data/N_L2Hr/kf.txt --kfolds=15 --trials_train=5"
)

for model_name in "${!jobs[@]}"; do
  args="${jobs[$model_name]}"
  log_file="${model_name}${suffix}.log"
  model_dir="models/${model_name}${suffix}"

  echo "🔸 Launching training: $model_name → $log_file"
  python hyper_optim.py $COMMON_ARGS $args \
    --model_dir="$model_dir" \
    --i_z="$i_z" &> "$log_file" &
done

sleep 60
nvidia-smi

wait
echo "✅ All jobs finished at: $(date)"
