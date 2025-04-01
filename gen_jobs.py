import os

# Redshift bins and PCA threshold
z_bins = list(range(1))  # Redshift indices: 0 to 33
min_pca = 0.999

# Format PCA string for log suffix, e.g., 0.99999 → p99999
pca_tag = f"p{str(min_pca).replace('.', '')}"

# Job script template
template = """#!/bin/bash
#SBATCH --job-name=z{i_z:02d}
#SBATCH --time=0-48:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gh
#SBATCH -A AST21005
#SBATCH --output=%x-%j.out

source ~/.bashrc
conda activate pytorch-env
which python

export PYTHONUNBUFFERED=1

echo "🔹 Running on: $(hostname)"
echo "🔹 Start time: $(date)"

i_z="{i_z}"

# Build suffix like _z00_p99999
suffix="_z{suffix}_{pca_tag}"

COMMON_ARGS="--trials=80 --save_kfold --save_best --lr=0.01 --zero_centering --min_pca={min_pca}"

TEST_FOLDS="144,145,146,168,169,170,195,196,197,204,205,206,336,337,338"

declare -A jobs
jobs=(
  ["N_L1A"]="--data_x=./data/N_L1A/train_input.txt --data_y=./data/N_L1A/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --lgk=./data/N_L1A/kf.txt --kfolds=564 --test_folds=$TEST_FOLDS --k2r"
  ["N_L1HAr"]="--data_x=./data/N_L1HAr/train_input.txt --data_y=./data/N_L1HAr/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --lgk=./data/N_L1HAr/kf.txt --kfolds=15 --trials_train=5"
  ["N_L2"]="--data_x=./data/N_L2/train_input.txt --data_y=./data/N_L2/train_output.txt --bound_x=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/input_limits.txt --lgk=./data/N_L2/kf.txt --kfolds=564 --test_folds=$TEST_FOLDS --k2r"
  ["N_L2Hr"]="--data_x=./data/N_L2Hr/train_input.txt --data_y=./data/N_L2Hr/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --lgk=./data/N_L2Hr/kf.txt --kfolds=15 --trials_train=5"
)

for model_name in "${{!jobs[@]}}"; do
  args="${{jobs[$model_name]}}"
  log_file="${{model_name}}${{suffix}}.log"
  model_dir="models/${{model_name}}${{suffix}}"

  echo "🔸 Launching training: $model_name → $log_file"
  python hyper_optim.py $COMMON_ARGS $args \\
    --model_dir="$model_dir" \\
    --i_z="$i_z" &> "$log_file" &
done

sleep 60
nvidia-smi

wait
echo "✅ All jobs finished at: $(date)"
"""

# Generate job scripts
for z in z_bins:
    suffix = f"{z:02d}"
    job_script = template.format(i_z=z, suffix=suffix, pca_tag=pca_tag, min_pca=min_pca)
    filename = f"./opt_submit_z{suffix}_{pca_tag}.sh"
    with open(filename, "w") as f:
        f.write(job_script)

print(f"✅ Generated {len(z_bins)} job scripts with min_pca={min_pca}")

