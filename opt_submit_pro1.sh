#!/bin/bash
#SBATCH --job-name=hyper-pro
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

hostname
date

# production sims

# L1A:
python hyper_optim.py --trials=80 --data_x=./data/N_L1A/train_input.txt --data_y=./data/N_L1A/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L1A_p0999 --save_best --lr=0.01 --kfolds=564 --lgk=./data/N_L1A/kf.txt --zero_centering --test_folds="144, 145, 146, 168, 169, 170, 195, 196, 197, 204, 205, 206, 336, 337, 338" --k2r --min_pca=.999 &> N_L1A_p0999.log &

# L1HAr:
python hyper_optim.py --trials=80 --data_x=./data/N_L1HAr/train_input.txt --data_y=./data/N_L1HAr/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L1HAr_p0999 --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L1HAr/kf.txt --zero_centering --trials_train=5 --min_pca=.999 &> N_L1HAr_p0999.log &
# L2:
python hyper_optim.py --trials=80 --data_x=./data/N_L2/train_input.txt --data_y=./data/N_L2/train_output.txt --bound_x=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L2_p0999 --save_best --lr=0.01 --kfolds=564 --lgk=./data/N_L2/kf.txt --zero_centering --test_folds="144, 145, 146, 168, 169, 170, 195, 196, 197, 204, 205, 206, 336, 337, 338" --k2r --min_pca=.999 &> N_L2_p0999.log &

# LHr:
python hyper_optim.py --trials=80 --data_x=./data/N_L2Hr/train_input.txt --data_y=./data/N_L2Hr/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L2Hr_p0999 --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.999 &> N_L2Hr_p0999.log &

sleep 100
nvidia-smi


wait
date

