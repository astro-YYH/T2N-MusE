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
# L1:
# python hyper_optim.py --trials=80 --data_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/train_input_fidelity_0.txt --data_y=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/train_output_fidelity_0.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L1_z0_pca --save_best --lr=0.01 --kfolds=564 --lgk=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/kf.txt --zero_centering --test_folds="144, 145, 146, 168, 169, 170, 195, 196, 197, 204, 205, 206, 336, 337, 338" --k2r --min_pca=.99999 &> N_L1_z0_pca.log &

# L1A:
python hyper_optim.py --trials=80 --data_x=./data/N_L1A_z0/train_input.txt --data_y=./data/N_L1A_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L1A_z0_pca --save_best --lr=0.01 --kfolds=564 --lgk=./data/N_L1A_z0/kf.txt --zero_centering --test_folds="144, 145, 146, 168, 169, 170, 195, 196, 197, 204, 205, 206, 336, 337, 338" --k2r --min_pca=.99999 &> N_L1A_z0_pca.log &

# L1HA:
# python hyper_optim.py --trials=80 --data_x=./data/N_LHA_z0/train_input.txt --data_y=./data/N_LHA_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_LHA_z0_0c --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L1A_z0/kf.txt --zero_centering --trials_train=5 &> N_LHA_z0_0c.log &
# L1HAr:
# python hyper_optim.py --trials=80 --data_x=./data/N_LHAr_z0/train_input.txt --data_y=./data/N_LHAr_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_LHAr_z0_pca --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L1A_z0/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> N_LHAr_z0_pca.log &
# L1Hr:
# python hyper_optim.py --trials=80 --data_x=./data/N_L1Hr_z0/train_input.txt --data_y=./data/N_L1Hr_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L1Hr_z0_pca --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L1Hr_z0/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> N_L1Hr_z0_pca.log &
# L1HAr:
python hyper_optim.py --trials=80 --data_x=./data/N_L1HAr_z0/train_input.txt --data_y=./data/N_L1HAr_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L1HAr_z0_pca --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L1A_z0/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> N_L1HAr_z0_pca.log &

# L2:
# python hyper_optim.py --trials=60 --data_x=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/train_input_fidelity_0.txt --data_y=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/train_output_fidelity_0.txt --bound_x=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L2_z0_0c --save_best --lr=0.01 --kfolds=30 --lgk=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/kf.txt --zero_centering &> N_L2_z0_0c.log &
# select folds
# python hyper_optim.py --trials=80 --data_x=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/train_input_fidelity_0.txt --data_y=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/train_output_fidelity_0.txt --bound_x=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L2_z0_pca --save_best --lr=0.01 --kfolds=564 --lgk=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/kf.txt --zero_centering --test_folds="144, 145, 146, 168, 169, 170, 195, 196, 197, 204, 205, 206, 336, 337, 338" --k2r --min_pca=.99999 &> N_L2_z0_pca.log &

# LHr:
# python hyper_optim.py --trials=80 --data_x=./data/N_LHr_stitch_z0/train_input.txt --data_y=./data/N_LHr_stitch_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_LHr_stitch_z0_pca --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_xLH_stitch_z0/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> N_LHr_stitch_z0_pca.log &
# L2Hr:
# python hyper_optim.py --trials=80 --data_x=./data/N_L2Hr_z0/train_input.txt --data_y=./data/N_L2Hr_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L2Hr_z0_pca --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L2Hr_z0/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> N_L2Hr_z0_pca.log &

sleep 100
nvidia-smi


wait
date

