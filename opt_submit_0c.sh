#!/bin/bash
#SBATCH --job-name=hyper-0c
#SBATCH --time=0-24:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gh
#SBATCH -A AST21005
#SBATCH --output=%x-%j.out

source ~/.bashrc
conda activate pytorch-env
which python

hostname
date

# L1A:
python hyper_optim.py --trials=80 --data_x=./data/pre_N_L1A_z0/train_input.txt --data_y=./data/pre_N_L1A_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_L1A_z0_0c --save_best --lr=0.02 --kfolds=27 --lgk=./data/pre_N_L1A_z0/kf.txt --zero_centering &

# L1HA:
python hyper_optim.py --trials=80 --data_x=./data/pre_N_LHA_z0/train_input.txt --data_y=./data/pre_N_LHA_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_LHA_z0_0c --save_best --lr=0.02 --kfolds=27 --lgk=./data/pre_N_L1A_z0/kf.txt --zero_centering &

# L2:
python hyper_optim.py --trials=80 --data_x=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/train_input_fidelity_0.txt --data_y=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/train_output_fidelity_0.txt --bound_x=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_L2_z0_0c --save_best --lr=0.02 --kfolds=27 --lgk=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/kf.txt --zero_centering &

# LH:
python hyper_optim.py --trials=80 --data_x=./data/pre_N_xL-H_stitch_z0/train_input.txt --data_y=./data/pre_N_xL-H_stitch_z0/train_output.txt --bound_x=./data/pre_N_xL-H_stitch_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_xLH_stitch_z0_0c --save_best --lr=0.01 --kfolds=27 --lgk=./data/pre_N_L-H_stitch_z0/kf.txt --zero_centering & 

sleep 100 
nvidia-smi

wait
date

