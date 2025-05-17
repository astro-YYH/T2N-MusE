#!/bin/bash
#SBATCH --job-name=hyper-muse
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

# L2:
# NNL-3
python hyper_optim.py --trials=80 --trials_fine=0 --trials_train=3 --data_x=./data/muse_L2/train_input.txt --data_y=./data/muse_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-NNL-3_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/muse_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --min_pca=.99999 &> muse-NNL-3_L2.log &
# All-2
python hyper_optim.py --k2r --hidden_size=272 --decay=1.912504e-09 --num_layers=7 --fine_only --trials_fine=40 --trials_train=15 --data_x=./data/muse_L2/train_input.txt --data_y=./data/muse_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-All-2_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/muse_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --min_pca=.99999 &> muse-All-2_L2_fine.log &

# L2H:
# Arch-1
# python hyper_optim.py --trials=80 --trials_fine=0 --data_x=./data/muse_L2H_pca_l/train_input.txt --data_y=./data/muse_L2H_pca_l/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-Arch-1_L2H --save_best --lr=0.01 --kfolds=21 --lgk=./data/muse_L2H_pca_l/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> muse-Arch-1_L2H.log &
# All-1
# python hyper_optim.py --trials=80 --trials_fine=0 --data_x=./data/muse_L2H_pca_g/train_input.txt --data_y=./data/muse_L2H_pca_g/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-All-1_L2H --save_best --lr=0.01 --kfolds=21 --lgk=./data/muse_L2H_pca_g/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> muse-All-1_L2H.log &

sleep 100
nvidia-smi


wait
date

