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
# HO-3 one
python hyper_optim.py --trian_one --hidden_size=448 --decay=2.438003e-07 --num_layers=4 --trials_train=1 --data_x=./data/muse_L2/train_input.txt --data_y=./data/muse_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-HO-3_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/muse_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --min_pca=.99999 &> muse-HO-3_L2_one.log &
# PCA-1 one
python hyper_optim.py --trian_one --hidden_size=176 --decay=4.676687e-08 --num_layers=2 --pca_allz --trials_train=1 --data_x=./data/muse_L2/train_input.txt --data_y=./data/muse_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-PCA-1_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/muse_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --min_pca=.99999 &> muse-PCA-1_L2_one.log &

# L2Hr:
# HO-3 one
python hyper_optim.py

# PCA-1 one
python hyper_optim.py --trian_one --hidden_size=208 --decay=8.848311e-08 --num_layers=7 --pca_allz --data_x=./data/muse_L2Hr/train_input.txt --data_y=./data/muse_L2Hr/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-PCA-1_L2Hr --save_best --lr=0.01 --kfolds=21 --lgk=./data/muse_L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> muse-PCA-1_L2Hr_one.log &

sleep 100
nvidia-smi


wait
date

