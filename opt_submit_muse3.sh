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
# python hyper_optim.py --trials=80 --trials_fine=0 --trials_train=3 --data_x=./data/muse_L2/train_input.txt --data_y=./data/muse_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-NNL-3_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/muse_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --min_pca=.99999 &> muse-NNL-3_L2.log &
# All-2
# python hyper_optim.py --k2r --trials=80 --trials_train=15 --data_x=./data/muse_L2/train_input.txt --data_y=./data/muse_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-All-2_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/muse_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --min_pca=.99999 &> muse-All-2_L2.log &
# HO-2 fine
python hyper_optim.py --fine_only --decay=2.4380034792160635e-07 --hidden_size=448 --num_layers=4 --trials=80 --trials_train=1 --data_x=./data/muse_L2/train_input.txt --data_y=./data/muse_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-HO-2_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/muse_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --min_pca=.99999 &> muse-HO-2_L2_fine.log &
# Mid one
# python hyper_optim.py --train_one --decay=2.4380034792160635e-07 --hidden_size=448 --num_layers=4 --trials_train=1 --data_x=./data/muse_L2/train_input.txt --data_y=./data/muse_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-Mid_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/muse_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --min_pca=.99999 &> muse-Mid_L2_one.log &


# L2H:
# Arch-1
# python hyper_optim.py --trials=80 --trials_fine=0 --data_x=./data/muse_L2H_pca_l/train_input.txt --data_y=./data/muse_L2H_pca_l/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-Arch-1_L2H --save_best --lr=0.01 --kfolds=21 --lgk=./data/muse_L2H_pca_l/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> muse-Arch-1_L2H.log &
# All-1
# python hyper_optim.py --trials=80 --trials_fine=0 --data_x=./data/muse_L2H_pca_g/train_input.txt --data_y=./data/muse_L2H_pca_g/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-All-1_L2H --save_best --lr=0.01 --kfolds=21 --lgk=./data/muse_L2H_pca_g/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> muse-All-1_L2H.log &
# pre N search space changed
# python hyper_optim.py --max_layers=8 --min_layers=2 --min_lambda=1e-8 --max_lambda=1e-5 --trials=80 --data_x=./data/muse_pre_N_L2Hr/train_input.txt --data_y=./data/muse_pre_N_L2Hr/train_output.txt --bound_x=./data/input_limits-N.txt --save_kfold --model_dir=models/muse_pre_N_L2Hr_hyper1 --save_best --lr=0.01 --kfolds=27 --lgk=./data/muse_pre_N_L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> muse_pre_N_L2Hr_hyper1.log &
# HO-2 (Mid)
# python hyper_optim.py --fine_only --hidden_size=16 --decay=2.959426e-08 --num_layers=7 --trials=80 --data_x=./data/muse_L2Hr/train_input.txt --data_y=./data/muse_L2Hr/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-HO-2_L2Hr --save_best --lr=0.01 --kfolds=21 --lgk=./data/muse_L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> muse-HO-2_L2Hr_fine.log &
sleep 100
nvidia-smi


wait
date

