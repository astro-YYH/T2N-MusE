#!/bin/bash
#SBATCH --job-name=hyper-dev
#SBATCH --time=0-2:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gh-dev
#SBATCH -A AST21005
#SBATCH --output=%x-%j.out

source ~/.bashrc
conda activate pytorch-env
which python

export PYTHONUNBUFFERED=1

hostname
date
# L2:
# # HO-3 one
# python hyper_optim.py --train_one --hidden_size=448 --decay=2.438003e-07 --num_layers=4 --trials_train=1 --data_x=./data/muse_L2/train_input.txt --data_y=./data/muse_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-HO-3_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/muse_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --min_pca=.99999 &> muse-HO-3_L2_one.log &
# one
python hyper_optim.py --train_one --trials_train=15 --hidden_size=464 --decay=2.3560447270270254e-07 --num_layers=6 --data_x=./data/L2/train_input.npy --data_y=./data/L2/train_output.npy --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/L2_1 --save_best --lr=0.01 --kfolds=1128 --lgk=./data/L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524, 708, 709, 710, 732, 733, 734, 759, 760, 761, 768, 769, 770, 900, 901, 902" --k2r --min_pca=.99999 &> L2_one.log &

sleep 100
nvidia-smi


wait
date

