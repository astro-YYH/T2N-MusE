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
# HO-2 (Mid)
python hyper_optim.py --trials=80 --trials_train=1 --data_x=./data/muse_L2/train_input.txt --data_y=./data/muse_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-HO-2_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/muse_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --min_pca=.99999 &> muse-HO-2_L2.log &
# HO-3
python hyper_optim.py --trials=120 --trials_fine=0 --trials_train=1 --data_x=./data/muse_L2/train_input.txt --data_y=./data/muse_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-HO-3_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/muse_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --min_pca=.99999 &> muse-HO-3_L2.log &
# combined
# python hyper_optim.py --trials=60 --data_x=./data/L2/train_input.npy --data_y=./data/L2/train_output.npy --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/L2 --save_best --lr=0.01 --kfolds=1128 --lgk=./data/L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524, 708, 709, 710, 732, 733, 734, 759, 760, 761, 768, 769, 770, 900, 901, 902" --k2r --min_pca=.99999 &> L2.log &
# only stage-2 search
# python hyper_optim.py --trials=60 --fine_only --hidden_size=224 --decay=1.1379857856909346e-08 --num_layers=6 --data_x=./data/L2/train_input.npy --data_y=./data/L2/train_output.npy --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/L2 --save_best --lr=0.01 --kfolds=1128 --lgk=./data/L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524, 708, 709, 710, 732, 733, 734, 759, 760, 761, 768, 769, 770, 900, 901, 902" --k2r --min_pca=.99999 &> L2_fine.log &

# L2Hr:
# python hyper_optim.py --trials=80 --data_x=./data/N_L2Hr_z0/train_input.txt --data_y=./data/N_L2Hr_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L2Hr_z0_pca --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L2Hr_z0/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> N_L2Hr_z0_pca.log &
# HO-2 (Mid)
python hyper_optim.py --trials=80 --data_x=./data/muse_L2Hr/train_input.txt --data_y=./data/muse_L2Hr/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-HO-2_L2Hr --save_best --lr=0.01 --kfolds=21 --lgk=./data/muse_L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> muse-HO-2_L2Hr.log &
# HO-3
python hyper_optim.py --trials=120 --trials_fine=0 --data_x=./data/muse_L2Hr/train_input.txt --data_y=./data/muse_L2Hr/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-HO-3_L2Hr --save_best --lr=0.01 --kfolds=21 --lgk=./data/muse_L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> muse-HO-3_L2Hr.log &
# python hyper_optim.py --trials=60 --data_x=./data/L2Hr/train_input.npy --data_y=./data/L2Hr/train_output.npy --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/L2Hr --save_best --lr=0.01 --kfolds=36 --lgk=./data/L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> L2Hr.log &

sleep 100
nvidia-smi


wait
date

