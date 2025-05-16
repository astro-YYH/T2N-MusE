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
# PCA-1
# python hyper_optim.py --pca_allz --trials=80 --trials_fine=0 --trials_train=1 --data_x=./data/muse_L2/train_input.txt --data_y=./data/muse_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-PCA-1_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/muse_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --min_pca=.99999 &> muse-PCA-1_L2.log &
# NNL-2
# python hyper_optim.py --k2r --trials=80 --trials_fine=0 --trials_train=15 --data_x=./data/muse_L2/train_input.txt --data_y=./data/muse_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-NNL-2_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/muse_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --min_pca=.99999 &> muse-NNL-2_L2.log &
# NNL-2 one
# python hyper_optim.py --k2r --train_one --hidden_size=272 --decay=1.912504e-09 --num_layers=7 --trials_train=15 --data_x=./data/muse_L2/train_input.txt --data_y=./data/muse_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-NNL-2_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/muse_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --min_pca=.99999 &> muse-NNL-2_L2_one.log &
# pre
# python hyper_optim.py --trials=80 --trials_train=15 --data_x=./data/muse_pre_L2/train_input.txt --data_y=./data/muse_pre_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse_pre_L2 --save_best --lr=0.01 --kfolds=297 --lgk=./data/muse_pre_L2/kf.txt --zero_centering --test_folds="3, 4, 5, 24, 25, 26, 51, 52, 53, 60, 61, 62, 105, 106, 107, 156, 157, 158, 180, 181, 182, 255, 256, 257, 267, 268, 269" --k2r --min_pca=.99999 &> muse_pre_L2.log &
# pre one
python hyper_optim.py --train_one --retrain --trials_train=15 --data_x=./data/muse_pre_L2/train_input.txt --data_y=./data/muse_pre_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse_pre_L2 --save_best --lr=0.01 --kfolds=297 --lgk=./data/muse_pre_L2/kf.txt --zero_centering --test_folds="3, 4, 5, 24, 25, 26, 51, 52, 53, 60, 61, 62, 105, 106, 107, 156, 157, 158, 180, 181, 182, 255, 256, 257, 267, 268, 269" --k2r --min_pca=.99999 &> muse_pre_L2_one.log &

# L2Hr:
# PCA-1
# python hyper_optim.py --pca_allz --trials=80 --trials_fine=0 --data_x=./data/muse_L2Hr/train_input.txt --data_y=./data/muse_L2Hr/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-PCA-1_L2Hr --save_best --lr=0.01 --kfolds=21 --lgk=./data/muse_L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> muse-PCA-1_L2Hr.log &
# NNL-2
# python hyper_optim.py --trials=80 --trials_fine=0 --data_x=./data/muse_L2Hr/train_input.txt --data_y=./data/muse_L2Hr/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-NNL-2_L2Hr --save_best --lr=0.01 --kfolds=21 --lgk=./data/muse_L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> muse-NNL-2_L2Hr.log &
# pre
# python hyper_optim.py --trials=80 --data_x=./data/muse_pre_L2Hr/train_input.txt --data_y=./data/muse_pre_L2Hr/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse_pre_L2Hr --save_best --lr=0.01 --kfolds=27 --lgk=./data/muse_pre_L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> muse_pre_L2Hr.log &
# pre one
python hyper_optim.py --train_one --retrain --data_x=./data/muse_pre_L2Hr/train_input.txt --data_y=./data/muse_pre_L2Hr/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse_pre_L2Hr --save_best --lr=0.01 --kfolds=27 --lgk=./data/muse_pre_L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> muse_pre_L2Hr_one.log &
# NNL-2 one
# python hyper_optim.py --train_one --hidden_size=336 --decay=6.150068e-09 --num_layers=5 --data_x=./data/muse_L2Hr/train_input.txt --data_y=./data/muse_L2Hr/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/muse-NNL-2_L2Hr --save_best --lr=0.01 --kfolds=21 --lgk=./data/muse_L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> muse-NNL-2_L2Hr_one.log &

sleep 100
nvidia-smi


wait
date

