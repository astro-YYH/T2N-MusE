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

# cp -r models/W_L1A models/W_L1A_add
# cp -r models/W_L1HAr models/W_L1HAr_add
# cp -r models/W_L2 models/W_L2_add
# cp -r models/W_L2Hr models/W_L2Hr_add

# production sims
# L1:
# python hyper_optim.py --trials=80 --data_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/train_input_fidelity_0.txt --data_y=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/train_output_fidelity_0.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L1_z0_pca --save_best --lr=0.01 --kfolds=564 --lgk=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/kf.txt --zero_centering --test_folds="144, 145, 146, 168, 169, 170, 195, 196, 197, 204, 205, 206, 336, 337, 338" --k2r --min_pca=.99999 &> N_L1_z0_pca.log &

# L1A:
# python hyper_optim.py --trials=80 --data_x=./data/N_L1A_z0/train_input.txt --data_y=./data/N_L1A_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L1A_z0_pca --save_best --lr=0.01 --kfolds=564 --lgk=./data/N_L1A_z0/kf.txt --zero_centering --test_folds="144, 145, 146, 168, 169, 170, 195, 196, 197, 204, 205, 206, 336, 337, 338" --k2r --min_pca=.99999 &> N_L1A_z0_pca.log &
# python hyper_optim.py --trials=80 --trials_train=15 --data_x=./data/N_L1A/train_input.txt --data_y=./data/N_L1A/train_output.txt --bound_x=./data/input_limits-N.txt --save_kfold --model_dir=models/N_L1A --save_best --lr=0.01 --kfolds=564 --lgk=./data/N_L1A/kf.txt --zero_centering --test_folds="144, 145, 146, 168, 169, 170, 195, 196, 197, 204, 205, 206, 336, 337, 338" --k2r --min_pca=.99999 &> N_L1A.log &
# cp -r models/N_L1A models/N_L1A_add
# python hyper_optim.py --train_one --retrain --trials_train=300 --data_x=./data/N_L1A/train_input.txt --data_y=./data/N_L1A/train_output.txt --bound_x=./data/input_limits-N.txt --save_kfold --model_dir=models/N_L1A_add --save_best --lr=0.01 --kfolds=564 --lgk=./data/N_L1A/kf.txt --zero_centering --test_folds="144, 145, 146, 168, 169, 170, 195, 196, 197, 204, 205, 206, 336, 337, 338" --k2r --min_pca=.99999 &> N_L1A_add.log &
# # cut at 1.5
# python hyper_optim.py --trials=80 --data_x=./data/N_L1A_c1.5/train_input.txt --data_y=./data/N_L1A_c1.5/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L1A_c1.5 --save_best --lr=0.01 --kfolds=564 --lgk=./data/N_L1A_c1.5/kf.txt --zero_centering --test_folds="144, 145, 146, 168, 169, 170, 195, 196, 197, 204, 205, 206, 336, 337, 338" --k2r --min_pca=.99999 &> N_L1A_c1.5.log &
# wide
# python hyper_optim.py --trials=80 --trials_train=15 --data_x=./data/W_L1A/train_input.txt --data_y=./data/W_L1A/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/W_L1A --save_best --lr=0.01 --kfolds=564 --lgk=./data/W_L1A/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --k2r --min_pca=.99999 &> W_L1A.log &
# python hyper_optim.py --train_one --retrain --trials_train=300 --data_x=./data/W_L1A/train_input.txt --data_y=./data/W_L1A/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/W_L1A_add --save_best --lr=0.01 --kfolds=564 --lgk=./data/W_L1A/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --k2r --min_pca=.99999 &> W_L1A_add.log &

# combined
# python hyper_optim.py --trials=60 --data_x=./data/L1A/train_input.npy --data_y=./data/L1A/train_output.npy --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/L1A --save_best --lr=0.01 --kfolds=1128 --lgk=./data/L1A/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524, 708, 709, 710, 732, 733, 734, 759, 760, 761, 768, 769, 770, 900, 901, 902" --k2r --min_pca=.99999 &> L1A.log &
#only stage-2 search
# python hyper_optim.py --trials=60 --fine_only --hidden_size=160 --decay=1.542976e-09 --num_layers=2 --data_x=./data/L1A/train_input.npy --data_y=./data/L1A/train_output.npy --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/L1A --save_best --lr=0.01 --kfolds=1128 --lgk=./data/L1A/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524, 708, 709, 710, 732, 733, 734, 759, 760, 761, 768, 769, 770, 900, 901, 902" --k2r --min_pca=.99999 &> L1A_fine.log &
# one
python hyper_optim.py --train_one --retrain --trials_train=15 --data_x=./data/L1A/train_input.npy --data_y=./data/L1A/train_output.npy --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/L1A --save_best --lr=0.01 --kfolds=1128 --lgk=./data/L1A/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524, 708, 709, 710, 732, 733, 734, 759, 760, 761, 768, 769, 770, 900, 901" --k2r --min_pca=.99999 &> L1A_one.log &

# L1HA:
# python hyper_optim.py --trials=80 --data_x=./data/N_LHA_z0/train_input.txt --data_y=./data/N_LHA_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_LHA_z0_0c --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L1A_z0/kf.txt --zero_centering --trials_train=5 &> N_LHA_z0_0c.log &
# L1HAr:
# python hyper_optim.py --trials=80 --data_x=./data/N_LHAr_z0/train_input.txt --data_y=./data/N_LHAr_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_LHAr_z0_pca --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L1A_z0/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> N_LHAr_z0_pca.log &
# L1Hr:
# python hyper_optim.py --trials=80 --data_x=./data/N_L1Hr_z0/train_input.txt --data_y=./data/N_L1Hr_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L1Hr_z0_pca --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L1Hr_z0/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> N_L1Hr_z0_pca.log &
# L1HAr:
# python hyper_optim.py --trials=80 --data_x=./data/N_L1HAr_z0/train_input.txt --data_y=./data/N_L1HAr_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L1HAr_z0_pca --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L1A_z0/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> N_L1HAr_z0_pca.log &
# python hyper_optim.py --trials=80 --data_x=./data/N_L1HAr/train_input.txt --data_y=./data/N_L1HAr/train_output.txt --bound_x=./data/input_limits-N.txt --save_kfold --model_dir=models/N_L1HAr --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L1HAr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> N_L1HAr.log &
# cp -r models/N_L1HAr models/N_L1HAr_add
# python hyper_optim.py --train_one --retrain --data_x=./data/N_L1HAr/train_input.txt --data_y=./data/N_L1HAr/train_output.txt --bound_x=./data/input_limits-N.txt --save_kfold --model_dir=models/N_L1HAr_add --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L1HAr/kf.txt --zero_centering --trials_train=100 --min_pca=.99999 &> N_L1HAr_add.log &
# # cut at 1.5
# python hyper_optim.py --trials=80 --data_x=./data/N_L1HAr_c1.5/train_input.txt --data_y=./data/N_L1HAr_c1.5/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L1HAr_c1.5 --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L1HAr_c1.5/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> N_L1HAr_c1.5.log &
# wide
# python hyper_optim.py --trials=80 --data_x=./data/W_L1HAr/train_input.txt --data_y=./data/W_L1HAr/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/W_L1HAr --save_best --lr=0.01 --kfolds=21 --lgk=./data/W_L1HAr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> W_L1HAr.log &
# python hyper_optim.py --train_one --retrain --data_x=./data/W_L1HAr/train_input.txt --data_y=./data/W_L1HAr/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/W_L1HAr_add --save_best --lr=0.01 --kfolds=21 --lgk=./data/W_L1HAr/kf.txt --zero_centering --trials_train=100 --min_pca=.99999 &> W_L1HAr_add.log &
# combined
# python hyper_optim.py --trials=60 --trials_train=5 --data_x=./data/L1HAr/train_input.npy --data_y=./data/L1HAr/train_output.npy --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/L1HAr --save_best --lr=0.01 --kfolds=36 --lgk=./data/L1HAr/kf.txt --zero_centering --min_pca=.99999 &> L1HAr.log &
# one
python hyper_optim.py --train_one --retrain --data_x=./data/L1HAr/train_input.npy --data_y=./data/L1HAr/train_output.npy --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/L1HAr --save_best --lr=0.01 --kfolds=36 --lgk=./data/L1HAr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> L1HAr_one.log &

# L2:
# python hyper_optim.py --trials=60 --data_x=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/train_input_fidelity_0.txt --data_y=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/train_output_fidelity_0.txt --bound_x=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L2_z0_0c --save_best --lr=0.01 --kfolds=30 --lgk=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/kf.txt --zero_centering &> N_L2_z0_0c.log &
# select folds
# python hyper_optim.py --trials=80 --data_x=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/train_input_fidelity_0.txt --data_y=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/train_output_fidelity_0.txt --bound_x=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L2_z0_pca --save_best --lr=0.01 --kfolds=564 --lgk=./data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000_z0/kf.txt --zero_centering --test_folds="144, 145, 146, 168, 169, 170, 195, 196, 197, 204, 205, 206, 336, 337, 338" --k2r --min_pca=.99999 &> N_L2_z0_pca.log &
# python hyper_optim.py --trials=80 --trials_train=15 --data_x=./data/N_L2/train_input.txt --data_y=./data/N_L2/train_output.txt --bound_x=./data/input_limits-N.txt --save_kfold --model_dir=models/N_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/N_L2/kf.txt --zero_centering --test_folds="144, 145, 146, 168, 169, 170, 195, 196, 197, 204, 205, 206, 336, 337, 338" --k2r --min_pca=.99999 &> N_L2.log &
# N fine
python hyper_optim.py --k2r --hidden_size=304 --decay=3.112687e-08 --num_layers=5 --fine_only --trials_fine=40 --trials_train=15 --data_x=./data/N_L2/train_input.txt --data_y=./data/N_L2/train_output.txt --bound_x=./data/input_limits-N.txt --save_kfold --model_dir=models/N_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/N_L2/kf.txt --zero_centering --test_folds="144, 145, 146, 168, 169, 170, 195, 196, 197, 204, 205, 206, 336, 337, 338" --min_pca=.99999 &> N_L2_fine.log &

# wide
# python hyper_optim.py --trials=80 --trials_train=15 --data_x=./data/W_L2/train_input.txt --data_y=./data/W_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/W_L2 --save_best --lr=0.01 --kfolds=564 --lgk=./data/W_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --k2r --min_pca=.99999 &> W_L2.log &
# python hyper_optim.py --train_one --retrain --trials_train=300 --data_x=./data/W_L2/train_input.txt --data_y=./data/W_L2/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/W_L2_add --save_best --lr=0.01 --kfolds=564 --lgk=./data/W_L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" --k2r --min_pca=.99999 &> W_L2_add.log &

# combined
# python hyper_optim.py --trials=60 --data_x=./data/L2/train_input.npy --data_y=./data/L2/train_output.npy --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/L2 --save_best --lr=0.01 --kfolds=1128 --lgk=./data/L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524, 708, 709, 710, 732, 733, 734, 759, 760, 761, 768, 769, 770, 900, 901, 902" --k2r --min_pca=.99999 &> L2.log &
# only stage-2 search
# python hyper_optim.py --trials=60 --fine_only --trials_train=15 --hidden_size=464 --decay=2.356045e-07 --num_layers=6 --data_x=./data/L2/train_input.npy --data_y=./data/L2/train_output.npy --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/L2 --save_best --lr=0.01 --kfolds=1128 --lgk=./data/L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524, 708, 709, 710, 732, 733, 734, 759, 760, 761, 768, 769, 770, 900, 901, 902" --k2r --min_pca=.99999 &> L2_fine.log &
# one
python hyper_optim.py --train_one --hidden_size=464 --decay=2.356045e-07 --num_layers=6 --trials_train=15 --data_x=./data/L2/train_input.npy --data_y=./data/L2/train_output.npy --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/L2 --save_best --lr=0.01 --kfolds=1128 --lgk=./data/L2/kf.txt --zero_centering --test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524, 708, 709, 710, 732, 733, 734, 759, 760, 761, 768, 769, 770, 900, 901" --k2r --min_pca=.99999 &> L2_one.log &

# LHr:
# python hyper_optim.py --trials=80 --data_x=./data/N_LHr_stitch_z0/train_input.txt --data_y=./data/N_LHr_stitch_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_LHr_stitch_z0_pca --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_xLH_stitch_z0/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> N_LHr_stitch_z0_pca.log &
# L2Hr:
# python hyper_optim.py --trials=80 --data_x=./data/N_L2Hr_z0/train_input.txt --data_y=./data/N_L2Hr_z0/train_output.txt --bound_x=./data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt --save_kfold --model_dir=models/N_L2Hr_z0_pca --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L2Hr_z0/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> N_L2Hr_z0_pca.log &
# python hyper_optim.py --trials=80 --data_x=./data/N_L2Hr/train_input.txt --data_y=./data/N_L2Hr/train_output.txt --bound_x=./data/input_limits-N.txt --save_kfold --model_dir=models/N_L2Hr --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> N_L2Hr.log &
# cp -r models/N_L2Hr models/N_L2Hr_add
# python hyper_optim.py --train_one --retrain --data_x=./data/N_L2Hr/train_input.txt --data_y=./data/N_L2Hr/train_output.txt --bound_x=./data/input_limits-N.txt --save_kfold --model_dir=models/N_L2Hr_add --save_best --lr=0.01 --kfolds=15 --lgk=./data/N_L2Hr/kf.txt --zero_centering --trials_train=100 --min_pca=.99999 &> N_L2Hr_add.log &
# wide
# python hyper_optim.py --trials=80 --data_x=./data/W_L2Hr/train_input.txt --data_y=./data/W_L2Hr/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/W_L2Hr --save_best --lr=0.01 --kfolds=21 --lgk=./data/W_L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> W_L2Hr.log &
# python hyper_optim.py --train_one --retrain --data_x=./data/W_L2Hr/train_input.txt --data_y=./data/W_L2Hr/train_output.txt --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/W_L2Hr_add --save_best --lr=0.01 --kfolds=21 --lgk=./data/W_L2Hr/kf.txt --zero_centering --trials_train=100 --min_pca=.99999 &> W_L2Hr_add.log &

# combined
# python hyper_optim.py --trials=60 --data_x=./data/L2Hr/train_input.npy --data_y=./data/L2Hr/train_output.npy --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/L2Hr --save_best --lr=0.01 --kfolds=36 --lgk=./data/L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> L2Hr.log &
# one
python hyper_optim.py --train_one --retrain --data_x=./data/L2Hr/train_input.npy --data_y=./data/L2Hr/train_output.npy --bound_x=./data/input_limits-W.txt --save_kfold --model_dir=models/L2Hr --save_best --lr=0.01 --kfolds=36 --lgk=./data/L2Hr/kf.txt --zero_centering --trials_train=5 --min_pca=.99999 &> L2Hr_one.log &

sleep 100
nvidia-smi


wait
date

