goku-pre:
L1:
python hyper_optim.py --trials=100 --data_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/train_input_fidelity_0.txt --data_y=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/train_output_fidelity_0.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_L1_z0 --save_best --lr=0.02 --kfolds=27
L1A:
python hyper_optim.py --trials=50 --data_x=./data/pre_N_L1A_z0/train_input.txt --data_y=./data/pre_N_L1A_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_L1A_z0 --save_best --lr=0.02 --kfolds=27
python hyper_optim.py --trials=80 --data_x=./data/pre_N_L1A_z0/train_input.txt --data_y=./data/pre_N_L1A_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_L1A_z0_0c --save_best --lr=0.02 --kfolds=27 --lgk=./data/pre_N_L1A_z0/kf.txt --zero_centering
python train_one.py --data_x=./data/pre_N_L1A_z0/train_input.txt --data_y=./data/pre_N_L1A_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --model_dir=models/pre_N_L1A_z0 --lr=0.01 --num_layers=2 --hidden_size=144 --decay=0.0001682765513440639 --model_name=best_model --lgk=./data/pre_N_L1A_z0/kf.txt 
python train_one.py --data_x=./data/pre_N_L1A_z0/train_input.txt --data_y=./data/pre_N_L1A_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --model_dir=models/pre_N_L1A_z0_0c --lr=0.01 --retrain --model_name=best_model --lgk=./data/pre_N_L1A_z0/kf.txt --zero_centering

L2:
python hyper_optim.py --trials=80 --data_x=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/train_input_fidelity_0.txt --data_y=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/train_output_fidelity_0.txt --bound_x=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_L2_z0 --save_best --lr=0.02 --kfolds=27 --lgk=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/kf.txt 
python hyper_optim.py --trials=80 --data_x=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/train_input_fidelity_0.txt --data_y=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/train_output_fidelity_0.txt --bound_x=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_L2_z0_0c --save_best --lr=0.02 --kfolds=27 --lgk=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/kf.txt --zero_centering
python train_one.py --data_x=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/train_input_fidelity_0.txt --data_y=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/train_output_fidelity_0.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --model_dir=models/pre_N_L2_z0 --lr=0.01 --model_name=best_model --retrain
python train_one.py --data_x=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/train_input_fidelity_0.txt --data_y=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/train_output_fidelity_0.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --model_dir=models/pre_N_L2_z0_0c --lr=0.01 --model_name=best_model --zero_centering --lgk=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/kf.txt --decay=3.699903060283192e-07 --hidden_size=176 --num_layers=4

L2B:
python hyper_optim.py --trials=50 --data_x=./data/pre_N_L2B_z0/train_input.txt --data_y=./data/pre_N_L2B_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_L2B_z0 --save_best --lr=0.02 --kfolds=27
python train_one.py --data_x=./data/pre_N_L2B_z0/train_input.txt --data_y=./data/pre_N_L2B_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --model_dir=models/pre_N_L2B_z0 --lr=0.01 --num_layers=3 --hidden_size=80 --decay=0.003877336150281618 --model_name=best_model --lgk=./data/pre_N_L2B_z0/kf.txt 

L1AL2B:
python hyper_optim.py --trials=50 --data_x=./data/pre_N_L1AL2B_z0/train_input.txt --data_y=./data/pre_N_L1AL2B_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_L1AL2B_z0 --save_best --lr=0.02 --kfolds=27 --lgk=./data/pre_N_L1AL2B_z0/kf.txt 
python train_one.py --data_x=./data/pre_N_L1AL2B_z0/train_input.txt --data_y=./data/pre_N_L1AL2B_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/input_limits.txt --model_dir=models/pre_N_L1AL2B_z0 --lr=0.01 --model_name=best_model --lgk=./data/pre_N_L1AL2B_z0/kf.txt --retrain

LF-HF
python hyper_optim.py --trials=100 --data_x=./data/pre_N_xL-H_z0/train_input.txt --data_y=./data/pre_N_xL-H_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_LH_z0 --save_best --lr=0.01 --kfolds=27
# xL-H cut-stitch
python hyper_optim.py --trials=60 --data_x=./data/pre_N_xL-H_stitch_z0/train_input.txt --data_y=./data/pre_N_xL-H_stitch_z0/train_output.txt --bound_x=./data/pre_N_xL-H_stitch_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_xLH_stitch_z0 --save_best --lr=0.01 --kfolds=27 --lgk=./data/pre_N_L-H_stitch_z0/kf.txt
python hyper_optim.py --trials=80 --data_x=./data/pre_N_xL-H_stitch_z0/train_input.txt --data_y=./data/pre_N_xL-H_stitch_z0/train_output.txt --bound_x=./data/pre_N_xL-H_stitch_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_xLH_stitch_z0_0c --save_best --lr=0.01 --kfolds=27 --lgk=./data/pre_N_L-H_stitch_z0/kf.txt --zero_centering
python train_one.py --data_x=./data/pre_N_xL-H_stitch_z0/train_input.txt --data_y=./data/pre_N_xL-H_stitch_z0/train_output.txt --bound_x=./data/pre_N_xL-H_stitch_z0/input_limits.txt --model_dir=models/pre_N_xLH_stitch_z0 --lr=0.01 --model_name=best_model --lgk=./data/pre_N_L-H_stitch_z0/kf.txt --retrain
python train_one.py --data_x=./data/pre_N_xL-H_stitch_z0/train_input.txt --data_y=./data/pre_N_xL-H_stitch_z0/train_output.txt --bound_x=./data/pre_N_xL-H_stitch_z0/input_limits.txt --model_dir=models/pre_N_xLH_stitch_z0_0c --lr=0.01 --model_name=best_model --lgk=./data/pre_N_L-H_stitch_z0/kf.txt --retrain --zero_centering
python train_one.py --data_x=./data/pre_N_xL-H_stitch_z0/train_input.txt --data_y=./data/pre_N_xL-H_stitch_z0/train_output.txt --bound_x=./data/pre_N_xL-H_stitch_z0/input_limits.txt --model_dir=models/pre_N_xLH_stitch_z0_0c --lr=0.01 --model_name=best_model --lgk=./data/pre_N_L-H_stitch_z0/kf.txt --hidden_size=128 --decay=1.6438530100725147e-05 --num_layers=2 --zero_centering

L1HA:
python hyper_optim.py --trials=80 --data_x=./data/pre_N_LHA_z0/train_input.txt --data_y=./data/pre_N_LHA_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_LHA_z0 --save_best --lr=0.02 --kfolds=27 --lgk=./data/pre_N_L1A_z0/kf.txt
python hyper_optim.py --trials=60 --data_x=./data/pre_N_LHA_z0/train_input.txt --data_y=./data/pre_N_LHA_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_LHA_z0_0c --save_best --lr=0.02 --kfolds=27 --lgk=./data/pre_N_L1A_z0/kf.txt --zero_centering
python hyper_optim_second.py --trials=40 --data_x=./data/pre_N_LHA_z0/train_input.txt --data_y=./data/pre_N_LHA_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_LHA_z0_0c --save_best --lr=0.02 --kfolds=27 --lgk=./data/pre_N_L1A_z0/kf.txt --zero_centering
python train_one.py --data_x=./data/pre_N_LHA_z0/train_input.txt --data_y=./data/pre_N_LHA_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --model_dir=models/pre_N_LHA_z0 --lr=0.01 --num_layers=2 --hidden_size=32 --decay=1.0016262573705154e-06 --model_name=best_model --lgk=./data/pre_N_L1A_z0/kf.txt 
python train_one.py --data_x=./data/pre_N_LHA_z0/train_input.txt --data_y=./data/pre_N_LHA_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --model_dir=models/pre_N_LHA_z0_0c --lr=0.01 --retrain --model_name=best_model --lgk=./data/pre_N_L1A_z0/kf.txt --zero_centering --kfolds=27
python train_one.py --data_x=./data/pre_N_LHA_z0/train_input.txt --data_y=./data/pre_N_LHA_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --model_dir=models/pre_N_LHA_z0_0c --lr=0.01 --hidden_size=496 --decay=3.09090682346284e-05 --num_layers=1 --model_name=best_model --lgk=./data/pre_N_L1A_z0/kf.txt --zero_centering --kfolds=27

linear:
python hyper_optim.py --trials=80 --data_x=./data/pre_N_LHA_z0/train_input.txt --data_y=./data/pre_N_LHA_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_LHA_3slin_z0 --save_best --lr=0.02 --kfolds=27 --activation=None --lgk=./data/pre_N_L1A_z0/kf.txt
python train_one.py --data_x=./data/pre_N_LHA_z0/train_input.txt --data_y=./data/pre_N_LHA_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --model_dir=models/pre_N_LHA_3slin_z0 --lr=0.01 --num_layers=1 --hidden_size=336 --decay=1.738581966057867e-06 --model_name=best_model --lgk=./data/pre_N_L1A_z0/kf.txt --activation=None
nonlinear:
python hyper_optim.py --trials=80 --data_x=./data/pre_N_LHA_3snonlin_z0/train_input.txt --data_y=./data/pre_N_LHA_3snonlin_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_LHA_3snonlin_z0 --save_best --lr=0.02 --kfolds=27 --lgk=./data/pre_N_L1A_z0/kf.txt

L2HB:
python hyper_optim.py --trials=50 --data_x=./data/pre_N_LHB_z0/train_input.txt --data_y=./data/pre_N_LHB_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_LHB_z0 --save_best --lr=0.02 --kfolds=27
python train_one.py --data_x=./data/pre_N_LHB_z0/train_input.txt --data_y=./data/pre_N_LHB_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --model_dir=models/pre_N_LHB_z0 --lr=0.01 --num_layers=1 --hidden_size=304 --decay=0.00016120080578261664 --model_name=best_model --lgk=./data/pre_N_L2B_z0/kf.txt 
python train_one.py --data_x=./data/pre_N_LHB_z0/train_input.txt --data_y=./data/pre_N_LHB_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --model_dir=models/pre_N_LHB_z0 --lr=0.01 --model_name=best_model --lgk=./data/pre_N_L2B_z0/kf.txt --retrain

L2H:
python hyper_optim.py --trials=50 --data_x=./data/pre_N_xL2-H_z0/train_input.txt --data_y=./data/pre_N_xL2-H_z0/train_output.txt --bound_x=./data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_L2H_z0 --save_best --lr=0.02 --kfolds=27 --lgk=./data/pre_N_L-H_stitch_z0/kf.txt

# 3-step
linear:
python hyper_optim.py --trials=50 --data_x=./data/pre_N_xL-H_stitch_z0/train_input.txt --data_y=./data/pre_N_xL-H_stitch_z0/train_output.txt --bound_x=./data/pre_N_xL-H_stitch_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_xLH_stitch_3slin_z0 --save_best --lr=0.01 --kfolds=27 --activation=None --lgk=./data/pre_N_xL-H_stitch_z0/kf.txt # linear
python train_one.py --data_x=./data/pre_N_xL-H_stitch_z0/train_input.txt --data_y=./data/pre_N_xL-H_stitch_z0/train_output.txt --bound_x=./data/pre_N_xL-H_stitch_z0/input_limits.txt --model_dir=models/pre_N_xLH_stitch_3slin_z0 --lr=0.01 --num_layers=1 --hidden_size=512 --decay=2.1157746193730135e-05 --model_name=best_model --lgk=./data/pre_N_xL-H_stitch_z0/kf.txt --activation=None
nonlinear:
python hyper_optim.py --trials=50 --data_x=./data/pre_N_xL-H_stitch_3snonlin_z0/train_input.txt --data_y=./data/pre_N_xL-H_stitch_3snonlin_z0/train_output.txt --bound_x=./data/pre_N_xL-H_stitch_z0/input_limits.txt --save_kfold --model_dir=models/pre_N_xLH_stitch_3snonlin_z0 --save_best --lr=0.01 --kfolds=27 --lgk=./data/pre_N_L-H_stitch_z0/kf.txt # nonlinear
python train_one.py --data_x=./data/pre_N_xL-H_stitch_3snonlin_z0/train_input.txt --data_y=./data/pre_N_xL-H_stitch_3snonlin_z0/train_output.txt --model_dir=models/pre_N_xLH_stitch_3snonlin_z0 --lr=0.01 --num_layers=1 --hidden_size=384 --decay=1.5447005920068177e-05 --model_name=best_model --lgk=./data/pre_N_L-H_stitch_z0/kf.txt


# L-H cut-stitch
python hyper_optim.py --trials=100 --data_x=./data/pre_N_L-H_stitch_z0/train_input.txt --data_y=./data/pre_N_L-H_stitch_z0/train_output.txt --save_kfold --model_dir=models/pre_N_LH_stitch_z0 --save_best --lr=0.01 --kfolds=27 --lgk=./data/pre_N_L-H_stitch_z0/kf.txt


# restart NVIDIA driver
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm

# monitor GPU:
watch -n 1 nvidia-smi