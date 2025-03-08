import os
import time
import argparse
import numpy as np
import torch
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from train_model import train_NN, train_model_kfold
import torch.nn as nn

if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser(description='Train a model with a given set of hyperparameters')

    parser.add_argument('--model_dir', type=str, default='./', help='Path to save the model')
    # set the number of folds
    parser.add_argument('--kfolds', type=int, default=0, help='Number of folds for K-Fold CV')
    # shuffle k-folds or not
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the data for K-Fold CV')
    # save k-fold models
    parser.add_argument('--save_kfold', action='store_true', help='Save the models for each fold')
    parser.add_argument('--model_name', type=str, default='best_model', help='Name of the model')
    parser.add_argument('--retrain', action='store_true', help='Retrain the model')
    # parser.add_argument('--save_model', action='store_true', help='Save the model')
    # learning rate
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    # input data
    parser.add_argument('--data_x', type=str, default=None, help='Path to the input data')
    parser.add_argument('--data_y', type=str, default=None, help='Path to the target data')
    # input bounds
    parser.add_argument('--bound_x', type=str, default=None, help='Bounds for the input data')
    # epochs
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    # epochs per neuron
    parser.add_argument('--epochs_neuron', type=int, default=200, help='Number of epochs per neuron')
    # activation function
    parser.add_argument('--activation', type=str, default='SiLU', help='Activation function')
    # number of hidden layers
    parser.add_argument('--num_layers', type=int, default=1, help='Number of hidden layers')
    # hidden size
    parser.add_argument('--hidden_size', type=int, default=16, help='Number of neurons per hidden layer')
    # decay
    parser.add_argument('--decay', type=float, default=1e-4, help='AdamW weight decay')
    # lgk file
    parser.add_argument('--lgk', type=str, default=None, help='Path to the lgk file')
    parser.add_argument('--zero_centering', action='store_true', help='Zero-center the output data')
    parser.add_argument('--standardize', action='store_true', help='Standardize the output data')



    args = parser.parse_args()

    # load the lgk file
    if args.lgk is not None:
        lgk = np.loadtxt(args.lgk)

    # activation function choices: 'ReLU', 'SiLU', 'Tanh', None
    act_dict = {'ReLU': nn.ReLU(), 'SiLU': nn.SiLU(), 'Tanh': nn.Tanh(), 'None': None}
    if args.activation not in act_dict:
        raise ValueError(f"Activation function should be one of {list(act_dict.keys())}")

    # Start the timer
    start_time = time.time()  # Track start time

    # mkidr if model_dir does not exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    model_path = os.path.join(args.model_dir, f"{args.model_name}.pth")

    # Define the choices explicitly
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    decay = args.decay
    # if retrain is True, retrain the model with the hyperparameters from the old model
    if args.retrain:
        # load the old model
        print(f"Retraining the model with the hyperparameters from the old model...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        hidden_size = checkpoint['hidden_size']
        num_layers = checkpoint['num_layers']
        decay = checkpoint['decay']
        # lgk
        lgk = checkpoint['lgk']
        # activation
        activation = checkpoint['activation']
        # and make a copy of the old model
        old_model_path = os.path.join(args.model_dir, f"{args.model_name}_old.pth")
        os.system(f"cp {model_path} {old_model_path}")

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # test
    # device = torch.device("cpu")
    print("Using device:", device)

    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    
    # Load the data
    x = np.loadtxt(args.data_x)
    y = np.loadtxt(args.data_y)
    # normalize the input data if bounds are provided
    if args.bound_x is not None:
        bounds = np.loadtxt(args.bound_x)
        # normalize the input data (only the first matched columns)
        dimx_original = bounds.shape[0]
        x[:,:dimx_original] = (x[:,:dimx_original] - bounds[:,0]) / (bounds[:,1] - bounds[:,0])

    # Convert to tensors on the CPU 
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Move tensors to the device
    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)

    # Create a trials object to store optimization history
    trials = Trials()

    print("\n🎯 Hyperparameters:")
    print('hidden_size', hidden_size, 'decay', decay, 'num_layers', num_layers)

    # if kfolds > 0, perform K-Fold CV
    if args.kfolds > 0:
        final_val_loss = train_model_kfold(num_layers, hidden_size, decay=decay, x_data=x_tensor, y_data=y_tensor, k=args.kfolds, save_kf_model=args.save_kfold, model_dir=args.model_dir, lr=args.lr, device=device, epochs=args.epochs, epochs_neuron=args.epochs_neuron, shuffle=args.shuffle, activation=act_dict[args.activation], zero_centering=args.zero_centering, lgk=lgk)

    # train and save the model with the best hyperparameters
    # Save the model if required
    # train the model on the full dataset

    epochs = args.epochs if args.epochs is not None else args.epochs_neuron * hidden_size * num_layers
    train_loss, _ = train_NN(num_layers, hidden_size, x_tensor, y_tensor, decay=decay, device=device, save_model=True, model_path=model_path, lr=args.lr, epochs=epochs, activation=act_dict[args.activation], lgk=lgk, zero_centering=args.zero_centering)

    # print(f"⏱ Elapsed time: {time.time() - start_time:.2f} seconds\n")
    elapsed_time = time.time() - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"⏱ Elapsed time: {formatted_time}\n")