import os
import time
import argparse
import numpy as np
import torch
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from train_model import train_NN, train_model_kfold
from mfbox import act_dict

# Function to evaluate a given set of hyperparameters
def objective(params):
    print(f"Testing with: {params}")
    
    # Train the model with K-Fold CV
    val_loss = train_model_kfold(params['num_layers'], params['hidden_size'], x_tensor, y_tensor, decay=params['decay'], k=args.kfolds, epochs=args.epochs, epochs_neuron=args.epochs_neuron, lr=args.lr, device=device, shuffle=args.shuffle, activation=activation)

    print(f"Validation Loss: {val_loss:.6f}\n")

    return {'loss': val_loss, 'status': STATUS_OK}

if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')
    parser.add_argument('--trials', type=int, default=200, help='Number of trials for optimization')
    parser.add_argument('--model_dir', type=str, default='./', help='Path to save the model')
    # set the number of folds
    parser.add_argument('--kfolds', type=int, default=5, help='Number of folds for K-Fold CV')
    # shuffle k-folds or not
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the data for K-Fold CV')
    # save k-fold models
    parser.add_argument('--save_kfold', action='store_true', help='Save the models for each fold')
    parser.add_argument('--save_best', action='store_true', help='Save the best model')
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
    parser.add_argument('--epochs_neuron', type=int, default=20, help='Number of epochs per neuron')
    # activation function
    parser.add_argument('--activation', type=str, default='SiLU', help='Activation function')
    # lgk file
    parser.add_argument('--lgk', type=str, default=None, help='Path to the lgk file')


    args = parser.parse_args()

    # load the lgk file
    if args.lgk is not None:
        lgk = np.loadtxt(args.lgk)

    activation = act_dict[args.activation]

    # Start the timer
    start_time = time.time()  # Track start time

    # mkidr if model_dir does not exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Define the choices explicitly
    hidden_size_choices = list(range(16, 513, 16))  # Generates [16, 32, 48, ..., 512]
    num_layers_choices = [1, 2, 3, 4, 5]
    # activation_choices = [nn.ReLU, nn.Tanh, nn.Sigmoid]
    decay_lower, decay_upper = 1e-6, 1e-2

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

    # Define the hyperparameter search space
    space = {
        'num_layers': hp.choice('num_layers', num_layers_choices),  # Number of hidden layers
        'hidden_size': hp.choice('hidden_size', hidden_size_choices),  # Neurons per layer
        'decay': hp.loguniform('decay', np.log(decay_lower), np.log(decay_upper))  # L2 regularization strength
    }

    # find the optimal batch size according to the largest model
    # model = SimpleNN(num_layers=np.max(num_layers_choices), hidden_size=np.max(num_layers_choices), dim_x=x_tensor.shape[1], dim_y=y_tensor.shape[1]).to(device)
    # best_batch = find_max_batch_size(model, TensorDataset(x_tensor, y_tensor), device=device)  # our dataset is small, so we can use full-batch training

    # Create a trials object to store optimization history
    trials = Trials()

    # Run Bayesian optimization
    best_hyperparams = fmin(
        fn=objective,        # Function to minimize (validation loss)
        space=space,         # Hyperparameter search space
        algo=tpe.suggest,    # Tree-structured Parzen Estimator (TPE)
        max_evals=args.trials,        # Number of trials to run
        trials=trials        # Store results
    )

    print("\n🎯 Best Hyperparameters Found:")
    print('hidden_size', hidden_size_choices[best_hyperparams['hidden_size']], 'decay', best_hyperparams['decay'], 'num_layers', num_layers_choices[best_hyperparams['num_layers']])

    # Evaluate the model with the best hyperparameters
    best_params = {'hidden_size': hidden_size_choices[best_hyperparams['hidden_size']], 'decay': best_hyperparams['decay'], 'num_layers': num_layers_choices[best_hyperparams['num_layers']]}
    final_val_loss = train_model_kfold(**best_params, x_data=x_tensor, y_data=y_tensor, k=args.kfolds, save_kf_model=args.save_kfold, model_dir=args.model_dir, lr=args.lr, device=device, epochs=args.epochs, epochs_neuron=args.epochs_neuron, shuffle=args.shuffle, activation=activation)

    # train and save the model with the best hyperparameters
    # Save the model if required
    # train the model on the full dataset

    model_path = os.path.join(args.model_dir, 'best_model.pth')

    print(f"Training the model on the full dataset with the best hyperparameters...")
    epochs = args.epochs if args.epochs is not None else args.epochs_neuron * best_params['hidden_size'] * best_params['num_layers']
    train_loss, _ = train_NN(best_params['num_layers'], best_params['hidden_size'], x_tensor, y_tensor, decay=best_params['decay'], device=device, save_model=args.save_best, model_path=model_path, lr=args.lr, epochs=epochs, activation=activation, lgk=lgk)

    # print(f"⏱ Elapsed time: {time.time() - start_time:.2f} seconds\n")
    elapsed_time = time.time() - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"⏱ Elapsed time: {formatted_time}\n")