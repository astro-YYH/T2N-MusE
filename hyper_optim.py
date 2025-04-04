import os
import time
import argparse
import numpy as np
import torch
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from train_model import train_NN, train_model_kfold, train_model_kfold_2r
from mfbox import act_dict
# import tqdm
import sys
from sklearn.decomposition import PCA

# Automatically disable tqdm progress bar if not running interactively
show_progress = sys.stdout.isatty()  # True if running interactively, False if output is redirected
# tqdm.tqdm.disable = not show_progress  # Disable if not interactive

# Function to evaluate a given set of hyperparameters
def objective(params):
    # Determine which trials object is active
    if len(trials_fine.trials) > 0:  # If fine-tuning has started, count from trials_fine
        trial_number = len(trials_fine.trials)
        round_name = "Fine-Tuning"
        trials_max = trials_fine_max
        best_loss = min([trial['result']['loss'] for trial in trials_fine.trials[:-1]]) if len(trials_fine.trials) > 1 else float('inf')
    else:
        trial_number = len(trials.trials)
        round_name = "Initial Search"
        trials_max = args.trials
        best_loss = min([trial['result']['loss'] for trial in trials.trials[:-1]]) if len(trials.trials) > 1 else float('inf')

    print(f"\n🔹 {round_name} | Trial {trial_number}/{trials_max} | Best loss {best_loss} | Testing with: {params}")
    
    # Train the model with K-Fold CV
    train_loss, val_loss, _, _ = train_kfold(params['num_layers'], params['hidden_size'], x_tensor, y_tensor, decay=params['decay'], k=args.kfolds, epochs=args.epochs, epochs_neuron=args.epochs_neuron, lr=args.lr, device=device, shuffle=args.shuffle, activation=activation, zero_centering=args.zero_centering, test_folds=test_folds, num_trials=args.trials_train, mean_std=mean_std)

    # optimize the average of training loss and validation loss

    tv_loss = (train_loss + val_loss) / 2
    # print(f"Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, sum: {sum_loss:.6f}\n")

    if args.opt_val:
        return {'loss': val_loss, 'status': STATUS_OK}
    else:
        return {'loss': tv_loss, 'status': STATUS_OK}

def pca_decomp(y, explained_min=0.999, standardize=False):
        pca = PCA()
        pca.fit(y)
        # Explained variance ratio for each PC
        explained_variance = pca.explained_variance_ratio_

        # Cumulative explained variance
        cumulative_variance = np.cumsum(explained_variance)

        # Find the number of PCs that explain args.min_pca of the variance
        n_components = np.argmax(cumulative_variance > explained_min) + 1

        # Transform the data, only keep the first n_components
        y = pca.transform(y)[:, :n_components]

        # standardize the output data (PCA coefficients) (not recommended)
        if standardize:
            y_mean = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            y = (y - y_mean) / y_std
        else:
            y_mean = None
            y_std = None

        # save the PCA components
        mean_std = {'pca_components': pca.components_[:n_components], 'pca_mean': pca.mean_, 'mean': y_mean, 'std': y_std}

        # print the number of PCs and the explained variance
        print(f"Number of PCs: {n_components}")
        print(f"Explained variance: {cumulative_variance[n_components - 1]}")

        return y, mean_std

if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials for optimization')
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
    parser.add_argument('--epochs_neuron', type=int, default=1000, help='Number of epochs per neuron')
    # activation function
    parser.add_argument('--activation', type=str, default='SiLU', help='Activation function')
    # lgk file
    parser.add_argument('--lgk', type=str, default=None, help='Path to the lgk file')
    parser.add_argument('--zero_centering', action='store_true', help='Zero-center the output data')
    # optimize training + validation loss
    parser.add_argument('--opt_val', action='store_true', help='Optimize validation loss')  # if False, optimize training loss + validation loss
    parser.add_argument('--test_folds', type=str, default=None, help='Comma-separated list of fold indices to test (e.g., "0,2,4")')
    parser.add_argument('--trials_train', type=int, default=1, help='Number of trials per k-fold training')
    parser.add_argument('--k2r', action='store_true', help='Use the 2-round k-fold training')
    parser.add_argument('--min_pca', type=float, default=1, help='Minimum explained variance ratio of the PCA')
    parser.add_argument('--standardize', action='store_true', help='Standardize the output data')
    parser.add_argument('--i_z', type=str, default=None, help='Comma-separated list of indices for the z bins we want to train (e.g., "0,1,2")')

    # train one model with a set of hyperparameters (not used if we are doing hyperparameter optimization)
    parser.add_argument('--num_layers', type=int, default=None, help='Number of hidden layers')
    # hidden size
    parser.add_argument('--hidden_size', type=int, default=None, help='Number of neurons per hidden layer')
    # decay
    parser.add_argument('--decay', type=float, default=None, help='AdamW weight decay')
    parser.add_argument('--retrain', action='store_true', help='Retrain the model')  # ignore provided hyperparameters
    parser.add_argument('--train_one', action='store_true', help='train the model with a given set of hyperparameters')

    parser.add_argument('--pca_allz', action='store_true', help='PCA on all redshifts')  # PCA on all redshifts

    args = parser.parse_args()

    train_kfold = train_model_kfold_2r if args.k2r else train_model_kfold

    if args.test_folds is not None:
        test_folds = list(map(int, args.test_folds.split(',')))  # Convert CSV input into a list of ints
    else:
        test_folds = None  # Default to None (test all folds)

    # load the lgk file
    if args.lgk is not None:
        lgk = np.loadtxt(args.lgk)

    n_k = lgk.shape[0]

    activation = act_dict[args.activation]

    # Start the timer
    start_time = time.time()  # Track start time

    # mkidr if model_dir does not exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    n_trials = args.trials

    model_path = os.path.join(args.model_dir, 'best_model.pth')


    hidden_size = args.hidden_size
    num_layers = args.num_layers
    decay = args.decay
    
    if args.train_one:
        n_trials = 1
        if args.trials > 1:
            print(f"Warning: Number of trials set to 1 for training a single model.")
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
            activation = act_dict[checkpoint['activation']]
            # and make a copy of the old model
            old_model_path = os.path.join(args.model_dir, f"best_model_old.pth")
            os.system(f"cp {model_path} {old_model_path}")
        elif hidden_size is None or num_layers is None or decay is None:
            raise ValueError("Please provide hidden_size, num_layers and decay for training a single model.")


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

    n_z = y.shape[1] // n_k  # total number of redshifts

    # extract the z bins we want to train
    if args.i_z is not None:
        i_z = list(map(int, args.i_z.split(',')))
        # check if the indices are valid
        if any(i < 0 or i >= n_z for i in i_z):
            raise ValueError(f"Invalid z bin indices. Please provide indices between 0 and {n_z - 1}.")
        # check if the indices are unique
        if len(i_z) != len(set(i_z)):
            raise ValueError(f"Duplicate z bin indices found. Please provide unique indices.")
        # check if the indices are sorted
        if not all(i_z[i] < i_z[i + 1] for i in range(len(i_z) - 1)):
            i_z.sort()
        # extract the z bins we want to train
        y_temp = y.reshape(-1, n_z, n_k)
        y = y_temp[:, i_z, :].reshape(-1, len(i_z) * n_k)
        n_z = len(i_z)  # update the number of redshifts

    print(f"Number of target redshifts: {n_z}")

    # normalize the input data if bounds are provided
    if args.bound_x is not None:
        bounds = np.loadtxt(args.bound_x)
        # normalize the input data (only the first matched columns)
        dimx_original = bounds.shape[0]
        x[:,:dimx_original] = (x[:,:dimx_original] - bounds[:,0]) / (bounds[:,1] - bounds[:,0])

    # y PCA if the minimum explained variance ratio is not 1
    if args.min_pca < 1:
        # PCA decomposition
        if args.pca_allz:
            # PCA on all redshifts
            y, mean_std = pca_decomp(y, explained_min=args.min_pca, standardize=args.standardize)
        else:
            # PCA on each redshift separately
            # pca per redshift

            # the number of coefficients we keep
            n_pc_zs = []
            y_pca = []
            mean_std = {'pca_components': [], 'pca_mean': [], 'mean': [], 'std': [], 'n_pc_zs': []}
            # y = y.reshape(-1, n_z, n_k)
            for i in range(n_z):
                ik_start = i * n_k
                ik_end = (i + 1) * n_k
                y_pca_z, mean_std_z = pca_decomp(y[:, ik_start:ik_end], explained_min=args.min_pca, standardize=args.standardize)
                # n_pc_zs.append(y_pca_z.shape[1])
                y_pca.append(y_pca_z)
                mean_std['pca_components'].append(mean_std_z['pca_components'])
                mean_std['pca_mean'].append(mean_std_z['pca_mean'])
                mean_std['mean'].append(mean_std_z['mean'])
                mean_std['std'].append(mean_std_z['std'])
                mean_std['n_pc_zs'].append(y_pca_z.shape[1])

            # concatenate the PCA coefficients
            y = np.concatenate(y_pca, axis=1)

    else:
        mean_std = {'pca_components': None, 'pca_mean': None, 'mean': None, 'std': None, 'n_pc_zs': None}

    # Convert to tensors on the CPU 
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Move tensors to the device
    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)

    if args.train_one:
        # if we are training a single model, use the provided hyperparameters
        best_params = {'hidden_size': hidden_size, 'decay': decay, 'num_layers': num_layers}
        print("\n🎯 Hyperparameters:")
        print('hidden_size', hidden_size, 'decay', decay, 'num_layers', num_layers)
    else:
        # if hyperparameters are provided, use them to train the model
        # Define the choices explicitly
        hidden_size_choices = list(range(16, 513, 16))  # Generates [16, 32, 48, ..., 512]
        num_layers_choices = [1, 2, 3, 4, 5, 6, 7]
        # activation_choices = [nn.ReLU, nn.Tanh, nn.Sigmoid]
        decay_lower, decay_upper = 1e-9, 5e-6
        
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

        trials_fine = Trials()
        trials_fine_max = max(n_trials // 2,1)

        # Run Bayesian optimization
        best_hyperparams = fmin(
            fn=objective,        # Function to minimize (training + validation loss)
            space=space,         # Hyperparameter search space
            algo=tpe.suggest,    # Tree-structured Parzen Estimator (TPE)
            max_evals=n_trials,        # Number of trials to run
            trials=trials,        # Store results
            show_progressbar=show_progress
        )
        best_hidden_size = hidden_size_choices[best_hyperparams['hidden_size']]
        best_num_layers = num_layers_choices[best_hyperparams['num_layers']]
        best_decay = best_hyperparams['decay']

        print("\n🎯 Best Hyperparameters Found in First Round:")
        print(f"hidden_size: {best_hidden_size}, decay: {best_decay:.6e}, num_layers: {best_num_layers}")

        # Define a refined search space
        hidden_size_choices_fine = list(range(max(16, best_hidden_size - 24), (best_hidden_size + 24), 2))  
        # num_layers_choices_fine = list(range(max(1, best_num_layers - 1), ((best_num_layers + 1) + 1)))
        # do not change the number of layers (changing the number of layers often leads to a worse model)
        num_layers_choices_fine = [best_num_layers]  # Keep the number of layers fixed
        decay_lower_fine = best_decay / 3  # Search around the best decay
        decay_upper_fine = best_decay * 3  

        space_fine = {
            'num_layers': hp.choice('num_layers', num_layers_choices_fine),
            'hidden_size': hp.choice('hidden_size', hidden_size_choices_fine),
            'decay': hp.loguniform('decay', np.log(decay_lower_fine), np.log(decay_upper_fine))
        }

        # Run the second optimization round
        
        best_hyperparams_fine = fmin(
            fn=objective,
            space=space_fine,
            algo=tpe.suggest,
            max_evals=trials_fine_max,  # Fewer trials for fine-tuning
            trials=trials_fine,
            show_progressbar=show_progress
        )

        # ✅ Directly assign the fine-tuned best hyperparameters
        best_hyperparams = best_hyperparams_fine
        best_params_fine = {'hidden_size': hidden_size_choices_fine[best_hyperparams['hidden_size']], 'decay': best_hyperparams['decay'], 'num_layers': num_layers_choices_fine[best_hyperparams['num_layers']]}
        
        
        print("\n🎯 Best Hyperparameters Found:")
        print('hidden_size', best_params_fine['hidden_size'], 'decay', best_params_fine['decay'], 'num_layers', best_params_fine['num_layers'])
        best_params = best_params_fine

    # Evaluate the model with the best hyperparameters
    _, _, best_fold, lr_best = train_kfold(**best_params, x_data=x_tensor, y_data=y_tensor, k=args.kfolds, save_kf_model=args.save_kfold, model_dir=args.model_dir, lr=args.lr, device=device, epochs=args.epochs, epochs_neuron=args.epochs_neuron, shuffle=args.shuffle, activation=activation, zero_centering=args.zero_centering, lgk=lgk, test_folds=test_folds, num_trials=args.trials_train, mean_std=mean_std)

    # train and save the model with the best hyperparameters
    # Save the model if required
    # train the model on the full dataset

    print(f"Training the model on the full dataset with the best hyperparameters...")
    epochs = args.epochs if args.epochs is not None else args.epochs_neuron * best_params['hidden_size'] * best_params['num_layers']
    train_loss, _, _, _ = train_NN(best_params['num_layers'], best_params['hidden_size'], x_tensor, y_tensor, decay=best_params['decay'], device=device, save_model=args.save_best, model_path=model_path, lr=lr_best, epochs=epochs, activation=activation, lgk=lgk, zero_centering=args.zero_centering, initial_model=best_fold,mean_std=mean_std)

    # print(f"⏱ Elapsed time: {time.time() - start_time:.2f} seconds\n")
    elapsed_time = time.time() - start_time

    total_seconds = int(elapsed_time)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    print(f"⏱ Elapsed time: {formatted_time}")
