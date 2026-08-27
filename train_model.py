import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import os
import random
import copy  # <-- Import copy for deep copying the model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures reproducibility

# Define a neural network model
class SimpleNN(nn.Module):
    def __init__(self, num_layers, hidden_size, dim_x=1, dim_y=1, activation=nn.SiLU()):   # num_layers: number of hidden layers
        super(SimpleNN, self).__init__()
        layers = [nn.Linear(dim_x, hidden_size)]
        if activation is not None:
            layers.append(activation)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if activation is not None:
                layers.append(activation)
        layers.append(nn.Linear(hidden_size, dim_y))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class EarlyStopping:
    def __init__(self, patience=50, fraction=0.005):
        """
        Early stopping with a relative threshold.

        Parameters:
        - patience (int): Number of epochs to wait for improvement.
        - fraction (float): Minimum percentage decrease required to reset patience.
        """
        self.patience = patience
        self.fraction = fraction
        self.best_loss = float("inf")
        self.wait = 0

    def step(self, val_loss):
        """Check if training should stop."""
        if val_loss < self.best_loss * (1 - self.fraction):  # ✅ Relative improvement
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1

        return self.wait >= self.patience

# Select the seed with the lowest mean regularized loss across its folds.
def select_best_seed(seed_results):
    if not seed_results or any(not results for results in seed_results.values()):
        raise ValueError("seed_results must contain at least one fold result per seed.")

    seed_reg_losses = {
        seed: float(np.mean([result['regularized_training_loss'] for result in results]))
        for seed, results in seed_results.items()
    }
    seed_val_losses = {
        seed: float(np.mean([result['validation_loss'] for result in results]))
        for seed, results in seed_results.items()
    }
    best_seed = min(seed_reg_losses, key=seed_reg_losses.get)
    best_results = seed_results[best_seed]
    completed_epochs = [
        result['epochs_completed'] for result in best_results
        if result['epochs_completed'] is not None
    ]
    return {
        'best_seed': best_seed,
        'mean_training_loss': float(np.mean([result['training_loss'] for result in best_results])),
        'mean_validation_loss': float(np.mean([result['validation_loss'] for result in best_results])),
        'mean_regularized_loss': seed_reg_losses[best_seed],
        'seed_regularized_losses': seed_reg_losses,
        'seed_validation_losses': seed_val_losses,
        'max_completed_epochs': max(completed_epochs) if completed_epochs else None,
    }

# Select the exact minimum of the validation curve averaged across folds.
def select_epoch_from_validation_curves(validation_curves):
    if not validation_curves:
        raise ValueError("validation_curves must contain at least one fold curve.")
    common_epochs = min(len(curve) for curve in validation_curves)
    if common_epochs < 1:
        raise ValueError("Every validation curve must contain at least one epoch.")

    mean_validation_curve = np.mean(
        [curve[:common_epochs] for curve in validation_curves],
        axis=0,
    )
    selected_epoch = int(np.argmin(mean_validation_curve))
    return {
        'selected_epoch': selected_epoch,
        'selected_epochs': selected_epoch + 1,
        'common_epochs': common_epochs,
        'mean_validation_loss': float(mean_validation_curve[selected_epoch]),
    }

# Copy nested training state to CPU so fold continuations do not retain GPU memory.
def copy_training_state_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: copy_training_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_training_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_training_state_to_cpu(item) for item in value)
    return copy.deepcopy(value)

# Capture random-number-generator states needed for an exact continuation.
def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state['cuda'] = [cuda_state.cpu() for cuda_state in torch.cuda.get_rng_state_all()]
    return state

# Restore random-number-generator states saved at a fold's stopping point.
def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])

def find_max_batch_size(model, dataset, device, start=32, step=2):
    """
    Dynamically find the largest batch size that fits in memory.

    Parameters:
    - model: PyTorch model
    - dataset: Dataset to test
    - device: "cuda" or "cpu"
    - start: Initial batch size to test
    - step: Factor to increase batch size (default: double each step)

    Returns:
    - Largest batch size that fits in memory
    """
    batch_size = start
    best_batch = batch_size
    max_batch = len(dataset)  # Limit to dataset size

    print(f"🔹 Starting batch size search on {device}...")

    # ✅ **1st Attempt: Try the largest possible batch size**
    try:
        loader = DataLoader(dataset, batch_size=max_batch)
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        model.to(device)(x)  # Check if model can process batch
        print(f"✅ Batch Size {max_batch} fits in memory on {device}.")
        return max_batch
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "memory" in str(e).lower():
            print(f"❌ Batch Size {max_batch} is too large, reducing...")
        else:
            print(f"⚠ Unexpected error: {e}")

    # ✅ **2nd Attempt: Find the largest batch size incrementally**
    while batch_size <= max_batch:
        try:
            # Create DataLoader with current batch size
            loader = DataLoader(dataset, batch_size=batch_size)
            
            # Try multiple batches to ensure stability
            for _ in range(3):  
                x, y = next(iter(loader))
                x, y = x.to(device), y.to(device)
                model.to(device)(x)  # Check model compatibility

            print(f"✅ Batch Size {batch_size} fits in memory on {device}. Trying larger size...")
            best_batch = batch_size
            batch_size *= step  # Increase batch size

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "memory" in str(e).lower():
                print(f"❌ Batch Size {batch_size} is too large, stopping search.")
                break  # Stop increasing when OOM occurs
            else:
                print(f"⚠ Unexpected error: {e}")
                break  # Stop on unknown errors

    print(f"🎯 Optimal Batch Size Found: {best_batch}")
    return best_batch

def train_NN(num_layers, hidden_size, train_x, train_y, val_x=None, val_y=None, decay=0, epochs=1000, lr=0.1, device='cuda', save_model=False, model_path='model.pth', activation=nn.SiLU(), lgk=None, zero_centering=False, L2_reg=True, initial_model=None, random_seed=42, mean_std=None, train_loss_lower=0, fold_val_weight=0, early_stopping_patience=300, early_stopping_fraction=0.005, use_early_stopping=True, return_history=False, return_resume_state=False, resume_state=None, restore_best_model=True):
    set_seed(random_seed) # Set seed for reproducibility

    if not 0 <= fold_val_weight <= 1:
        raise ValueError("fold_val_weight must be between 0 and 1.")
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1.")
    if not 0 <= early_stopping_fraction < 1:
        raise ValueError("early_stopping_fraction must be between 0 (inclusive) and 1 (exclusive).")
    if resume_state is not None and initial_model is not None:
        raise ValueError("initial_model and resume_state cannot be used together.")
    if resume_state is not None and epochs < resume_state['epochs_completed']:
        raise ValueError("epochs must not be smaller than the resumed epoch count.")
    val_provided = val_x is not None and val_y is not None
    effective_val_weight = fold_val_weight if val_provided else 0

    center_x = None
    center_y = None
    if zero_centering: # x and y
        center_x = train_x.mean(dim=0, keepdim=True)
        train_x = train_x - center_x

        # check if x includes y_LF 
        # if train_x.shape[1] > train_y.shape[1]:
        #     center_y = center_x[:, -train_y.shape[1]:]
        # else:
        #     center_y = train_y.mean(dim=0, keepdim=True)
        center_y = train_y.mean(dim=0, keepdim=True)

        train_y = train_y - center_y

        if val_x is not None and val_y is not None:
            val_x = val_x - center_x
            val_y = val_y - center_y

    # lgk is not used for training, but saved for later use

    if resume_state is not None:
        model = SimpleNN(num_layers=num_layers, hidden_size=hidden_size,
                         dim_x=train_x.shape[1], dim_y=train_y.shape[1],
                         activation=activation).to(device)
        model.load_state_dict(resume_state['model_state_dict'])
    elif initial_model is not None:
        model = initial_model
    else:
        # Create the model with the given hyperparameters
        model = SimpleNN(num_layers=num_layers, hidden_size=hidden_size, dim_x=train_x.shape[1], dim_y=train_y.shape[1], activation=activation).to(device)

    criterion = nn.MSELoss()
    penalty = 0

    def penalty():
        if L2_reg:
            l2_norm = sum(torch.sum(param ** 2) for param in model.parameters())
            penalty = decay * l2_norm
            return penalty
        else:
            return 0

        
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0 if L2_reg else decay)  # Use weight decay for L2 regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=60)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        scheduler.load_state_dict(resume_state['scheduler_state_dict'])

    # **Check if full-batch training is possible**
    # use_full_batch = best_batch >= len(train_dataset)
    use_full_batch = True
    best_batch = len(train_x) # enforce full-batch training for now (small dataset)

    # if not device yet, move to device
    if str(device)[:3] != str(train_x.device)[:3] and val_x is not None and val_y is not None:
        val_x, val_y = val_x.to(device), val_y.to(device)

    if use_full_batch:
        print(f"🔹 Using full-batch training (batch_size={best_batch})")
        # Convert to PyTorch tensors if not already
        if str(train_x.device)[:3] != str(device)[:3]:  # uss [:3] because device is cuda:0 or cuda when using GPU
            print("Converting to device tensors...")
            train_x, train_y = train_x.to(device), train_y.to(device)
    else:
        print(f"🔹 Using mini-batch training (batch_size={best_batch})")
        # Create PyTorch dataset
        train_x, train_y = train_x.cpu(), train_y.cpu()  # ✅ Ensure CPU tensors for DataLoader
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=best_batch, shuffle=True, num_workers=2, pin_memory=True)

    # Usage in training loop
    early_stopping = EarlyStopping(patience=early_stopping_patience,
                                   fraction=early_stopping_fraction)
    collect_history = return_history or return_resume_state
    if resume_state is not None:
        early_stopping.best_loss = resume_state['early_stopping_best_loss']
        early_stopping.wait = resume_state['early_stopping_wait']
        best_fold_objective = resume_state['best_fold_objective']
        best_epoch = resume_state['best_epoch']
        best_state_dict = resume_state['best_state_dict']
        history = copy.deepcopy(resume_state['history']) if collect_history else None
        completed_epochs = resume_state['epochs_completed']
        restore_rng_state(resume_state['rng_state'])
    else:
        best_fold_objective = float("inf")
        best_epoch = -1
        best_state_dict = copy.deepcopy(model.state_dict())
        history = {
            'training_loss': [],
            'validation_loss': [],
            'regularized_training_loss': [],
        } if collect_history else None
        completed_epochs = 0

    # Training loop with mini-batches
    for epoch in range(completed_epochs, epochs):
        completed_epochs = epoch + 1
        model.train()
        
        if use_full_batch:
            optimizer.zero_grad()
            y_pred = model(train_x)
            train_loss = criterion(y_pred, train_y)
            loss = train_loss + penalty()
            loss.backward()
            optimizer.step()
        else:
            for batch_x, batch_y in train_loader:  # Loop over mini-batches
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                y_pred = model(batch_x)
                loss_batch = criterion(y_pred, batch_y) + penalty()
                loss_batch.backward()
                optimizer.step()

        # Training and Validation losses
        model.eval()
        with torch.no_grad():
            train_pred = model(train_x.to(device))
            train_loss = criterion(train_pred, train_y.to(device)).item()
            if val_x is not None and val_y is not None:
                val_pred = model(val_x)
                val_loss = criterion(val_pred, val_y).item()
            else:
                val_loss = train_loss
            regularization = penalty()
            reg_train_loss = train_loss + (regularization.item() if torch.is_tensor(regularization) else regularization)
            fold_objective = ((1 - effective_val_weight) * reg_train_loss
                              + effective_val_weight * val_loss)

        if history is not None:
            history['training_loss'].append(train_loss)
            history['validation_loss'].append(val_loss)
            history['regularized_training_loss'].append(reg_train_loss)

        if fold_objective < best_fold_objective:
            best_fold_objective = fold_objective
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        scheduler.step(fold_objective)

        # Check early stopping condition
        early_stop = use_early_stopping and early_stopping.step(fold_objective)
        if early_stop or train_loss < train_loss_lower:
            print(f"Stopping early at epoch {epoch}")
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}, Train loss with L2: {reg_train_loss:.6e}, Fold objective: {fold_objective:.6e}, LR: {optimizer.param_groups[0]['lr']:.6e}")
            break

        if epoch==0 or (epoch+1) % 100 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}, Train loss with L2: {reg_train_loss:.6e}, Fold objective: {fold_objective:.6e}, LR: {optimizer.param_groups[0]['lr']:.6e}")

        # if epoch reached the maximum number of epochs, warn the user that the model is not converged
        if epoch == epochs - 1:
            if use_early_stopping:
                print("⚠ Maximum number of epochs reached. The model may not have converged.\n")
            else:
                print("✅ Completed the fixed epoch budget.\n")

    continuation_state = None
    if return_resume_state:
        continuation_state = {
            'model_state_dict': copy_training_state_to_cpu(model.state_dict()),
            'optimizer_state_dict': copy_training_state_to_cpu(optimizer.state_dict()),
            'scheduler_state_dict': copy_training_state_to_cpu(scheduler.state_dict()),
            'early_stopping_best_loss': early_stopping.best_loss,
            'early_stopping_wait': early_stopping.wait,
            'best_fold_objective': best_fold_objective,
            'best_epoch': best_epoch,
            'best_state_dict': copy_training_state_to_cpu(best_state_dict),
            'epochs_completed': completed_epochs,
            'history': copy.deepcopy(history),
            'rng_state': capture_rng_state(),
        }

    if restore_best_model:
        model.load_state_dict(best_state_dict)
    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(train_x.to(device)), train_y.to(device)).item()
        if val_x is not None and val_y is not None:
            val_loss = criterion(model(val_x), val_y).item()
        else:
            val_loss = train_loss
        regularization = penalty()
        reg_loss = train_loss + (regularization.item() if torch.is_tensor(regularization) else regularization)
        fold_objective = ((1 - effective_val_weight) * reg_loss
                          + effective_val_weight * val_loss)
    if restore_best_model:
        print(f"Restored best model from epoch {best_epoch} with fold objective {fold_objective:.6e}")
    else:
        print(f"Kept model at epoch {completed_epochs - 1} with fold objective {fold_objective:.6e}")

    # print('type of pca_components:', type(pca_components))
    if save_model:
        torch.save({
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'activation': activation.__class__.__name__ if activation is not None else 'None', # save activation function as string
            'decay': decay,
            'lgk': lgk,
            'training_loss': train_loss,
            'validation_loss': val_loss,
            'regularized_training_loss': reg_loss,
            'fold_val_weight': effective_val_weight,
            'fold_objective': fold_objective,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_fraction': early_stopping_fraction,
            'early_stopping_enabled': use_early_stopping,
            'random_seed': random_seed,
            'epochs_requested': epochs,
            'epochs_completed': completed_epochs,
            'best_epoch': best_epoch,
            'restored_best_model': restore_best_model,
            'center_x': center_x.cpu().numpy() if center_x is not None else None,  # Ensure it's a NumPy array
            'center_y': center_y.cpu().numpy() if center_y is not None else None,  # Convert before saving
            'state_dict': model.state_dict(),
            'mean_std': mean_std,
        }, model_path)
        print(f"Model saved to {model_path}\n")

    result = (train_loss, val_loss, model, optimizer.param_groups[0]['lr'], reg_loss)
    if return_history:
        result += (history,)
    if return_resume_state:
        result += (continuation_state,)
    return result

# Resume only shorter folds and select the minimum of their completed mean validation curve.
def complete_resumed_validation_curves(num_layers, hidden_size, x_data, y_data,
                                       fold_runs, decay, curve_epochs, lr, device,
                                       activation, zero_centering, fold_val_weight,
                                       early_stopping_patience,
                                       early_stopping_fraction, label):
    validation_curves = []
    print(f"🔄 Completing validation curves for {label} over {curve_epochs} epochs")
    for run_index, fold_run in enumerate(fold_runs, start=1):
        fold = fold_run['fold']
        history = fold_run['history']
        completed_epochs = len(history['validation_loss'])
        if completed_epochs < curve_epochs:
            print(f"🔹 {label} | Fold {run_index}/{len(fold_runs)}: resuming fold "
                  f"{fold} from {completed_epochs} to {curve_epochs} epochs 🔹")
            train_idx = fold_run['train_idx']
            val_idx = fold_run['val_idx']
            train_x, train_y = x_data[train_idx], y_data[train_idx]
            val_x, val_y = x_data[val_idx], y_data[val_idx]
            result = train_NN(
                num_layers, hidden_size, train_x, train_y, val_x, val_y,
                decay=decay, epochs=curve_epochs, lr=lr, device=device,
                activation=activation, zero_centering=zero_centering,
                random_seed=fold_run['seed'], fold_val_weight=fold_val_weight,
                early_stopping_patience=early_stopping_patience,
                early_stopping_fraction=early_stopping_fraction,
                use_early_stopping=False, return_history=True,
                resume_state=fold_run['resume_state'],
                restore_best_model=False,
            )
            history = result[5]
        else:
            print(f"🔹 {label} | Fold {run_index}/{len(fold_runs)}: fold {fold} "
                  f"already reached {curve_epochs} epochs 🔹")
        validation_curves.append(
            np.asarray(history['validation_loss'], dtype=np.float64)
        )

    return select_epoch_from_validation_curves(validation_curves)

def train_model_kfold_2r_old(num_layers, hidden_size, x_data, y_data, decay=0, k=5, epochs=None, 
                      epochs_neuron=10, lr=0.1, model_dir='./', save_kf_model=False, 
                      device='cuda', shuffle=False, activation=nn.SiLU(), zero_centering=False, 
                      lgk=None, test_folds=None, num_trials=1, mean_std=None, trials_k1=None,
                      fold_val_weight=0, early_stopping_patience=300,
                      early_stopping_fraction=0.005):
    """
    Train model using K-Fold Cross-Validation with an option to specify test folds.

    Parameters:
        num_layers: int - Number of layers in the NN.
        hidden_size: int - Number of neurons per layer.
        x_data: np.array - Input data.
        y_data: np.array - Target data.
        decay: float - Weight decay.
        k: int - Number of folds.
        epochs: int or None - Number of training epochs.
        epochs_neuron: int - Epochs per neuron.
        lr: float - Learning rate.
        model_dir: str - Directory to save models.
        save_kf_model: bool - Whether to save models.
        device: str - Device for computation ('cuda' or 'cpu').
        shuffle: bool - Shuffle data before splitting.
        activation: nn.Module - Activation function.
        zero_centering: bool - Whether to zero-center data.
        lgk: any - Additional parameter.
        test_folds: list or None - List of fold indices to test. If None, all folds are tested.

    Returns:
        tuple: (avg_train_loss, avg_val_loss)
    """
    epochs = epochs if epochs is not None else epochs_neuron * hidden_size * num_layers
    kf = KFold(n_splits=k, shuffle=True, random_state=42) if shuffle else KFold(n_splits=k)
    fold_results = []

    if test_folds is None:
        test_folds = list(range(k))  # Use all folds if not specified

    total_folds_to_test = len(test_folds)  # Total number of folds to test
    tested_count = 0  # Counter for completed folds

    # first round of training: independent training for each fold

    # exclude test folds from training x_data and y_data
    # x_data and y_data are PyTorch tensors
    mask = torch.ones(len(x_data), dtype=torch.bool)  # Create a mask of all True values
    mask[test_folds] = False  # Set test fold indices to False

    # Apply the mask to exclude the points we want to test against
    inds = np.arange(len(x_data))
    inds_1 = inds[mask]  # Indices of the points we will use in the first round of training
    x_data_1 = x_data[mask]
    y_data_1 = y_data[mask]

    # print excluded folds
    print(f"🔹 Excluded the {total_folds_to_test} target test points from the first round of training 🔹")

    # the number of folds in the first round of training
    k1 = k - len(test_folds)
    kf_1 = KFold(n_splits=k1, shuffle=True, random_state=42) if shuffle else KFold(n_splits=k1)

    trials_k1 = trials_k1 if trials_k1 is not None else total_folds_to_test

    print(f"🔹 Starting Round 1 of K-Fold Training: the first {trials_k1} folds from the {k1} folds will be used 🔹")

    for fold, (train_idx, val_idx) in enumerate(kf_1.split(x_data_1)):
        if tested_count == trials_k1: # for now, only test the first few folds
            break

        tested_count += 1
        print(f"🔹 Fold {tested_count}/{trials_k1}: Testing fold index {fold}/{k1-1} (point {inds_1[fold]}/{k-1}) 🔹")

        train_x, train_y = x_data_1[train_idx], y_data_1[train_idx]
        val_x, val_y = x_data_1[val_idx], y_data_1[val_idx]
        
        train_loss, val_loss, model, lr_fine, _ = train_fold_multiple_times(num_layers, hidden_size, train_x, train_y, val_x, val_y,
                 num_trials=num_trials, decay=decay, epochs=epochs, lr=lr, device=device, 
                 activation=activation, zero_centering=zero_centering, fold_val_weight=fold_val_weight,
                 early_stopping_patience=early_stopping_patience,
                 early_stopping_fraction=early_stopping_fraction)
        
        fold_results.append((train_loss, val_loss, model, lr_fine))
    # find the best model fold index
    idx_best = np.argmin([train_loss + val_loss for train_loss, val_loss, _, _ in fold_results])
    # best_model = fold_results[idx_best][2]
    best_model = copy.deepcopy(fold_results[idx_best][2])
    lr_best = fold_results[idx_best][3]

    # print the best model fold
    print(f"\n✅ Best Model Selected: model fold {idx_best}")

    fold_results = []  # Reset fold results for second round
    tested_count = 0  # Reset tested count for second round

    # second round of training: retrain using the best model's weights
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):
        if fold not in test_folds:
            continue  # Skip unselected folds

        tested_count += 1
        print(f"🔹 Fold {tested_count}/{total_folds_to_test}: Testing fold index {fold}/{k-1} 🔹")

        train_x, train_y = x_data[train_idx], y_data[train_idx]
        val_x, val_y = x_data[val_idx], y_data[val_idx]

        kf_model_path = os.path.join(model_dir, f"model_fold{fold}.pth")

        train_loss, val_loss, model, _, _ = train_NN(num_layers, hidden_size, train_x, train_y, val_x, val_y, 
                                        decay=decay, epochs=epochs, lr=lr_best, device=device, 
                                        save_model=save_kf_model, model_path=kf_model_path, 
                                        activation=activation, zero_centering=zero_centering, lgk=lgk, initial_model=copy.deepcopy(best_model), mean_std=mean_std,
                                        fold_val_weight=fold_val_weight,
                                        early_stopping_patience=early_stopping_patience,
                                        early_stopping_fraction=early_stopping_fraction)

        fold_results.append((train_loss, val_loss, model))
    
    # best model
    best_model = min(fold_results, key=lambda x: x[0] + x[1])[2]

    if fold_results:
        avg_val_loss = np.mean([val_loss for _, val_loss, _ in fold_results])
        avg_train_loss = np.mean([train_loss for train_loss, _, _ in fold_results])
        print(f"✅ Average Loss Across Selected Folds: training: {avg_train_loss:.6e}, validation: {avg_val_loss:.6e}, mean(training,validation): {.5*(avg_train_loss+avg_val_loss):.6e}\n")
        return avg_train_loss, avg_val_loss, best_model, lr_best
    else:
        print("⚠️ No folds were selected for testing. Returning None.")
        return None, None, None, None

# update 2-round training: no k-fold training in the first round, but only multiple trials with distinct random seeds
# and then use the best model to train on the selected folds in the second round
def train_model_kfold_2r(num_layers, hidden_size, x_data, y_data, decay=0, k=5, epochs=None, 
                      epochs_neuron=10, lr=0.1, model_dir='./', save_kf_model=False, 
                      device='cuda', shuffle=False, activation=nn.SiLU(), zero_centering=False, 
                      lgk=None, test_folds=None, num_trials=None, mean_std=None,
                      fold_val_weight=0, early_stopping_patience=300,
                      early_stopping_fraction=0.005, select_duration=False,
                      exact_duration_model=False):
    """
    Train model using K-Fold Cross-Validation with an option to specify test folds.

    Parameters:
        num_layers: int - Number of layers in the NN.
        hidden_size: int - Number of neurons per layer.
        x_data: np.array - Input data.
        y_data: np.array - Target data.
        decay: float - Weight decay.
        k: int - Number of folds.
        epochs: int or None - Number of training epochs.
        epochs_neuron: int - Epochs per neuron.
        lr: float - Learning rate.
        model_dir: str - Directory to save models.
        save_kf_model: bool - Whether to save models.
        device: str - Device for computation ('cuda' or 'cpu').
        shuffle: bool - Shuffle data before splitting.
        activation: nn.Module - Activation function.
        zero_centering: bool - Whether to zero-center data.
        lgk: any - Additional parameter.
        test_folds: list or None - List of fold indices to test. If None, all folds are tested.
        select_duration: bool - Whether to select the Round 2 duration from validation curves.
        exact_duration_model: bool - Keep saved models at the selected epoch instead of their best L2 epoch.

    Returns:
        tuple: (avg_train_loss, avg_val_loss, initialization, training_value).
        With duration selection, initialization is the Round 1 model and training_value
        contains the Round 2 learning rate and selected epoch count. Otherwise, the
        existing representative Round 2 model and learning-rate return is preserved.
    """

    epochs = epochs if epochs is not None else epochs_neuron * hidden_size * num_layers
    kf = KFold(n_splits=k, shuffle=True, random_state=42) if shuffle else KFold(n_splits=k)

    if test_folds is None:
        test_folds = list(range(k))  # Use all folds if not specified

    fold_splits = [
        (fold, train_idx, val_idx)
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_data))
        if fold in test_folds
    ]
    if not fold_splits:
        print("⚠️ No folds were selected for testing. Returning None.")
        return None, None, None, None

    if num_trials is None:
        print("⚠️ No trials specified. Setting num_trials to the number of test points by default.")
        num_trials = len(test_folds)  # Number of trials in the first round

    total_folds_to_test = len(fold_splits)  # Total number of folds to test
    tested_count = 0  # Counter for completed folds

    # first round of training: train on the data excluding the test points

    mask = torch.ones(len(x_data), dtype=torch.bool)  # Create a mask of all True values
    mask[test_folds] = False  # Set test fold indices to False

    # Apply the mask to exclude the points we want to test against
    inds = np.arange(len(x_data))
    # inds_1 = inds[mask]  # Indices of the points we will use in the first round of training
    x_data_1 = x_data[mask]
    y_data_1 = y_data[mask]

    # use the excluded test points as the validation set in the first round
    x_data_1_val = x_data[test_folds]
    y_data_1_val = y_data[test_folds]

    # print training and validation data
    print(f"🔹 Excluded the {total_folds_to_test} target test points from training and test on them  🔹")

    print(f"🔹 Starting Round 1 of Training: searching for a good minimum 🔹")
    # no real validation loss here, just training
    train_loss, val_loss, model, lr_fine, _ = train_fold_multiple_times(num_layers, hidden_size, x_data_1, y_data_1, x_data_1_val, y_data_1_val,
                 num_trials=num_trials, decay=decay, epochs=epochs, lr=lr, device=device, 
                 activation=activation, zero_centering=zero_centering, fold_val_weight=fold_val_weight,
                 early_stopping_patience=early_stopping_patience,
                 early_stopping_fraction=early_stopping_fraction)
        
    # best_model = fold_results[idx_best][2]
    round1_model = copy.deepcopy(model)
    lr_best = lr_fine

    fold_results = []  # Reset fold results for second round
    tested_count = 0  # Reset tested count for second round

    # second round of training: retrain using the best model's weights
    for fold, train_idx, val_idx in fold_splits:
        tested_count += 1
        print(f"🔹 Fold {tested_count}/{total_folds_to_test}: Testing fold index {fold}/{k-1} 🔹")

        train_x, train_y = x_data[train_idx], y_data[train_idx]
        val_x, val_y = x_data[val_idx], y_data[val_idx]

        kf_model_path = os.path.join(model_dir, f"model_fold{fold}.pth")

        result = train_NN(num_layers, hidden_size, train_x, train_y, val_x, val_y,
                          decay=decay, epochs=epochs, lr=lr_best, device=device,
                          save_model=save_kf_model and not select_duration,
                          model_path=kf_model_path, activation=activation,
                          zero_centering=zero_centering, lgk=lgk,
                          initial_model=copy.deepcopy(round1_model), mean_std=mean_std,
                          fold_val_weight=fold_val_weight,
                          early_stopping_patience=early_stopping_patience,
                          early_stopping_fraction=early_stopping_fraction,
                          return_history=select_duration,
                          return_resume_state=select_duration)
        train_loss, val_loss, model, _, reg_loss = result[:5]
        history = result[5] if select_duration else None
        resume_state = result[6] if select_duration else None

        fold_results.append((train_loss, val_loss, model, reg_loss, history,
                             resume_state, fold, train_idx, val_idx))
    
    # best model
    # best_model = min(fold_results, key=lambda x: x[0] + x[1])[2]

    # print the best model fold
    # idx_best = np.argmin([train_loss + val_loss for train_loss, val_loss, _, _ in fold_results])
    # should select the best model fold based on regularized loss instead of train_loss + val_loss, because individual val loss is highly dependent on the tested point, reg loss is more stable
    # choose the model with the regularized loss that is closest to the mean of the regularized loss
    del_loss = np.abs(
        np.array([result[3] for result in fold_results])
        - np.mean([result[3] for result in fold_results])
    )
    idx_best = np.argmin(del_loss)
    best_model = copy.deepcopy(fold_results[idx_best][2])
    print(f"\n✅ Best Model Selected: model fold {fold_splits[idx_best][0]} (with regularized loss closest to the mean)")

    if fold_results:
        avg_val_loss = np.mean([result[1] for result in fold_results])
        avg_train_loss = np.mean([result[0] for result in fold_results])
        print(f"✅ Average Loss Across Selected Folds: training: {avg_train_loss:.6e}, validation: {avg_val_loss:.6e}, mean(training,validation): {.5*(avg_train_loss+avg_val_loss):.6e}\n")

        if select_duration:
            fold_runs = [
                {
                    'fold': result[6],
                    'train_idx': result[7],
                    'val_idx': result[8],
                    'seed': 42,
                    'history': result[4],
                    'resume_state': result[5],
                }
                for result in fold_results
            ]
            curve_epochs = max(len(run['history']['validation_loss'])
                               for run in fold_runs)
            duration = complete_resumed_validation_curves(
                num_layers, hidden_size, x_data, y_data, fold_runs,
                decay=decay, curve_epochs=curve_epochs, lr=lr_best,
                device=device, activation=activation,
                zero_centering=zero_centering,
                fold_val_weight=fold_val_weight,
                early_stopping_patience=early_stopping_patience,
                early_stopping_fraction=early_stopping_fraction,
                label="Round 2",
            )
            selected_epochs = duration['selected_epochs']
            print(f"✅ Round 2 validation curve selected epoch {duration['selected_epoch']} "
                  f"({selected_epochs} training epochs), with mean validation loss "
                  f"{duration['mean_validation_loss']:.6e}\n")

            if save_kf_model:
                print("🔄 Saving Round 2 fold models at the validation-selected duration")
                for fold, train_idx, val_idx in fold_splits:
                    train_x, train_y = x_data[train_idx], y_data[train_idx]
                    val_x, val_y = x_data[val_idx], y_data[val_idx]
                    kf_model_path = os.path.join(model_dir, f"model_fold{fold}.pth")
                    train_NN(
                        num_layers, hidden_size, train_x, train_y, val_x, val_y,
                        decay=decay, epochs=selected_epochs, lr=lr_best,
                        device=device, save_model=True, model_path=kf_model_path,
                        activation=activation, zero_centering=zero_centering,
                        lgk=lgk, initial_model=copy.deepcopy(round1_model),
                        mean_std=mean_std, fold_val_weight=fold_val_weight,
                        early_stopping_patience=early_stopping_patience,
                        early_stopping_fraction=early_stopping_fraction,
                        use_early_stopping=False,
                        restore_best_model=not exact_duration_model,
                    )

            training_config = {
                'lr': lr_best,
                'epochs': selected_epochs,
            }
            returned_val_loss = (duration['mean_validation_loss']
                                 if exact_duration_model else avg_val_loss)
            return (avg_train_loss, returned_val_loss,
                    round1_model, training_config)

        return avg_train_loss, avg_val_loss, best_model, lr_best
    

def train_fold_multiple_times(num_layers, hidden_size, train_x, train_y, val_x=None, val_y=None, 
                              num_trials=3, decay=0, epochs=1000, lr=0.1, device='cuda', 
                              activation=nn.SiLU(), zero_centering=False, save_model=False,
                              model_path='model.pth', lgk=None, mean_std=None,
                              fold_val_weight=0, early_stopping_patience=300,
                              early_stopping_fraction=0.005):
    """
    Train a single fold multiple times with different seeds and return the best model.
    
    Parameters:
        num_trials: int - Number of times to train with different random seeds.
        
    Returns:
        best_model - The best-performing model for this fold.
    """
    val_provided = True  # Check if validation data is provided
    if val_x is None or val_y is None:
        val_provided = False
        print(f"⚠️ No validation data provided. Regularized loss will be used to select the best model.")

    best_model = None
    best_summed_loss = float("inf")

    for trial in range(num_trials):
        seed = 42 + trial  # Change seed for each trial
        
        print(f"🔄 Training fold with seed {seed} (Trial {trial+1}/{num_trials})...")
        
        train_loss, val_loss, model, lr_fine, reg_loss = train_NN(num_layers, hidden_size, train_x, train_y, val_x, val_y, 
                                                  decay=decay, epochs=epochs, lr=lr, device=device, 
                                                  activation=activation, zero_centering=zero_centering, random_seed=seed,
                                                  fold_val_weight=fold_val_weight,
                                                  early_stopping_patience=early_stopping_patience,
                                                  early_stopping_fraction=early_stopping_fraction)

        effective_val_weight = fold_val_weight if val_provided else 0
        summed_loss = ((1 - effective_val_weight) * reg_loss
                       + effective_val_weight * val_loss)
            
        if summed_loss < best_summed_loss:
            best_model = model
            best_summed_loss = summed_loss
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_reg_loss = reg_loss
            lr_best = lr_fine
            seed_best = seed
            # print(f"✅ Best model selected for this fold (Validation Loss + Training Loss: {best_summed_loss:.6e})")
    #retrain and save the best model

    print(f"✅ Best model selected with fold objective: {best_summed_loss:.6e}")
    if save_model and best_model is not None:
        print(f"🔄 Retraining the best model with seed {seed_best}... (usually leads to a slightly better model)")
        # retrain and save the best model
        train_NN(num_layers, hidden_size, train_x, train_y, val_x, val_y,
                 decay=decay, epochs=epochs, lr=lr_best, device=device, 
                 activation=activation, zero_centering=zero_centering, random_seed=seed_best, save_model=save_model, model_path=model_path, lgk=lgk, initial_model=best_model, mean_std=mean_std,
                 fold_val_weight=fold_val_weight,
                 early_stopping_patience=early_stopping_patience,
                 early_stopping_fraction=early_stopping_fraction)

    return best_train_loss, best_val_loss, best_model, lr_best, best_reg_loss  # Return the loss with L2 regularization

def train_model_kfold(num_layers, hidden_size, x_data, y_data, decay=0, k=5, epochs=None, 
                      epochs_neuron=10, lr=0.1, model_dir='./', save_kf_model=False, 
                      device='cuda', shuffle=True, activation=nn.SiLU(), zero_centering=False, 
                      lgk=None, test_folds=None, num_trials=1, mean_std=None, trials_k1=None,
                      fold_val_weight=0, early_stopping_patience=300,
                      early_stopping_fraction=0.005,
                      select_duration=False, fixed_seed=None,
                      exact_duration_model=False):  # trials_k1 is not used here
    """
    Train model using K-Fold Cross-Validation with an option to specify test folds.

    Parameters:
        num_layers: int - Number of layers in the NN.
        hidden_size: int - Number of neurons per layer.
        x_data: np.array - Input data.
        y_data: np.array - Target data.
        decay: float - Weight decay.
        k: int - Number of folds.
        epochs: int or None - Number of training epochs.
        epochs_neuron: int - Epochs per neuron.
        lr: float - Learning rate.
        model_dir: str - Directory to save models.
        save_kf_model: bool - Whether to save models.
        device: str - Device for computation ('cuda' or 'cpu').
        shuffle: bool - Shuffle data before splitting.
        activation: nn.Module - Activation function.
        zero_centering: bool - Whether to zero-center data.
        lgk: any - Additional parameter.
        test_folds: list or None - List of fold indices to test. If None, all folds are tested.
        select_duration: bool - Whether to run the post-selection validation-curve pass.
        fixed_seed: int or None - Restrict training to an already selected seed.
        exact_duration_model: bool - Keep saved models at the selected epoch instead of their best L2 epoch.

    Returns:
        tuple: (avg_train_loss, avg_val_loss, best_seed, selected_epochs or None)
    """
    epochs = epochs if epochs is not None else epochs_neuron * hidden_size * num_layers
    kf = KFold(n_splits=k, shuffle=True, random_state=42) if shuffle else KFold(n_splits=k)

    if test_folds is None:
        test_folds = list(range(k))  # Use all folds if not specified
    if fixed_seed is None and num_trials < 1:
        raise ValueError("num_trials must be at least 1.")

    fold_splits = [
        (fold, train_idx, val_idx)
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_data))
        if fold in test_folds
    ]
    if not fold_splits:
        print("⚠️ No folds were selected for testing. Returning None.")
        return None, None, None, None

    seed_results = {}
    best_duration_seed_loss = float("inf")
    best_duration_fold_runs = None
    seeds = [fixed_seed] if fixed_seed is not None else [42 + trial for trial in range(num_trials)]
    for trial, seed in enumerate(seeds):
        seed_results[seed] = []
        seed_fold_runs = []
        print(f"\n🔄 Seed {seed} ({trial + 1}/{len(seeds)}): training all selected folds")

        for tested_count, (fold, train_idx, val_idx) in enumerate(fold_splits, start=1):
            print(f"🔹 Seed {seed} | Fold {tested_count}/{len(fold_splits)}: Testing fold index {fold}/{k-1} 🔹")
            train_x, train_y = x_data[train_idx], y_data[train_idx]
            val_x, val_y = x_data[val_idx], y_data[val_idx]

            train_kwargs = dict(
                decay=decay, epochs=epochs, lr=lr, device=device,
                activation=activation, zero_centering=zero_centering,
                random_seed=seed, fold_val_weight=fold_val_weight,
                early_stopping_patience=early_stopping_patience,
                early_stopping_fraction=early_stopping_fraction,
                return_history=select_duration,
                return_resume_state=select_duration,
            )
            result = train_NN(
                num_layers, hidden_size, train_x, train_y, val_x, val_y,
                **train_kwargs,
            )
            train_loss, val_loss, _, _, reg_loss = result[:5]
            history = result[5] if select_duration else None
            resume_state = result[6] if select_duration else None
            seed_results[seed].append({
                'training_loss': train_loss,
                'validation_loss': val_loss,
                'regularized_training_loss': reg_loss,
                'epochs_completed': len(history['validation_loss']) if history is not None else None,
            })
            if select_duration:
                seed_fold_runs.append({
                    'fold': fold,
                    'train_idx': train_idx,
                    'val_idx': val_idx,
                    'seed': seed,
                    'history': history,
                    'resume_state': resume_state,
                })

        if select_duration:
            seed_reg_loss = float(np.mean([
                result['regularized_training_loss']
                for result in seed_results[seed]
            ]))
            if seed_reg_loss < best_duration_seed_loss:
                best_duration_seed_loss = seed_reg_loss
                best_duration_fold_runs = seed_fold_runs

    selection = select_best_seed(seed_results)
    best_seed = selection['best_seed']
    for seed, reg_loss in selection['seed_regularized_losses'].items():
        val_loss = selection['seed_validation_losses'][seed]
        print(f"   Seed {seed}: mean regularized loss {reg_loss:.6e}, "
              f"mean validation loss {val_loss:.6e}")
    print(f"✅ Best seed: {best_seed} with mean regularized loss "
          f"{selection['mean_regularized_loss']:.6e}")
    print(f"✅ Best-seed average losses: training: {selection['mean_training_loss']:.6e}, "
          f"validation: {selection['mean_validation_loss']:.6e}\n")

    selected_epochs = None
    selected_validation_loss = selection['mean_validation_loss']
    if select_duration:
        curve_epochs = selection['max_completed_epochs']
        if best_duration_fold_runs is None:
            raise RuntimeError("Missing continuation states for the selected seed.")
        duration = complete_resumed_validation_curves(
            num_layers, hidden_size, x_data, y_data, best_duration_fold_runs,
            decay=decay, curve_epochs=curve_epochs, lr=lr, device=device,
            activation=activation, zero_centering=zero_centering,
            fold_val_weight=fold_val_weight,
            early_stopping_patience=early_stopping_patience,
            early_stopping_fraction=early_stopping_fraction,
            label=f"seed {best_seed}",
        )
        selected_epochs = duration['selected_epochs']
        if exact_duration_model:
            selected_validation_loss = duration['mean_validation_loss']
        print(f"✅ Validation curve selected epoch {duration['selected_epoch']} "
              f"({selected_epochs} training epochs), with mean validation loss "
              f"{duration['mean_validation_loss']:.6e}\n")

    if save_kf_model:
        save_epochs = selected_epochs if selected_epochs is not None else epochs
        print(f"🔄 Saving fold models for selected seed {best_seed}")
        for fold, train_idx, val_idx in fold_splits:
            train_x, train_y = x_data[train_idx], y_data[train_idx]
            val_x, val_y = x_data[val_idx], y_data[val_idx]
            kf_model_path = os.path.join(model_dir, f"model_fold{fold}.pth")
            train_NN(
                num_layers, hidden_size, train_x, train_y, val_x, val_y,
                decay=decay, epochs=save_epochs, lr=lr, device=device,
                save_model=True, model_path=kf_model_path, activation=activation,
                zero_centering=zero_centering, lgk=lgk, random_seed=best_seed,
                mean_std=mean_std, fold_val_weight=fold_val_weight,
                early_stopping_patience=early_stopping_patience,
                early_stopping_fraction=early_stopping_fraction,
                use_early_stopping=selected_epochs is None,
                restore_best_model=not (selected_epochs is not None
                                        and exact_duration_model),
            )

    return (selection['mean_training_loss'],
            selected_validation_loss,
            best_seed, selected_epochs)
    
def train_model_kfold_with_initial(num_layers, hidden_size, x_data, y_data, decay=0, k=5, epochs=None, 
                      epochs_neuron=10, lr=0.1, model_dir='./', save_kf_model=False, 
                      device='cuda', shuffle=True, activation=nn.SiLU(), zero_centering=False, 
                      lgk=None, test_folds=None, initial_model=None, fold_val_weight=0,
                      early_stopping_patience=300, early_stopping_fraction=0.005):

    epochs = epochs if epochs is not None else epochs_neuron * hidden_size * num_layers
    kf = KFold(n_splits=k, shuffle=True, random_state=42) if shuffle else KFold(n_splits=k)
    fold_results = []

    if test_folds is None:
        test_folds = list(range(k))  # Use all folds if not specified

    total_folds_to_test = len(test_folds)  # Total number of folds to test
    tested_count = 0  # Counter for completed folds


    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):
        if fold not in test_folds:
            continue  # Skip unselected folds

        tested_count += 1
        print(f"🔹 Fold {tested_count}/{total_folds_to_test}: Testing fold index {fold}/{k-1} 🔹")

        train_x, train_y = x_data[train_idx], y_data[train_idx]
        val_x, val_y = x_data[val_idx], y_data[val_idx]

        kf_model_path = os.path.join(model_dir, f"model_fold{fold}.pth")

        train_loss, val_loss, model, lr_fine, _ = train_NN(num_layers, hidden_size, train_x, train_y, val_x, val_y, 
                                        decay=decay, epochs=epochs, lr=lr, device=device, 
                                        save_model=save_kf_model, model_path=kf_model_path, 
                                        activation=activation, zero_centering=zero_centering, lgk=lgk, initial_model=initial_model,
                                        fold_val_weight=fold_val_weight,
                                        early_stopping_patience=early_stopping_patience,
                                        early_stopping_fraction=early_stopping_fraction)

        fold_results.append((train_loss, val_loss, model, lr_fine))
    # find the best model fold index
    idx_best = np.argmin([train_loss + val_loss for train_loss, val_loss, _, _ in fold_results])
    # best_model = fold_results[idx_best][2]
    best_model = copy.deepcopy(fold_results[idx_best][2])
    lr_best = fold_results[idx_best][3]

    # print the best model fold
    print(f"\n✅ Best Model Selected: model fold {test_folds[idx_best]}")

    if fold_results:
        avg_val_loss = np.mean([val_loss for _, val_loss, _, _ in fold_results])
        avg_train_loss = np.mean([train_loss for train_loss, _, _, _ in fold_results])
        print(f"✅ Average Loss Across Selected Folds: training: {avg_train_loss:.6e}, validation: {avg_val_loss:.6e}, mean(training,validation): {.5*(avg_train_loss+avg_val_loss):.6e}\n")
        return avg_train_loss, avg_val_loss, best_model, lr_best
    else:
        print("⚠️ No folds were selected for testing. Returning None.")
        return None, None, None, None
