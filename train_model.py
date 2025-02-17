import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import os

# Define a neural network model
class SimpleNN(nn.Module):
    def __init__(self, num_layers, hidden_size, dim_x=1, dim_y=1, activation=nn.SiLU()):   # num_layers: number of hidden layers
        super(SimpleNN, self).__init__()
        layers = [nn.Linear(dim_x, hidden_size), activation]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
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

def train_NN(num_layers, hidden_size, train_x, train_y, val_x=None, val_y=None, decay=0, epochs=1000, lr=0.1, device='cuda', save_model=False, model_path='model.pth'):

    # Create the model with the given hyperparameters
    model = SimpleNN(num_layers=num_layers, hidden_size=hidden_size, dim_x=train_x.shape[1], dim_y=train_y.shape[1]).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=40)

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
    early_stopping = EarlyStopping(patience=100)

    # Training loop with mini-batches
    for epoch in range(epochs):
        model.train()
        
        if use_full_batch:
            optimizer.zero_grad()
            y_pred = model(train_x)
            loss = criterion(y_pred, train_y)
            loss.backward()
            optimizer.step()
        else:
            for batch_x, batch_y in train_loader:  # Loop over mini-batches
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                y_pred = model(batch_x)
                loss_batch = criterion(y_pred, batch_y)
                loss_batch.backward()
                optimizer.step()

        # Training and Validation losses
        model.eval()
        with torch.no_grad():
            if not use_full_batch:
                train_pred = model(train_x.to(device))
                loss = criterion(train_pred, train_y.to(device))
            train_loss = loss.item()
            if val_x is not None and val_y is not None:
                val_pred = model(val_x)
                val_loss = criterion(val_pred, val_y).item()
            else:
                val_loss = train_loss

        scheduler.step(val_loss+train_loss)

        # Check early stopping condition
        if early_stopping.step(val_loss+train_loss):   # sum of train and val loss (should be more stable)
            print(f"Stopping early at epoch {epoch}")
            print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            break

        if epoch==0 or (epoch+1) % 100 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # if epoch reached the maximum number of epochs, warn the user that the model is not converged
        if epoch == epochs - 1:
            print("⚠ Maximum number of epochs reached. The model may not have converged.\n")

    if save_model:
        torch.save({
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'state_dict': model.state_dict()
        }, model_path)
        print(f"Model saved to {model_path}\n")

    return train_loss, val_loss

# Training function with K-Fold CV
def train_model_kfold(num_layers, hidden_size, x_data, y_data, decay=0, k=5, epochs=1000, lr=0.1, model_dir='./', save_kf_model=False, device='cuda', shuffle=True):
    kf = KFold(n_splits=k, shuffle=True, random_state=42) if shuffle else KFold(n_splits=k)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):
        print(f"🔹 Fold {fold + 1}/{k} 🔹")

        train_x, train_y = x_data[train_idx], y_data[train_idx]
        val_x, val_y = x_data[val_idx], y_data[val_idx]

        kf_model_path = os.path.join(model_dir,f"model_fold{fold}.pth")

        train_loss, val_loss = train_NN(num_layers, hidden_size, train_x, train_y, val_x, val_y, decay=decay, epochs=epochs, lr=lr, device=device, save_model=save_kf_model, model_path=kf_model_path)

        fold_results.append((train_loss, val_loss))

        

    avg_val_loss = np.mean([val_loss for _, val_loss in fold_results])
    print(f"✅ Average Validation Loss Across {k} Folds: {avg_val_loss:.6f}\n")

    return avg_val_loss



    
