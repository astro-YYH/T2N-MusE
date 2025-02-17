from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
import time  # Added for timing

# Define a neural network model
class SimpleNN(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(SimpleNN, self).__init__()
        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience
    
# Training function with K-Fold CV and Time Tracking
def train_model_kfold(num_layers, hidden_size, decay=0, k=5, epochs=500, lr=0.01):
    start_time = time.time()  # Track start time
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_tensor)):
        print(f"\n🔹 Fold {fold + 1}/{k} 🔹")

        train_x, train_y = x_tensor[train_idx], y_tensor[train_idx]
        val_x, val_y = x_tensor[val_idx], y_tensor[val_idx]

        # Reinitialize model for each fold to avoid weight leakage
        model = SimpleNN(num_layers=num_layers, hidden_size=hidden_size).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

        early_stopping = EarlyStopping(patience=50)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = model(train_x)
            loss = criterion(y_pred, train_y)
            loss.backward()
            optimizer.step()

            # Validation loss
            model.eval()
            with torch.no_grad():
                val_pred = model(val_x)
                val_loss = criterion(val_pred, val_y).item()

            scheduler.step(val_loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            if early_stopping.step(val_loss):
                print(f"Stopping early at epoch {epoch}")
                break

        fold_results.append(val_loss)

    avg_val_loss = np.mean(fold_results)
    elapsed_time = time.time() - start_time  # Compute elapsed time
    print(f"\n✅ Average Validation Loss: {avg_val_loss:.6f} (Time: {elapsed_time:.2f} sec)")

    return avg_val_loss, elapsed_time

# Function to evaluate a given set of hyperparameters
def objective(params):
    print(f"Testing with: {params}")

    val_loss, elapsed_time = train_model_kfold(
        num_layers=params['num_layers'],
        hidden_size=params['hidden_size'],
        decay=params['decay']
    )

    return {'loss': val_loss, 'status': STATUS_OK, 'elapsed_time': elapsed_time}

if __name__ == "__main__":
    # Define hyperparameter choices
    hidden_size_choices = [32, 64, 128, 256, 512]
    num_layers_choices = [2, 3, 4, 5]
    decay_lower, decay_upper = 1e-6, 1e-3

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Generate synthetic data (y = sin(x) + noise)
    np.random.seed(42)
    x = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)
    y = np.sin(x) + 0.05 * np.random.randn(100, 1)

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    # Define the hyperparameter search space
    space = {
        'num_layers': hp.choice('num_layers', num_layers_choices),
        'hidden_size': hp.choice('hidden_size', hidden_size_choices),
        'decay': hp.loguniform('decay', np.log(decay_lower), np.log(decay_upper))
    }

    # Create a trials object to store optimization history
    trials = Trials()

    # Track overall optimization time
    total_start_time = time.time()

    # Run Bayesian optimization
    best_hyperparams = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials
    )

    total_time = time.time() - total_start_time  # Compute total time
    print(f"\n🎯 Best Hyperparameters Found: {best_hyperparams}")
    print(f"⏳ Total Optimization Time: {total_time:.2f} sec")

    # Convert hyperopt indexes to actual values
    best_params = {
        'hidden_size': hidden_size_choices[best_hyperparams['hidden_size']],
        'decay': best_hyperparams['decay'],
        'num_layers': num_layers_choices[best_hyperparams['num_layers']]
    }

    final_val_loss, final_time = train_model_kfold(**best_params)

    print(f"\n🎯 Final Validation Loss: {final_val_loss:.6f} (Time: {final_time:.2f} sec)")
