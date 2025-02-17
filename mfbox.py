import os
import torch
import numpy as np
from train_model import SimpleNN

def extract_input_output_dims(state_dict):
    dim_x, dim_y = None, None
    for key in state_dict:
        if "weight" in key and "network" in key:
            if dim_x is None:
                dim_x = state_dict[key].shape[1]  # First Linear Layer (input size)
            dim_y = state_dict[key].shape[0]  # Last Linear Layer (output size)
    return dim_x, dim_y

def load_model(path, device='cpu'):
    checkpoint = torch.load(path, weights_only=True, map_location=device)
    dim_x, dim_y = extract_input_output_dims(state_dict=checkpoint['state_dict'])
    model = SimpleNN(num_layers=checkpoint['num_layers'], hidden_size=checkpoint['hidden_size'], dim_x=dim_x, dim_y=dim_y).to(device)

    # Load the model weights
    model.load_state_dict(checkpoint['state_dict'])

    return model


# load L1, L2 and LF-HF models, and make predictions
class mfbox:
    def __init__(self, path_L1, path_L2, path_LH, device='cpu'):
        self.device = torch.device(device)

        # Load the saved model dictionary
        self.model_L1 = load_model(path_L1, device=self.device)
        self.model_L2 = load_model(path_L2, device=self.device)
        self.model_LH = load_model(path_LH, device=self.device)
        self.model_L1.eval()
        self.model_L2.eval()
        self.model_LH.eval()
    
    def predict(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y_L1 = self.model_L1(x_tensor)
        y_L2 = self.model_L2(x_tensor)

        # Concatenate the predictions
        x_xyL1L2 = torch.cat((x_tensor, y_L1, y_L2), dim=1)
        y = self.model_LH(x_xyL1L2).detach().cpu().numpy()

        return y
    
    def predict_L1(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y = self.model_L1(x_tensor).detach().cpu().numpy()

        return y
    
    def predict_L2(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y = self.model_L2(x_tensor).detach().cpu().numpy()

        return y
    

# define gokunet model based on mfbox, normalize the input data

class gokunet(mfbox):
    def __init__(self, path_L1, path_L2, path_LH, device='cpu', bounds_path="./data/pre_N_L-H_z0/input_limits.txt"):
        super().__init__(path_L1, path_L2, path_LH, device=device)
        self.bounds = np.loadtxt(bounds_path)
    
    def predict(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        y = super().predict(x)
        # return 10**y # Convert back to linear scale
        return 10**y
    
    def predict_L1(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        y = super().predict_L1(x)
        # return 10**y # Convert back to linear scale
        return 10**y
    
    def predict_L2(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        y = super().predict_L2(x)
        # return 10**y # Convert back to linear scale
        return 10**y
    

