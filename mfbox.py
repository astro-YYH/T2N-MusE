import os
import torch
import numpy as np
from train_model import SimpleNN
import torch.nn as nn

act_dict = {'ReLU': nn.ReLU(), 'SiLU': nn.SiLU(), 'Tanh': nn.Tanh(), 'None': None}

def extract_input_output_dims(state_dict):
    dim_x, dim_y = None, None
    for key in state_dict:
        if "weight" in key and "network" in key:
            if dim_x is None:
                dim_x = state_dict[key].shape[1]  # First Linear Layer (input size)
            dim_y = state_dict[key].shape[0]  # Last Linear Layer (output size)
    return dim_x, dim_y

def load_model(path, device='cpu'):
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    dim_x, dim_y = extract_input_output_dims(state_dict=checkpoint['state_dict'])
    # activation
    activation = act_dict[checkpoint['activation']]
    model = SimpleNN(num_layers=checkpoint['num_layers'], hidden_size=checkpoint['hidden_size'], dim_x=dim_x, dim_y=dim_y, activation=activation).to(device)

    # Load the model weights
    model.load_state_dict(checkpoint['state_dict'])

    return model


# load L1, L2 and LF-HF models, and make predictions
class mfbox:
    def __init__(self, path_L1, path_L2, path_LH, path_LHf=None, device='cpu', stitch="XL1L2"):
        self.device = torch.device(device)

        if stitch not in ['XL1L2', 'XL', 'L']:
            raise ValueError("stitch should be one of 'XL1L2', 'XL' and 'L'")
        self.stitch = stitch

        # Load the saved model dictionary
        self.model_L1 = load_model(path_L1, device=self.device)
        self.model_L2 = load_model(path_L2, device=self.device)
        self.model_LH = load_model(path_LH, device=self.device)
        self.model_L1.eval()
        self.model_L2.eval()
        self.model_LH.eval()
        # load the final LH model if provided
        self.model_LHf = load_model(path_LHf, device=self.device) if path_LHf is not None else None

        self.lgk_L1 = np.loadtxt("./data/pre_N_L-H_stitch_z0/kf.txt") # use this for now, will be replaced later
        self.lgk_L2 = np.loadtxt("./data/pre_N_L-H_stitch_z0/kf.txt")
        self.lgk_H = np.loadtxt("./data/pre_N_L-H_stitch_z0/kf.txt")
    
    def predict(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y_L1 = self.model_L1(x_tensor)
        y_L2 = self.model_L2(x_tensor)

        # Concatenate the predictions with the input according to the stitching method
        if self.stitch == 'XL1L2':
            x_LH = torch.cat((x_tensor, y_L1, y_L2), dim=1)
        else:
            # cut and stitch L1 and L2
            middle = y_L1.shape[1] // 2
            y_L1_interp = np.array([np.interp(self.lgk_H[:middle], self.lgk_L1, y_L1[i, :].detach().numpy()) for i in range(y_L1.shape[0])])
            y_L2_interp = np.array([np.interp(self.lgk_H[middle:], self.lgk_L2, y_L2[i, :].detach().numpy()) for i in range(y_L2.shape[0])])
            # to tensor
            y_L1_interp = torch.tensor(y_L1_interp, dtype=torch.float32).to(self.device)
            y_L2_interp = torch.tensor(y_L2_interp, dtype=torch.float32).to(self.device)

            if self.stitch == 'XL':
                x_LH = torch.cat((x_tensor, y_L1_interp, y_L2_interp), dim=1)
            else: # 'L'
                x_LH = torch.cat((y_L1_interp, y_L2_interp), dim=1)

        y = self.model_LH(x_LH)

        if self.model_LHf is not None:
            x_LHf = torch.cat((x_LH, y), dim=1)
            y = self.model_LHf(x_LHf).detach().cpu().numpy()
        else:
            y = y.detach().cpu().numpy()

        return self.lgk_H, y
    
    def predict_L1(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y = self.model_L1(x_tensor).detach().cpu().numpy()

        return self.lgk_L1, y
    
    def predict_L2(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y = self.model_L2(x_tensor).detach().cpu().numpy()

        return self.lgk_L2, y


# define gokunet model based on mfbox, normalize the input data

class gokunet(mfbox):
    def __init__(self, path_L1, path_L2, path_LH, path_LHf=None, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt", stitch="XL1L2"):
        super().__init__(path_L1, path_L2, path_LH, path_LHf, device=device, stitch=stitch)
        self.bounds = np.loadtxt(bounds_path)
    
    def predict(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        lgk, y = super().predict(x)
        # return 10**y # Convert back to linear scale
        return 10**lgk, 10**y
    
    def predict_L1(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        lgk, y = super().predict_L1(x)
        # return 10**y # Convert back to linear scale
        return 10**lgk, 10**y
    
    def predict_L2(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        lgk, y = super().predict_L2(x)
        # return 10**y # Convert back to linear scale
        return 10**lgk, 10**y
    
# define doublefid: LF and HF
class doublefid:
    def __init__(self, path_LF, path_LH, device='cpu', k_region='A'): # k_region: 'A' or 'B', temporary solution
        self.device = torch.device(device)

        # Load the saved model dictionary
        self.model_LF = load_model(path_LF, device=self.device)
        self.model_HF = load_model(path_LH, device=self.device)
        self.model_LF.eval()
        self.model_HF.eval()

        k_mid = 32
        if k_region == 'A':
            self.lgk = np.loadtxt("./data/pre_N_L-H_stitch_z0/kf.txt")[:k_mid] # use this for now, will be replaced later
        else:
            self.lgk = np.loadtxt("./data/pre_N_L-H_stitch_z0/kf.txt")[k_mid:]
    
    def predict(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y_LF = self.model_LF(x_tensor)
        x_LH = torch.cat((x_tensor, y_LF), dim=1)
        y_HF = self.model_HF(x_LH)

        return self.lgk, y_HF.detach().cpu().numpy()
    
    def predict_LF(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y = self.model_LF(x_tensor).detach().cpu().numpy()

        return self.lgk, y
    
# define gokunet_df based on doublefid
class gokunet_df(doublefid):
    def __init__(self, path_LF, path_LH, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt", k_region='A'):
        super().__init__(path_LF, path_LH, device=device, k_region=k_region)
        self.bounds = np.loadtxt(bounds_path)
    
    def predict(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        lgk, y = super().predict(x)
        # return 10**y # Convert back to linear scale
        return 10**lgk, 10**y
    
    def predict_LF(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        lgk, y = super().predict_LF(x)
        # return 10**y # Convert back to linear scale
        return 10**lgk, 10**y

# define gokunet-split based on gokunet_df
class gokunet_split():
    def __init__(self, path_LA, path_HA, path_LB, path_HB, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.part_A = gokunet_df(path_LA, path_HA, device=device, bounds_path=bounds_path, k_region='A')
        self.part_B = gokunet_df(path_LB, path_HB, device=device, bounds_path=bounds_path, k_region='B')

    def predict_LA(self, x):
        return self.part_A.predict_LF(x)
    
    def predict_LB(self, x):
        return self.part_B.predict_LF(x)
    
    def predict(self, x):
        k_A, y_A = self.part_A.predict(x)
        k_B, y_B = self.part_B.predict(x)
        # concatenate the results
        k = np.concatenate((k_A, k_B))
        y = np.concatenate((y_A, y_B), axis=1)
        return k, y

