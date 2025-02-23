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
    # load lgk
    lgk = checkpoint['lgk']

    return lgk, model

class singlefid:
    def __init__(self, path, device='cpu'):
        self.device = torch.device(device)

        # Load the saved model dictionary
        self.lgk, self.model = load_model(path, device=self.device)
        self.model.eval()

    def predict(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y = self.model(x_tensor)

        return self.lgk, y.detach().cpu().numpy()

# define doublefid: LF and HF
class doublefid:
    def __init__(self, path_LF, path_LH, device='cpu'): # k_region: 'A' or 'B', temporary solution
        self.device = torch.device(device)

        # Load the saved model dictionary
        self.lgk, self.model_LF = load_model(path_LF, device=self.device)
        _, self.model_HF = load_model(path_LH, device=self.device)
        self.model_LF.eval()
        self.model_HF.eval()
    
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

    
# define gokunet_df based on doublefid
class gokunet_df(doublefid):
    def __init__(self, path_LF, path_LH, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        super().__init__(path_LF, path_LH, device=device)
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
        self.part_A = gokunet_df(path_LA, path_HA, device=device, bounds_path=bounds_path)
        self.part_B = gokunet_df(path_LB, path_HB, device=device, bounds_path=bounds_path)

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

# class gokunet_sf
class gokunet_sf(singlefid):
    def __init__(self, path, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt", output_lg=False):
        super().__init__(path, device=device)
        self.bounds = np.loadtxt(bounds_path)
        self.output_lg = output_lg
    
    def predict(self, x):
        # Normalize the input data
        dimx_original = self.bounds.shape[0]
        # print('before normalization',x)
        # copy the original x
        x_norm = np.copy(x)
        x_norm[:,:dimx_original] = (x_norm[:,:dimx_original] - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        # x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        # print('after normalization',x_norm)
        lgk, y = super().predict(x_norm)
        # return 10**y # Convert back to linear scale
        if self.output_lg:
            print('output lg: True')
            print("one of the y values larger than 1000") if np.any(y>1000) else None
            return lgk, y
        else:
            return 10**lgk, 10**y
    
# define gokunet_alpha; A is trained separately, B is trained with range A included (L2); LH part: XL1A-HA, XL1AL2B-H(B)
class gokunet_alpha():
    def __init__(self, path_LA, path_HA, path_L2, path_LH, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.part_A = gokunet_df(path_LA, path_HA, device=device, bounds_path=bounds_path)
        self.model_L2 = gokunet_sf(path_L2, device=device, bounds_path=bounds_path)
        self.model_LH = gokunet_sf(path_LH, device=device, bounds_path=bounds_path)

    def predict_LA(self, x):
        return self.part_A.predict_LF(x)
    
    def predict_L2(self, x):
        return self.model_L2.predict(x)
    
    def predict(self, x):
        k_A, y_A = self.part_A.predict(x)
        k_L2, y_L2 = self.model_L2.predict(x)

        lgy_A = np.log10(y_A) 
        lgy_L2 = np.log10(y_L2) 
        x_xL1AL2B = np.concatenate((x, lgy_A, lgy_L2[:,lgy_L2.shape[1]//2:]), axis=1) # temporary solution 
        _, y_H_forB = self.model_LH.predict(x_xL1AL2B)
        # _, y_H_forB = self.model_LH.predict(x)
        # combine with y_A
        y = np.concatenate((y_A, y_H_forB[:, (y_H_forB.shape[1])//2:]), axis=1)
        k = np.concatenate((k_A, k_L2[len(k_L2)//2:]))  # temporary solution
        return k, y
    
# alpha 3-step version
class gokunet_alpha3s():
    def __init__(self, path_LA, path_HAlin, path_HAnonl, path_L2, path_LHlin, path_LHnonl, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.model_LA = gokunet_sf(path_LA, device=device, bounds_path=bounds_path, output_lg=True)
        self.model_HA_lin = gokunet_sf(path_HAlin, device=device, bounds_path=bounds_path, output_lg=True)
        self.model_HA_nonl = gokunet_sf(path_HAnonl, device=device, bounds_path=bounds_path)
        self.model_L2 = gokunet_sf(path_L2, device=device, bounds_path=bounds_path, output_lg=True)
        self.model_LH_lin = gokunet_sf(path_LHlin, device=device, bounds_path=bounds_path, output_lg=True)
        self.model_LH_nonl = gokunet_sf(path_LHnonl, device=device, bounds_path=bounds_path)


    def predict_LA(self, x):
        lgk, lgy = self.model_LA.predict(x)
        return 10**lgk, 10**lgy
    
    def predict_L2(self, x):
        lgk, lgy = self.model_L2.predict(x)
        return 10**lgk, 10**lgy
    
    def predict(self, x):
        # part A
        k_A, y_LA = self.model_LA.predict(x)
        x_xLA = np.concatenate((x, y_LA), axis=1)
        _, y_HA_lin = self.model_HA_lin.predict(x_xLA)
        x_xLAlin = np.concatenate((x_xLA, y_HA_lin), axis=1)
        _, y_A = self.model_HA_nonl.predict(x_xLAlin)

        # part B
        k_L2, y_L2 = self.model_L2.predict(x)
        x_xLAB = np.concatenate((x, y_LA, y_L2[:, y_L2.shape[1]//2:]), axis=1)
        # print('x_xLAB',x_xLAB[0])
        _, y_LH_lin = self.model_LH_lin.predict(x_xLAB)
        x_xLABlin = np.concatenate((x_xLAB, y_LH_lin), axis=1)
        # print x_xLABlin all elements 
        print('x_xLABlin',x_xLABlin)
        print(x_xLABlin[0])
        _, y_H_forB = self.model_LH_nonl.predict(x_xLABlin)
        print('y_H_forB',y_H_forB)
        # concatenate the results
        k = np.concatenate((10**k_A, 10**k_L2[len(k_L2)//2:]))  # temporary solution
        y = np.concatenate((y_A, y_H_forB[:, (y_H_forB.shape[1])//2:]), axis=1)
        # y = np.concatenate((y_A, 10**y_LH_lin[:, (y_H_forB.shape[1])//2:]), axis=1)
        # y = np.concatenate((10**y_HA_lin, 10**y_LH_lin[:, (y_H_forB.shape[1])//2:]), axis=1)
        return k, y

# beta based on split
class gokunet_beta():
    def __init__(self, path_LA, path_HA, path_L2, path_HB, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.part_A = gokunet_df(path_LA, path_HA, device=device, bounds_path=bounds_path)
        self.model_L2 = gokunet_sf(path_L2, device=device, bounds_path=bounds_path)
        self.model_HB = gokunet_sf(path_HB, device=device, bounds_path=bounds_path)

    def predict_LA(self, x):
        return self.part_A.predict_LF(x)
    
    def predict_L2(self, x):
        return self.model_L2.predict(x)
    
    def predict(self, x):
        k_A, y_A = self.part_A.predict(x)
        
        k_2, y_2 = self.model_L2.predict(x)
        # print('x',x)
        lgy_2 = np.log10(y_2)  # lg values used in training
        x_xL2B = np.concatenate((x, lgy_2[:,y_2.shape[1]//2:]), axis=1) # temporary solution
        k_B, y_B = self.model_HB.predict(x_xL2B)
        # print('x_xL2B',x_xL2B)
        # concatenate the results
        k = np.concatenate((k_A, k_B))
        y = np.concatenate((y_A, y_B), axis=1)
        return k, y

# define gokunet_gamma; A is trained separately, B is trained with range A included (L2); LH part: XL1A-HA, XL2-H(B)
class gokunet_gamma():
    def __init__(self, path_LA, path_HA, path_L2, path_L2H, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.part_A = gokunet_df(path_LA, path_HA, device=device, bounds_path=bounds_path)
        self.part_B = gokunet_df(path_L2, path_L2H, device=device, bounds_path=bounds_path)

    def predict_LA(self, x):
        return self.part_A.predict_LF(x)
    
    def predict_L2(self, x):
        return self.part_B.predict_LF(x)
    
    def predict(self, x):  # combine the A result with the last half of B
        k_A, y_A = self.part_A.predict(x)
        k_B, y_B = self.part_B.predict(x)
        # concatenate the results
        k = np.concatenate((k_A, k_B[len(k_B)//2:]))  # temporary solution
        y = np.concatenate((y_A, y_B[:, len(k_B)//2:]), axis=1)
        return k, y

# LAB and LH for B
class gokunet_kappa(gokunet_gamma):
    def __init__(self, path_LA, path_HA, path_L, path_LH, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        super().__init__(path_LA, path_HA, path_L, path_LH, device=device, bounds_path=bounds_path)
    
    def predict(self, x): 
        return super().predict(x)
    
    def predict_L(self, x):
        return super().predict_L2(x)
    
    def predict_LA(self, x):
        return super().predict_LA(x)
    
# LAB for A and B, LH for B
class gokunet_lambda():
    def __init__(self, path_L, path_HA, path_LH, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.model_L = gokunet_sf(path_L, device=device, bounds_path=bounds_path)
        self.model_HA = gokunet_sf(path_HA, device=device, bounds_path=bounds_path)
        self.part_B = gokunet_df(path_L, path_LH, device=device, bounds_path=bounds_path)
    
    def predict_LA(self, x):
        k_L, y_L = self.model_L.predict(x)
        k_A = k_L[:len(k_L)//2]
        y_A = y_L[:,:y_L.shape[1]//2]
        return k_A, y_A
    
    def predict_LB(self, x):
        k_L, y_L = self.model_L.predict(x)
        k_B = k_L[len(k_L)//2:]
        y_B = y_L[:,y_L.shape[1]//2:]
        return k_B, y_B
    
    def predict(self, x):
        # part A
        k_A, y_LA = self.predict_LA(x)
        x_xLA = np.concatenate((x, np.log10(y_LA)), axis=1)
        _, y_HA = self.model_HA.predict(x_xLA)

        # part B
        k_B, y_H_forB = self.part_B.predict(x)
        # concatenate the results
        k = np.concatenate((k_A, k_B[len(k_B)//2:]))  # temporary solution
        y = np.concatenate((y_HA, y_H_forB[:, (y_H_forB.shape[1])//2:]), axis=1)
        return k, y
