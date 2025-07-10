# T2N-MusE

The Triple-2 neural Network Multifidelity cosmological Emulation framework (T2N-MusE) is designed to enable efficient training of neural networks for regression.

## Dependencies

pytorch, numpy

## Usage

Prepare training data and then use `hyper_optim.py' to optimize hyperparameters: the number of hidden layers, the number of neurons per layer (i.e., layer width/hidden size), and the strength of regularization (lambda). Each trial includes k-fold training and validation for the set of hyperparameters evaluated at. Below introduction to the parameters that be specified for the program:

### Data

Specify input and output data (can be txt or npy files):

```
--data_x=/path/to/input
--data_y=/path/to/output
```

Lower and upper bounds of input (used for normalization):
```
--bound_x=input_limit.txt
```

### Hyperparameter optimization

#### Stage 1

Set the number of trials:
```
--trials=80
```

Customize the search space if needed (or use the default space):
```
--min_layers=1
--max_layers=7
--min_hidden_size=16
--max_hidden_size=512
--min_lambda=1e-9
--max_lambda=5e-6
```

#### Stage 2 (Optional)

Fine-tuning around the best point found from stage 1.
Number of trials:
```
--trials_fine=40
```

Optional:

If you want to do the fine-tuning without the first stage, add the flag:
```
--fine_only
```
and remember to specify the point you want to search around:
```
--num_layers=[ ]
--hidden_size=[ ]
--decay=[ ]
```
If you want to change the size of fine-tuning space:
```
--f_lambda_fine=3
--r_hidden_size_fine=24
```

### k-fold parameters

#### Regular
The number of folds:
```
--kfolds=5
```

#### 2-Phase strategy

Besides the number of folds, a flag should be added, and the folds you want to test against need to be specified:
```
--k2r
--test_folds="6,7,9"
```

# Other parameters for training

The number of random seeds you want to try for initializing each training run:
```
--trials_train=15
```

Initial learning rate:
```
--lr=0.2
```

(Optional, no need to change usually) The maximum number of epochs:
```
--epochs=[ ]
```
or set an upper limit that depends on the complexity of the NN via specifying the number of epochs per neuron:
```
---epochs_neuron=1000
```

Activation function:
```
--activation=SiLU
```

### PCA
Minimum explained variance ratio of the PCA:
```
--min_pca=0.99999
```

### Model directory
Where to save the models:
```
--model_dir=models
```

### Other parameters/flags
Wavenumbers (not for training, just save in the model file for the convenience of use):
```
--lgk=kf.txt
```

Save the k-fold models and the final model trained on the whole set:
```
--save_kfold
--save_best
```

Zero-center the output data (may not be needed if using PCA):
```
--zero_centering
```

### Train with given hyperparameters

If you want to train the NN with certain hyperparameters (without Bayesian optimization), use 
```
--train_one
```
and remember to specify the hyperparameters (see Section Hyperparameter optimization Stage 2).

If you want to retrain the NN (without changing hyperparameters), add
```
--retrain
```
and remember to specify the model directory.

### Example commands