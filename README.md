# T2N-MusE

**T2N-MusE** (Triple-2 Neural Network Multifidelity Cosmological Emulation Framework) enables efficient neural network training for regression tasks in cosmological modeling.

## Dependencies

- `pytorch`
- `numpy`

## Usage

First, prepare your training data, then run `hyper_optim.py` to perform hyperparameter optimization. The script tunes:

- Number of hidden layers  
- Number of neurons per layer (hidden size)  
- Regularization strength (lambda)

Each trial includes k-fold training and validation for the selected hyperparameter set.

---

### Data Input

Specify input and output datasets (`.txt` or `.npy`):

```bash
--data_x=/path/to/input
--data_y=/path/to/output
```

Input normalization bounds (required):

```bash
--bound_x=input_limit.txt
```

---

### Hyperparameter Optimization

#### Stage 1: Coarse Search

Set the number of optimization trials:

```bash
--trials=80
```

(Optional) Customize the search space:

```bash
--min_layers=1
--max_layers=7
--min_hidden_size=16
--max_hidden_size=512
--min_lambda=1e-9
--max_lambda=5e-6
```

#### Stage 2: Fine-Tuning

Fine-tune around the best configuration found in Stage 1.

Set number of fine-tuning trials (default: `trials/2`):

```bash
--trials_fine=40
```

Run fine-tuning *only*, skipping Stage 1:

```bash
--fine_only
--num_layers=[ ]
--hidden_size=[ ]
--decay=[ ]
```

Customize the fine-tuning region:

```bash
--f_lambda_fine=3
--r_hidden_size_fine=24
```

---

### K-Fold Cross-Validation

#### Standard K-Fold

Set number of folds:

```bash
--kfolds=5
```

#### 2-Phase Strategy

Enable 2-phase training and specify test folds:

```bash
--k2r
--test_folds="6,7,9"
```

---

### Training Parameters

Set number of training trials per hyperparameter configuration (used only in phase 1 for 2-phase):

```bash
--trials_train=15
```

Initial learning rate:

```bash
--lr=0.2
```

Maximum training epochs (optional):

```bash
--epochs=[ ]
```

Or specify epoch count per neuron:

```bash
--epochs_neuron=1000
```

Activation function:

```bash
--activation=SiLU
```

---

### PCA Compression

Set the minimum explained variance ratio:

```bash
--min_pca=0.99999
```
Local (per-z) PCA is used by default. If global PCA is desired:
```
--pca_allz
```


---

### Output Directory

Specify where models will be saved:

```bash
--model_dir=models
```

---

### Miscellaneous Options

Attach metadata (e.g., wavenumbers, also used to determine the number of redshift bins):

```bash
--lgk=kf.txt
```

Save models from each fold and the best full-set model:

```bash
--save_kfold
--save_best
```

Zero-center output data (optional when PCA is used):

```bash
--zero_centering
```

---

### Direct Training (No Optimization)

Train with user-defined hyperparameters:

```bash
--train_one
--num_layers=[ ] --hidden_size=[ ] --decay=[ ]
```

Retrain using saved configuration:

```bash
--retrain
--model_dir=models/your_model
```

---

## Example Commands

Train a low-fidelity model with 2-phase training:

```bash
python hyper_optim.py --k2r --trials=80 --trials_train=15 \
--data_x=./data/muse_L2/train_input.txt \
--data_y=./data/muse_L2/train_output.txt \
--bound_x=./data/input_limits-W.txt \
--save_kfold --save_best \
--model_dir=models/muse-All-2_L2 \
--lr=0.01 --kfolds=564 \
--lgk=./data/muse_L2/kf.txt \
--zero_centering \
--test_folds="24, 25, 26, 54, 55, 56, 72, 73, 74, 207, 208, 209, 240, 241, 242, 300, 301, 302, 522, 523, 524" \
--min_pca=0.99999 &> muse-All-2_L2.log &
```

Train a low- to high-fidelity correction model (NN_LH):

```bash
python hyper_optim.py --trials=80 \
--data_x=./data/muse_L2Hr/train_input.txt \
--data_y=./data/muse_L2Hr/train_output.txt \
--bound_x=./data/input_limits-W.txt \
--save_kfold --save_best \
--model_dir=models/muse-HO-2_L2Hr \
--lr=0.01 --kfolds=21 \
--lgk=./data/muse_L2Hr/kf.txt \
--zero_centering \
--trials_train=5 \
--min_pca=0.99999 &> muse-HO-2_L2Hr.log &
```

---

## Additional Resources

A Jupyter notebook for leave-one-out cross-validation (LOOCV) is available at:

```
loocv/muse-All-2.ipynb
```

This corresponds to the **Optimal** model discussed in the T2N-MusE paper.

