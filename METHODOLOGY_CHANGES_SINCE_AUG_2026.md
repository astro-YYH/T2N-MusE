# Post-paper methodology changes since August 2026

## Scope

This note records changes made after the methodology described in the paper. It is
intended to prevent the current repository behavior from being read as an exact
description of the paper methods.

For the comparison below:

- **Before** means the last repository commit before these changes,
  `686bb53` (2025-11-12).
- **Now** means the post-change implementation at `de9a280` (2026-08-27).

This is a comparison of the implementation history. It does not independently
verify which archived checkpoints or outputs were used for each paper figure.

## Essential changes

| Part of the method | Before (paper-era implementation) | Now (post-paper implementation) |
| --- | --- | --- |
| **Hyperparameter-optimization objective** | By default, Hyperopt minimized the average of mean training and validation losses, `(train_loss + val_loss) / 2`. Passing `--opt_val` changed the objective to validation loss alone. | Hyperopt always minimizes validation loss averaged over the selected folds. `--opt_val` is retained only as a hidden compatibility argument and no longer changes the objective. With `--hyperopt_validation_curve`, the objective is the minimum of the completed mean validation curve rather than the mean of the folds' individually restored validation losses. |
| **Loss controlling a training run** | The learning-rate scheduler and early stopping were driven by `validation_loss + regularized_training_loss`. The final epoch reached by the run was returned and, when requested, saved. | A configurable fold objective is used: `J_fold = (1 - w) L_train,reg + w L_val`, where `w = --fold_val_weight`. The default is `w = 0`, so scheduling, early stopping, and checkpoint selection use regularized training loss only. A run restores the weights from the epoch with the lowest `J_fold`, except when an exact validation-selected duration is explicitly required. If there is no validation set, `w` is forced to zero. |
| **Early-stopping rule** | Patience was fixed at 300 epochs and the patience counter reset after a relative improvement of at least `0.0005` (0.05%). | Patience and the relative-improvement threshold are configurable with `--early_stopping_patience` and `--early_stopping_fraction`. Their defaults are 300 epochs and `0.005` (0.5%), respectively. |
| **Selection among random seeds in standard K-fold training** | Multiple seeds were tried separately inside each fold. Each fold could therefore choose a different seed, selected by `training_loss + validation_loss` when validation data were present. | Seeds are evaluated outermost: each candidate seed is run across all selected folds, so one seed is used consistently across folds. The selected seed is the one with the lowest mean validation loss across those folds. |
| **Selection of final training duration** | There was no cross-fold validation-curve choice of epoch count. Final full-data training used the nominal epoch budget, early stopping, and an additional training-loss floor (`0.8` times the cross-validation training loss). | For the winning hyperparameters and seed, fold validation histories are aligned and averaged. The epoch at the minimum of this mean validation curve sets the full-data training budget, and early stopping is disabled for that fit. In default mode, the best fold-objective checkpoint within that budget is still restored; with `--hyperopt_validation_curve`, the model is kept exactly at the validation-selected epoch. Shorter fold histories are completed by resuming the raw model, optimizer, scheduler, early-stopping, and random-number-generator states up to the longest fold history. |
| **Initialization and learning rate of the final standard K-fold model** | A trained fold model whose regularized loss was closest to the mean fold loss was used to initialize the full-data model. Its terminal, scheduler-reduced learning rate was inherited. | Final full-data training starts again from the selected seed's random initialization and uses the original requested learning rate. It does not inherit fold-trained weights or a fold's terminal learning rate. |
| **Two-round (`--k2r`) final model** | After Round 2 cross-validation, a fold-adapted model was passed into full-data training, together with the selected fold learning rate and nominal/early-stopped duration. | Round 2 validation curves determine the final duration. Full-data Round 2 training starts from the shared Round 1 model and uses the Round 2 learning rate; it does not start from a fold-adapted Round 2 model. Round 1 has no equivalent fold-wise validation score, so its initialization is still selected using the regularized-training-based fold objective by default. |
| **Optional duration selection inside Hyperopt** | Not available. Duration selection was not part of a Hyperopt trial. | `--hyperopt_validation_curve` performs validation-curve duration selection inside every trial. Hyperopt receives the curve minimum, and the winning seed/epoch metadata (plus the Round 1 checkpoint and Round 2 learning rate for `--k2r`) is reused for final training. This avoids an additional post-Hyperopt selection pass and keeps final/saved models at exactly the selected epoch. |
| **Hyperopt trial bookkeeping** | Several best-loss calculations used `trials[:-1]`, which could omit the latest completed trial and produced `inf` when only one trial existed. This could also affect the comparison between the initial and fine-tuning searches. | Best loss is computed from all successful completed trials with finite losses. Failed, incomplete, missing, and non-finite results are ignored. |

## Current selection hierarchy

The current standard K-fold workflow deliberately uses different criteria for
different decisions:

1. **Within each fold run:** control the scheduler, early stopping, and restored
   checkpoint using `J_fold` (regularized training loss by default).
2. **Across random seeds:** select one seed using mean validation loss across
   folds.
3. **Across hyperparameters:** select the configuration using mean validation
   loss, or the minimum mean validation-curve loss when
   `--hyperopt_validation_curve` is enabled.
4. **For final full-data training:** reuse the selected seed, but start from its
   random initialization and use the validation-selected epoch count as the
   training budget. The optional in-Hyperopt curve mode keeps the terminal model
   at that exact epoch; default mode restores the best fold-objective checkpoint
   encountered within the budget.

Thus, validation data now directly determine seed, hyperparameter, and duration
selection, while the default within-run stopping signal remains regularized
training loss.

The resulting procedure is more stable and robust because seed quality is
assessed consistently across folds, training duration is selected from the mean
validation behavior, and the final fit no longer depends on one fold's trained
weights or terminal learning rate. Consequently, substantially fewer seed trials
are needed in practice. The current README recommendations reduce
`--trials_train` from 15 to 5 for the low-fidelity two-round example and from 5
to 2 for the high-fidelity example.

## Reproducibility implications

- Results produced by the current default pipeline are not methodologically
  identical to results produced by the pre-August implementation.
- The most consequential differences are the Hyperopt objective, consistent
  cross-fold seed selection, validation-selected final duration, and removal of
  fold-checkpoint inheritance for the final standard model.
- To inspect or run the repository exactly as it stood before these changes,
  check out `686bb53`. Confirm the exact checkpoint/command provenance separately
  before claiming numerical reproduction of a paper result.
- Current checkpoints contain more provenance fields, including validation loss,
  regularized training loss, fold objective and weight, requested/completed epoch
  counts, best epoch, and whether the best checkpoint was restored. Run arguments
  are also printed at startup.

## What did not change

The August 2026 update did not change the training data files, neural-network
architecture definition, base MSE data loss, AdamW optimizer family, K-fold split
construction, or the `ReduceLROnPlateau` scheduler type. The changes concern how
training runs are monitored and restored and how seeds, hyperparameters, and the
final training duration/initialization are selected.
