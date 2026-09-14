# Week 3 Predictor Training

`probe.data.train_week3_predictors` trains success-probability heads on the
post-processed Week 3 dataset.

## Model

Every head returns one raw logit. The reported probability is
`sigmoid(logit)`. Training uses the aggregate binomial negative log likelihood:

```text
- sum[s * log(sigmoid(logit)) + (N-s) * log(1-sigmoid(logit))] / sum(N)
```

The implementation evaluates this through `logsigmoid`, without explicitly
taking `log(sigmoid(logit))`.

The default run trains only the nonlinear MLP:

```text
mlp:    input -> 64 -> GELU -> 32 -> GELU -> 1
```

The MLP has dropout `0.1`, AdamW optimization, a maximum of 200 epochs, and
validation early stopping with patience 40. Three seeds are run by default;
the seed with the lowest validation NLL is selected for the main predictions.
The linear model remains available with `--model-type linear` or
`--model-type both`, but is not trained by the default command.

## Feature variants

The command supports:

```text
A       frozen SigLIP observation/instruction feature
B       one-call VLA hidden feature
S_K     eight action-distribution statistics at prefix K
B+S_K   B concatenated with S_K
```

For A and B, standardization and PCA are fitted using the training split only.
The default PCA output is 64 dimensions. Passing `--pca-dim 0` disables PCA and
keeps the standardized feature at its raw dimension. S is standardized without
PCA. For B+S, PCA is applied only to the B block and the S block is appended
afterward; `--pca-dim 0` disables PCA for the B block as well.
The fixed `splits.json` group assignment is reused; original, better, and worse
conditions therefore stay in the same split.

## Server command

```bash
export PROBE_ROOT=/home/nvidia/yutao/lyh/PROBE
source "$PROBE_ROOT/.venv-qwen/bin/activate"
export PYTHONPATH="$PROBE_ROOT"

python -m probe.data.train_week3_predictors \
  --dataset-dir "$PROBE_ROOT/data/week3_all_suites" \
  --output-dir "$PROBE_ROOT/data/week3_training" \
  --methods A B S B+S \
  --s-k 4 8 16 32 \
  --pca-dim 64 \
  --model-type mlp \
  --hidden-dims 64,32 \
  --dropout 0.1 \
  --max-epochs 200 \
  --patience 40 \
  --seeds 0 1 2 \
  --device auto
```

The active environment must contain PyTorch and NumPy. The training script
does not call the VLA or the simulator.

## Outputs

```text
week3_training/
  summary.json
  predictions.jsonl
  selection_results.jsonl
  preprocessors/
  models/<method>/<linear|mlp>/seed<seed>/
    model.pt
    history.json
    metrics.json
```

Each `history.json` row contains `epoch`, `train_nll`, `validation_nll`,
`test_nll`, and train/validation/test tie-aware accuracy. Test NLL and test
tie-aware accuracy are recorded for plotting and diagnostics only; neither is
used for early stopping or seed selection.

To plot the selected-seed curves for the first three experiments and the
average curve across all supplied experiments:

```bash
python -m probe.data.plot_week3_loss_curves \
  --run-dir "$PROBE_ROOT/data/week3_training_tie_aware" \
  "$PROBE_ROOT/data/week3_training_tie_aware_s345" \
  "$PROBE_ROOT/data/week3_training_tie_aware_s678" \
  "$PROBE_ROOT/data/week3_training_tie_aware_s091011" \
  "$PROBE_ROOT/data/week3_training_tie_aware_s121314" \
  "$PROBE_ROOT/data/week3_training_tie_aware_s151617" \
  "$PROBE_ROOT/data/week3_training_tie_aware_s181920" \
  "$PROBE_ROOT/data/week3_training_tie_aware_s212223" \
  "$PROBE_ROOT/data/week3_training_tie_aware_s242526" \
  --output-dir "$PROBE_ROOT/data/week3_loss_curves"
```

The average is computed separately at each epoch using only runs that reached
that epoch, because early stopping can make history lengths different.

To plot the corresponding train, validation, and test tie-aware accuracy
curves:

```bash
python -m probe.data.plot_week3_tie_accuracy_curves \
  --run-dir "$PROBE_ROOT/data/week3_loss_curve_runs/week3_training_tie_aware" \
  "$PROBE_ROOT/data/week3_loss_curve_runs/week3_training_tie_aware_s345" \
  "$PROBE_ROOT/data/week3_loss_curve_runs/week3_training_tie_aware_s678" \
  "$PROBE_ROOT/data/week3_loss_curve_runs/week3_training_tie_aware_s091011" \
  "$PROBE_ROOT/data/week3_loss_curve_runs/week3_training_tie_aware_s121314" \
  "$PROBE_ROOT/data/week3_loss_curve_runs/week3_training_tie_aware_s151617" \
  "$PROBE_ROOT/data/week3_loss_curve_runs/week3_training_tie_aware_s181920" \
  "$PROBE_ROOT/data/week3_loss_curve_runs/week3_training_tie_aware_s212223" \
  "$PROBE_ROOT/data/week3_loss_curve_runs/week3_training_tie_aware_s242526" \
  --model-kind mlp \
  --first-n 3 \
  --output-dir "$PROBE_ROOT/data/week3_tie_accuracy_curves"
```

The selected seed is the same validation-NLL-selected seed used by the loss
curve script. Accuracy curves do not change seed or epoch selection.

The selected predictions contain the record ID, group, condition, split,
method, selected seed, predicted probability, and the original `successes`
and `trials`. `selection_results.jsonl` reports fixed-split condition
selection and empirical regret; it is an offline diagnostic, not an online
execution result. If multiple conditions have the same highest observed
success rate, all of them are included in `oracle_conditions` and count as
correct selections; `oracle_condition` remains a deterministic representative
for compatibility.
