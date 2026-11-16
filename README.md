# APPR_project

## Prepare training data

The raw training archives in `data/raw/` can be combined into one CSV per
training set with:

```powershell
python src\appr_project\extract_data.py
```

This creates `data/csv/set_A.csv` and `data/csv/set_B.csv`.
Each PSV observation is preserved, with `patient_id` (for example,
`A_000001`) and `source_set` (`A` or `B`) added to the end of the row. To use a
different data directory containing `raw/` and `csv/`, pass
`--data-dir <path>`.

## Exploratory analysis (no preprocessing)

Install the development dependencies with `uv sync --group dev`, then select
the project `.venv` Python kernel and open `notebooks/exploration.ipynb`.
The notebook compares all recorded values in A and B, audits documented units,
and examines missingness, patient-level label prevalence, temporal integrity,
distributions and associations. It does not modify either CSV.

To print the complete per-column unit audit and refresh the observed-evidence
appendix in `docs/data_dictionary.md`:

```powershell
uv run python scripts/audit_units.py --update-dictionary
```

The notebook and audit script share the diagnostics in `src/appr_project/eda.py`.
Observed distributions can reveal scale discrepancies but cannot independently
certify measurement units. Suspected differences remain explicitly unconfirmed.

## Train the Keras tower model

Run `notebooks/cleaning.ipynb` through export to create the four CSVs under
`data/processed/train` and `data/processed/test`. Install the modeling dependencies
with `uv sync --extra model` using the project's Python 3.13 environment.

For a timing pilot with an 80/20 patient-level train/validation split:

```powershell
uv run --extra model python scripts/train.py --source A --sample-patients 300 --epochs 2
```

Choose `A`, `B`, or `A+B` with `--source`. With random search, omit
`--sample-patients` to use the full training partition. Pass
`--architecture semantic_grud` to train five
multivariate GRU-D towers with temporal attention and patient context. The
default `tower_grud` uses one tower per variable. Add `--trials 8` to try the base
configuration plus seven random alternatives for window length, GRU-D units, dense units, dropout,
learning rate, and batch size (64, 128, or 256). Set the base configuration's
batch size with `--batch-size`. Trials reuse the same patients in each partition. Epochs are
selected by validation PR AUC and trials by validation average precision. Add
`--folds 5` to use stratified CV instead of holdout. CV ranks trials by mean
fold average precision, selects an alert threshold on out-of-fold predictions,
and refits on the CV cohort for the median number of selected epochs.

To run the 2,000-patient TPE search with median pruning:

```powershell
uv run --extra model python scripts/train.py --architecture semantic_grud_3 --search tpe
```

Use `semantic_grud` for five towers. By default, the TPE pilot runs 20 trials
of up to 30 epochs with early stopping patience 4. It evaluates the base first,
then uses TPE after 10 startup trials. It searches windows of 8, 12, or 16 hours.
Median pruning checks validation PR AUC from trial 6 and epoch 8 onward when at
least three completed trials reported at that epoch. The trial CSV records the
state, epochs trained, and fit time for pruned trials. Only completed trials
receive an average precision score. To cross-validate the selected configuration,
pass its hyperparameters with `--search random --trials 1 --folds 3` and omit
`--sample-patients` to use the full development partition.

The model scores each recorded hour using only information available by that
hour. The input pipeline fits its scaler on training observations and builds
windows with separate masks for missing measurements and left padding. The
holdout model is saved from its best validation epoch. Each run stores
`best.keras`, `config.json`, patient assignments, validation predictions,
metrics, and training timings under `runs/`. The configuration records the
temporal variables in each tower, in model input order. `run_timing.json` records
elapsed seconds from the start of the experiment through the validation exports,
including the hyperparameter search and any CV refit. The validation metrics describe
model selection and threshold tuning. Use the reserved test splits to assess
the final model.

```powershell
uv run --extra model python scripts/evaluate.py 'runs/<run-directory>' --source 'A+B'
```

Evaluation reloads the saved scaler and alert threshold and reports metrics pooled
and by source. In a holdout run, the threshold is selected on validation
predictions. In CV, it is selected on pooled out-of-fold predictions. Test
utility and alert fraction use the saved threshold. Training and tuning only
read the training CSVs.

Validation, per-fold, and test metrics include `positive_hours` and
`positive_hour_fraction`. The latter is the average precision of a constant
score on the same set of hours. Compare models on the same patients and report
hourly average precision, AUROC, utility, and the fraction of hours with an alert.
`recall_at_threshold` measures the fraction of positive hours with an alert at the
reported threshold. It is undefined for cohorts with no positive hours.
