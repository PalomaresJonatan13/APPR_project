"""Holdout or cross-validation, hyperparameter search, and Keras artifacts."""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
from tensorflow import keras

from appr_project.data import (
    PatientHours,
    Standardization,
    WindowBatches,
    split_patients,
)
from appr_project.evaluation import predict_hours, select_threshold, summarize_metrics
from appr_project.models import semantic_grud, tower_grud, vs_gru

ARCHITECTURES = ("tower_grud", *semantic_grud.GROUPS_BY_ARCHITECTURE, "vs_gru")
WINDOW_CHOICES = (8, 12, 16)
UNIT_CHOICES = (8, 16, 32)
DENSE_CHOICES = (32, 64)
BATCH_CHOICES = (64, 128, 256)
DROPOUT_RANGE = (0.0, 0.3)
LEARNING_RATE_RANGE = (0.0001, 0.01)
TPE_STARTUP_TRIALS = 10
PRUNING_STARTUP_TRIALS = 5
PRUNING_WARMUP_STEPS = 7  # Keras numbers epochs from zero.
PRUNING_MIN_TRIALS = 3


@dataclass(frozen=True)
class Hyperparameters:
    window: int = 16
    units: int = 16
    dense_units: int = 64
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64
    use_hidden_dense: bool = False


def search_candidates(base, trials, seed, architecture="tower_grud"):
    """Evaluate the supplied configuration first, then sample distinct alternatives."""
    if trials < 1:
        raise ValueError("trials must be positive.")

    rng = np.random.default_rng(seed)
    candidates = [base]
    seen = {base}
    if architecture == "vs_gru" and trials > 1:
        alternate = replace(base, use_hidden_dense=not base.use_hidden_dense)
        candidates.append(alternate)
        seen.add(alternate)
    while len(candidates) < trials:
        window = int(rng.choice(WINDOW_CHOICES))
        if architecture == "vs_gru":
            model_options = {
                "units": base.units,
                "dense_units": base.dense_units,
                "dropout": base.dropout,
                "use_hidden_dense": bool(rng.integers(2)),
            }
        else:
            model_options = {
                "units": int(rng.choice(UNIT_CHOICES)),
                "dense_units": int(rng.choice(DENSE_CHOICES)),
                "dropout": rng.uniform(*DROPOUT_RANGE),
            }
        candidate = Hyperparameters(
            window=window,
            learning_rate=10.0 ** rng.uniform(*np.log10(LEARNING_RATE_RANGE)),
            batch_size=int(rng.choice(BATCH_CHOICES)),
            **model_options,
        )
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)
    return candidates


def _tpe_suggestions(trial, architecture, base):
    window = trial.suggest_categorical("window", WINDOW_CHOICES)
    if architecture == "vs_gru":
        model_options = {
            "use_hidden_dense": trial.suggest_categorical(
                "use_hidden_dense", [False, True]
            ),
        }
    else:
        model_options = {
            "units": trial.suggest_categorical("units", UNIT_CHOICES),
            "dense_units": trial.suggest_categorical("dense_units", DENSE_CHOICES),
            "dropout": trial.suggest_float("dropout", *DROPOUT_RANGE),
        }
    return replace(
        base,
        window=window,
        learning_rate=trial.suggest_float(
            "learning_rate", *LEARNING_RATE_RANGE, log=True
        ),
        batch_size=trial.suggest_categorical("batch_size", BATCH_CHOICES),
        **model_options,
    )


class _MedianPruningCallback(keras.callbacks.Callback):
    def __init__(self, trial):
        super().__init__()
        self.trial = trial
        self.pruned = False

    def on_epoch_end(self, epoch, logs=None):
        self.trial.report(float(logs["val_pr_auc"]), step=epoch)
        if self.trial.should_prune():
            self.pruned = True
            self.model.stop_training = True


def _fit(train, validation, hp, *, epochs, patience, seed, architecture, pruning=None):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    started = perf_counter()
    standardization = Standardization.fit(train)
    hours = PatientHours(train, standardization)
    train_batches = WindowBatches(
        hours, hp.window, hp.batch_size, shuffle=True, seed=seed
    )
    val_batches = (
        WindowBatches(
            PatientHours(validation, standardization), hp.window, hp.batch_size
        )
        if validation is not None
        else None
    )
    model_options = {
        "static_features": hours.static.shape[-1],
        "window": hp.window,
        "units": hp.units,
        "dense_units": hp.dense_units,
        "dropout": hp.dropout,
    }
    if architecture == "vs_gru":
        model = vs_gru.build_model(
            variables=len(standardization.temporal),
            static_features=hours.static.shape[-1],
            window=hp.window,
            use_hidden_dense=hp.use_hidden_dense,
        )
    elif architecture in semantic_grud.GROUPS_BY_ARCHITECTURE:
        model = semantic_grud.build_model(
            temporal_variables=standardization.temporal,
            architecture=architecture,
            **model_options,
        )
    else:
        model = tower_grud.build_model(
            variables=len(standardization.temporal), **model_options
        )
    model.compile(
        optimizer=keras.optimizers.Adam(hp.learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc", num_thresholds=1000)],
    )
    preparation_seconds = perf_counter() - started
    callbacks = []
    if val_batches is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=patience,
                restore_best_weights=True,
            )
        )
    if pruning is not None:
        callbacks.append(pruning)
    started = perf_counter()
    history = model.fit(
        train_batches,
        validation_data=val_batches,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=2,
    ).history
    fit_seconds = perf_counter() - started
    best_epoch = (
        int(np.argmax(history["val_pr_auc"])) + 1 if val_batches is not None else epochs
    )
    timing = {
        "train_windows": len(train),
        "validation_windows": len(validation) if validation is not None else 0,
        "preparation_seconds": preparation_seconds,
        "fit_seconds": fit_seconds,
        "epochs_run": len(history["loss"]),
        "best_epoch": best_epoch,
        "seconds_per_epoch": fit_seconds / len(history["loss"]),
    }
    predictions = (
        predict_hours(model, val_batches)
        if val_batches is not None and (pruning is None or not pruning.pruned)
        else None
    )
    return model, standardization, predictions, timing


def _write_model(run_dir, model, standardization, hp, threshold, settings):
    model.save(run_dir / "best.keras")
    groups = semantic_grud.GROUPS_BY_ARCHITECTURE.get(model.name)
    tower_variables = (
        list(groups.values())
        if groups is not None
        else [[variable] for variable in standardization.temporal]
    )
    metadata = {
        "architecture": model.name,
        "tower_variables": tower_variables,
        "hyperparameters": asdict(hp),
        "standardization": asdict(standardization),
        "threshold": threshold,
        **settings,
    }
    (run_dir / "config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def train_experiment(
    frame,
    run_dir: Path,
    *,
    source,
    base=None,
    trials=None,
    folds,
    validation_fraction,
    sample_patients,
    epochs,
    patience,
    seed,
    architecture="tower_grud",
    candidates=None,
    patient_splits=None,
    search="random",
):
    """Evaluate explicit candidates, random search, or TPE with holdout pruning."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    if search not in ("random", "tpe"):
        raise ValueError(f"Unknown search method: {search}")
    run_started = perf_counter()
    use_tpe = search == "tpe"
    if use_tpe:
        if candidates is not None or folds is not None:
            raise ValueError("TPE pruning requires a search with one validation split.")
        if base is None or trials is None or trials < 1:
            raise ValueError("Supply a base configuration and a positive trial count.")
    elif candidates is None:
        if base is None or trials is None:
            raise ValueError("Supply base and trials for a random search.")
        candidates = dict(
            enumerate(search_candidates(base, trials, seed, architecture), start=1)
        )
    if not use_tpe and not candidates:
        raise ValueError("Supply at least one trial.")
    splits = patient_splits
    if splits is None:
        splits = list(
            split_patients(
                frame,
                validation_fraction=validation_fraction,
                folds=folds,
                seed=seed,
                sample_patients=sample_patients,
            )
        )
    if use_tpe:
        if len(splits) != 1:
            raise ValueError("TPE pruning requires exactly one validation split.")
        import optuna

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=seed, n_startup_trials=TPE_STARTUP_TRIALS
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=PRUNING_STARTUP_TRIALS,
                n_warmup_steps=PRUNING_WARMUP_STEPS,
                n_min_trials=PRUNING_MIN_TRIALS,
            ),
        )
        base_in_space = (
            base.window in WINDOW_CHOICES
            and base.batch_size in BATCH_CHOICES
            and LEARNING_RATE_RANGE[0] <= base.learning_rate <= LEARNING_RATE_RANGE[1]
            and (
                architecture == "vs_gru"
                or (
                    base.units in UNIT_CHOICES
                    and base.dense_units in DENSE_CHOICES
                    and DROPOUT_RANGE[0] <= base.dropout <= DROPOUT_RANGE[1]
                )
            )
        )
        if base_in_space:
            params = asdict(base)
            if architecture == "vs_gru":
                for name in ("units", "dense_units", "dropout"):
                    del params[name]
            else:
                del params["use_hidden_dense"]
            study.enqueue_trial(params)

    total_trials = trials if use_tpe else len(candidates)
    trial_ids = range(1, total_trials + 1) if use_tpe else candidates
    run_dir.mkdir(parents=True, exist_ok=False)
    assignments = []
    for fold, (train_ids, val_ids) in enumerate(splits, start=1):
        for partition, ids in (("train", train_ids), ("validation", val_ids)):
            assignments.append(
                pd.DataFrame({"fold": fold, "partition": partition, "patient_id": ids})
            )
    pd.concat(assignments, ignore_index=True).to_csv(
        run_dir / "splits.csv", index=False
    )

    settings = {
        "source": source,
        "folds": folds,
        "validation_fraction": validation_fraction,
        "sample_patients": sample_patients,
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
        "trials": total_trials,
    }
    if use_tpe:
        settings.update(
            search="tpe",
            tpe_startup_trials=TPE_STARTUP_TRIALS,
            pruning_startup_trials=PRUNING_STARTUP_TRIALS,
            pruning_first_epoch=PRUNING_WARMUP_STEPS + 1,
            pruning_min_trials=PRUNING_MIN_TRIALS,
        )
    trial_rows, fold_rows = [], []
    best_score = -np.inf
    for position, trial in enumerate(trial_ids, start=1):
        pruning = None
        if use_tpe:
            optuna_trial = study.ask()
            # Keep an explicitly supplied reference outside the search space fixed.
            hp = (
                base
                if position == 1 and not base_in_space
                else _tpe_suggestions(optuna_trial, architecture, base)
            )
            pruning = _MedianPruningCallback(optuna_trial)
        else:
            hp = candidates[trial]
        predictions, scores, best_epochs = [], [], []
        for fold, (train_ids, val_ids) in enumerate(splits, start=1):
            print(
                f"Trial {trial} ({position}/{total_trials}), "
                f"fold {fold}/{len(splits)}: {hp}",
                flush=True,
            )
            train = frame.loc[frame.patient_id.isin(train_ids)]
            validation = frame.loc[frame.patient_id.isin(val_ids)]
            if train.SepsisLabel.nunique() < 2 or validation.SepsisLabel.nunique() < 2:
                raise ValueError(
                    "Training and validation need both labels; use a larger patient sample."
                )
            model, standardization, val, timing = _fit(
                train,
                validation,
                hp,
                epochs=epochs,
                patience=patience,
                seed=seed + fold,
                architecture=architecture,
                pruning=pruning,
            )
            if pruning is not None and pruning.pruned:
                break
            val["fold"] = fold
            threshold = select_threshold(val)
            metrics = summarize_metrics(val, threshold)
            scores.append(
                float(
                    metrics.loc[metrics.source.eq("all"), "average_precision"].iloc[0]
                )
            )
            best_epochs.append(timing["best_epoch"])
            predictions.append(val)
            fold_rows.extend(
                {"trial": trial, "fold": fold, **timing, **row}
                for row in metrics.to_dict("records")
            )

        pruned = pruning is not None and pruning.pruned
        score = None if pruned else float(np.mean(scores))
        trial_row = {
            "trial": trial,
            **asdict(hp),
            "mean_validation_ap": score,
            "std_validation_ap": None if pruned else float(np.std(scores)),
        }
        if use_tpe:
            state = (
                optuna.trial.TrialState.PRUNED
                if pruned
                else optuna.trial.TrialState.COMPLETE
            )
            study.tell(optuna_trial, score, state=state)
            trial_row.update(state=state.name, **timing)
            if pruned:
                print(
                    f"Trial {trial} pruned after {timing['epochs_run']} epochs.",
                    flush=True,
                )
        trial_rows.append(trial_row)
        pd.DataFrame(trial_rows).to_csv(run_dir / "trials.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics.csv", index=False)
        if score is not None and score > best_score:
            best_score, best_hp, best_trial = score, hp, trial
            best_predictions = pd.concat(predictions, ignore_index=True)
            best_threshold = select_threshold(best_predictions)
            refit_epochs = max(1, round(median(best_epochs)))
            if folds is None:
                _write_model(
                    run_dir,
                    model,
                    standardization,
                    hp,
                    best_threshold,
                    {**settings, "best_trial": trial, "fitted_epochs": refit_epochs},
                )
        del model

    if folds is not None:
        # Fit the selected configuration on all the patients used in CV.
        selected = np.concatenate(splits[0])
        model, standardization, _, _ = _fit(
            frame.loc[frame.patient_id.isin(selected)],
            None,
            best_hp,
            epochs=refit_epochs,
            patience=patience,
            seed=seed,
            architecture=architecture,
        )
        _write_model(
            run_dir,
            model,
            standardization,
            best_hp,
            best_threshold,
            {**settings, "best_trial": best_trial, "fitted_epochs": refit_epochs},
        )

    best_predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    metrics = summarize_metrics(best_predictions, best_threshold)
    metrics.to_csv(run_dir / "validation_metrics.csv", index=False)
    (run_dir / "run_timing.json").write_text(
        json.dumps({"total_seconds": perf_counter() - run_started}, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics
