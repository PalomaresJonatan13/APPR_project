"""Holdout or cross-validation, random search, and Keras artifacts."""

import json
from dataclasses import asdict, dataclass
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
from appr_project.models import semantic_grud, tower_grud

ARCHITECTURES = ("tower_grud", *semantic_grud.GROUPS_BY_ARCHITECTURE)


@dataclass(frozen=True)
class Hyperparameters:
    window: int = 15
    units: int = 16
    dense_units: int = 64
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64


def search_candidates(base, trials, seed):
    """Evaluate the supplied configuration first, then sample distinct alternatives."""
    if trials < 1:
        raise ValueError("trials must be positive.")

    rng = np.random.default_rng(seed)
    candidates = [base]
    seen = {base}
    while len(candidates) < trials:
        candidate = Hyperparameters(
            window=int(rng.integers(8, 24, endpoint=True)),
            units=int(rng.integers(8, 32, endpoint=True)),
            dense_units=int(rng.choice((32, 64))),
            dropout=rng.uniform(0.0, 0.3),
            learning_rate=10.0 ** rng.uniform(-4.0, -2.0),
            batch_size=int(rng.choice((64, 128, 256))),
        )
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)
    return candidates


def _fit(train, validation, hp, *, epochs, patience, seed, architecture):
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
    if architecture in semantic_grud.GROUPS_BY_ARCHITECTURE:
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
    predictions = predict_hours(model, val_batches) if val_batches is not None else None
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
):
    """Train random search candidates or an explicit mapping of trial IDs to settings."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    run_started = perf_counter()
    if candidates is None:
        if base is None or trials is None:
            raise ValueError("Supply base and trials for a random search.")
        candidates = dict(enumerate(search_candidates(base, trials, seed), start=1))
    if not candidates:
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
        "trials": len(candidates),
    }
    trial_rows, fold_rows = [], []
    best_score = -np.inf
    for position, (trial, hp) in enumerate(candidates.items(), start=1):
        predictions, scores, best_epochs = [], [], []
        for fold, (train_ids, val_ids) in enumerate(splits, start=1):
            print(
                f"Trial {trial} ({position}/{len(candidates)}), "
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
            )
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

        score = float(np.mean(scores))
        trial_rows.append(
            {
                "trial": trial,
                **asdict(hp),
                "mean_validation_ap": score,
                "std_validation_ap": float(np.std(scores)),
            }
        )
        pd.DataFrame(trial_rows).to_csv(run_dir / "trials.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics.csv", index=False)
        if score > best_score:
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
