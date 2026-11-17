"""Repeat saved trials using their original patient splits."""

import json
import re
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import pandas as pd

from appr_project.data import load_partition
from appr_project.training import ARCHITECTURES, Hyperparameters, train_experiment


def _read_group(runs_dir, *, nombre, origen, trials, epochs, patience):
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (nombre, origen)):
        raise ValueError("El nombre y el origen deben ser nombres de carpeta simples.")
    if (
        type(epochs) is not int
        or epochs < 1
        or type(patience) is not int
        or patience < 0
    ):
        raise ValueError(f"{nombre}: epochs debe ser positivo y patience no negativo.")
    if not trials or any(type(trial) is not int or trial < 1 for trial in trials):
        raise ValueError(f"{nombre}: indica los números de los trials a repetir.")
    if len(set(trials)) != len(trials):
        raise ValueError(f"{nombre}: hay números de trial repetidos.")

    original = runs_dir / origen
    config = json.loads((original / "config.json").read_text(encoding="utf-8"))
    architecture = config["architecture"]
    if architecture not in ARCHITECTURES:
        raise ValueError(f"{origen}: arquitectura desconocida: {architecture}")

    trial_rows = pd.read_csv(original / "trials.csv").set_index(
        "trial", verify_integrity=True
    )
    if "batch_size" not in trial_rows and "batch_size" in config:
        trial_rows["batch_size"] = config["batch_size"]
    if "use_hidden_dense" not in trial_rows:
        trial_rows["use_hidden_dense"] = Hyperparameters.use_hidden_dense
    for name in ("kernel_l2", "recurrent_l2"):
        if name not in trial_rows:
            trial_rows[name] = getattr(Hyperparameters, name)
    missing = set(trials) - set(trial_rows.index)
    if missing:
        raise ValueError(f"{origen}: no se encontraron los trials {sorted(missing)}.")
    hp_names = [field.name for field in fields(Hyperparameters)]
    candidates = {
        trial: Hyperparameters(**row)
        for trial, row in trial_rows.loc[list(trials), hp_names]
        .to_dict("index")
        .items()
    }

    assignments = pd.read_csv(original / "splits.csv", dtype={"patient_id": str})
    expected_folds = list(range(1, (config["folds"] or 1) + 1))
    splits = []
    for _, split in assignments.groupby("fold", sort=True):
        train = split.loc[split.partition.eq("train"), "patient_id"]
        validation = split.loc[split.partition.eq("validation"), "patient_id"]
        if (
            train.empty
            or validation.empty
            or len(train) + len(validation) != len(split)
            or split.patient_id.isna().any()
            or split.patient_id.duplicated().any()
        ):
            raise ValueError(
                f"{origen}: partición de pacientes inválida en splits.csv."
            )
        splits.append((train.to_numpy(), validation.to_numpy()))
    if (
        assignments.fold.isna().any()
        or sorted(assignments.fold.unique()) != expected_folds
    ):
        raise ValueError(
            f"{origen}: los folds de splits.csv no coinciden con config.json."
        )

    settings = {
        key: config[key]
        for key in ("source", "folds", "validation_fraction", "sample_patients", "seed")
    }
    settings.update(architecture=architecture, epochs=epochs, patience=patience)
    baseline = pd.read_csv(original / "fold_metrics.csv")
    baseline = baseline.loc[
        baseline.source.eq("all") & baseline.trial.isin(trials),
        ["trial", "fold", "average_precision"],
    ].rename(columns={"average_precision": "original_average_precision"})
    return candidates, splits, assignments.patient_id.unique(), settings, baseline


def repeat_trial_groups(data_dir: Path, runs_dir: Path, groups):
    """Train each group from scratch and return its batch directory and fold summary."""
    groups = list(groups)
    names = [group["nombre"].casefold() for group in groups]
    if not names or len(set(names)) != len(names):
        raise ValueError("Indica al menos un grupo y asigna nombres únicos.")
    prepared = [_read_group(runs_dir, **group) for group in groups]

    batch_dir = runs_dir / f"repeticiones-{datetime.now():%Y%m%d-%H%M%S-%f}"
    batch_dir.mkdir()
    (batch_dir / "grupos.json").write_text(
        json.dumps(groups, indent=2) + "\n", encoding="utf-8"
    )
    summaries = []
    loaded_source = None
    for group, (candidates, splits, patient_ids, settings, baseline) in zip(
        groups, prepared, strict=True
    ):
        if loaded_source != settings["source"]:
            loaded_source = settings["source"]
            frame = load_partition(data_dir, "train", loaded_source)
        missing = pd.Index(patient_ids).difference(frame.patient_id.unique())
        if len(missing):
            raise ValueError(
                f"{group['origen']}: faltan {len(missing)} pacientes en el train actual."
            )

        run_dir = batch_dir / group["nombre"]
        print(f"Grupo {group['nombre']} · {group['origen']}", flush=True)
        train_experiment(
            frame, run_dir, candidates=candidates, patient_splits=splits, **settings
        )
        metrics = pd.read_csv(run_dir / "fold_metrics.csv")
        summary = metrics.loc[
            metrics.source.eq("all"),
            [
                "trial",
                "fold",
                "epochs_run",
                "best_epoch",
                "average_precision",
                "preparation_seconds",
                "fit_seconds",
            ],
        ].merge(baseline, on=["trial", "fold"], how="left", validate="one_to_one")
        summary["delta_average_precision"] = (
            summary["average_precision"] - summary["original_average_precision"]
        )
        summary["epochs"] = settings["epochs"]
        summary["patience"] = settings["patience"]
        summary["architecture"] = settings["architecture"]
        summary.insert(0, "origen", group["origen"])
        summary.insert(0, "grupo", group["nombre"])
        summaries.append(summary)
        combined = pd.concat(summaries, ignore_index=True)
        combined.to_csv(batch_dir / "resumen.csv", index=False)
    return batch_dir, combined
