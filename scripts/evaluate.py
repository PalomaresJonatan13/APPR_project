"""Evaluate a saved Keras run on the fixed test partitions."""

import argparse
import json
from pathlib import Path

from tensorflow import keras

from appr_project.data import (
    PatientHours,
    Standardization,
    WindowBatches,
    load_partition,
)
from appr_project.evaluation import predict_hours, summarize_metrics
from appr_project.models import semantic_grud, tower_grud, vs_gru  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--source", choices=("A", "B", "A+B"), default="A+B")
    args = parser.parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    hp = config["hyperparameters"]
    standardization = Standardization(**config["standardization"])
    model = keras.models.load_model(args.run_dir / "best.keras", compile=False)
    hours = PatientHours(
        load_partition(args.data_dir, "test", args.source), standardization
    )
    batches = WindowBatches(
        hours, hp["window"], hp.get("batch_size", config.get("batch_size"))
    )
    predictions = predict_hours(model, batches)
    metrics = summarize_metrics(predictions, config["threshold"])
    predictions.to_csv(
        args.run_dir / f"test_{args.source}_predictions.csv", index=False
    )
    metrics.to_csv(args.run_dir / f"test_{args.source}_metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
