"""Train and optionally tune the Keras tower model on the cleaning split."""

import argparse
from datetime import datetime
from pathlib import Path

from appr_project.data import load_partition
from appr_project.training import Hyperparameters, train_experiment

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("tower_grud", "semantic_grud"),
        default="tower_grud",
    )
    parser.add_argument("--source", choices=("A", "B", "A+B"), default="A")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--sample-patients", type=int)
    parser.add_argument(
        "--folds", type=int, help="Enable k-fold CV instead of holdout."
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of random-search trials, including the base configuration.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for the base configuration.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--units", type=int, default=16)
    parser.add_argument("--dense-units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if (
        min(
            args.trials,
            args.epochs,
            args.batch_size,
            args.window,
            args.units,
            args.dense_units,
        )
        < 1
    ):
        parser.error("Trials, epochs, batch size, window and units must be positive.")
    if args.folds is not None and args.folds < 2:
        parser.error("--folds must be at least 2.")
    if args.sample_patients is not None and args.sample_patients < 1:
        parser.error("--sample-patients must be positive.")
    if not (
        0 < args.validation_fraction < 1
        and 0 <= args.dropout < 1
        and args.learning_rate > 0
    ):
        parser.error("Check the validation fraction, dropout and learning rate.")
    if args.patience < 0 or args.seed < 0:
        parser.error("Patience and seed must be nonnegative.")

    run_dir = (
        args.run_dir
        or ROOT / "runs" / f"{args.source}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    )
    frame = load_partition(args.data_dir, "train", args.source)
    hp = Hyperparameters(
        window=args.window,
        units=args.units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    metrics = train_experiment(
        frame,
        run_dir,
        source=args.source,
        base=hp,
        trials=args.trials,
        folds=args.folds,
        validation_fraction=args.validation_fraction,
        sample_patients=args.sample_patients,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        architecture=args.architecture,
    )
    print(metrics.to_string(index=False))
    print(f"Artifacts: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
