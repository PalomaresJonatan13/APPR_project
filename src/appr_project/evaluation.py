"""Hourly predictions, alert threshold selection and PhysioNet utility."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def predict_hours(model, batches):
    if batches.shuffle:
        raise ValueError("Evaluation batches must preserve patient and hour order.")
    frame = batches.hours.metadata.copy()
    frame["probability"] = model.predict(batches, verbose=0).ravel()
    return frame


def _utility_gain(frame):
    """Per-hour gain over no alert, and optimal gain on the full sequences."""
    frame = frame.sort_values(["patient_id", "ICULOS"])
    group = frame.groupby("patient_id", sort=False)
    positions = group.cumcount().to_numpy()
    first_positive = (
        pd.Series(positions, index=frame.index)
        .where(frame.SepsisLabel.eq(1))
        .groupby(frame.patient_id)
        .transform("min")
        .to_numpy()
    )
    # The official scorer anchors the utility peak at the first positive label.
    relative = positions - first_positive
    alert = np.where(
        relative <= 0, np.maximum(1 + relative / 6, -0.05), 1 - relative / 9
    )
    silence = np.where(relative > 0, -2 * relative / 9, 0)
    gain = np.where(
        np.isnan(relative), -0.05, np.where(relative <= 9, alert - silence, 0)
    )
    optimal = gain[(relative >= -6) & (relative <= 9)].sum()
    return frame.probability.to_numpy(dtype=np.float64), gain, optimal


def select_threshold(validation):
    """Maximize normalized utility on validation predictions only."""
    probability, gain, _ = _utility_gain(validation)
    order = np.argsort(-probability, kind="stable")
    scores = probability[order]
    ties_end = np.r_[scores[:-1] != scores[1:], True]
    thresholds = np.r_[np.nextafter(1.0, np.inf), scores[ties_end]]
    total_gain = np.r_[0.0, np.cumsum(gain[order])[ties_end]]
    return float(thresholds[np.argmax(total_gain)])


def summarize_metrics(predictions, threshold):
    """Return an aggregate row and one row for every source present."""
    rows = []
    groups = [("all", predictions), *predictions.groupby("source_set", sort=True)]
    for source, frame in groups:
        labels = frame.SepsisLabel
        positive = labels.eq(1)
        probability, gain, optimal = _utility_gain(frame)
        alerts = probability >= threshold
        rows.append(
            {
                "source": source,
                "patients": frame.patient_id.nunique(),
                "hours": len(frame),
                "positive_hours": int(positive.sum()),
                "positive_hour_fraction": positive.mean(),
                "average_precision": average_precision_score(labels, frame.probability)
                if labels.any()
                else np.nan,
                "auroc": roc_auc_score(labels, frame.probability)
                if labels.nunique() == 2
                else np.nan,
                "utility": gain[alerts].sum() / optimal if optimal else np.nan,
                "alert_fraction": alerts.mean(),
                "recall_at_threshold": frame.loc[positive, "probability"]
                .ge(threshold)
                .mean(),
                "threshold": threshold,
            }
        )
    return pd.DataFrame(rows)
