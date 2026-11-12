"""Patient splits, training statistics and causal windows for Keras."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow import keras

from appr_project.eda import CLINICAL

STATIC_NUMERIC = ["Age", "HospAdmTime"]
STATIC_BINARY = ["Gender", "Unit1"]


def load_partition(root: Path, partition: str, source: str) -> pd.DataFrame:
    sites = ("A", "B") if source == "A+B" else (source,)
    return pd.concat(
        [pd.read_csv(root / partition / f"set_{site}_clean.csv") for site in sites],
        ignore_index=True,
    )


def split_patients(frame, *, validation_fraction, folds, seed, sample_patients=None):
    patients = frame.groupby("patient_id", sort=True).agg(
        site=("source_set", "first"), positive=("SepsisLabel", "max")
    )
    strata = patients.site.astype(str) + ":" + patients.positive.astype(str)
    ids = patients.index.to_numpy()
    if sample_patients is not None and sample_patients < len(ids):
        ids, _ = train_test_split(
            ids, train_size=sample_patients, stratify=strata, random_state=seed
        )
        ids = np.sort(ids)
        strata = strata.loc[ids]

    if folds is None:
        yield train_test_split(
            ids, test_size=validation_fraction, stratify=strata, random_state=seed
        )
        return

    if strata.value_counts().min() < folds:
        raise ValueError("Each site/label stratum needs at least k patients for CV.")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, validation in splitter.split(ids, strata):
        yield ids[train], ids[validation]


@dataclass
class Standardization:
    temporal: list[str]
    mean: dict[str, float]
    scale: dict[str, float]

    @classmethod
    def fit(cls, train):
        temporal = [name for name in CLINICAL if name in train.columns]
        numeric = train[temporal + STATIC_NUMERIC]
        mean = numeric.mean().fillna(0)
        scale = numeric.std(ddof=0).replace(0, 1).fillna(1)
        return cls(temporal, mean.to_dict(), scale.to_dict())

    def transform(self, frame, columns):
        values = frame[columns].to_numpy(dtype=np.float32)
        means = np.array([self.mean[c] for c in columns], dtype=np.float32)
        scales = np.array([self.scale[c] for c in columns], dtype=np.float32)
        return (values - means) / scales


class PatientHours:
    """Store each hour once. Build overlapping windows only when requested."""

    def __init__(self, frame, standardization):
        frame = frame.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
        self.metadata = frame[["patient_id", "source_set", "ICULOS", "SepsisLabel"]]
        self.labels = frame.SepsisLabel.to_numpy(dtype=np.float32)

        static = np.column_stack(
            (
                standardization.transform(frame, STATIC_NUMERIC),
                frame[STATIC_BINARY].to_numpy(dtype=np.float32),
                np.log1p(frame.ICULOS.to_numpy(dtype=np.float32)),
            )
        )
        self.static = np.column_stack((np.nan_to_num(static), np.isnan(static))).astype(
            np.float32
        )

        values = standardization.transform(frame, standardization.temporal)
        observed = ~np.isnan(values)
        values = np.nan_to_num(values)
        elapsed = np.zeros_like(values)
        previous = np.zeros_like(values)
        hours = frame.ICULOS.to_numpy(dtype=np.float32)
        ids = frame.patient_id.to_numpy()
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
        ends = np.r_[starts[1:], len(frame)]
        self.patient_start = np.repeat(starts, ends - starts)

        for start, end in zip(starts, ends):
            # Last observed row strictly before the current one, within this patient.
            rows = np.arange(start, end)[:, None]
            last = np.maximum.accumulate(
                np.where(observed[start:end], rows, -1), axis=0
            )
            last = np.concatenate((np.full((1, values.shape[1]), -1), last[:-1]))
            safe_last = np.maximum(last, start)
            elapsed[start:end] = hours[start:end, None] - hours[safe_last]
            previous[start:end] = np.where(
                last >= 0, values[safe_last, np.arange(values.shape[1])], 0
            )

        self.temporal = np.stack((values, observed, elapsed, previous), axis=-1)


class WindowBatches(keras.utils.PyDataset):
    """One right-aligned window and one label per observed patient hour."""

    def __init__(self, hours, window, batch_size, *, shuffle=False, seed=0):
        super().__init__()
        self.hours = hours
        self.window = window
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(hours.labels))
        self.on_epoch_end()

    def __len__(self):
        return (len(self.order) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, batch):
        end = self.order[batch * self.batch_size : (batch + 1) * self.batch_size]
        indices = end[:, None] - np.arange(self.window - 1, -1, -1)
        start = self.hours.patient_start[end, None]
        valid = indices >= start
        windows = self.hours.temporal[np.maximum(indices, start)]
        windows[~valid] = 0
        inputs = {
            "temporal": windows,
            "valid_hours": valid,
            "static": self.hours.static[end],
        }
        return inputs, self.hours.labels[end, None]

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)
