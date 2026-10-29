"""Read-only diagnostics for the unmodified PhysioNet 2019 training tables.

Distribution compatibility is not proof of a physical unit. No conversion,
imputation, clipping, filtering of source rows, or feature engineering is done.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DOCUMENTED_UNITS = {
    "HR": "beats/min",
    "O2Sat": "%",
    "Temp": "°C",
    "SBP": "mmHg",
    "MAP": "mmHg",
    "DBP": "mmHg",
    "Resp": "breaths/min",
    "EtCO2": "mmHg",
    "BaseExcess": "mmol/L",
    "HCO3": "mmol/L",
    "FiO2": "%",
    "pH": "dimensionless",
    "PaCO2": "mmHg",
    "SaO2": "%",
    "AST": "IU/L",
    "BUN": "mg/dL",
    "Alkalinephos": "IU/L",
    "Calcium": "mg/dL",
    "Chloride": "mmol/L",
    "Creatinine": "mg/dL",
    "Bilirubin_direct": "mg/dL",
    "Glucose": "mg/dL",
    "Lactate": "mg/dL",
    "Magnesium": "mmol/dL",
    "Phosphate": "mg/dL",
    "Potassium": "mmol/L",
    "Bilirubin_total": "mg/dL",
    "TroponinI": "ng/mL",
    "Hct": "%",
    "Hgb": "g/dL",
    "PTT": "s",
    "WBC": "10³ cells/µL",
    "Fibrinogen": "mg/dL",
    "Platelets": "10³ cells/µL",
    "Age": "years (100 for age ≥90)",
    "Gender": "0=female, 1=male",
    "Unit1": "MICU administrative indicator",
    "Unit2": "SICU administrative indicator",
    "HospAdmTime": "h relative to ICU admission",
    "ICULOS": "h since ICU admission",
    "SepsisLabel": "binary 0/1",
    "patient_id": "identifier added during extraction",
    "source_set": "A/B provenance",
}
CLINICAL = list(DOCUMENTED_UNITS)[:34]
BINARY = ["Gender", "Unit1", "Unit2", "SepsisLabel"]


def load_sets(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load both complete CSVs; retain original row order, values and NaNs."""
    return {name: pd.read_csv(data_dir / f"set_{name}.csv") for name in ("A", "B")}


def unit_audit(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Audit all 43 columns separately by site, without asserting unit identity."""
    records = []
    for name, frame in sets.items():
        missing = set(DOCUMENTED_UNITS) - set(frame)
        extra = set(frame) - set(DOCUMENTED_UNITS)
        if missing or extra:
            raise ValueError(
                f"{name}: missing columns {missing}; unknown columns {extra}"
            )
        for column, unit in DOCUMENTED_UNITS.items():
            values = frame[
                column
            ].dropna()  # Summary denominator only; no data mutation.
            row = {
                "source_set": name,
                "column": column,
                "documented_unit": unit,
                "n_observed": len(values),
                "missing_pct": 100 * frame[column].isna().mean(),
                "dtype": str(frame[column].dtype),
            }
            note = (
                "Scale not contradicted by this audit." # unit not independently verified.
            )
            invalid = pd.Series(False, index=values.index)
            if pd.api.types.is_numeric_dtype(values):
                row.update(
                    values.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
                    .set_axis(["min", "p01", "q1", "median", "q3", "p99", "max"])
                    .to_dict()
                )
                row["n_infinite"] = int(np.isinf(values).sum())
                if column in BINARY:
                    invalid = ~values.isin([0, 1])
                    note = "Binary coding checked on observed values." # NaN is not zero
                elif column in ["O2Sat", "SaO2", "Hct"]:
                    invalid = ~values.between(0, 100)
                    note = "Checked percentage bounds [0,100]; fraction-scale mixing cannot be excluded."
                elif column == "pH":
                    invalid = ~values.between(0, 14)
                    note = "Checked broad aqueous pH scale [0,14]."
                elif column == "FiO2":
                    invalid = ~values.between(0, 100)
                    note = "Predominantly fractional scale despite documented %; inspect scale bins."
                elif column == "Lactate":
                    note = "Suspected mmol/L scale despite documented mg/dL; unconfirmed hypothesis."
                    invalid = values < 0
                elif column == "Magnesium":
                    note = "Suspected mg/dL scale despite documented mmol/dL; unconfirmed hypothesis."
                    invalid = values < 0
                elif column == "Calcium":
                    invalid = values < 0
                    row["n_calcium_below_3"] = int(values.lt(3).sum())
                    row["pct_calcium_below_3"] = 100 * values.lt(3).mean()
                    note = (
                        "Low-value cluster, especially B: possible mixed measurement types/scales; "
                        "unconfirmed. Threshold <3 is descriptive, not a clinical validity rule."
                    )
                elif column == "Age":
                    invalid = (values < 0) | ((values >= 90) & (values != 100))
                    note = "Years; check documented ≥90→100 encoding. Absence of 100 is not proof of error."
                elif column == "HospAdmTime":
                    note = "Hours relative to ICU admission; positive values flagged separately, not converted."
                elif column == "ICULOS":
                    invalid = (values < 1) | (values % 1 != 0)
                    note = "Positive integer hours checked; per-patient progression checked separately."
                elif column != "BaseExcess":
                    invalid = values < 0
                row["n_domain_flags"] = int(invalid.sum())
            else:
                row["n_domain_flags"] = (
                    int((values != name).sum())
                    if column == "source_set"
                    else int(
                        (~values.str.fullmatch(rf"{name}_\d{{6}}", na=False)).sum()
                    )
                )
                note = (
                    "Extraction metadata: no physical unit; format/provenance checked."
                )
            if not len(values):
                note = "No observed values: unit/scale cannot be assessed for this set."
            row["assessment"] = note
            records.append(row)
    return pd.DataFrame(records)


def fio2_scales(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Disjoint bins describe stored values, not inferred corrected measurements."""
    rows = []
    for name, frame in sets.items():
        x = frame.FiO2.dropna()
        bins = {
            "negative": x < 0,
            "zero": x == 0,
            "(0,1] fraction-like": (x > 0) & (x <= 1),
            "(1,20) ambiguous": (x > 1) & (x < 20),
            "[20,100] percent-like": x.between(20, 100),
            ">100": x > 100,
        }
        for label, mask in bins.items():
            rows.append(
                {
                    "source_set": name,
                    "stored_value_bin": label,
                    "count": int(mask.sum()),
                    "pct_observed": 100 * mask.mean(),
                }
            )
    return pd.DataFrame(rows)


def patient_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One diagnostic record per patient, not a model-ready dataset."""
    groups = frame.groupby("patient_id", sort=False)
    out = groups.agg(
        rows=("ICULOS", "size"),
        first_hour=("ICULOS", "min"),
        last_hour=("ICULOS", "max"),
        septic=("SepsisLabel", "max"),
        age=("Age", "first"),
        gender=("Gender", "first"),
        hosp_adm_time=("HospAdmTime", "first"),
    )
    first_positive = (
        frame.loc[frame.SepsisLabel.eq(1)].groupby("patient_id").ICULOS.min()
    )
    return out.join(first_positive.rename("first_positive_hour"))


def overview(
    sets: dict[str, pd.DataFrame], patients: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for name, frame in sets.items():
        p = patients[name]
        rows.append(
            {
                "source_set": name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "patients": len(p),
                "septic_patients": int(p.septic.sum()),
                "septic_patient_pct": 100 * p.septic.mean(),
                "positive_row_pct": 100 * frame.SepsisLabel.mean(),
                "median_recorded_hours": p.rows.median(),
                "median_patient_age": p.age.median(),
                "female_patient_pct": 100 * p.gender.eq(0).mean(),
                "clinical_missing_pct": 100 * frame[CLINICAL].isna().to_numpy().mean(),
            }
        )
    return pd.DataFrame(rows).set_index("source_set")


def integrity(sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in sets.items():
        groups = frame.groupby("patient_id", sort=False)
        delta = groups.ICULOS.diff()
        label_delta = groups.SepsisLabel.diff()
        stable = groups[["Age", "Gender", "HospAdmTime"]].nunique(dropna=True)
        rows.append(
            {
                "source_set": name,
                "missing_patient_id": int(frame.patient_id.isna().sum()),
                "duplicate_patient_hour": int(
                    frame.duplicated(["patient_id", "ICULOS"]).sum()
                ),
                "non_unit_hour_steps": int((delta.notna() & delta.ne(1)).sum()),
                "patients_not_starting_at_1": int(groups.ICULOS.first().ne(1).sum()),
                "label_reversals": int(label_delta.lt(0).sum()),
                "positive_hospital_admission_rows": int(frame.HospAdmTime.gt(0).sum()),
                "changing_demographics_patients": int(stable.gt(1).any(axis=1).sum()),
                "infinite_numeric_values": int(
                    np.isinf(frame.select_dtypes("number")).sum().sum()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("source_set")


def compare_sets(audit: pd.DataFrame) -> pd.DataFrame:
    """Descriptive contrasts, deliberately without iid row-level significance tests."""
    a, b = (audit.loc[audit.source_set.eq(s)].set_index("column") for s in ("A", "B"))
    out = pd.DataFrame(
        {
            "unit": a.documented_unit,
            "n_A": a.n_observed,
            "n_B": b.n_observed,
            "missing_A_pct": a.missing_pct,
            "missing_B_pct": b.missing_pct,
            "median_A": a["median"],
            "median_B": b["median"],
        }
    )
    out["missing_B_minus_A_pp"] = out.missing_B_pct - out.missing_A_pct
    out["median_B_minus_A"] = out.median_B - out.median_A
    return out
