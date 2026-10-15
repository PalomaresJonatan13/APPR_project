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
