# Data
This directory contains raw and processed data files.

## Structure
- `raw/` — Raw CSV files (synthetic or organizer-provided)
- `processed/` — Feature-engineered datasets ready for modeling

## Note
Large data files are gitignored. Run `python -m src.data.synthesize` to generate synthetic data,
or place organizer-provided files in `data/raw/`.
