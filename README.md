# ASBB: Adaptive Seasonal Baseline Blending for Weakness Trend Forecasting

Code and artifacts accompanying the paper "Adaptive Seasonal Baseline Blending for Weakness Trend Forecasting," accepted at IEEE BigData/BigCyber 2026.

## What is in this repository

- `scripts/` — the full pipeline. This includes building the monthly CWE panel from CISA CSAF advisories, the ASBB model and baselines, the walk-forward evaluation harness, Diebold-Mariano significance tests, CWE-family clustering, and topic-CWE semantic alignment.
- `data/family/cwe_auto_family_mapping.csv` — the full mapping of every CWE id to its automatically constructed family (15 families, Section 4.3 of the paper).
- `data/predictions/forecasts_2021.pkl` ... `forecasts_2024.pkl` — per-CWE one-step-ahead forecasts from every model, for each of the four hold-out years (2021-2024).
- `data/per_cwe_results/per_cwe_results_2021.csv` ... `per_cwe_results_2024.csv` — per-CWE MAE/RMSE for every model, each hold-out year. This is the table `scripts/build_result_tables.py` and `scripts/diebold_mariano_test.py` are built from.
- `data/topic_pairs/topic_cwe_xcorr_permtest.csv` — all 82 high-confidence topic-CWE pairs used in the semantic alignment analysis (Section 5.5), with their cosine similarity, lag, correlation, and permutation-test results.
- `data/failed_fits/sarimax_failed_{year}.csv` — the CWE-years where SARIMAX's one-step fit exceeded the error threshold and was excluded from SARIMAX's own reported average (Section 6, Discussion and Limitations).
- `data/csaf_snapshot_identifier.json` — identifies the exact local copy of the CISA CSAF corpus used (advisory file count, timestamps, and a filename-list hash for reproducibility checks).

## Running the pipeline

Scripts read from a data directory set by the `ASBB_DATA_DIR` environment variable (default `./data` for the files already in this repo, or `./data_raw` for the rebuilt CISA panel the scripts that start from raw advisory text expect). A typical run order:

1. `family_clustering_sweep.py` — selects the CWE-family clustering (Section 4.3).
2. `walk_forward_eval.py` — runs the walk-forward evaluation, producing `outputs/per_cwe_results_{year}.csv` and `outputs/forecasts_{year}.pkl`.
3. `build_result_tables.py` — builds the paper's main tables and the Forecastability Index from the walk-forward output.
4. `diebold_mariano_test.py` — significance testing against each baseline.
5. `threat_horizon.py` — the 60-month threat-horizon figure and long-horizon accuracy check.
6. `topic_model_fit.py` — fits the topic model and aligns topics to CWEs (Section 5.5).
7. `build_topic_intensity.py` — builds the family-level Topic Intensity Index from the fitted topic model.
8. `weight_distribution.py` — reports the distribution of ASBB's selected blend weight across all forecast origins (Section 4.5 Discussion).

## Reproducibility notes

- All UMAP clustering steps (CWE-family construction and BERTopic topic modeling) use `random_state=42`. HDBSCAN is deterministic given its inputs and parameters.
- The ASBB model itself is specified in Section 4.5 of the paper; `scripts/asbb_eval.py` is the reference implementation.
- `scripts/weight_distribution.py` and `scripts/build_topic_intensity.py` are two supplementary checks: the distribution of ASBB's selected blend weight across all forecast origins (Section 4.5), and a test of lagged topic proportions as forecasting predictors (end of Section 5.5).

## Requirements

See `requirements.txt`. Install with `pip install -r requirements.txt`.

## Data access

This repository does not redistribute the raw CISA advisory text itself; `data/csaf_snapshot_identifier.json` documents which advisory files were used so the panel can be rebuilt from CISA's own public CSAF repository.
