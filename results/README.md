# results/ - experiment outputs

Raw JSON outputs of all experiments behind the manuscript. Every file maps to a place in the
paper: `code/checks/final_data_check.py` (75 itemized checks, author-local) and
`code/checks/verify_results.py` (179 checks, reads only this directory, no manuscript sources
needed) compare them against the reported numbers; summary files can be regenerated from the
per-seed files with `code/checks/aggregate_results.py`.

## baselines/ - source-domain baselines

| File | Manuscript |
|---|---|
| baseline_3seed_final.json | Table 1, four backbones x 5-fold group CV (3 seeds; +/- is the sample std over seeds, ddof=1) |
| lobo_tcn.jsonl | Table 2, TCN row (original run, 119 folds, one per line) |
| lobo_tcn_s42/43/44.jsonl | 357-fold pooled distribution of Fig. 2 plus the Section 4.2 cross-seed check (reproduction runs) |
| lobo_lstm.jsonl / lobo_transformer.jsonl | Table 2, LSTM / Transformer rows (119 folds each, original run) |
| lobo_final_multiseed.json | Section 4.2 cross-seed check (TCN repeated 3 times, 92.45 +/- 0.98; LSTM/Transformer original single run) |
| lobo_3model_final.csv | Per-cell RMSEs of the last Table 2 row (per-cell best-of-three upper bound) |

> Note: summaries only cover runs that have per-fold jsonl files; "LSTM/Transformer/ensemble,
> 2 repeats each" is omitted for lack of jsonl support (aggregate_results.py regenerates only
> the supported parts).

## transfer/ - cross-chemistry transfer

| File | Manuscript |
|---|---|
| transfer_multiseed_v2.json | Table 3 + Fig. 3, three mechanisms (5 seeds) |
| raw_protocol_multiseed.json | First row of Table 4, original-protocol drift (seeds 42-46) |
| t3_soh_tcn_s42-46.json | Table 4 per-seed raw outputs (original protocol, from t3_transfer_local.py) |
| t3b_tcn/lstm_s42-46.json | Table 3 per-seed raw outputs (per-dataset standardization, from t3b_std_local.py) |
| t3b_tcn_s42_repeat1/2.json | Section 4.9 repeat pair of the same seed (CALCE zero-shot RMSE differs by 14%, GPU nondeterminism) |
| t3_rul_tcn_s42.json | RUL-scope control (basis for using SOH as the transfer target, end of Section 3.1) |

> Scope note: Table 3 (main table) = per-dataset standardization = t3b files; first row of
> Table 4 = original protocol = t3_soh files.

## ablation/ - feature-richness ablation

| File | Manuscript |
|---|---|
| ablation_v3c_multiseed.json | Table 6 + Fig. 6 (v3c modeling table, used in the paper) |
| ablation_multiseed.json / ablation_multiseed_v5.json | Data-version comparison archive (v3 / v5 tables; values not cited) |
| t3c/t3d/t3e_*_s42-46.json | Per-seed raw outputs of the three data versions (base7 / curve14 groups) |

## conformal/ - conformal intervals

| File | Manuscript |
|---|---|
| conformal_multiseed_summary.json | Table 5 + Fig. 5, dual-route comparison (5 seeds) |
| t4c_multiseed_summary.json | Section 4.5 conditional narrowing (Mondrian / weighted, negative result) |
| t4_tcn/lstm_s42-46.json | Table 5 per-seed raw outputs (with per-cell residual vectors) |
| t4c_mondrian_s42-46.json / t4c_weighted_s42-46.json | Per-seed outputs of the Section 4.5 extensions |
| t4d_per_cell_tcn/lstm_s42-46.json | Section 4.6 diagnostics (per-cell coverage, per-cycle residuals, aggregated conformal variants) |

> Scope note: the target-domain calibration quantile in t4_*.json / t4d_*.json is computed from
> the deployed model's own residuals on the calibration cells, the same-model premise of split
> conformal; per-cell residual vectors are stored with the JSON. t4c_weighted_*.json was rerun
> after removing a test-label leak and fixing an off-by-one in the weighted quantile;
> t4c_mondrian_*.json was defect-free and keeps the original run.

## Supplementary experiments (cited in Sections 4.5/4.9)

| File | Description |
|---|---|
| t4_split_sweep/ | coverage vs calibration-cell count (`t4_conformal_local.py --n-cal`): 7 test cells fixed, calibration cells 1 to 5, fine-tuning cells 8 down to 4, 2 backbones x 5 seeds; sweep_summary.json aggregates |
| t4_gru_s42-46.json | GRU backbone robustness check (dual route, 5 seeds, source pre-training rerun under the same protocol) |
| diag_deterministic/ | cross-pipeline reproducibility diagnostics: t3b/t3e reruns with `--deterministic` (cited in Section 4.9) |

## Other

- `early_pred_summary.json` / `early_pred_results.csv`: Section 4.8 early-life prediction (dQ
  features + ridge regression, per-cell details for 119 cells; log10 RMSE 0.117 / cycle RMSE
  141.7 / MAPE 19.5%; script `code/early_pred/t5_early_pred.py`)

## Conventions

- Per-seed files are named by seed (s42-s46); manuscript tables report 5 seeds (42-46) except
  single-run LOBO cells
- +/- is the sample standard deviation (ddof=1) throughout
- Table 3 gain ratios are per-seed ratios averaged afterwards, not ratios of column means; the
  median basis is the more conservative view under the heavy tail (disclosed in Section 4.3)
- Table 5 splits are hard-coded (CALCE 7/2/7, NASA 2/1/1); Mondrian / weighted use the 1/3
  protocol (CALCE 5/5/6); splits are redrawn per seed, so +/- mixes split and training variability
- Same-seed reruns move slightly (GPU nondeterminism), differences of a few tenths of a percent
  to a few percent are normal (Section 4.9). All scripts share one set of MIT pre-trained weights
  (unified source-model cache), so identical configurations agree across pipelines.

## CX2 extension (results_cx2/, submission scope)

All experiments rerun after the CALCE target domain grew from 8 to 16 cells; the layout mirrors
this directory:
- transfer/ Table 3 (5 seeds); t3_soh_* first row of Table 4 (original protocol)
- ablation/ Table 6 (base7/curve14 paired t)
- conformal/ Table 5 (split 7/2/7) + t4c (5/5/6) + t4d per-cell diagnostics
  + t4_split_sweep/ (calibration cells 1-5) + i9_seeds/ (50 split redraws)
  + t4zs_* (zero-shot route control of Section 3.4; produced with --zs-only, 20 runs)
- Recompute: python code/checks/verify_results.py --results results_cx2 (160 checks)
This directory (results/) is the frozen 8-cell-protocol history; it still passes
python code/checks/verify_results.py (179/179).
