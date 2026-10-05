# Pipeline guide

Scripts are listed in run order. Each script carries a short header comment with its design
choices; this file gives the global map and the typical commands.

## Run order

```
1. data_prep/          raw data -> per-cycle feature tables
   parse_calce.py        CALCE 8 cells (CADEX txt + Arbin xlsx) -> calce_features.csv
   parse_calce_v2.py     unified CALCE rebuild (current-integrated capacity + v_q curve features)
                         -> calce_full_v2.csv
   parse_nasa.py         NASA 4 cells -> nasa_capacity.csv (EOL/RUL definitions; NASA EOL = 70%)
   features_nasa.py      NASA cycle-level features (incl. ICA) -> nasa_features.csv
   extract_mit_dq_early.py  first-100-cycle dQ features of the MIT cells (input of the early-life
                         experiment); the CSV shipped with the repository is the frozen version
                         used for the manuscript
        -> one merged modeling table (uniform columns across datasets)
   build_modeling_table.py  merges the above into modeling_table_v3.csv / modeling_table_v3c.csv
        (the v3/v3c/v5 tables used in the manuscript are shipped as .csv.gz; see DATA.md)

2. baselines/          source-domain baselines (backbone selection)
   t2_train_local.py     4 backbones x 5-fold group CV (--smoke for a quick overfit self-check)
   t2_lobo_local.py      leave-one-battery-out, 119 folds (resumable; data source of Table 2 / Fig. 2)

3. transfer/           cross-chemistry transfer
   t3_transfer_local.py  zero-shot / fine-tune / target-only, original protocol (first row of Table 4)
   t3b_std_local.py      per-dataset standardization protocol (main Table 3 of the manuscript)

4. ablation/           feature ablation (Table 6)
   t3c_ablation_local.py base7 vs curve14 (v3 modeling table)
   t3d_ablation_v5.py    same protocol, v5 modeling table (data-version comparison)
   t3e_ablation_v3c.py   same protocol, v3c modeling table (used for the manuscript)

5. conformal/          conformal intervals
   t4_conformal_local.py dual-route comparison (Table 5); the target-domain calibration quantile
                         is computed from the deployed model's own residuals, keeping calibration
                         and evaluation on the same model; --src-cache reuses source models;
                         --n-cal N sweeps the calibration-cell count into t4_split_sweep/;
                         --deterministic enables deterministic algorithms
   t4c_mondrian_local.py conditional narrowing, part 1: SOH binning
   t4c_weighted_local.py conditional narrowing, part 2: density-ratio weighting (test-side uses
                         predicted SOH only, never the test labels)
   t4d_per_cell_diag.py  per-cell coverage and per-cell aggregation variants (Section 4.6);
                         the t4/t4d JSON files store per-cell residual vectors

6. early_pred/         early-life prediction (Section 4.8)
   t5_early_pred.py      dQ features + ridge regression, leave-one-cell-out over 119 cells
                         (input: data/mit_dq_early.csv)

7. checks/
   final_data_check.py   experiment JSONs vs manuscript numbers, 75 itemized checks (~3 s; needs
                         the manuscript source files, so it runs only on the author's machine)
   verify_results.py     manuscript numbers vs the results/ summary files, 179 checks
                         (no manuscript sources, no GPU)
   verify_results_cx2.py submission-scope check: manuscript numbers vs results_cx2/, 160 checks
   zeroshot_route.py     zero-shot route recomputation (the Section 3.4 control, 20 runs; no GPU)
   aggregate_results.py  regenerates all multi-seed summary files from the per-seed raw files (~1 s)
```

## Results

`results/` holds the raw per-seed JSON outputs of every experiment. After re-running a script,
overwrite the corresponding file and run `checks/aggregate_results.py` to rebuild the summaries;
`results/README.md` maps each file to the manuscript table it feeds.
`checks/final_data_check.py` is the author's local cross-check (hard-coded paths and manuscript
sources required); the result files themselves are readable without it.

## Conventions used in the manuscript

- **Seeds**: transfer/conformal/ablation use 5 independent runs (42-46); values are reported as
  mean +/- sample standard deviation (ddof=1). Gain ratios in the abstract are per-seed ratios
  averaged afterwards, not ratios of means; the NASA/LSTM median-basis value (3.2) is disclosed
  alongside in Section 4.3.
- **Splits**: Table 5 conformal uses the hard-coded CALCE 7/2/7 and NASA 2/1/1 splits;
  Mondrian/weighted use the 1/3 protocol (CALCE 5/5/6, NASA 1/1/2). `t4d_per_cell_diag.py`
  asserts it uses the same splits as Table 5. Splits are redrawn per seed, so +/- reflects both
  training randomness and split variability.
- **Standardization**: t3/t3c/t3d/t3e apply the source-domain scaler unchanged to the target
  domain (original protocol, first row of Table 4); t3b/t4*/t4d standardise each dataset with its
  own scaler (per-dataset protocol, main Table 3).
- **Conformal calibration**: the target-domain quantile always comes from the deployed
  (fine-tuned) model's residuals on the calibration cells, the same-model premise of split
  conformal.
- **GPU nondeterminism**: rerunning a seed moves metrics slightly (discussed in Section 4.9);
  comparisons at the third decimal are not meaningful. All scripts share one set of MIT
  pre-trained weights (a unified source-model cache), so the same configuration in Tables 3 and 6
  gives identical values across pipelines (diagnostics in `results/diag_deterministic/`).

## Reproduction boundaries

- The per-seed -> summary -> manuscript-number chain is fully recomputable (verify + aggregate).
- End-to-end retraining from raw data is limited in four ways: (i) eight MIT-side intermediate
  files (mit_capacity.csv etc.) are not distributed, so `build_modeling_table.py` cannot rerun
  from scratch (the frozen v3/v3c/v5 tables are shipped instead); (ii) training is GPU-
  nondeterministic and numbers move slightly; (iii) `data/mit_dq_early.csv` is frozen and
  `extract_mit_dq_early.py` is an independent implementation of the feature definition (variance
  and mean align, slope/extremes differ slightly; see its docstring); (iv) source-model caches
  (`results_cx2/src_cache/*.pt`, about 52 MB) are not distributed (`.gitignore` excludes `*.pt`):
  t3b/t3e/t4/t4c x2/t4d accept `--src-cache` and rebuild it automatically on first use; without a
  cache each script retrains the source domain itself and no longer shares one weight set.

## Common modifications

- New dataset: edit the `DATA` path and the `FEATS` column list at the top of each script; the
  modeling table needs `dataset` (MIT/CALCE/NASA), `battery_id`, `cycle`, `soh`, `rul` columns.
- New backbone: register it in `new_model()` of `t2_train_local.py`; the model-selection logic of
  the other scripts is a thin wrapper.
- Conformal miscoverage rate: `ALPHA` (0.10 -> 0.90 nominal coverage), top of each script.
- New diagnostics: the t4/t4d JSON files carry per-cell residual vectors, so evaluation changes
  need no retraining.

## CX2 extension (2026-10, submission scope)

- CALCE target domain extended from 8 to 16 cells (8 CS2 + 8 CX2); inclusion criteria and
  excluded cells are in Section 4.1. The parser sets RATED per cell series (CS2 1.1 Ah, CX2
  1.35 Ah); CX2_31 takes the CADEX txt branch. Raw zips live in data/raw/calce/ (not distributed).
- New scripts:
  data_prep/build_calce_curve_features.py  curve features (v_q/ICA shape)
  data_prep/build_modeling_table_cx2.py    appends v3_cx2 / v3c_cx2 (with V1-V3 self-checks)
- New data: data/modeling_table_v3_cx2.csv.gz, modeling_table_v3c_cx2.csv.gz, calce_full_v2_cx2.csv
- **Unified source-model cache**: all training scripts accept --src-cache (cache keys include the
  epoch count; the old t4d_src_* names are a fallback), and t3b/t3e/t4/t4d/t4c share one set of
  MIT pre-trained weights, removing the earlier up-to-57% discrepancy between pipelines.
- New results tree: results_cx2/ (mirrors results/; contains the t4_split_sweep calibration-cell
  sweep and i9_seeds, 50 split redraws). The old results/ tree (8-cell protocol) is frozen.
- Recompute: python code/checks/verify_results.py --results results_cx2 (160 checks); without
  arguments it still checks the old scope, 179/179.
- Table 5 splits (CX2): CALCE 7/2/7, NASA 2/1/1 (changeable via --split-calce/--split-nasa);
  t4c protocol (CX2): CALCE 5/5/6, NASA 1/1/2.
