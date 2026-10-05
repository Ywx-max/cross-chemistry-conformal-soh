# Cross-chemistry conformal state-of-health prediction of lithium-ion batteries

Code and derived results accompanying the manuscript:

> **Conformal coverage failure and target-domain recalibration in cross-chemistry state-of-health
> prediction of lithium-ion batteries** — Wangxing Yang (School of Automation, Beijing University
> of Posts and Telecommunications), submitted to *Journal of Energy Storage*.

This release (v1.0.0) contains everything needed to reproduce the numerical results of the
manuscript: source-domain baselines, cross-chemistry transfer, split-conformal calibration, the
supplementary experiments of Appendix E (alpha sweep, uncertainty baselines, paired statistics,
backbone cost, window/horizon sensitivity), and the paper figures.

## Repository structure

```
code/                 experimental pipelines (see code/README.md)
  checks/             verify_results_cx2.py recomputes the 160 reported numbers (~1 s, no GPU)
  conformal/ transfer/ baselines/ ablation/ ...  (main experiments)
  appendix_e/         Appendix E supplementary experiments (P0-1..P0-6) + verify_p0_tables.py
results_cx2/          per-seed raw results of the main experiments (JSON/JSONL)
results_p0/           Appendix E results: CSVs, per-seed NPZ, appendixE_tables.json
configs_p0/           YAML configurations of the Appendix E experiments
logs_p0/              run logs of the Appendix E experiments
figures/              Figure 1-6 (PNG/PDF/EPS) and the graphical abstract
data/                 modeling tables (compressed); raw third-party data are NOT redistributed
figures_reproduce/    figure-generation scripts of the main experiments
```

## Data (not redistributed)

The three public datasets are **not** redistributed here; download them from their original
repositories (see `DATA.md`). The compressed modeling tables under `data/` are derived tables
built from those datasets with the parsing scripts in `code/data_prep/`. To use them:

```bash
gunzip -k data/*.gz     # creates the .csv files expected by the scripts
```

## Reproduction

```bash
python -m pip install -r requirements.txt
python code/checks/verify_results_cx2.py          # 160 numbers of the manuscript
python code/appendix_e/p0_3_stats.py              # paired statistics (CPU, ~seconds)
python code/appendix_e/p0_2_alpha_sweep.py        # alpha sweep (GPU recommended)
python code/appendix_e/verify_p0_tables.py        # consistency of Appendix E tables vs CSVs
```

Source-model caches (`results_cx2/src_cache/*.pt`) are **not** distributed; the Appendix E
scripts retrain the source models automatically when a cache is missing (GPU recommended; see
Appendix E.3 for the cost on an NVIDIA RTX 4060 Laptop GPU, CUDA 12.8, PyTorch 2.11.0+cu128).

## Licenses

- **Code**: MIT License (see `LICENSE`).
- **Derived results and figures** (`results/`, `results_cx2/`, `results_p0/`, `figures/`,
  `data/` modeling tables): CC-BY-4.0 (see `DATA_LICENSE.md`).
- Third-party datasets remain under their own licenses and are not redistributed.

## Citation

See `CITATION.cff`. A Zenodo DOI will be added upon acceptance of the manuscript.

## Contact

2024212097@bupt.cn (ORCID 0009-0006-1813-333X)
