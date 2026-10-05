# Datasets

The three public datasets used in the manuscript are **not redistributed** in this repository.
Download them from their original repositories:

| Dataset | Role | Download |
|---|---|---|
| NASA PCoE battery data (Saha & Goebel, 2007) | Target (4 LCO 18650 cells) | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ |
| MIT-Stanford (Severson et al., 2019) | Source (124 LFP cells) | https://data.matr.io/1/ |
| CALCE battery data (University of Maryland) | Target (16 LCO cells) | https://calce.umd.edu/battery-data |

The compressed modeling tables under `data/` are derived from these datasets with the parsing
scripts in `code/data_prep/`; uncompress them with `gunzip -k data/*.gz` before running the
pipelines.

For CALCE-based prognostics context, the benchmark of He et al. (J. Power Sources 196 (2011)
10314-10321, https://doi.org/10.1016/j.jpowsour.2011.08.040) is cited as [29] in the manuscript.
