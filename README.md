# SUMO Emergency-Vehicle Signal Pre-emption — Entrecampos, Lisbon

Simulation code and data for the MSc thesis **“Traffic Lights Scenario for Emergency Vehicle Response Time Optimisation”** (Francisco Cruz, NOVA IMS — Universidade Nova de Lisboa, 2026, supervised by Mijail Naranjo Zolotov). A rule-based, distance-triggered traffic-light pre-emption policy for emergency vehicles is evaluated on a SUMO model of the Entrecampos junction complex in Lisbon, measuring both the benefit to emergency vehicles and the cost imposed on background traffic. Across 30 paired trials, a 50 m trigger distance reduced mean emergency travel time by 23.3% while adding +6.3 s (+4.7%) to mean background travel time; larger trigger distances produced no additional emergency benefit but a higher background cost.

## Repository structure

```
run_experiments.py    Experiment driver: builds demand, runs 30 trials × 5 conditions
                      (baseline + trigger distances 50/100/150/200 m) via TraCI,
                      writes emergency_metrics_priority.csv and background_metrics.csv
data_analysis.py      Statistical analysis: produces report_01–05 CSVs and all figures
entrecampos.net.xml   SUMO network of Entrecampos (OpenStreetMap import, refined in NetEdit)
entreampos.rou.xml    Vehicle's Configuration
thesis_analysis/      Output CSVs and figures as reported in the thesis
LICENSE               MIT
```

## Requirements

- SUMO 1.27.0 with the `SUMO_HOME` environment variable set (the script uses `$SUMO_HOME/tools` for TraCI and `randomTrips.py`)
- Python 3.10+ with: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`

```bash
pip install pandas numpy scipy matplotlib seaborn
```

## How to run

```bash
python run_experiments.py    # ~30 trials × 5 conditions, 1,800 s each, step length 1.0 s
python data_analysis.py      # regenerates all report CSVs and figures from the metrics files
```

## Reproducibility

All stochastic elements derive from three fixed, recorded seeds (see also Appendix B of the thesis):

1. **Emergency departure times** — for each trial *t* (1–30), the ten departure times are drawn with Python's `random.Random(42 + t)`, i.e. seeds 43–72. These draws are the only element that varies between trials.
2. **Background demand** — `randomTrips.py` is invoked without an explicit `--seed` and therefore uses its documented default (42): background demand is identical across the 30 trials of a demand scenario, which is what makes the within-trial ON/OFF pairing exact.
3. **Microsimulation** — the SUMO configuration sets no seed, so runs use SUMO's documented default (23423) and are deterministic given identical inputs.

Given the same SUMO version, the full campaign is exactly reproducible.

## Citation

> Cruz, F. (2026). *Traffic Lights Scenario for Emergency Vehicle Response Time Optimisation.* MSc thesis, NOVA Information Management School, Universidade Nova de Lisboa.

Released as **v1.0-thesis** (the state of the code and data at thesis submission). Licensed under the MIT License.
