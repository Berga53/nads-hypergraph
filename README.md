# Hypergraph influence maximization

This repository builds yearly municipality-company hypergraphs, derives one
real company-size parameter per hyperedge and one population-derived weight per
municipality node, simulates generalized influence propagation (GIP), and
optimizes the municipality seed set with NaDS.

The NaDS implementation has no MG phase and no connectivity requirement on
seed sets.

The default and only supported evaluator uses a
shared company gate, excludes direct self-return, and scores every state with
population weights and discount `(1-gamma)^t`, including seeds at `t=0`.
The active results folder contains only this model. Previous outputs are
archived outside it under `misc/legacy_outputs/`. See [model and runner notes](MODEL_MIGRATION.md).

## Project layout

```text
data/
  raw/                 original input datasets
  processed/           generated clean networks and parameters
notebooks/             interactive analysis
results/
  experiment/          current-model results.csv checkpoint
scripts/               experiment runner
src/                   preparation, GIP, and NaDS source modules
```

Core code:

- `src/prepare_data.py` — creates clean yearly incidences and parameters;
- `src/gip_model.py` — GIP diffusion and parameter alignment;
- `src/nads.py` — fixed-budget NaDS seed search;
- `scripts/run_experiment.py` — the sole experiment runner, with an alpha/seed-budget
  grid across selected years and one progressively saved CSV;
- `notebooks/graph_analysis.ipynb` — interactive longitudinal analysis;
- `notebooks/influence_maximization.ipynb` — inspect saved propagation or call the runner interactively;
- `notebooks/results.ipynb` — compare the current saved experiment results.

The complete mathematical specification, modeling assumptions, hypergraph
propagation equations, and NaDS search definition are in
[`MODEL_AND_ALGORITHM.md`](MODEL_AND_ALGORITHM.md).

Exploratory notebooks, superseded scripts, and previous generated outputs are
archived under the gitignored `misc/` directory.

## 1. Prepare the data

The required inputs are:

- `data/raw/rete_IPL.dta` and `data/raw/dataset_partecipate.xlsx` for graph
  incidences and company data;
- `data/raw/Anagrafe_comuni.csv` for the municipality fiscal-code (`CF`) to BDAP
  identifier (`Id_Ente`) crosswalk;
- `data/raw/popolazione.csv`, whose `BDAP` column is the `Id_Ente` key and whose
  other columns are annual populations.

```bash
python src/prepare_data.py
```

This writes:

- `data/processed/rete_YYYY.csv` — municipality-company incidences and quotas;
- `data/processed/hyperedge_parameters.csv` — one row per hyperedge/year;
- `data/processed/node_parameters.csv` — one row per municipality/year with the
  fiscal code, BDAP identifier, raw population, and model-ready node weight;
- `data/processed/data_quality_by_year.csv` — missing-data diagnostics;
- `data/processed/run_summary.json` — build metadata.

Missing company values are taken only from another year of the same company.
Each metric retains its source year, distance, and imputation flag. Companies
without any quantitative size information receive the neutral within-year
median parameter and are explicitly flagged.

Municipality population is joined in two explicit steps:

```text
rete_IPL.codfisc_ente -> Anagrafe_comuni.CF
Anagrafe_comuni.Id_Ente -> popolazione.BDAP
```

Fiscal codes and BDAP identifiers are kept as strings during this join to
preserve leading zeroes. Historic or merged municipalities that have no exact
BDAP population match retain a missing raw `population` and receive the
within-year median only in `node_weight_population`; `node_weight_basis` flags
every such fallback.

For prediction or causal work, avoid future-year information with:

```bash
python src/prepare_data.py --imputation-direction past
```

## 2. Analyze the yearly graphs

Open `notebooks/graph_analysis.ipynb` and run it from the top. It compares graph size,
hyperedge size, municipality degree, connected components, year-to-year
turnover, and degree distributions. It also provides company and municipality
rankings for a configurable year and reports parameter data quality.
The detailed view defaults to the latest processed year. Structural summaries
include singleton companies; experiment counts use the runner's company-size filter.

## 3. Run influence maximization

Open `notebooks/influence_maximization.ipynb` and run it from the top. By default
it reads the saved latest-year result with alpha 0.1 and seed budget 20, then
reconstructs starting and optimized propagation. It shows selected municipalities,
active and ever-reached counts, discounted score, stopping reason, and update count.
Coverage includes seeds and uses the filtered experiment network as its denominator.

The notebook imports `ExperimentConfig` and `load_year_model` from
`scripts/run_experiment.py`. Edit `CONFIG = replace(ExperimentConfig(), ...)`
to choose another configuration. Defaults match the CLI: beta 0.5 for both
weights, stress level 0, and horizon 20. `RUN_SEARCH = False` inspects a saved
row matching all settings. `RUN_SEARCH = True` calls the runner's `run_year`
search and displays the outcome in memory; the default allocation is 800 seconds.
Use the CLI below to save checkpoints or run multiple settings.

`CONFIG.d = 2` permits one-for-one exchanges; 4 also permits two-for-two exchanges.
Set these frozen configuration fields through `replace`, including
`max_neighbors_per_phase` and `search_seconds_per_year` to control search runtime.
Population weights value states in the objective; they do not alter transmission
or thresholds. Beta 1 uses full population differences, 0 gives equal valuation,
and the weights are normalized to mean one on the filtered network.

## 4. Run the experiment

Set up the environment if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`run_experiment.py` is the only experiment command. It contains its own
configuration, data loading, per-year search, and checkpoint logic, and calls
`src/gip_model.py` and `src/nads.py` directly.

The experiment crosses:

- `alpha`: 0.05, 0.10, and 0.20;
- seed budget: 10, 20, and 40 municipalities;
- the latest processed year by default, or any explicit year list;
- the tempered model (`edge_weight_beta=node_weight_beta=0.5`).

It assigns a round 800 seconds to every scenario/year search:

```text
3 alpha values × 3 budgets × 800 seconds = 2 search hours per year
```

The tempered weights use square-root company-size and population differences,
with both means equal to one. Initial seeds follow the existing degree-first
rule. NaDS uses one-for-one exchanges, up to 5,000 candidates per phase, and
restarts with reproducible neighbor orderings if time remains after convergence.

The default GIP settings retain `h0=l0=1`, `theta_l=2`, `theta_h=50`, and
`alpha=0.1`. They imply a lower path `0.2^t` and an upper path
`5 * 0.2^(t-1)`, so the cap remains 25 times the activation floor at every
step. The runner defaults to a fixed `--horizon 20`, with exactly 20 updates
for every seed set. `gamma=0.1` discounts scores by `0.9^t`. Use
`--early-stopping --max-iter 999` for numerical truncation with `eps=0.01`;
this is not a certified bound on omitted score. The stress
gate remains zero because ownership quotas span many orders of magnitude; a
positive gate would need an externally justified unit-specific calibration.

Run the latest available year with:

```bash
source .venv/bin/activate
caffeinate -i python scripts/run_experiment.py
```

Choose one year or a list of years with:

```bash
python scripts/run_experiment.py --years 2020
python scripts/run_experiment.py --years 2018 2019 2020
```

Every completed year/parameter combination is immediately checkpointed to
`results/experiment/results.csv`. Each row contains the year, model and search
parameters, `start_spread`, `finish_spread`, `start_list`, `finish_list`, stopping
reason, and actual diffusion updates. Seed lists are JSON arrays of unique
municipality IDs; the notebook joins them to readable municipality names.
Completed year/parameter instances are skipped on resume. Model version tags,
input hashes, and score keys are not written to the CSV. After changing the
processed input data, use a fresh output directory to avoid reusing old scores.

For a short validation run:

```bash
python scripts/run_experiment.py \
  --years 2023 --alphas 0.1 --seed-budgets 20 --search-seconds 10 \
  --output-dir /tmp/alpha_seed_smoke
```

## 5. Inspect stored experiment results

Open `notebooks/results.ipynb`. It reads `results/experiment/results.csv`, identifies which
hyperparameters actually vary, checks the experiment grid, and compares spread
levels, optimization gains, interactions, and optimized-set stability across
alpha and seed budget. It also shows robust municipality selections and
compares company propagation activity and ablation importance across alpha
settings for a configurable year and seed budget.
Reconstruction uses the shared runner loader and every saved propagation setting.
Set `RESULT_FILTERS` when multiple horizons or other settings occur within a year,
so alpha and seed-budget comparisons hold the remaining parameters fixed.


## 6. Small example and automated verification

```python
import numpy as np
from scipy.sparse import csr_matrix
from src.gip_model import GIPParameters, gip, prepare_incidence

B = prepare_incidence(csr_matrix([[1.0], [1.0]]))
p = GIPParameters(h0=10, l0=0, theta_l=1, theta_h=1, gamma=0.5, eps=0.75)
result = gip(B, np.array([1.0]), np.array([1.0, 0.0]), p,
             node_weights=np.array([2.0, 3.0]), alpha=1, horizon=2)
assert result.total_spread == 4.0  # 2 + 0.5*3 + 0.25*2
assert result.iterations == 2
```

Use `horizon=0` to score seeds alone. With `horizon=None`, `max_iter` is the
finite update cap; the norm test before step j uses exponent j-1. Equality
never terminates evaluation. `prepare_incidence` holds a private sparse
snapshot and cached elementwise square; prepare a new snapshot if B changes.
SciPy is an explicit dependency. No diffusion-level normalization is performed.

Run the automated synthetic suite (no empirical experiment):

```bash
MPLCONFIGDIR=/tmp/rete-ipl-mpl MPLBACKEND=Agg python -m unittest discover -s tests -v
```

It checks the explicit componentwise reference, graph reduction, discounting,
seed contribution, gate semantics, stopping and bound indexing, fixed-horizon
monotonicity, input validation, sparse storage, in-memory cache identity, checkpoint
resume, and runner/notebook/NaDS integration on three synthetic nodes.
