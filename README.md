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
Population weights enter source pressure and company gates as well as scoring.
The numerical objective is the infinite discounted sum, stopped by tolerance.
New outputs use a compact `results/results.csv` layout. Every accepted NaDS
improvement is stored inline, and fixed parameters and provenance are grouped
in typed JSON columns at the end of each row.

## Project layout

```text
data/
  raw/                 original input datasets
  processed/           generated clean networks and parameters
notebooks/             interactive analysis
results/
  results.csv          compact experiment checkpoint with NaDS histories
  geographical_results.csv  optional ISTAT macro-area experiment checkpoint
scripts/               experiment runner
src/                   preparation, GIP, and NaDS source modules
```

Core code:

- `src/prepare_data.py` — creates clean yearly incidences and parameters;
- `src/gip_model.py` — GIP diffusion and parameter alignment;
- `src/nads.py` — fixed-budget NaDS seed search;
- `scripts/run_experiment.py` — the national alpha/seed-budget experiment;
- `scripts/run_geographical_experiment.py` — the same experiment on the five
  municipality-induced ISTAT macro-area networks;
- `notebooks/graph_analysis.ipynb` — interactive longitudinal analysis;
- `notebooks/influence_maximization.ipynb` — inspect saved propagation or call the runner interactively;
- `notebooks/results.ipynb` — compare versioned revised-model results.

Propagation formulas and numerical conventions are documented in
[`src/gip_model.py`](src/gip_model.py); the exchange search is in
[`src/nads.py`](src/nads.py).

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

The notebook also detects Louvain communities on each unweighted bipartite
municipality-company graph after applying the experiment's minimum company-size
filter. It compares the municipality assignments with the five ISTAT geographical
ripartizioni using purity, homogeneity, completeness, normalized mutual information,
and adjusted Rand index. The longitudinal diagnostics and selected-year composition
plot distinguish geographically concentrated communities from a one-to-one recovery
of the five macro-areas.

## 3. Run influence maximization

Open `notebooks/influence_maximization.ipynb` and run it from the top. By default
it performs one verbose NaDS search for the latest year with alpha 0.1 and seed
budget 20, without repeated restarts from the initial point. It shows selected municipalities,
active and ever-reached counts, discounted score, stopping reason, and update count.
Coverage includes seeds and uses the filtered experiment network as its denominator.

The notebook imports `ExperimentConfig` and `load_year_model` from
`scripts/run_experiment.py`. Edit `CONFIG = replace(ExperimentConfig(), ...)`
to choose another configuration. Defaults match the CLI: beta 0.5 for both
weights, stress level 0, and discounted-state tolerance `eps=0.01`. Set
`RUN_SEARCH = False` to inspect a saved row matching all settings instead.
The interactive search displays its outcome in memory; the default allocation
is 300 seconds and `VERBOSE_NADS = True` shows live NaDS progress.
Use the CLI below to save checkpoints or run multiple settings.

`CONFIG.d = 2` permits one-for-one exchanges; 4 also permits two-for-two exchanges.
Set these frozen configuration fields through `replace`, including
`max_neighbors_per_phase` and `search_seconds_per_year` to control search runtime.
Population weights enter both transmission pressure and valuation. Beta 1 uses
full relative population differences, 0 gives equal weights, and the existing
construction normalizes them to mean one on the filtered network. Recalibration
can now change gates and trajectories. `CONFIG.h0` is thesis `u0`, used for both
seed amplitudes and reference caps; no independent seed intensity remains.

## 4. Run the experiment

Set up the environment if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`run_experiment.py` is the national experiment command. It contains its own
configuration, data loading, per-year search, and checkpoint logic, and calls
`src/gip_model.py` and `src/nads.py` directly.

The experiment crosses:

- `alpha`: 0.05, 0.10, and 0.20;
- seed budget: 10, 20, and 40 municipalities;
- the latest processed year by default, or any explicit year list;
- the tempered model (`edge_weight_beta=node_weight_beta=0.5`).

It assigns 300 seconds to every scenario/year search:

```text
3 alpha values × 3 budgets × 300 seconds = 45 search minutes per year
```

The tempered weights use square-root company-size and population differences,
with both means equal to one. Initial seeds follow the existing degree-first
rule. NaDS uses one-for-one exchanges, up to 5,000 candidates per phase, and
restarts with reproducible neighbor orderings if time remains after convergence.

The default GIP settings retain `h0=l0=1`, `theta_l=2`, `theta_h=50`, and
`alpha=0.1`. They imply a lower path `0.2^t` and an upper path
`5 * 0.2^(t-1)`, so the cap remains 25 times the activation floor at every
step. The runner evaluates until `norm((1-gamma)^t*q_t,2) <= eps`, with
`eps=0.01` and `gamma=0.1`. No fixed horizon exists. Optional `--max-iter N`
is an operational guard; hitting it above tolerance aborts ranking/checkpointing
as incomplete. The norm is not a certified bound on omitted score. The stress
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
`results/results.csv`. The ordinary columns contain the varying dimensions
(`year`, `alpha`, and `seed_budget`) and the main outcome fields. Seed lists are
JSON arrays of unique municipality IDs; the notebook joins them to readable
municipality names.

`nads_history` contains one entry for every NaDS restart and one event for the
initial evaluation plus each strict improvement. Every event records the
objective-call number, elapsed seconds within that search, objective value, and
complete municipality seed list. The final columns group fixed settings and
provenance as `network_parameters`, `gip_parameters`, `nads_parameters`,
`graph_metadata`, `model_metadata`, and `run_metadata`. Optional values such as
`max_iter` are omitted from their JSON object when unset, so obsolete blank
columns such as `seed_intensity` and `horizon` are gone. Resume validation uses
the inline fingerprints, settings, and score keys. Files using the earlier wide
schema are deliberately not reused or overwritten; start a new output file or
move the old file before rerunning.

### Starting-point sensitivity on one fixed setup

To test whether NaDS reaches different solutions from different initial seed
sets, use the dedicated single-year runner. For example, this compares the
degree-ranked baseline with nine distinct random starts while holding the 2023
model, alpha, seed budget, and neighborhood-order seed fixed:

```bash
python scripts/run_starting_point_experiment.py \
  --year 2023 --alpha 0.1 --seed-budget 20 \
  --num-starts 10 --search-seconds 300
```

Each starting point gets one NaDS run; the time is a per-start safety cap and
the optimizer does not restart repeatedly from the same point. This isolates
starting-set sensitivity more cleanly than changing the normal runner's
`random_seed`, which changes neighborhood ordering but not its starting set.
Use `--random-only` to omit the degree-ranked baseline. Random starting sets
are reproducible through `--initialization-seed`; every optimization uses the
same `--search-random-seed`.

All runs and setups append to one checkpointed file,
`results/starting_point_results.csv`. It uses the complete compact results
schema, including `nads_history`, followed by one final
`starting_point_metadata` JSON column for the experiment identity, run number,
and initialization method. Timing and model provenance remain in the shared
`run_metadata` and `model_metadata` columns. Re-running the same command resumes
completed starting points. The **Starting-point sensitivity** section of
`notebooks/results.ipynb` derives score ranges, distinct-solution counts,
Jaccard overlaps, plots, and municipality selection frequencies. Time-capped
final sets are incumbents rather than demonstrated local optima. Use `--output
another_file.csv` to write to a separate consolidated file.

For a short validation run:

```bash
python scripts/run_experiment.py \
  --years 2023 --alphas 0.1 --seed-budgets 20 --search-seconds 10 \
  --output-dir /tmp/alpha_seed_smoke
```

### Experiment by ISTAT geographical macro-area

`run_geographical_experiment.py` applies the same alpha/seed-budget grid separately
to `NORD-OVEST`, `NORD-EST`, `CENTRO`, `SUD`, and `ISOLE`. Municipalities are assigned
from `Anagrafe_comuni.csv` using the official `Codice_Zona`/`Dizione_zona` fields.
Each area is a municipality-induced sub-hypergraph: a company that spans areas may
appear in more than one area, but it is retained only where at least two municipal
shareholders remain. Company and population weights are realigned and normalized
inside each area.

Run all five areas for the latest year with:

```bash
python scripts/run_geographical_experiment.py
```

Choose years or macro-areas with:

```bash
python scripts/run_geographical_experiment.py --years 2018 2019 2020
python scripts/run_geographical_experiment.py --years 2023 --macroareas NORD-OVEST CENTRO SUD
```

The default grid takes up to 225 search minutes per year
(`5 areas × 3 alpha values × 3 budgets × 300 seconds`). Completed runs are
checkpointed and resumed from `results/geographical_results.csv`. Each row has a
plain `macro_area` column; `graph_metadata` records the partition rule, canonical
geography fingerprint, induced graph size, and partition-specific input fingerprint.

For a short validation run that does not touch the main checkpoint:

```bash
python scripts/run_geographical_experiment.py \
  --years 2023 --macroareas CENTRO --alphas 0.1 --seed-budgets 20 \
  --search-seconds 10 --output /tmp/geographical_smoke.csv
```

## 5. Inspect stored experiment results

Open `notebooks/results.ipynb`. It reads `results/results.csv`, identifies which
hyperparameters actually vary, checks the experiment grid, and compares spread
levels, optimization gains, interactions, and optimized-set stability across
alpha and seed budget. It also shows robust municipality selections and
compares company propagation activity and ablation importance across alpha
settings for a configurable year and seed budget. The inspected-result section
also expands the saved NaDS event history and plots accepted objective
improvements over elapsed search time.
Reconstruction uses the shared runner loader and every saved propagation setting.
Set `RESULT_FILTERS` when multiple tolerances or other settings occur within a year,
so alpha and seed-budget comparisons hold the remaining parameters fixed.


## 6. Small example and automated verification

```python
from scipy.sparse import csr_matrix
from src.gip_model import GIPParameters, gip_from_seeds, prepare_incidence

B = prepare_incidence(csr_matrix([[1.0], [1.0]]))
p = GIPParameters(u0=1, l0=0.1, theta_l=1, theta_h=2,
                  gamma=0.1, eps=1e-10)
result = gip_from_seeds(B, [0.4], [1, 0], p, budget=1,
                        node_weights=[1, 1], alpha=0.4).require_complete()
assert result.status == "tolerance_reached"
assert abs(result.score - 1 / (1 - 0.36)) < 1e-10
```

`h0` remains an alias of `u0`; scalars broadcast and node-wise vectors are
supported by the core API. Require `0 < l0 <= u0`, `theta_l*alpha < 1`,
`theta_h*u0 >= theta_l*l0`, `0 < gamma < 1`, and `eps > 0`.
`gip` accepts explicit `q0=u0*z`; `gip_from_seeds` accepts binary `z`.
`prepare_incidence` caches private CSR/CSC data, incident/owner lists, lazy
neighbors, and the elementwise square. Rebuild it after changing the network.
The evaluator applies no ownership or population normalization.

Run the synthetic suite, without an empirical experiment:

```bash
MPLCONFIGDIR=/tmp/rete-ipl-mpl MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v
```

It verifies sparse incidence, independent source sums, effective weights and
activity transitions; population-aware gates; graph reduction; bounds and geometric
convergence; numerical stopping and cutoff rejection; one-step/common-prefix
order; sparse storage; cache identity; checkpoint resume; and runner/notebook/NaDS
integration on three synthetic nodes. Historical notebook outputs are explicitly
labelled preserved and not rerun. Historical CSVs without the compact inline
schema and matching model metadata are not accepted for resuming the current
experiment.
