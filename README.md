# Hypergraph influence maximization

This repository builds yearly municipality-company hypergraphs, derives one
real company-size parameter per hyperedge and one population-derived weight per
municipality node, simulates generalized influence propagation (GIP), and
optimizes the municipality seed set with NaDS.

The NaDS implementation has no MG phase and no connectivity requirement on
seed sets.

## Project layout

```text
data/
  raw/                 original input datasets
  processed/           generated clean networks and parameters
notebooks/             interactive analysis
results/
  experiment/          one checkpointed results.csv experiment table
scripts/               executable experiment runners
src/                   preparation, GIP, and NaDS source modules
```

Core code:

- `src/prepare_data.py` — creates clean yearly incidences and parameters;
- `src/gip_model.py` — GIP diffusion and parameter alignment;
- `src/nads.py` — fixed-budget NaDS seed search;
- `scripts/run_yearly_influence_experiment.py` — comparable seed selection for
  every year in one run;
- `scripts/run_experiment.py` — configurable tempered alpha/seed-budget run
  with one progressively saved CSV;
- `notebooks/graph_analysis.ipynb` — interactive longitudinal analysis;
- `notebooks/influence_maximization.ipynb` — interactive single-year experiment.

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

## 3. Run influence maximization

Open `notebooks/influence_maximization.ipynb` and run it from the top. The notebook:

1. builds the selected year's HyperNetX hypergraph;
2. aligns one real company parameter with each incidence-matrix column and one
   population weight with each row;
3. defines population-weighted total GIP spread as the optimization objective;
4. runs unrestricted fixed-budget NaDS exchanges;
5. reports the selected municipalities and diffusion path.

`D = 2` gives one-for-one exchanges. `D = 4` additionally permits two-for-two
exchanges. `MAX_NEIGHBORS_PER_PHASE` and `SEARCH_SECONDS` control runtime.

`NODE_WEIGHT_BETA = 1` uses full relative population differences, while `0`
recovers the original equal-node objective. Node weights are normalized to mean
one, so their scale stays comparable across years. They change how activated
municipalities are valued in the objective; they do not alter GIP transmission
or thresholds.

## 4. Compare influence-central municipalities across all years

Create/activate the local environment and run the complete experiment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_yearly_influence_experiment.py
```

The default run has a four-hour total NaDS budget. With the current 14 years
and two profiles, this gives about 514 seconds to each of the 28 profile/year
runs. Small post-search reporting overhead means actual elapsed time can be a
little over four hours. If a NaDS search converges early, the runner restarts it
with another reproducible neighborhood ordering and retains the best result.

The two profiles provide a controlled sensitivity comparison:

- `tempered` (primary): company-size and population exponents are both 0.5.
  The square-root transform preserves empirical ordering but limits domination
  by extreme values and reduces sensitivity to measurement/imputation noise.
- `full`: both exponents are 1.0, so the complete relative differences in the
  empirical variables enter the model. This tests whether the selections are
  robust to stronger heterogeneity.

Both profiles normalize node and edge weights to mean one and use the same
graphs, GIP dynamics, 20-seed budget, deterministic degree-first initial set,
and NaDS settings. The search uses one-for-one exchanges (`d=2`), evaluates up
to 5,000 candidates per phase, and remembers 50,000 recent seed sets. Keeping
the optimizer fixed makes profile differences attributable to model weighting,
not unequal search effort.

The shared GIP settings retain `h0=l0=1`, `theta_l=2`, `theta_h=50`, and
`alpha=0.1`. They imply a lower path `0.2^t` and an upper path
`5 * 0.2^(t-1)`, so the cap remains 25 times the activation floor at every
step. `gamma=0.1` and `eps=0.01` provide a finite numerical horizon. The stress
gate remains zero because ownership quotas span many orders of magnitude; a
positive gate would need an externally justified unit-specific calibration.

For a short smoke test, or to run only the primary profile:

```bash
python scripts/run_yearly_influence_experiment.py --years 2023 --search-seconds 30
python scripts/run_yearly_influence_experiment.py --profiles tempered
```

Outputs under `results/yearly_influence/` are:

- `year_summary.csv` — graph size, profile parameters, spread improvement,
  restarts, objective calls, and runtime;
- `selected_municipalities.csv` — the 20 chosen municipalities per profile and
  year, ranked by individual population-weighted GIP spread;
- `profile_comparison.csv` — within-year overlap and Jaccard similarity between
  profile selections (spread columns use each profile's own objective);
- `selection_turnover.csv` — retained/new/lost selections and consecutive-year
  Jaccard similarity within each profile;
- `selection_frequency.csv` — municipalities most consistently selected within
  each profile over the complete period;
- `experiment_config.json` — exact settings, profiles, and budget allocation.

Here “central” means membership in the fixed-budget set that maximizes
population-weighted cumulative GIP spread. Structural degree and ownership-
weighted degree are included beside each selected municipality for comparison.

## 5. Run the experiment

The experiment crosses:

- `alpha`: 0.05, 0.10, and 0.20;
- seed budget: 10, 20, and 40 municipalities;
- the latest processed year by default, or any explicit year list;
- the tempered model (`edge_weight_beta=node_weight_beta=0.5`).

It assigns a round 800 seconds to every scenario/year search:

```text
3 alpha values × 3 budgets × 800 seconds = 2 search hours per year
```

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

Every completed year/parameter combination is immediately checkpointed to the
single file `results/experiment/results.csv`. Each row contains the year, every
model and search parameter, `start_spread`, `finish_spread`, `start_list`, and
`finish_list`. The two lists are JSON arrays of unique municipality IDs inside
the CSV; the notebook joins them to readable municipality names. A matching
year/parameter row is replaced rather than duplicated; when an interrupted
command is started again, already saved combinations are skipped.
For a short validation run:

```bash
python scripts/run_experiment.py \
  --years 2023 --alphas 0.1 --seed-budgets 20 --search-seconds 10 \
  --output-dir /tmp/alpha_seed_smoke
```

## 6. Inspect stored experiment results

Open `notebooks/results.ipynb`. It reads `results.csv`, identifies which
hyperparameters actually vary, checks the experiment grid, and compares spread
levels, optimization gains, interactions, and optimized-set stability across
alpha and seed budget. It also shows robust municipality selections and
compares company propagation activity and ablation importance across alpha
settings for a configurable year and seed budget.
