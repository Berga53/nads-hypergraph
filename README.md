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
  yearly_influence/    generated all-years experiment tables
scripts/               executable experiment runners
src/                   preparation, GIP, and NaDS source modules
```

Core code:

- `src/prepare_data.py` — creates clean yearly incidences and parameters;
- `src/gip_model.py` — GIP diffusion and parameter alignment;
- `src/nads.py` — fixed-budget NaDS seed search;
- `scripts/run_yearly_influence_experiment.py` — comparable seed selection for
  every year in one run;
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

Run the fixed-parameter experiment once for every processed year:

```bash
python scripts/run_yearly_influence_experiment.py
```

The default experiment uses 20 seeds and five seconds of NaDS search per year.
Every year uses the same graph filter, GIP thresholds, company-size weighting,
population weighting, seed budget, and search configuration. The initial set is
deterministic: highest municipality degree, then population weight, then fiscal
code. To use a longer common search budget:

```bash
python scripts/run_yearly_influence_experiment.py --search-seconds 30
```

Outputs under `results/yearly_influence/` are:

- `year_summary.csv` — graph size, spread improvement, calls, and runtime;
- `selected_municipalities.csv` — the 20 chosen municipalities per year, ranked
  by their individual population-weighted GIP spread;
- `selection_turnover.csv` — retained/new/lost selections and consecutive-year
  Jaccard similarity;
- `selection_frequency.csv` — municipalities most consistently selected across
  the complete period;
- `experiment_config.json` — exact fixed settings and experiment definition.

Here “central” means membership in the fixed-budget set that maximizes
population-weighted cumulative GIP spread. Structural degree and ownership-
weighted degree are included beside each selected municipality for comparison.
