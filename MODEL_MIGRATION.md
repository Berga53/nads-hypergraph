# Model behavior and experiment runner

`src/gip_model.py` is the sole supported evaluator. The company gate uses total
pressure from all shareholders and the strict test p > tau. Returned influence
excludes the recipient's own contribution. Scores sum population-weighted states
with discount (1-gamma)^t, including seed value at t=0. Gamma and population
weights do not alter a fixed-horizon trajectory.

Ownership units, economic parameter construction, and NaDS search logic are
unchanged. The mathematical definition is in `MODEL_AND_ALGORITHM.md`.

## Evaluation

- `gip(..., horizon=T)` performs exactly T updates and scores T+1 states, including
  T=0. The runner and influence notebook default to T=20.
- `horizon=None` selects numerical early stopping, with finite `max_iter=999` by
  default. Before update j, the norm uses exponent j-1. Equal consecutive states
  do not stop evaluation. This truncation is not a bound on the omitted score.
- Gamma must satisfy 0 < gamma < 1. Shapes, finite/nonnegative inputs, and valid
  lower/upper bounds are checked.
- `prepare_incidence(B)` caches the elementwise square in a private sparse
  snapshot. Construct a new snapshot when B changes. Materially negative raw
  influence or non-finite arithmetic fails; only scale-aware negative roundoff
  is set to zero.
- `GIPResult` reports cumulative discounted scores, states, stopping reason,
  horizon, maximum updates, and actual iterations.

## Plain result files

`run_experiment.py` checkpoints `results/experiment/results.csv`. Rows retain
all numerical model/search settings, seed lists, start/finish scores, stopping
reason, and update count. There are no model-version, input-fingerprint, or
score-key columns. Resume compares year and parameters. If processed input data
change, select a fresh output directory before rerunning the experiment.

The results notebook reads only the current experiment table. Historical result
tables, old notebook outputs, and diagnostics comparing both recurrences are
archived under `misc/legacy_outputs/`, outside the active results tree. Current
scores and seed lists were retained during cleanup; historical scores were not
relabelled or regenerated. Active notebook outputs show the current data and model.

## Input fingerprint and in-memory caches

An input fingerprint is a SHA-256 checksum of the yearly network and the two
company/municipality parameter CSVs. It detects changes in file content; it is
not an economic parameter and never affects diffusion or score values.

It is now used only inside the notebook's in-memory cache and is not saved in
results. The cache key also covers the actual incidence, company weights, node
weights, seed state, and every evaluation setting. Notebook reconstruction
compares recalculated and saved scores as a consistency check.

## Experiment runner

`run_experiment.py` is the sole experiment entry point. Its configuration,
data-loading helpers, per-year NaDS search, and checkpoint logic live in that
file. It calls `src/gip_model.py` and `src/nads.py` directly.

The runner varies alpha and seed budget with tempered weights. Use `--years`
with one or more years; omitting it selects the latest processed year. Existing
horizon, runtime, and output-directory options are unchanged. There is no separate
profile-comparison runner or secondary experiment output directory.

Both propagation notebooks import the runner's `load_year_model` helper. The
influence notebook also uses `ExperimentConfig` and calls `run_year` when
`RUN_SEARCH=True`; its default saved-result mode reconstructs starting and optimized
trajectories without optimization. Its coverage plots distinguish active states,
ever-reached municipalities, and discounted scores. The results notebook uses each
row's saved evaluation settings and checks reconstructed scores.
