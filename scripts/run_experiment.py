"""Run the current GIP model on the tempered alpha/seed-budget grid.

Completed year/parameter instances are checkpointed to results.csv and skipped
on resume. This is the sole experiment runner; it contains the configuration,
network loading, per-year search, and checkpoint orchestration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import hypernetx as hnx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gip_model import (
    GIPParameters,
    aligned_edge_parameters,
    aligned_node_weights,
    gip_from_seeds,
    seed_state,
    prepare_incidence,
)
from src.model_identity import MODEL_VERSION, MODEL_METADATA, evaluation_cache_key, input_fingerprint
from src.nads import nads

DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
DEFAULT_ALPHAS = (0.05, 0.10, 0.20)
DEFAULT_SEED_BUDGETS = (10, 20, 40)
DEFAULT_SEARCH_SECONDS = 800.0
EDGE_WEIGHT_BETA = 0.5
NODE_WEIGHT_BETA = 0.5

# Every ExperimentConfig value is saved with a short, direct column name.
PARAMETER_COLUMNS = [
    "alpha",
    "seed_budget",
    "edge_weight_beta",
    "node_weight_beta",
    "min_edge_size",
    "edge_weight_mean",
    "node_weight_mean",
    "h0",
    "l0",
    "theta_l",
    "theta_h",
    "gamma",
    "eps",
    "stress_level",
    "max_iter",
    "delta",
    "xi",
    "d",
    "search_seconds",
    "max_neighbors_per_phase",
    "buffer_dim",
    "max_search_iterations",
    "random_seed",
]
IDENTITY_COLUMNS = [*MODEL_METADATA, "input_fingerprint"]
SCORE_KEY_COLUMNS = ["start_score_key", "finish_score_key"]
# Keep the historical public CSV columns. These two columns describe the new
# model rather than introducing independent seed intensity or a finite horizon.
CSV_PARAMETER_COLUMNS = PARAMETER_COLUMNS.copy()
CSV_PARAMETER_COLUMNS[CSV_PARAMETER_COLUMNS.index("max_iter"):CSV_PARAMETER_COLUMNS.index("max_iter")] = [
    "seed_intensity", "horizon",
]
RESULT_COLUMNS = [
    "year", *CSV_PARAMETER_COLUMNS,
    "start_spread", "finish_spread", "start_list", "finish_list",
    "stopping_reason", "diffusion_iterations",
]
INTERNAL_RESULT_COLUMNS = [
    *IDENTITY_COLUMNS, *SCORE_KEY_COLUMNS,
    *[name for name in RESULT_COLUMNS if name not in ("seed_intensity", "horizon")],
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Scalar settings shared across years; h0 is thesis u0 and seed amplitude."""

    seed_budget: int = 20
    min_edge_size: int = 2
    edge_weight_mean: float = 1.0
    edge_weight_beta: float = 0.5
    node_weight_mean: float = 1.0
    node_weight_beta: float = 0.5
    h0: float = 1.0
    l0: float = 1.0
    theta_l: float = 2.0
    theta_h: float = 50.0
    gamma: float = 0.1
    eps: float = 0.01
    alpha: float = 0.1
    stress_level: float = .0
    max_iter: int | None = None  # Optional operational guard, never a horizon.
    delta: float = 0.5
    xi: float = 0.01
    d: int = 2
    search_seconds_per_year: float = DEFAULT_SEARCH_SECONDS
    max_neighbors_per_phase: int = 5_000
    buffer_dim: int = 50_000
    max_search_iterations: int = 1_000
    random_seed: int = 42

    @property
    def gip_parameters(self) -> GIPParameters:
        return GIPParameters(
            h0=self.h0,
            l0=self.l0,
            theta_l=self.theta_l,
            theta_h=self.theta_h,
            gamma=self.gamma,
            eps=self.eps,
        )


def add_stopping_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Optional operational update guard; incomplete runs cannot be ranked/saved.")


def available_years(data_dir: Path) -> list[int]:
    return sorted(
        int(path.stem.removeprefix("rete_"))
        for path in data_dir.glob("rete_*.csv")
        if path.stem.removeprefix("rete_").isdigit()
    )


def deterministic_initial_seeds(
    node_ids: np.ndarray,
    structural_degree: pd.Series,
    node_weights: np.ndarray,
    budget: int,
) -> np.ndarray:
    """Start every year from the same degree-first deterministic rule."""
    candidates = pd.DataFrame(
        {
            "node_index": np.arange(len(node_ids)),
            "municipality_id": node_ids.astype("int64"),
            "structural_degree": structural_degree.reindex(node_ids).to_numpy(),
            "node_weight": node_weights,
        }
    ).sort_values(
        ["structural_degree", "node_weight", "municipality_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    initial = np.zeros(len(node_ids), dtype=float)
    initial[candidates.head(budget)["node_index"].to_numpy(dtype=int)] = 1.0
    return initial


def load_year_model(
    year: int,
    config: ExperimentConfig,
    *,
    data_dir: Path | None = None,
) -> dict[str, object]:
    """Prepare the same filtered, aligned network for the CLI and notebooks."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    incidences = pd.read_csv(data_dir / f"rete_{year}.csv")
    edge_sizes = incidences.groupby("CF Partecipata")["CF Comune"].nunique()
    valid_edges = edge_sizes.loc[edge_sizes.ge(config.min_edge_size)].index
    filtered = incidences.loc[
        incidences["CF Partecipata"].isin(valid_edges)
    ].copy()
    if filtered.empty:
        raise ValueError(f"No incidences remain for {year}")

    hypergraph = hnx.Hypergraph(
        filtered,
        edge_col="CF Partecipata",
        node_col="CF Comune",
        cell_weight_col="Quota",
    )
    incidence_matrix, matrix_node_ids, matrix_edge_ids = hypergraph.incidence_matrix(
        index=True,
        weights="weight",
    )
    node_ids = np.asarray(matrix_node_ids, dtype="int64")
    incidence_matrix = prepare_incidence(incidence_matrix)

    edge_weights, edge_audit = aligned_edge_parameters(
        matrix_edge_ids,
        year,
        data_dir / "hyperedge_parameters.csv",
        target_mean=config.edge_weight_mean,
        beta=config.edge_weight_beta,
    )
    node_weights, node_audit = aligned_node_weights(
        matrix_node_ids,
        year,
        data_dir / "node_parameters.csv",
        target_mean=config.node_weight_mean,
        beta=config.node_weight_beta,
    )
    structural_degree = filtered.groupby("CF Comune")["CF Partecipata"].nunique()
    ownership_weighted_degree = filtered.groupby("CF Comune")["Quota"].sum()
    return {
        "input_fingerprint": input_fingerprint(data_dir, year),
        "incidence": incidence_matrix,
        "node_ids": node_ids,
        "edge_ids": np.asarray(matrix_edge_ids, dtype="int64"),
        "edge_weights": edge_weights,
        "node_weights": node_weights,
        "edge_audit": edge_audit,
        "node_audit": node_audit,
        "structural_degree": structural_degree,
        "ownership_weighted_degree": ownership_weighted_degree,
        "incidence_count": len(filtered),
    }


def result_identity(year: int, *, data_dir: Path | None = None) -> dict[str, str]:
    return {**MODEL_METADATA, "input_fingerprint": input_fingerprint(DATA_DIR if data_dir is None else data_dir, year)}


def seed_score_key(model, seeds, config):
    return evaluation_cache_key(
        model["incidence"], model["edge_weights"], seed_state(seeds, config.gip_parameters),
        config.gip_parameters, node_weights=model["node_weights"],
        stress_level=config.stress_level, alpha=config.alpha, max_iter=config.max_iter,
    )


def config_from_row(row) -> ExperimentConfig:
    values = {name: row[name] for name in PARAMETER_COLUMNS}
    values["search_seconds_per_year"] = values.pop("search_seconds")
    for name in ("seed_budget", "min_edge_size", "d", "max_neighbors_per_phase", "buffer_dim",
                 "max_search_iterations", "random_seed"):
        values[name] = int(values[name])
    values["max_iter"] = None if pd.isna(values["max_iter"]) else int(values["max_iter"])
    return ExperimentConfig(**values)


def validate_saved_seeds(row, model, config):
    """Validate stored seeds and full evaluation keys before reuse/reconstruction."""
    for column, expected in MODEL_METADATA.items():
        if row[column] != expected:
            raise ValueError(f"Incompatible saved model: {column}")
    if row["stopping_reason"] != "tolerance_reached":
        raise ValueError("Incomplete saved evaluation cannot be ranked")
    if row["input_fingerprint"] != model["input_fingerprint"]:
        raise ValueError("Saved input fingerprint does not match current data; preserve the old result")
    output = {}
    for label in ("start", "finish"):
        ids = row[f"{label}_list"]
        ids = json.loads(ids) if isinstance(ids, str) else list(ids)
        if len(set(ids)) != len(ids) or len(ids) != config.seed_budget or set(ids) - set(model["node_ids"]):
            raise ValueError("Saved seeds do not match the network/budget")
        seeds = np.isin(model["node_ids"], ids).astype(float)
        if seed_score_key(model, seeds, config) != row[f"{label}_score_key"]:
            raise ValueError("Saved score key does not match seeds/network/parameters; preserve the old result")
        output[label] = seeds
    return output


def run_year(
    year: int,
    config: ExperimentConfig,
    *,
    verbose: bool,
) -> tuple[dict[str, object], pd.DataFrame, set[int]]:
    model = load_year_model(year, config)
    incidence_matrix = model["incidence"]
    node_ids = model["node_ids"]
    edge_weights = model["edge_weights"]
    node_weights = model["node_weights"]
    node_audit = model["node_audit"]
    structural_degree = model["structural_degree"]
    ownership_weighted_degree = model["ownership_weighted_degree"]

    if config.seed_budget > len(node_ids):
        raise ValueError(
            f"Seed budget {config.seed_budget} exceeds {len(node_ids)} nodes in {year}"
        )
    initial_seeds = deterministic_initial_seeds(
        node_ids,
        structural_degree,
        node_weights,
        config.seed_budget,
    )

    config.gip_parameters.validate(len(node_ids), config.alpha)
    if config.seed_budget < 1:
        raise ValueError("seed_budget must be positive")
    objective_calls = 0

    def simulate(seed_indicator: np.ndarray) -> float:
        result = gip_from_seeds(
            incidence_matrix,
            edge_weights,
            seed_indicator,
            config.gip_parameters,
            node_weights=node_weights,
            stress_level=config.stress_level,
            alpha=config.alpha,
            max_iter=config.max_iter,
        )
        return result.require_complete().total_spread

    def search_objective(seed_indicator: np.ndarray) -> float:
        nonlocal objective_calls
        objective_calls += 1
        return simulate(seed_indicator)

    # A single NaDS run may converge before its wall-clock limit. Restarting
    # with a new deterministic random seed uses the remaining allocation and
    # makes the final solution less dependent on one neighborhood ordering.
    started = time.monotonic()
    best_spread_history: list[float] | None = None
    best_seed_history: list[np.ndarray] | None = None
    best_restart = 0
    search_restarts = 0
    accepted_seed_sets_all_restarts = 0
    while True:
        remaining = config.search_seconds_per_year - (time.monotonic() - started)
        if remaining <= 0.01:
            break
        spread_history, seed_history, _ = nads(
            objective=search_objective,
            x0=initial_seeds,
            delta=config.delta,
            xi=config.xi,
            d=config.d,
            max_time=remaining,
            buffer_dim=config.buffer_dim,
            max_neighbors_per_phase=config.max_neighbors_per_phase,
            max_iterations=config.max_search_iterations,
            neighbor_graph=incidence_matrix,
            random_seed=config.random_seed + search_restarts,
            verbose=int(verbose),
        )
        search_restarts += 1
        accepted_seed_sets_all_restarts += len(seed_history)
        if (
            best_spread_history is None
            or spread_history[-1] > best_spread_history[-1]
        ):
            best_spread_history = spread_history
            best_seed_history = seed_history
            best_restart = search_restarts

    if best_spread_history is None or best_seed_history is None:
        raise RuntimeError("NaDS search budget was too short to evaluate a seed set")
    elapsed = time.monotonic() - started
    best_seeds = best_seed_history[-1]
    selected_indices = np.flatnonzero(best_seeds)
    selected_ids = node_ids[selected_indices]

    # Rank the chosen municipalities by their individual influence under the
    # same yearly graph and fixed parameters. This adds only `seed_budget` runs.
    individual_spread: dict[int, float] = {}
    for node_index, municipality_id in zip(selected_indices, selected_ids):
        single_seed = np.zeros(len(node_ids), dtype=float)
        single_seed[int(node_index)] = 1.0
        individual_spread[int(municipality_id)] = simulate(single_seed)

    selected = node_audit.reindex(selected_ids).reset_index()
    selected["year"] = int(year)
    selected["individual_weighted_spread"] = selected["municipality_id"].map(
        individual_spread
    )
    selected["structural_degree"] = selected["municipality_id"].map(
        structural_degree
    )
    selected["ownership_weighted_degree"] = selected["municipality_id"].map(
        ownership_weighted_degree
    )
    selected = selected.sort_values(
        ["individual_weighted_spread", "structural_degree", "municipality_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    selected.insert(1, "selection_rank", np.arange(1, len(selected) + 1))

    selected_columns = [
        "year",
        "selection_rank",
        "municipality_id",
        "municipality_tax_code",
        "municipality_bdap_id",
        "municipality_name",
        "population",
        "population_data_available",
        "node_weight",
        "node_weight_basis",
        "individual_weighted_spread",
        "structural_degree",
        "ownership_weighted_degree",
    ]
    selected = selected[selected_columns]

    initial_ids = node_ids[np.flatnonzero(initial_seeds)]
    initial_selected = node_audit.reindex(initial_ids).reset_index()
    initial_selected["structural_degree"] = initial_selected[
        "municipality_id"
    ].map(structural_degree)
    initial_selected = initial_selected.sort_values(
        ["structural_degree", "node_weight", "municipality_id"],
        ascending=[False, False, True],
        kind="stable",
    )

    best_result = gip_from_seeds(
        incidence_matrix,
        edge_weights,
        best_seeds,
        config.gip_parameters,
        node_weights=node_weights,
        stress_level=config.stress_level,
        alpha=config.alpha,
        max_iter=config.max_iter,
    )
    best_result.require_complete()
    summary: dict[str, object] = {
        **MODEL_METADATA,
        "input_fingerprint": model["input_fingerprint"],
        "start_score_key": seed_score_key(model, initial_seeds, config),
        "finish_score_key": seed_score_key(model, best_seeds, config),
        **asdict(config),
        "stopping_reason": best_result.stopping_reason,
        "evaluation_mode": "infinite_discounted_tolerance",
        "year": int(year),
        "nodes": incidence_matrix.shape[0],
        "hyperedges": incidence_matrix.shape[1],
        "incidences": model["incidence_count"],
        "seed_budget": int(config.seed_budget),
        "edge_weight_beta": float(config.edge_weight_beta),
        "node_weight_beta": float(config.node_weight_beta),
        "search_budget_seconds": float(config.search_seconds_per_year),
        "initial_spread": float(best_spread_history[0]),
        "best_spread": float(best_spread_history[-1]),
        "relative_improvement_pct": float(
            100.0
            * ((best_spread_history[-1] / best_spread_history[0] - 1.0)
            if best_spread_history[0] else 0.0)
        ),
        "accepted_seed_sets": len(best_seed_history),
        "accepted_seed_sets_all_restarts": accepted_seed_sets_all_restarts,
        "search_restarts": search_restarts,
        "best_restart": best_restart,
        "search_objective_calls": int(objective_calls),
        "search_elapsed_seconds": float(elapsed),
        "diffusion_iterations": int(best_result.iterations),
        "population_match_pct_selected": float(
            100.0 * selected["population_data_available"].mean()
        ),
        "start_list": initial_selected["municipality_id"].astype(int).tolist(),
        "finish_list": selected["municipality_id"].astype(int).tolist(),
    }
    return summary, selected, set(int(value) for value in selected_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the tempered alpha and seed-budget experiment."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="One year or a space-separated list; default: latest available year.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=list(DEFAULT_ALPHAS),
        help="Threshold scales to compare (default: 0.05 0.10 0.20).",
    )
    parser.add_argument(
        "--seed-budgets",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEED_BUDGETS),
        help="Seed-set sizes to compare (default: 10 20 40).",
    )
    parser.add_argument(
        "--search-seconds",
        type=float,
        default=DEFAULT_SEARCH_SECONDS,
        help="NaDS allocation for each year/parameter search (default: 800).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing results.csv.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show live NaDS objective evaluations.",
    )
    add_stopping_arguments(parser)
    args = parser.parse_args()
    if args.max_iter is not None and args.max_iter < 0:
        parser.error("--max-iter must be non-negative")
    if any(not np.isfinite(alpha) or alpha < 0 or ExperimentConfig.theta_l * alpha >= 1 for alpha in args.alphas):
        parser.error("Every --alphas value must be non-negative with theta_l * alpha < 1")
    if any(budget <= 0 for budget in args.seed_budgets):
        parser.error("Every --seed-budgets value must be positive")
    if not np.isfinite(args.search_seconds) or args.search_seconds <= 0:
        parser.error("--search-seconds must be positive")
    args.alphas = sorted(set(float(value) for value in args.alphas))
    args.seed_budgets = sorted(set(int(value) for value in args.seed_budgets))
    return args


def parameter_values(config: ExperimentConfig) -> dict[str, object]:
    values = asdict(config)
    values["search_seconds"] = values.pop("search_seconds_per_year")
    return {column: values[column] for column in PARAMETER_COLUMNS}


def instance_mask(
    results: pd.DataFrame,
    instance: dict[str, object],
) -> pd.Series:
    if results.empty:
        return pd.Series(False, index=results.index, dtype=bool)
    mask = results["year"].eq(int(instance["year"]))
    for column in PARAMETER_COLUMNS:
        if instance[column] is None:
            mask &= results[column].isna()
        else:
            mask &= pd.to_numeric(results[column], errors="coerce").eq(instance[column])
    for column in IDENTITY_COLUMNS:
        mask &= results[column].eq(instance[column])
    mask &= results["stopping_reason"].eq("tolerance_reached")
    return mask


def result_metadata_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.metadata.json")


def _validate_result_metadata(results: pd.DataFrame, path: Path) -> None:
    missing = [column for column in INTERNAL_RESULT_COLUMNS if column not in results.columns]
    if missing:
        raise ValueError(f"{path} has an incompatible result format; missing {missing}")
    for column, expected in MODEL_METADATA.items():
        if not results[column].eq(expected).all():
            raise ValueError(f"{path} has incompatible model metadata ({column}); preserve the historical file")
    if not results["stopping_reason"].eq("tolerance_reached").all():
        raise ValueError("Incomplete evaluations cannot be loaded/saved as completed results")
    if results[IDENTITY_COLUMNS + SCORE_KEY_COLUMNS].isna().any().any():
        raise ValueError("Missing result identity or score keys")


def load_results(path: Path) -> pd.DataFrame:
    """Read the familiar CSV, restoring internal identities from its hidden sidecar.

    Rich CSVs from the first v2 runner remain readable for lossless migration.
    Historical plain CSVs without model provenance are never silently relabelled.
    """
    if not path.exists():
        return pd.DataFrame(columns=INTERNAL_RESULT_COLUMNS)
    results = pd.read_csv(path, float_precision="round_trip")
    if not all(name in results.columns for name in IDENTITY_COLUMNS + SCORE_KEY_COLUMNS):
        metadata_path = result_metadata_path(path)
        if not metadata_path.exists():
            raise ValueError(f"{path} is incompatible: missing current-model metadata sidecar")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("csv_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("CSV and metadata checksum differ; do not reuse stale or edited scores")
        records = metadata.get("rows", [])
        if metadata.get("format_version") != 1 or len(records) != len(results):
            raise ValueError("Incompatible result metadata format or row count")
        if any(name not in results.columns for name in RESULT_COLUMNS):
            raise ValueError("Incompatible public result columns")
        for name in IDENTITY_COLUMNS + SCORE_KEY_COLUMNS:
            results[name] = [record[name] for record in records]
        if results["horizon"].notna().any() or not results["seed_intensity"].eq(results["h0"]).all():
            raise ValueError("Current results require blank horizon and seed_intensity equal to h0=u0")
    _validate_result_metadata(results, path)
    return results


def save_results(path: Path, results: pd.DataFrame) -> None:
    """Save the original CSV layout; keep provenance in .results.metadata.json."""
    if path.exists():
        load_results(path)  # Refuse to overwrite an unversioned historical checkpoint.
    _validate_result_metadata(results, path)
    ordered = results.sort_values(["year", "alpha", "seed_budget"], kind="stable").copy()
    ordered["seed_intensity"] = ordered["h0"]  # Derived alias, not a separate parameter.
    ordered["horizon"] = None                  # Infinite objective, never a fixed horizon.
    public = ordered.drop(columns=IDENTITY_COLUMNS + SCORE_KEY_COLUMNS)
    extra_columns = [name for name in public.columns if name not in RESULT_COLUMNS]
    public = public[[*RESULT_COLUMNS, *extra_columns]]
    temporary_path = path.with_name(f".{path.name}.tmp")
    public.to_csv(temporary_path, index=False)
    metadata = {
        "format_version": 1,
        "csv_sha256": hashlib.sha256(temporary_path.read_bytes()).hexdigest(),
        "rows": ordered[IDENTITY_COLUMNS + SCORE_KEY_COLUMNS].to_dict(orient="records"),
    }
    metadata_path = result_metadata_path(path)
    temporary_metadata = metadata_path.with_suffix(".tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    # If interrupted between replacements, the digest fails closed on the next load.
    temporary_path.replace(path)
    temporary_metadata.replace(metadata_path)


def upsert_result(
    results: pd.DataFrame,
    row: dict[str, object],
) -> pd.DataFrame:
    if row["stopping_reason"] != "tolerance_reached":
        raise ValueError("Incomplete evaluations cannot be checkpointed/ranked")
    if instance_mask(results, row).any():
        return results  # Preserve the original completed result.
    return pd.concat([results, pd.DataFrame([row])], ignore_index=True)


def progress_parameters(year: int, config: ExperimentConfig) -> str:
    return (
        f"year={year}, alpha={config.alpha:g}, "
        f"seed_budget={config.seed_budget}, "
        f"edge_weight_beta={config.edge_weight_beta:g}, "
        f"node_weight_beta={config.node_weight_beta:g}, "
        f"search_seconds={config.search_seconds_per_year:g}"
    )


def main() -> None:
    args = parse_args()
    years = available_years(DATA_DIR)
    if not years:
        raise FileNotFoundError(
            "No processed yearly networks found; run `python src/prepare_data.py`."
        )
    if args.years is not None:
        missing = sorted(set(args.years) - set(years))
        if missing:
            raise ValueError(f"Unavailable requested years: {missing}")
        years = sorted(set(args.years))
    else:
        years = [years[-1]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.csv"
    results = load_results(results_path)
    run_specs = [
        (year, alpha, seed_budget)
        for year in years
        for alpha in args.alphas
        for seed_budget in args.seed_budgets
    ]
    total = len(run_specs)

    for number, (year, alpha, seed_budget) in enumerate(run_specs, start=1):
        config = ExperimentConfig(
            seed_budget=seed_budget,
            max_iter=args.max_iter,
            edge_weight_beta=EDGE_WEIGHT_BETA,
            node_weight_beta=NODE_WEIGHT_BETA,
            alpha=alpha,
            search_seconds_per_year=float(args.search_seconds),
        )
        parameters = parameter_values(config)
        instance: dict[str, object] = {
            "year": year, **parameters, **result_identity(year),
        }
        label = progress_parameters(year, config)
        if instance_mask(results, instance).any():
            model = load_year_model(year, config)
            for _, saved in results.loc[instance_mask(results, instance)].iterrows():
                validate_saved_seeds(saved, model, config)
            print(f"[{number}/{total}] already saved: {label}")
            continue

        print(f"[{number}/{total}] {label}")
        summary, _, _ = run_year(year, config, verbose=args.verbose)
        if summary["input_fingerprint"] != instance["input_fingerprint"]:
            raise ValueError("Input files changed during evaluation; no checkpoint saved")
        row: dict[str, object] = {
            **instance,
            **{key: summary[key] for key in SCORE_KEY_COLUMNS},
            **{key: summary[key] for key in ("stopping_reason", "diffusion_iterations")},
            "start_spread": float(summary["initial_spread"]),
            "finish_spread": float(summary["best_spread"]),
            "start_list": json.dumps(summary["start_list"], ensure_ascii=False),
            "finish_list": json.dumps(summary["finish_list"], ensure_ascii=False),
        }
        results = upsert_result(results, row)
        save_results(results_path, results)
        print(
            f"[{number}/{total}] saved: "
            f"start_spread={row['start_spread']:.3f}, "
            f"finish_spread={row['finish_spread']:.3f}"
        )

    print(f"Saved {len(results)} row(s) to {results_path}")


if __name__ == "__main__":
    main()
