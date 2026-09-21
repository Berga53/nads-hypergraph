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
from src.model_identity import MODEL_METADATA, evaluation_cache_key, input_fingerprint
from src.nads import nads

DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
DEFAULT_ALPHAS = (0.05, 0.10, 0.20)
DEFAULT_SEED_BUDGETS = (10, 20, 40)
DEFAULT_SEARCH_SECONDS = 300.0
EDGE_WEIGHT_BETA = 0.5
NODE_WEIGHT_BETA = 0.5

# Only values varied by the experiment grid remain as ordinary CSV columns.
# Fixed settings and provenance are grouped into typed JSON columns at the end.
NETWORK_PARAMETER_FIELDS = [
    "min_edge_size",
    "edge_weight_beta",
    "node_weight_beta",
    "edge_weight_mean",
    "node_weight_mean",
]
GIP_PARAMETER_FIELDS = [
    "h0",
    "l0",
    "theta_l",
    "theta_h",
    "gamma",
    "eps",
    "stress_level",
    "max_iter",
]
NADS_PARAMETER_FIELDS = [
    "delta",
    "xi",
    "d",
    "search_seconds",
    "max_neighbors_per_phase",
    "buffer_dim",
    "max_search_iterations",
    "random_seed",
]
GROUPED_PARAMETER_COLUMNS = [
    "network_parameters",
    "gip_parameters",
    "nads_parameters",
]
METADATA_COLUMNS = ["graph_metadata", "model_metadata", "run_metadata"]
IDENTITY_COLUMNS = ["graph_metadata", "model_metadata"]
SCORE_KEY_COLUMNS = ["start_score_key", "finish_score_key"]
RESULT_COLUMNS = [
    "year", "alpha", "seed_budget",
    "start_spread", "finish_spread", "start_list", "finish_list",
    "stopping_reason", "diffusion_iterations",
    "nads_history",
    *GROUPED_PARAMETER_COLUMNS,
    *METADATA_COLUMNS,
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
    municipality_ids: np.ndarray | list[int] | set[int] | None = None,
    partition_identity: str | None = None,
) -> dict[str, object]:
    """Prepare a filtered, aligned national or municipality-induced network."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    incidences = pd.read_csv(data_dir / f"rete_{year}.csv")
    fingerprint = input_fingerprint(data_dir, year)
    if municipality_ids is not None:
        selected_ids = np.asarray(
            sorted(set(int(value) for value in municipality_ids))
        )
        if selected_ids.ndim != 1 or not len(selected_ids):
            raise ValueError("municipality_ids must select at least one municipality")
        incidences = incidences.loc[
            incidences["CF Comune"].isin(selected_ids)
        ].copy()
        digest = hashlib.sha256()
        digest.update(fingerprint.encode())
        digest.update(selected_ids.astype("<i8").tobytes())
        digest.update((partition_identity or "municipality-induced").encode())
        fingerprint = digest.hexdigest()
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
        "input_fingerprint": fingerprint,
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


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_dump(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_object(value: object, column: str) -> dict[str, object]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError(f"{column} must contain a JSON object")
    return parsed


def _json_list(value: object, column: str) -> list[object]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError(f"{column} must contain a JSON list")
    return parsed


def _model_metadata() -> dict[str, object]:
    return {
        "format_version": 2,
        "evaluation_mode": "infinite_discounted_tolerance",
        **MODEL_METADATA,
    }


def result_identity(year: int, *, data_dir: Path | None = None) -> dict[str, str]:
    data_dir = DATA_DIR if data_dir is None else data_dir
    return {
        "graph_metadata": _json_dump(
            {"input_fingerprint": input_fingerprint(data_dir, year)}
        ),
        "model_metadata": _json_dump(_model_metadata()),
    }


def seed_score_key(model, seeds, config):
    return evaluation_cache_key(
        model["incidence"], model["edge_weights"], seed_state(seeds, config.gip_parameters),
        config.gip_parameters, node_weights=model["node_weights"],
        stress_level=config.stress_level, alpha=config.alpha, max_iter=config.max_iter,
    )


def config_from_row(row) -> ExperimentConfig:
    network = _json_object(row["network_parameters"], "network_parameters")
    gip_parameters = _json_object(row["gip_parameters"], "gip_parameters")
    nads_parameters = _json_object(row["nads_parameters"], "nads_parameters")
    values = {
        "alpha": float(row["alpha"]),
        "seed_budget": int(row["seed_budget"]),
        **network,
        **gip_parameters,
        **nads_parameters,
    }
    values["search_seconds_per_year"] = values.pop("search_seconds")
    return ExperimentConfig(**values)


def validate_saved_seeds(row, model, config):
    """Validate stored seeds and full evaluation keys before reuse/reconstruction."""
    metadata = _json_object(row["model_metadata"], "model_metadata")
    graph = _json_object(row["graph_metadata"], "graph_metadata")
    run = _json_object(row["run_metadata"], "run_metadata")
    for field, expected in _model_metadata().items():
        if metadata.get(field) != expected:
            raise ValueError(f"Incompatible saved model metadata: {field}")
    if row["stopping_reason"] != "tolerance_reached":
        raise ValueError("Incomplete saved evaluation cannot be ranked")
    if graph.get("input_fingerprint") != model["input_fingerprint"]:
        raise ValueError("Saved input fingerprint does not match current data; preserve the old result")
    output = {}
    for label in ("start", "finish"):
        ids = row[f"{label}_list"]
        ids = json.loads(ids) if isinstance(ids, str) else list(ids)
        if len(set(ids)) != len(ids) or len(ids) != config.seed_budget or set(ids) - set(model["node_ids"]):
            raise ValueError("Saved seeds do not match the network/budget")
        seeds = np.isin(model["node_ids"], ids).astype(float)
        if seed_score_key(model, seeds, config) != run.get(f"{label}_score_key"):
            raise ValueError("Saved score key does not match seeds/network/parameters; preserve the old result")
        output[label] = seeds
    return output


def run_year(
    year: int,
    config: ExperimentConfig,
    *,
    verbose: bool,
    initial_seeds: np.ndarray | None = None,
    restart_search: bool = True,
    data_dir: Path | None = None,
    municipality_ids: np.ndarray | list[int] | set[int] | None = None,
    partition_identity: str | None = None,
) -> tuple[dict[str, object], pd.DataFrame, set[int]]:
    """Run one yearly search, optionally from a caller-supplied seed set.

    ``restart_search`` retains the historical runner behavior by default.  Set
    it to false for a starting-point sensitivity experiment: NaDS then runs
    once from the supplied point, with the time allocation acting as a cap.
    """
    model = load_year_model(
        year,
        config,
        data_dir=data_dir,
        municipality_ids=municipality_ids,
        partition_identity=partition_identity,
    )
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
    initialization = "provided"
    if initial_seeds is None:
        initialization = "degree_ranked"
        initial_seeds = deterministic_initial_seeds(
            node_ids,
            structural_degree,
            node_weights,
            config.seed_budget,
        )
    else:
        initial_seeds = np.asarray(initial_seeds, dtype=float)
        if initial_seeds.ndim != 1 or len(initial_seeds) != len(node_ids):
            raise ValueError("initial_seeds must have one entry per yearly network node")
        if (
            not np.isfinite(initial_seeds).all()
            or not np.isin(initial_seeds, (0.0, 1.0)).all()
        ):
            raise ValueError("initial_seeds must be a finite binary indicator")
        if np.count_nonzero(initial_seeds) != config.seed_budget:
            raise ValueError("initial_seeds must contain exactly seed_budget selected nodes")
        initial_seeds = initial_seeds.copy()

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
    nads_history: list[dict[str, object]] = []
    while True:
        remaining = config.search_seconds_per_year - (time.monotonic() - started)
        if remaining <= 0.01:
            break
        search_number = search_restarts + 1
        search_random_seed = config.random_seed + search_restarts
        calls_before_search = objective_calls
        search_started = time.monotonic()
        spread_history, seed_history, event_history = nads(
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
            random_seed=search_random_seed,
            verbose=int(verbose),
        )
        search_restarts += 1
        accepted_seed_sets_all_restarts += len(seed_history)
        events = []
        for event in event_history:
            seed_indices = np.asarray(event["seed_indices"], dtype=int)
            events.append(
                {
                    "call": int(event["call"]),
                    "elapsed_seconds": float(event["elapsed_seconds"]),
                    "seed_list": sorted(
                        int(value) for value in node_ids[seed_indices]
                    ),
                    "value": float(event["value"]),
                }
            )
        nads_history.append(
            {
                "accepted_seed_sets": len(seed_history),
                "elapsed_seconds": time.monotonic() - search_started,
                "events": events,
                "objective_calls": objective_calls - calls_before_search,
                "random_seed": search_random_seed,
                "search_number": search_number,
            }
        )
        if (
            best_spread_history is None
            or spread_history[-1] > best_spread_history[-1]
        ):
            best_spread_history = spread_history
            best_seed_history = seed_history
            best_restart = search_restarts
        if not restart_search:
            break

    if best_spread_history is None or best_seed_history is None:
        raise RuntimeError("NaDS search budget was too short to evaluate a seed set")
    elapsed = time.monotonic() - started
    search_time_limit_reached = config.search_seconds_per_year - elapsed <= 0.01
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
        "search_time_limit_reached": bool(search_time_limit_reached),
        "initialization": initialization,
        "restart_search_until_time": bool(restart_search),
        "nads_history": nads_history,
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
        help="NaDS allocation for each year/parameter search (default: 300).",
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
    network = {
        name: _json_value(values[name]) for name in NETWORK_PARAMETER_FIELDS
    }
    gip_parameters = {
        name: _json_value(values[name])
        for name in GIP_PARAMETER_FIELDS
        if values[name] is not None
    }
    nads_parameters = {
        name: _json_value(values[name])
        for name in NADS_PARAMETER_FIELDS
        if name != "search_seconds"
    }
    nads_parameters["search_seconds"] = float(config.search_seconds_per_year)
    return {
        "alpha": float(config.alpha),
        "seed_budget": int(config.seed_budget),
        "network_parameters": _json_dump(network),
        "gip_parameters": _json_dump(gip_parameters),
        "nads_parameters": _json_dump(nads_parameters),
    }


def result_row(summary: dict[str, object], config: ExperimentConfig) -> dict[str, object]:
    """Convert a rich in-memory run summary to the compact public CSV schema."""
    graph_metadata = {
        "hyperedges": int(summary["hyperedges"]),
        "incidences": int(summary["incidences"]),
        "input_fingerprint": summary["input_fingerprint"],
        "nodes": int(summary["nodes"]),
    }
    run_metadata = {
        "accepted_seed_sets": int(summary["accepted_seed_sets"]),
        "accepted_seed_sets_all_restarts": int(
            summary["accepted_seed_sets_all_restarts"]
        ),
        "best_restart": int(summary["best_restart"]),
        "finish_score_key": summary["finish_score_key"],
        "initialization": summary["initialization"],
        "population_match_pct_selected": float(
            summary["population_match_pct_selected"]
        ),
        "relative_improvement_pct": float(summary["relative_improvement_pct"]),
        "restart_search_until_time": bool(summary["restart_search_until_time"]),
        "search_elapsed_seconds": float(summary["search_elapsed_seconds"]),
        "search_objective_calls": int(summary["search_objective_calls"]),
        "search_restarts": int(summary["search_restarts"]),
        "search_time_limit_reached": bool(summary["search_time_limit_reached"]),
        "start_score_key": summary["start_score_key"],
    }
    return {
        "year": int(summary["year"]),
        **parameter_values(config),
        "start_spread": float(summary["initial_spread"]),
        "finish_spread": float(summary["best_spread"]),
        "start_list": _json_dump(summary["start_list"]),
        "finish_list": _json_dump(summary["finish_list"]),
        "stopping_reason": summary["stopping_reason"],
        "diffusion_iterations": int(summary["diffusion_iterations"]),
        "nads_history": _json_dump(summary["nads_history"]),
        "graph_metadata": _json_dump(graph_metadata),
        "model_metadata": _json_dump(_model_metadata()),
        "run_metadata": _json_dump(run_metadata),
    }


def instance_mask(
    results: pd.DataFrame,
    instance: dict[str, object],
) -> pd.Series:
    if results.empty:
        return pd.Series(False, index=results.index, dtype=bool)
    mask = results["year"].eq(int(instance["year"]))
    mask &= pd.to_numeric(results["alpha"], errors="coerce").eq(instance["alpha"])
    mask &= pd.to_numeric(results["seed_budget"], errors="coerce").eq(
        instance["seed_budget"]
    )
    for column in GROUPED_PARAMETER_COLUMNS:
        expected = _json_dump(_json_object(instance[column], column))
        mask &= results[column].map(
            lambda value: _json_dump(_json_object(value, column)) == expected
        )
    expected_model = _json_object(instance["model_metadata"], "model_metadata")
    expected_graph = _json_object(instance["graph_metadata"], "graph_metadata")
    mask &= results["model_metadata"].map(
        lambda value: _json_object(value, "model_metadata") == expected_model
    )
    mask &= results["graph_metadata"].map(
        lambda value: all(
            _json_object(value, "graph_metadata").get(name) == expected
            for name, expected in expected_graph.items()
        )
    )
    mask &= results["stopping_reason"].eq("tolerance_reached")
    return mask


def _validate_nads_history(value: object, seed_budget: int) -> None:
    searches = _json_list(value, "nads_history")
    if not searches:
        raise ValueError("nads_history must contain at least one NaDS search")
    for expected_number, search in enumerate(searches, start=1):
        if not isinstance(search, dict):
            raise ValueError("Each nads_history search must be a JSON object")
        required = {
            "accepted_seed_sets",
            "elapsed_seconds",
            "events",
            "objective_calls",
            "random_seed",
            "search_number",
        }
        if required - set(search):
            raise ValueError("A nads_history search is missing required fields")
        if int(search["search_number"]) != expected_number:
            raise ValueError("nads_history search numbers must be consecutive")
        events = search["events"]
        if not isinstance(events, list) or not events:
            raise ValueError("Every NaDS search must contain its initial event")
        objective_calls = int(search["objective_calls"])
        search_elapsed = float(search["elapsed_seconds"])
        if (
            int(search["accepted_seed_sets"]) < 1
            or objective_calls < 1
            or not np.isfinite(search_elapsed)
            or search_elapsed < 0
        ):
            raise ValueError("Invalid nads_history search summary")
        previous_value = None
        previous_call = 0
        previous_time = -1.0
        for event in events:
            if not isinstance(event, dict) or {
                "call", "elapsed_seconds", "seed_list", "value"
            } - set(event):
                raise ValueError("A nads_history event is missing required fields")
            seeds = event["seed_list"]
            value_number = float(event["value"])
            call = int(event["call"])
            elapsed = float(event["elapsed_seconds"])
            if (
                not isinstance(seeds, list)
                or len(seeds) != seed_budget
                or len(set(seeds)) != seed_budget
                or not np.isfinite(value_number)
                or call <= previous_call
                or elapsed < previous_time
                or (previous_value is not None and value_number <= previous_value)
            ):
                raise ValueError("Invalid or non-improving nads_history event")
            previous_value = value_number
            previous_call = call
            previous_time = elapsed
        if previous_call > objective_calls or previous_time > search_elapsed:
            raise ValueError("nads_history events exceed their search summary")


def _validate_results(results: pd.DataFrame, path: Path) -> None:
    missing = [column for column in RESULT_COLUMNS if column not in results.columns]
    if missing:
        raise ValueError(f"{path} has an incompatible result format; missing {missing}")
    if not results["stopping_reason"].eq("tolerance_reached").all():
        raise ValueError("Incomplete evaluations cannot be loaded/saved as completed results")
    for _, row in results.iterrows():
        config = config_from_row(row)
        expected_parameters = parameter_values(config)
        for column in GROUPED_PARAMETER_COLUMNS:
            if _json_dump(_json_object(row[column], column)) != expected_parameters[column]:
                raise ValueError(f"Non-canonical or incompatible {column}")
        model = _json_object(row["model_metadata"], "model_metadata")
        if model != _model_metadata():
            raise ValueError(f"{path} has incompatible model metadata")
        graph = _json_object(row["graph_metadata"], "graph_metadata")
        if not isinstance(graph.get("input_fingerprint"), str):
            raise ValueError("Missing graph input fingerprint")
        run = _json_object(row["run_metadata"], "run_metadata")
        if not all(isinstance(run.get(name), str) for name in SCORE_KEY_COLUMNS):
            raise ValueError("Missing run score keys")
        scores = np.asarray([row["start_spread"], row["finish_spread"]], dtype=float)
        if not np.isfinite(scores).all() or scores[1] < scores[0]:
            raise ValueError("Invalid start/finish spread")
        seed_lists = {}
        for label in ("start", "finish"):
            seeds = _json_list(row[f"{label}_list"], f"{label}_list")
            if len(seeds) != config.seed_budget or len(set(seeds)) != len(seeds):
                raise ValueError(f"Invalid {label}_list")
            seed_lists[label] = set(seeds)
        _validate_nads_history(row["nads_history"], config.seed_budget)
        searches = _json_list(row["nads_history"], "nads_history")
        if any(
            set(search["events"][0]["seed_list"]) != seed_lists["start"]
            for search in searches
        ):
            raise ValueError("NaDS initial history seeds do not match start_list")
        if any(
            not np.isclose(
                float(search["events"][0]["value"]), scores[0], rtol=1e-12, atol=1e-9
            )
            for search in searches
        ):
            raise ValueError("NaDS initial history value does not match start_spread")
        best_restart = int(run.get("best_restart", 0))
        if not 1 <= best_restart <= len(searches) or not np.isclose(
            float(searches[best_restart - 1]["events"][-1]["value"]),
            scores[1],
            rtol=1e-12,
            atol=1e-9,
        ):
            raise ValueError("NaDS best history value does not match finish_spread")
        if (
            set(searches[best_restart - 1]["events"][-1]["seed_list"])
            != seed_lists["finish"]
        ):
            raise ValueError("NaDS best history seeds do not match finish_list")
        if int(run.get("search_restarts", 0)) != len(searches):
            raise ValueError("NaDS search count does not match run_metadata")
        if int(run.get("search_objective_calls", 0)) != sum(
            int(search["objective_calls"]) for search in searches
        ):
            raise ValueError("NaDS call count does not match run_metadata")


def load_results(path: Path) -> pd.DataFrame:
    """Read a current result CSV whose grouped settings and metadata are inline."""
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    results = pd.read_csv(path, float_precision="round_trip")
    _validate_results(results, path)
    return results


def save_results(path: Path, results: pd.DataFrame) -> None:
    """Atomically save results with grouped parameters and metadata at the end."""
    if path.exists():
        load_results(path)  # Refuse to overwrite an incompatible checkpoint.
    _validate_results(results, path)
    ordered = results.sort_values(["year", "alpha", "seed_budget"], kind="stable").copy()
    trailing = ["nads_history", *GROUPED_PARAMETER_COLUMNS, *METADATA_COLUMNS]
    visible = [name for name in RESULT_COLUMNS if name not in trailing]
    extra_columns = [name for name in ordered.columns if name not in RESULT_COLUMNS]
    public = ordered[[*visible, *extra_columns, *trailing]]
    temporary_path = path.with_name(f".{path.name}.tmp")
    public.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


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
        expected_graph = _json_object(instance["graph_metadata"], "graph_metadata")
        if summary["input_fingerprint"] != expected_graph["input_fingerprint"]:
            raise ValueError("Input files changed during evaluation; no checkpoint saved")
        row = result_row(summary, config)
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
