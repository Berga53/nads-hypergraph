"""Run the current GIP model on the tempered alpha/seed-budget grid.

Completed year/parameter instances are checkpointed to results.csv and skipped
on resume. This is the sole experiment runner; it contains the configuration,
network loading, per-year search, and checkpoint orchestration.
"""

from __future__ import annotations

import argparse
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
    gip,
    prepare_incidence,
)
from src.nads import nads

DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "experiment"
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
    "seed_intensity",
    "horizon",
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
RESULT_COLUMNS = [
    "year",
    *PARAMETER_COLUMNS,
    "start_spread",
    "finish_spread",
    "start_list",
    "finish_list",
    "stopping_reason",
    "diffusion_iterations",
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Fixed settings shared by every year in the experiment."""

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
    seed_intensity: float = 1.0
    horizon: int | None = 20
    max_iter: int = 999
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


def add_horizon_arguments(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--horizon", type=int, default=ExperimentConfig.horizon,
                      help="Exactly T updates, including T=0 (default: 20).")
    mode.add_argument("--early-stopping", action="store_true",
                      help="Use the discounted-state tolerance instead of a fixed horizon.")
    parser.add_argument("--max-iter", type=int, default=ExperimentConfig.max_iter,
                        help="Finite J_max for early stopping (default: 999).")


def validate_horizon_arguments(parser, args) -> None:
    if args.horizon < 0:
        parser.error("--horizon must be non-negative")
    if args.max_iter < 1:
        parser.error("--max-iter must be positive")
    if args.early_stopping:
        args.horizon = None


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

    objective_calls = 0

    def simulate(seed_indicator: np.ndarray) -> float:
        result = gip(
            incidence_matrix,
            edge_weights,
            seed_indicator * config.seed_intensity,
            config.gip_parameters,
            node_weights=node_weights,
            stress_level=config.stress_level,
            alpha=config.alpha,
            horizon=config.horizon,
            max_iter=config.max_iter,
        )
        return result.total_spread

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

    best_result = gip(
        incidence_matrix,
        edge_weights,
        best_seeds * config.seed_intensity,
        config.gip_parameters,
        node_weights=node_weights,
        stress_level=config.stress_level,
        alpha=config.alpha,
        horizon=config.horizon,
        max_iter=config.max_iter,
    )
    summary: dict[str, object] = {
        **asdict(config),
        "stopping_reason": best_result.stopping_reason,
        "evaluation_mode": "fixed" if config.horizon is not None else "early",
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
            * (best_spread_history[-1] / best_spread_history[0] - 1.0)
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
    add_horizon_arguments(parser)
    args = parser.parse_args()
    validate_horizon_arguments(parser, args)
    if any(not np.isfinite(alpha) or alpha <= 0 for alpha in args.alphas):
        parser.error("Every --alphas value must be positive")
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
    return mask


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    results = pd.read_csv(path, float_precision="round_trip")
    missing = [column for column in RESULT_COLUMNS if column not in results.columns]
    if missing:
        raise ValueError(
            f"{path} has an incompatible result format; missing {missing}. "
            "Use a results file generated by the current experiment runner."
        )
    return results


def save_results(path: Path, results: pd.DataFrame) -> None:
    ordered = results.sort_values(
        ["year", "alpha", "seed_budget"], kind="stable"
    )
    temporary_path = path.with_name(f".{path.name}.tmp")
    ordered.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def upsert_result(
    results: pd.DataFrame,
    row: dict[str, object],
) -> pd.DataFrame:
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
            horizon=args.horizon,
            max_iter=args.max_iter,
            edge_weight_beta=EDGE_WEIGHT_BETA,
            node_weight_beta=NODE_WEIGHT_BETA,
            alpha=alpha,
            search_seconds_per_year=float(args.search_seconds),
        )
        parameters = parameter_values(config)
        instance: dict[str, object] = {
            "year": year, **parameters,
        }
        label = progress_parameters(year, config)
        if instance_mask(results, instance).any():
            print(f"[{number}/{total}] already saved: {label}")
            continue

        print(f"[{number}/{total}] {label}")
        summary, _, _ = run_year(year, config, verbose=args.verbose)
        row: dict[str, object] = {
            **instance,
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
