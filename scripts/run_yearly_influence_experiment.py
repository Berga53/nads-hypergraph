"""Run one comparable influence-centrality experiment for every network year.

The same graph filter, GIP parameters, population weighting, seed budget, and
NaDS configuration are applied to every requested year. Results describe which
municipalities are selected, how their selection changes between years, and
which municipalities are selected most consistently over the full period.
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

from src.gip_model import (  # noqa: E402
    GIPParameters,
    aligned_edge_parameters,
    aligned_node_weights,
    gip,
)
from src.nads import nads  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "yearly_influence"


@dataclass(frozen=True)
class ExperimentConfig:
    """Fixed settings shared by every year in the experiment."""

    seed_budget: int = 20
    min_edge_size: int = 2
    edge_weight_mean: float = 1.0
    edge_weight_beta: float = 1.0
    node_weight_mean: float = 1.0
    node_weight_beta: float = 1.0
    h0: float = 1.0
    l0: float = 1.0
    theta_l: float = 2.0
    theta_h: float = 50.0
    gamma: float = 0.1
    eps: float = 0.01
    alpha: float = 0.1
    stress_level: float = .0
    seed_intensity: float = 1.0
    delta: float = 0.5
    xi: float = 0.01
    d: int = 2
    search_seconds_per_year: float = 5.0
    max_neighbors_per_phase: int = 250
    buffer_dim: int = 10_000
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select population-weighted influence-central municipalities with "
            "one fixed parameter configuration across all years."
        )
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset; by default every processed rete_YYYY.csv is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--search-seconds",
        type=float,
        default=ExperimentConfig.search_seconds_per_year,
        help="NaDS time budget per year; the same value is used for every year.",
    )
    parser.add_argument(
        "--seed-budget",
        type=int,
        default=ExperimentConfig.seed_budget,
        help="Number of municipalities selected in every year.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show live NaDS objective evaluations.",
    )
    args = parser.parse_args()
    if args.search_seconds <= 0:
        parser.error("--search-seconds must be positive")
    if args.seed_budget <= 0:
        parser.error("--seed-budget must be positive")
    return args


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


def run_year(
    year: int,
    config: ExperimentConfig,
    *,
    verbose: bool,
) -> tuple[dict[str, float | int], pd.DataFrame, set[int]]:
    incidences = pd.read_csv(DATA_DIR / f"rete_{year}.csv")
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
    incidence_matrix = hypergraph.incidence_matrix(weights="weight")
    node_ids = np.asarray([int(node) for node in hypergraph.nodes], dtype="int64")

    edge_weights, _ = aligned_edge_parameters(
        hypergraph.edges,
        year,
        DATA_DIR / "hyperedge_parameters.csv",
        target_mean=config.edge_weight_mean,
        beta=config.edge_weight_beta,
    )
    node_weights, node_audit = aligned_node_weights(
        hypergraph.nodes,
        year,
        DATA_DIR / "node_parameters.csv",
        target_mean=config.node_weight_mean,
        beta=config.node_weight_beta,
    )
    structural_degree = filtered.groupby("CF Comune")["CF Partecipata"].nunique()
    ownership_weighted_degree = filtered.groupby("CF Comune")["Quota"].sum()

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
        )
        return result.total_spread

    def search_objective(seed_indicator: np.ndarray) -> float:
        nonlocal objective_calls
        objective_calls += 1
        return simulate(seed_indicator)

    started = time.monotonic()
    spread_history, seed_history, _ = nads(
        objective=search_objective,
        x0=initial_seeds,
        delta=config.delta,
        xi=config.xi,
        d=config.d,
        max_time=config.search_seconds_per_year,
        buffer_dim=config.buffer_dim,
        max_neighbors_per_phase=config.max_neighbors_per_phase,
        max_iterations=config.max_search_iterations,
        random_seed=config.random_seed,
        verbose=int(verbose),
    )
    elapsed = time.monotonic() - started
    best_seeds = seed_history[-1]
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

    best_result = gip(
        incidence_matrix,
        edge_weights,
        best_seeds * config.seed_intensity,
        config.gip_parameters,
        node_weights=node_weights,
        stress_level=config.stress_level,
        alpha=config.alpha,
    )
    summary: dict[str, float | int] = {
        "year": int(year),
        "nodes": len(hypergraph.nodes),
        "hyperedges": len(hypergraph.edges),
        "incidences": len(filtered),
        "seed_budget": int(config.seed_budget),
        "initial_spread": float(spread_history[0]),
        "best_spread": float(spread_history[-1]),
        "relative_improvement_pct": float(
            100.0 * (spread_history[-1] / spread_history[0] - 1.0)
        ),
        "accepted_seed_sets": len(seed_history),
        "search_objective_calls": int(objective_calls),
        "search_elapsed_seconds": float(elapsed),
        "diffusion_iterations": int(best_result.iterations),
        "population_match_pct_selected": float(
            100.0 * selected["population_data_available"].mean()
        ),
    }
    return summary, selected, set(int(value) for value in selected_ids)


def build_turnover(selected_sets: dict[int, set[int]]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    previous_year: int | None = None
    previous_set: set[int] | None = None
    for year, current_set in sorted(selected_sets.items()):
        if previous_year is not None and previous_set is not None:
            intersection = previous_set & current_set
            union = previous_set | current_set
            rows.append(
                {
                    "previous_year": previous_year,
                    "year": year,
                    "year_gap": year - previous_year,
                    "retained": len(intersection),
                    "new": len(current_set - previous_set),
                    "lost": len(previous_set - current_set),
                    "jaccard": len(intersection) / len(union) if union else 1.0,
                }
            )
        previous_year = year
        previous_set = current_set
    return pd.DataFrame(rows)


def build_frequency(selected: pd.DataFrame) -> pd.DataFrame:
    frequency = (
        selected.groupby("municipality_id", sort=False)
        .agg(
            municipality_name=("municipality_name", "last"),
            years_selected=("year", lambda values: "|".join(map(str, sorted(values)))),
            selection_count=("year", "size"),
            first_selected_year=("year", "min"),
            last_selected_year=("year", "max"),
            mean_selection_rank=("selection_rank", "mean"),
            mean_individual_weighted_spread=("individual_weighted_spread", "mean"),
            mean_population=("population", "mean"),
            mean_structural_degree=("structural_degree", "mean"),
        )
        .reset_index()
    )
    return frequency.sort_values(
        ["selection_count", "mean_selection_rank", "municipality_id"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    years = available_years(DATA_DIR)
    if not years:
        raise FileNotFoundError(
            "No processed yearly networks found; run `python src/prepare_data.py`."
        )
    if args.years:
        missing = sorted(set(args.years) - set(years))
        if missing:
            raise ValueError(f"Unavailable requested years: {missing}")
        years = sorted(set(args.years))

    config = ExperimentConfig(
        seed_budget=args.seed_budget,
        search_seconds_per_year=args.search_seconds,
    )
    print(
        f"Running {len(years)} year(s), {years[0]}-{years[-1]}, "
        f"with {config.seed_budget} seeds and "
        f"{config.search_seconds_per_year:g}s NaDS per year."
    )

    summaries: list[dict[str, float | int]] = []
    selections: list[pd.DataFrame] = []
    selected_sets: dict[int, set[int]] = {}
    experiment_started = time.monotonic()
    for year in years:
        summary, selected, selected_set = run_year(year, config, verbose=args.verbose)
        summaries.append(summary)
        selections.append(selected)
        selected_sets[year] = selected_set
        print(
            f"{year}: spread {summary['initial_spread']:.3f} -> "
            f"{summary['best_spread']:.3f}; "
            f"improvement {summary['relative_improvement_pct']:.2f}%; "
            f"calls {summary['search_objective_calls']}"
        )

    year_summary = pd.DataFrame(summaries)
    selected_municipalities = pd.concat(selections, ignore_index=True)
    turnover = build_turnover(selected_sets)
    frequency = build_frequency(selected_municipalities)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    year_summary.to_csv(args.output_dir / "year_summary.csv", index=False)
    selected_municipalities.to_csv(
        args.output_dir / "selected_municipalities.csv", index=False
    )
    turnover.to_csv(args.output_dir / "selection_turnover.csv", index=False)
    frequency.to_csv(args.output_dir / "selection_frequency.csv", index=False)
    metadata = {
        "years": years,
        "elapsed_seconds": round(time.monotonic() - experiment_started, 3),
        "centrality_definition": (
            "membership in the fixed-budget seed set maximizing population-weighted "
            "cumulative GIP spread"
        ),
        "initial_seed_rule": (
            "highest structural degree; ties use population weight then municipality ID"
        ),
        "configuration": asdict(config),
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Results written to {args.output_dir}")
    print("Most consistently selected municipalities:")
    print(
        frequency[
            ["municipality_name", "selection_count", "years_selected"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
