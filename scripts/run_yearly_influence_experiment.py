"""Run comparable influence-centrality experiments for every network year.

The same graph filter, GIP dynamics, seed budget, and NaDS configuration are
applied to every requested year and model profile. Results describe which
municipalities are selected, how their selection changes between years, and
how robust selections are to empirical-weight heterogeneity.
"""

from __future__ import annotations

import argparse
import itertools
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
DEFAULT_TOTAL_HOURS = 4.0


@dataclass(frozen=True)
class ModelProfile:
    """Named sensitivity specification for empirical model heterogeneity."""

    edge_weight_beta: float
    node_weight_beta: float
    description: str


MODEL_PROFILES = {
    "tempered": ModelProfile(
        edge_weight_beta=0.5,
        node_weight_beta=0.5,
        description=(
            "Square-root transformed company-size and population heterogeneity; "
            "preserves empirical ordering while limiting domination by extremes."
        ),
    ),
    "full": ModelProfile(
        edge_weight_beta=1.0,
        node_weight_beta=1.0,
        description=(
            "Full empirical company-size and population heterogeneity; used as a "
            "sensitivity comparison with the tempered primary specification."
        ),
    ),
}
DEFAULT_PROFILES = ("tempered", "full")


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
    delta: float = 0.5
    xi: float = 0.01
    d: int = 2
    search_seconds_per_year: float = 600.0
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare population-weighted influence-central municipalities under "
            "fixed, named model profiles across all years."
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
        "--profiles",
        nargs="+",
        choices=tuple(MODEL_PROFILES),
        default=list(DEFAULT_PROFILES),
        help=(
            "Model sensitivity profiles to compare (default: tempered full). "
            "Search settings are identical across profiles."
        ),
    )
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument(
        "--total-hours",
        type=float,
        default=None,
        help=(
            "Approximate total NaDS budget, divided evenly across every selected "
            "profile/year run (default: 4)."
        ),
    )
    budget_group.add_argument(
        "--search-seconds",
        type=float,
        default=None,
        help=(
            "Override with a NaDS budget per profile/year, useful for short test "
            "runs. Cannot be combined with --total-hours."
        ),
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
    if args.total_hours is None and args.search_seconds is None:
        args.total_hours = DEFAULT_TOTAL_HOURS
    if args.total_hours is not None and args.total_hours <= 0:
        parser.error("--total-hours must be positive")
    if args.search_seconds is not None and args.search_seconds <= 0:
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
    profile_name: str,
    year: int,
    config: ExperimentConfig,
    *,
    verbose: bool,
) -> tuple[dict[str, object], pd.DataFrame, set[int]]:
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
    incidence_matrix, matrix_node_ids, matrix_edge_ids = hypergraph.incidence_matrix(
        index=True,
        weights="weight",
    )
    node_ids = np.asarray(matrix_node_ids, dtype="int64")

    edge_weights, _ = aligned_edge_parameters(
        matrix_edge_ids,
        year,
        DATA_DIR / "hyperedge_parameters.csv",
        target_mean=config.edge_weight_mean,
        beta=config.edge_weight_beta,
    )
    node_weights, node_audit = aligned_node_weights(
        matrix_node_ids,
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
    selected["profile"] = profile_name
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
        "profile",
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
    )
    summary: dict[str, object] = {
        "profile": profile_name,
        "year": int(year),
        "nodes": len(hypergraph.nodes),
        "hyperedges": len(hypergraph.edges),
        "incidences": len(filtered),
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


def build_turnover(
    selected_sets: dict[str, dict[int, set[int]]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for profile, profile_sets in selected_sets.items():
        previous_year: int | None = None
        previous_set: set[int] | None = None
        for year, current_set in sorted(profile_sets.items()):
            if previous_year is not None and previous_set is not None:
                intersection = previous_set & current_set
                union = previous_set | current_set
                rows.append(
                    {
                        "profile": profile,
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
    columns = [
        "profile",
        "previous_year",
        "year",
        "year_gap",
        "retained",
        "new",
        "lost",
        "jaccard",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_frequency(selected: pd.DataFrame) -> pd.DataFrame:
    frequency = (
        selected.groupby(["profile", "municipality_id"], sort=False)
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
        ["profile", "selection_count", "mean_selection_rank", "municipality_id"],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def build_profile_comparison(
    selected_sets: dict[str, dict[int, set[int]]],
    year_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare selected sets pairwise within each year."""
    spread_lookup = year_summary.set_index(["profile", "year"])["best_spread"]
    rows: list[dict[str, float | int | str]] = []
    for profile_a, profile_b in itertools.combinations(selected_sets, 2):
        common_years = sorted(
            set(selected_sets[profile_a]) & set(selected_sets[profile_b])
        )
        for year in common_years:
            set_a = selected_sets[profile_a][year]
            set_b = selected_sets[profile_b][year]
            intersection = set_a & set_b
            union = set_a | set_b
            spread_a = float(spread_lookup.loc[(profile_a, year)])
            spread_b = float(spread_lookup.loc[(profile_b, year)])
            rows.append(
                {
                    "year": year,
                    "profile_a": profile_a,
                    "profile_b": profile_b,
                    "shared_selections": len(intersection),
                    "only_profile_a": len(set_a - set_b),
                    "only_profile_b": len(set_b - set_a),
                    "selection_jaccard": (
                        len(intersection) / len(union) if union else 1.0
                    ),
                    "best_spread_a": spread_a,
                    "best_spread_b": spread_b,
                    "spread_difference_pct_b_vs_a": (
                        100.0 * (spread_b / spread_a - 1.0)
                    ),
                }
            )
    columns = [
        "year",
        "profile_a",
        "profile_b",
        "shared_selections",
        "only_profile_a",
        "only_profile_b",
        "selection_jaccard",
        "best_spread_a",
        "best_spread_b",
        "spread_difference_pct_b_vs_a",
    ]
    return pd.DataFrame(rows, columns=columns)


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

    profile_names = list(dict.fromkeys(args.profiles))
    run_count = len(years) * len(profile_names)
    if args.search_seconds is not None:
        search_seconds = float(args.search_seconds)
        budget_mode = "seconds_per_profile_year"
    else:
        search_seconds = float(args.total_hours) * 3_600.0 / run_count
        budget_mode = "total_hours"

    configs = {
        name: ExperimentConfig(
            seed_budget=args.seed_budget,
            search_seconds_per_year=search_seconds,
            edge_weight_beta=MODEL_PROFILES[name].edge_weight_beta,
            node_weight_beta=MODEL_PROFILES[name].node_weight_beta,
        )
        for name in profile_names
    }
    planned_search_seconds = search_seconds * run_count
    print(
        f"Running {len(profile_names)} profile(s) x {len(years)} year(s) "
        f"({years[0]}-{years[-1]}) with {args.seed_budget} seeds."
    )
    print(
        f"NaDS allocation: {search_seconds:.1f}s per profile/year; "
        f"planned search time {planned_search_seconds / 3_600.0:.2f}h."
    )

    summaries: list[dict[str, float | int | str]] = []
    selections: list[pd.DataFrame] = []
    selected_sets: dict[str, dict[int, set[int]]] = {
        profile_name: {} for profile_name in profile_names
    }
    experiment_started = time.monotonic()
    for profile_name, config in configs.items():
        profile = MODEL_PROFILES[profile_name]
        print(
            f"Profile {profile_name}: edge beta={profile.edge_weight_beta:g}, "
            f"node beta={profile.node_weight_beta:g}."
        )
        for year in years:
            summary, selected, selected_set = run_year(
                profile_name,
                year,
                config,
                verbose=args.verbose,
            )
            summaries.append(summary)
            selections.append(selected)
            selected_sets[profile_name][year] = selected_set
            print(
                f"{profile_name}/{year}: spread {summary['initial_spread']:.3f} -> "
                f"{summary['best_spread']:.3f}; "
                f"improvement {summary['relative_improvement_pct']:.2f}%; "
                f"calls {summary['search_objective_calls']}; "
                f"restarts {summary['search_restarts']}"
            )

    year_summary = pd.DataFrame(summaries)
    selected_municipalities = pd.concat(selections, ignore_index=True)
    turnover = build_turnover(selected_sets)
    frequency = build_frequency(selected_municipalities)
    profile_comparison = build_profile_comparison(selected_sets, year_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    year_summary.to_csv(args.output_dir / "year_summary.csv", index=False)
    selected_municipalities.to_csv(
        args.output_dir / "selected_municipalities.csv", index=False
    )
    turnover.to_csv(args.output_dir / "selection_turnover.csv", index=False)
    frequency.to_csv(args.output_dir / "selection_frequency.csv", index=False)
    profile_comparison.to_csv(
        args.output_dir / "profile_comparison.csv", index=False
    )
    metadata = {
        "years": years,
        "profiles": profile_names,
        "elapsed_seconds": round(time.monotonic() - experiment_started, 3),
        "budget": {
            "mode": budget_mode,
            "requested_total_hours": args.total_hours,
            "search_seconds_per_profile_year": search_seconds,
            "planned_search_seconds": planned_search_seconds,
            "profile_year_runs": run_count,
        },
        "centrality_definition": (
            "membership in the fixed-budget seed set maximizing population-weighted "
            "cumulative GIP spread"
        ),
        "initial_seed_rule": (
            "highest structural degree; ties use population weight then municipality ID"
        ),
        "restart_rule": (
            "restart NaDS from the deterministic initial set with incremented random "
            "seeds until each profile/year wall-clock allocation is consumed"
        ),
        "profile_comparison_note": (
            "selection overlap is directly comparable; spread values use each "
            "profile's own mean-one-normalized objective and are not model scores"
        ),
        "matrix_parameter_alignment": (
            "node and edge parameters aligned to labels returned by "
            "Hypergraph.incidence_matrix(index=True)"
        ),
        "model_profiles": {
            name: asdict(MODEL_PROFILES[name]) for name in profile_names
        },
        "profile_configurations": {
            name: asdict(config) for name, config in configs.items()
        },
    }
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Results written to {args.output_dir}")
    print("Most consistently selected municipalities by profile:")
    print(
        frequency[
            ["profile", "municipality_name", "selection_count", "years_selected"]
        ]
        .groupby("profile", sort=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
