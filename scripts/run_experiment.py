"""Run the tempered alpha/seed-budget experiment and save one simple CSV.

Each completed year/parameter combination is checkpointed immediately. A row
with the same year and parameters is replaced, so the file never contains
duplicate instances and an interrupted run can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_yearly_influence_experiment import (
    DATA_DIR,
    ExperimentConfig,
    available_years,
    run_year,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
]


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
    args = parser.parse_args()
    if any(alpha <= 0 for alpha in args.alphas):
        parser.error("Every --alphas value must be positive")
    if any(budget <= 0 for budget in args.seed_budgets):
        parser.error("Every --seed-budgets value must be positive")
    if args.search_seconds <= 0:
        parser.error("--search-seconds must be positive")
    args.alphas = sorted(set(float(value) for value in args.alphas))
    args.seed_budgets = sorted(set(int(value) for value in args.seed_budgets))
    return args


def parameter_values(config: ExperimentConfig) -> dict[str, float | int]:
    values = asdict(config)
    values["search_seconds"] = values.pop("search_seconds_per_year")
    return {column: values[column] for column in PARAMETER_COLUMNS}


def instance_mask(
    results: pd.DataFrame,
    instance: dict[str, float | int],
) -> pd.Series:
    if results.empty:
        return pd.Series(False, index=results.index, dtype=bool)
    mask = results["year"].eq(int(instance["year"]))
    for column in PARAMETER_COLUMNS:
        mask &= np.isclose(
            pd.to_numeric(results[column], errors="coerce"),
            float(instance[column]),
            rtol=0.0,
            atol=1e-12,
        )
    return mask


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    results = pd.read_csv(path)
    missing = [column for column in RESULT_COLUMNS if column not in results.columns]
    if missing:
        raise ValueError(
            f"{path} does not use the expected result format; missing {missing}"
        )
    return results[RESULT_COLUMNS]


def save_results(path: Path, results: pd.DataFrame) -> None:
    ordered = results[RESULT_COLUMNS].sort_values(
        ["year", "alpha", "seed_budget"], kind="stable"
    )
    temporary_path = path.with_name(f".{path.name}.tmp")
    ordered.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def upsert_result(
    results: pd.DataFrame,
    row: dict[str, object],
) -> pd.DataFrame:
    results = results.loc[~instance_mask(results, row)]
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
            edge_weight_beta=EDGE_WEIGHT_BETA,
            node_weight_beta=NODE_WEIGHT_BETA,
            alpha=alpha,
            search_seconds_per_year=float(args.search_seconds),
        )
        parameters = parameter_values(config)
        instance: dict[str, float | int] = {"year": year, **parameters}
        label = progress_parameters(year, config)
        if instance_mask(results, instance).any():
            print(f"[{number}/{total}] already saved: {label}")
            continue

        print(f"[{number}/{total}] {label}")
        summary, _, _ = run_year("tempered", year, config, verbose=args.verbose)
        row: dict[str, object] = {
            **instance,
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
