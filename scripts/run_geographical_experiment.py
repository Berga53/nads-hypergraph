"""Run the alpha/seed-budget experiment on ISTAT macro-area sub-hypergraphs.

Each yearly graph is partitioned by municipality. A company is represented in
every macro-area containing its municipal shareholders, but is retained in an
area only when at least ``min_edge_size`` shareholders remain there. Edge and
node parameters are then aligned and normalized within that induced network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import (
    DATA_DIR,
    DEFAULT_ALPHAS,
    DEFAULT_SEARCH_SECONDS,
    DEFAULT_SEED_BUDGETS,
    EDGE_WEIGHT_BETA,
    NODE_WEIGHT_BETA,
    ExperimentConfig,
    _json_dump,
    _json_object,
    available_years,
    instance_mask,
    load_results,
    load_year_model,
    parameter_values,
    progress_parameters,
    result_identity,
    result_row,
    run_year,
    save_results,
    upsert_result,
    validate_saved_seeds,
)
from src.geography import (
    DEFAULT_REGISTRY_PATH,
    ISTAT_MACROAREAS,
    geography_fingerprint,
    load_municipality_geography,
    normalize_macroarea,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "geographical_results.csv"
EXPERIMENT_TYPE = "istat_macroarea_induced_subgraph"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standard experiment separately on ISTAT macro-areas."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="One year or a space-separated list; default: latest available year.",
    )
    parser.add_argument(
        "--macroareas",
        nargs="+",
        default=list(ISTAT_MACROAREAS),
        help="Subset of NORD-OVEST NORD-EST CENTRO SUD ISOLE (default: all).",
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS)
    )
    parser.add_argument(
        "--seed-budgets", nargs="+", type=int, default=list(DEFAULT_SEED_BUDGETS)
    )
    parser.add_argument(
        "--search-seconds", type=float, default=DEFAULT_SEARCH_SECONDS
    )
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="BDAP municipality registry containing ISTAT geography.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Consolidated checkpoint CSV (default: results/geographical_results.csv).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    try:
        args.macroareas = list(
            dict.fromkeys(normalize_macroarea(value) for value in args.macroareas)
        )
    except ValueError as error:
        parser.error(str(error))
    if any(
        not np.isfinite(alpha)
        or alpha < 0
        or ExperimentConfig.theta_l * alpha >= 1
        for alpha in args.alphas
    ):
        parser.error(
            "Every --alphas value must satisfy 0 <= alpha and theta_l * alpha < 1"
        )
    if any(budget <= 0 for budget in args.seed_budgets):
        parser.error("Every --seed-budgets value must be positive")
    if not np.isfinite(args.search_seconds) or args.search_seconds <= 0:
        parser.error("--search-seconds must be positive")
    if args.max_iter is not None and args.max_iter < 0:
        parser.error("--max-iter must be non-negative")
    args.alphas = sorted(set(float(value) for value in args.alphas))
    args.seed_budgets = sorted(set(int(value) for value in args.seed_budgets))
    return args


def partition_identity(macroarea: str, mapping_fingerprint: str) -> str:
    return f"{EXPERIMENT_TYPE}:{macroarea}:{mapping_fingerprint}"


def geographical_graph_metadata(
    model: dict[str, object], macroarea: str, mapping_fingerprint: str
) -> dict[str, object]:
    return {
        "experiment_type": EXPERIMENT_TYPE,
        "geography_fingerprint": mapping_fingerprint,
        "hyperedges": int(model["incidence"].shape[1]),
        "incidences": int(model["incidence_count"]),
        "input_fingerprint": model["input_fingerprint"],
        "macro_area": macroarea,
        "nodes": int(model["incidence"].shape[0]),
        "partition_rule": "municipality_induced_then_min_edge_size",
    }


def _validate_geographical_results(results: pd.DataFrame) -> None:
    if results.empty:
        return
    if "macro_area" not in results:
        raise ValueError("Geographical results must contain a macro_area column")
    canonical_areas = results["macro_area"].map(normalize_macroarea)
    if not canonical_areas.eq(results["macro_area"]).all():
        raise ValueError("Checkpoint macro-area values must use canonical ISTAT labels")
    for _, row in results.iterrows():
        graph = _json_object(row["graph_metadata"], "graph_metadata")
        if (
            graph.get("experiment_type") != EXPERIMENT_TYPE
            or graph.get("macro_area") != row["macro_area"]
            or graph.get("partition_rule")
            != "municipality_induced_then_min_edge_size"
        ):
            raise ValueError("Checkpoint graph metadata is not a geographical experiment")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    years = available_years(DATA_DIR)
    if not years:
        raise FileNotFoundError(
            "No processed yearly networks found; run `python src/prepare_data.py`."
        )
    if args.years is None:
        years = [years[-1]]
    else:
        missing = sorted(set(args.years) - set(years))
        if missing:
            raise ValueError(f"Unavailable requested years: {missing}")
        years = sorted(set(args.years))

    geography = load_municipality_geography(args.registry)
    mapping_fingerprint = geography_fingerprint(geography)
    ids_by_area = {
        area: geography.loc[
            geography["istat_macroarea"].eq(area), "municipality_id"
        ].to_numpy(dtype="int64")
        for area in args.macroareas
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = load_results(args.output)
    _validate_geographical_results(results)
    specs = [
        (year, area, alpha, budget)
        for year in years
        for area in args.macroareas
        for alpha in args.alphas
        for budget in args.seed_budgets
    ]

    for number, (year, area, alpha, budget) in enumerate(specs, start=1):
        config = ExperimentConfig(
            seed_budget=budget,
            max_iter=args.max_iter,
            edge_weight_beta=EDGE_WEIGHT_BETA,
            node_weight_beta=NODE_WEIGHT_BETA,
            alpha=alpha,
            search_seconds_per_year=float(args.search_seconds),
        )
        identity = partition_identity(area, mapping_fingerprint)
        model = load_year_model(
            year,
            config,
            data_dir=DATA_DIR,
            municipality_ids=ids_by_area[area],
            partition_identity=identity,
        )
        base_identity = result_identity(year, data_dir=DATA_DIR)
        instance = {
            "year": year,
            "macro_area": area,
            **parameter_values(config),
            "graph_metadata": _json_dump(
                geographical_graph_metadata(model, area, mapping_fingerprint)
            ),
            "model_metadata": base_identity["model_metadata"],
        }
        matches = instance_mask(results, instance)
        label = f"macro_area={area}, {progress_parameters(year, config)}"
        if matches.any():
            for _, saved in results.loc[matches].iterrows():
                validate_saved_seeds(saved, model, config)
            print(f"[{number}/{len(specs)}] already saved: {label}")
            continue

        print(f"[{number}/{len(specs)}] {label}")
        summary, _, _ = run_year(
            year,
            config,
            verbose=args.verbose,
            data_dir=DATA_DIR,
            municipality_ids=ids_by_area[area],
            partition_identity=identity,
        )
        if summary["input_fingerprint"] != model["input_fingerprint"]:
            raise ValueError("Input files changed during evaluation; no checkpoint saved")
        row = result_row(summary, config)
        row["macro_area"] = area
        row["graph_metadata"] = instance["graph_metadata"]
        results = upsert_result(results, row)
        _validate_geographical_results(results)
        save_results(args.output, results)
        print(
            f"[{number}/{len(specs)}] saved: "
            f"start_spread={row['start_spread']:.3f}, "
            f"finish_spread={row['finish_spread']:.3f}"
        )

    print(f"Saved {len(results)} geographical row(s) to {args.output}")


if __name__ == "__main__":
    main()
