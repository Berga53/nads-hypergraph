"""Test NaDS sensitivity to its initial seed set on one fixed yearly model.

The degree-ranked baseline and uniformly sampled starting sets are optimized
with identical model/search settings.  Each starting point receives one NaDS
run with the same neighborhood-order seed; the time limit is a safety cap, not
a request to perform repeated restarts from the same point.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import (
    DATA_DIR,
    DEFAULT_SEARCH_SECONDS,
    ExperimentConfig,
    RESULT_COLUMNS,
    available_years,
    config_from_row,
    deterministic_initial_seeds,
    load_results,
    load_year_model,
    parameter_values,
    result_row,
    run_year,
    validate_saved_seeds,
)
from src.model_identity import MODEL_METADATA


# Use the normal result schema, including NaDS history and grouped settings. The
# final JSON column carries only starting-point experiment metadata.
RUN_COLUMNS = [*RESULT_COLUMNS, "starting_point_metadata"]


def _canonical_seed_ids(values) -> list[int]:
    return sorted(int(value) for value in values)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def generate_starting_points(
    node_ids: np.ndarray,
    structural_degree: pd.Series,
    node_weights: np.ndarray,
    budget: int,
    num_starts: int,
    initialization_seed: int,
    *,
    include_degree_start: bool,
) -> list[dict[str, object]]:
    """Return distinct, reproducible fixed-budget starting seed sets."""
    n_nodes = len(node_ids)
    if not 1 <= budget <= n_nodes:
        raise ValueError(f"seed_budget must be between 1 and {n_nodes}")
    if num_starts < 1:
        raise ValueError("num_starts must be positive")
    possible = math.comb(n_nodes, budget)
    if num_starts > possible:
        raise ValueError(
            f"Requested {num_starts} distinct starts, but only {possible} seed sets exist"
        )

    points: list[dict[str, object]] = []
    seen: set[tuple[int, ...]] = set()
    if include_degree_start:
        degree_start = deterministic_initial_seeds(
            node_ids, structural_degree, node_weights, budget
        )
        ids = tuple(_canonical_seed_ids(node_ids[np.flatnonzero(degree_start)]))
        seen.add(ids)
        points.append(
            {"run_index": 0, "start_type": "degree_ranked", "seed_ids": list(ids)}
        )

    rng = np.random.default_rng(initialization_seed)
    # Rejection sampling is efficient for the intended regime (few starts from
    # a large municipality set).  Enumerate only when nearly all combinations
    # of a genuinely small instance were requested.
    if possible <= 100_000 and num_starts > possible // 2:
        combinations = list(itertools.combinations(range(n_nodes), budget))
        rng.shuffle(combinations)
        candidate_indices = iter(combinations)
        while len(points) < num_starts:
            indices = next(candidate_indices)
            ids = tuple(_canonical_seed_ids(node_ids[list(indices)]))
            if ids in seen:
                continue
            seen.add(ids)
            points.append(
                {"run_index": len(points), "start_type": "random", "seed_ids": list(ids)}
            )
    else:
        while len(points) < num_starts:
            indices = rng.choice(n_nodes, size=budget, replace=False)
            ids = tuple(_canonical_seed_ids(node_ids[indices]))
            if ids in seen:
                continue
            seen.add(ids)
            points.append(
                {"run_index": len(points), "start_type": "random", "seed_ids": list(ids)}
            )
    return points


def _experiment_spec(
    year: int,
    config: ExperimentConfig,
    model: dict[str, object],
    *,
    num_starts: int,
    initialization_seed: int,
    include_degree_start: bool,
) -> dict[str, object]:
    return {
        "format_version": 2,
        "year": int(year),
        "num_starts": int(num_starts),
        "initialization_seed": int(initialization_seed),
        "include_degree_start": bool(include_degree_start),
        "search_protocol": "one_nads_run_per_start_same_neighbor_order_seed",
        "parameters": parameter_values(config),
        **MODEL_METADATA,
        "input_fingerprint": model["input_fingerprint"],
    }


def _experiment_id(spec: dict[str, object]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _parse_seed_list(value) -> list[int]:
    return _canonical_seed_ids(json.loads(value) if isinstance(value, str) else value)


def _starting_point_metadata(
    spec: dict[str, object],
    *,
    experiment_id: str,
    run_index: int,
    start_type: str,
) -> str:
    metadata = {
        "experiment_id": experiment_id,
        "run_index": int(run_index),
        "start_type": start_type,
        "num_starts": int(spec["num_starts"]),
        "initialization_seed": int(spec["initialization_seed"]),
        "include_degree_start": bool(spec["include_degree_start"]),
        "search_protocol": spec["search_protocol"],
    }
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def _parse_starting_point_metadata(value) -> dict[str, object]:
    metadata = json.loads(value)
    required = {
        "experiment_id",
        "run_index",
        "start_type",
        "num_starts",
        "initialization_seed",
        "include_degree_start",
        "search_protocol",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"starting_point_metadata is missing {missing}")
    return metadata


def _validate_completed_runs(
    runs: pd.DataFrame,
    experiment_id: str,
    starts: list[dict[str, object]],
    model: dict[str, object],
    config: ExperimentConfig,
    spec: dict[str, object],
) -> None:
    if runs.empty:
        return
    missing = [column for column in RUN_COLUMNS if column not in runs.columns]
    if missing:
        raise ValueError(f"The results CSV has an incompatible format; missing {missing}")
    metadata = runs["starting_point_metadata"].map(_parse_starting_point_metadata)
    identities = pd.DataFrame(metadata.tolist(), index=runs.index)
    if identities.duplicated(["experiment_id", "run_index"]).any():
        raise ValueError("The results CSV contains duplicate experiment/run rows")
    current_indices = identities.index[identities["experiment_id"].eq(experiment_id)]
    planned = {int(point["run_index"]): point for point in starts}
    for row_index in current_indices:
        row = runs.loc[row_index]
        row_metadata = metadata.loc[row_index]
        run_index = int(row_metadata["run_index"])
        if run_index not in planned:
            raise ValueError(f"The results CSV contains unplanned run_index {run_index}")
        if _parse_seed_list(row["start_list"]) != planned[run_index]["seed_ids"]:
            raise ValueError(
                f"Starting seeds differ from the reproducible plan for run {run_index}"
            )
        finish_ids = _parse_seed_list(row["finish_list"])
        if (
            len(finish_ids) != config.seed_budget
            or len(set(finish_ids)) != config.seed_budget
            or set(finish_ids) - set(int(value) for value in model["node_ids"])
        ):
            raise ValueError(f"Final seeds are invalid for run {run_index}")
        if int(row["year"]) != int(spec["year"]) or config_from_row(row) != config:
            raise ValueError(f"Saved parameters differ for run {run_index}")
        scores = np.asarray([row["start_spread"], row["finish_spread"]], dtype=float)
        if not np.isfinite(scores).all() or scores[1] < scores[0]:
            raise ValueError(f"Saved scores are invalid for run {run_index}")
        if (
            int(row_metadata["num_starts"]) != int(spec["num_starts"])
            or int(row_metadata["initialization_seed"]) != int(spec["initialization_seed"])
            or bool(row_metadata["include_degree_start"])
            != bool(spec["include_degree_start"])
            or row_metadata["search_protocol"] != spec["search_protocol"]
            or row_metadata["start_type"] != planned[run_index]["start_type"]
        ):
            raise ValueError(f"Saved starting-point setup differs for run {run_index}")
        validate_saved_seeds(row, model, config)


def experiment_summary(spec: dict[str, object], runs: pd.DataFrame) -> dict[str, object]:
    finish_sets = [frozenset(_parse_seed_list(value)) for value in runs["finish_list"]]
    scores = runs["finish_spread"].astype(float).to_numpy()
    pairwise_jaccard = [
        len(left & right) / len(left | right)
        for left, right in itertools.combinations(finish_sets, 2)
    ]
    unique_solutions = len(set(finish_sets))
    completed = len(runs)
    time_limited = sum(
        bool(json.loads(value)["search_time_limit_reached"])
        for value in runs["run_metadata"]
    )
    if completed < int(spec["num_starts"]):
        conclusion = "Experiment is incomplete."
    elif completed == 1:
        conclusion = "Only one starting point was evaluated."
    elif unique_solutions == 1:
        conclusion = "All starting points produced the same final seed set."
    elif np.isclose(scores.min(), scores.max(), rtol=1e-9, atol=1e-9):
        conclusion = "Different final seed sets have equivalent objective values."
    else:
        conclusion = "Starting points produced different seed sets and objective values."
    if time_limited:
        conclusion += f" {time_limited} run(s) reached the time cap."
    return {
        "completed_starts": completed,
        "unique_final_seed_sets": unique_solutions,
        "finish_spread_range": float(scores.max() - scores.min()),
        "finish_jaccard_mean": (
            float(np.mean(pairwise_jaccard)) if pairwise_jaccard else None
        ),
        "conclusion": conclusion,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare NaDS solutions from distinct starting seed sets on one year/setup."
    )
    parser.add_argument(
        "--year", type=int, required=True, help="One processed network year."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1,
        help="Fixed threshold scale (default: 0.1).",
    )
    parser.add_argument(
        "--seed-budget", type=int, default=20,
        help="Fixed seed-set size (default: 20).",
    )
    parser.add_argument(
        "--num-starts", type=int, default=10,
        help="Total distinct starting points (default: 10).",
    )
    parser.add_argument(
        "--random-only",
        action="store_true",
        help="Use only random starts; otherwise run 0 is the degree-ranked baseline.",
    )
    parser.add_argument(
        "--initialization-seed",
        type=int,
        default=12345,
        help="Seed used only to draw random starting points (default: 12345).",
    )
    parser.add_argument(
        "--search-random-seed",
        type=int,
        default=42,
        help="Common NaDS neighborhood-order seed for every start (default: 42).",
    )
    parser.add_argument(
        "--search-seconds",
        type=float,
        default=DEFAULT_SEARCH_SECONDS,
        help="Maximum seconds for each starting point (default: 300).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Optional GIP operational update guard; incomplete evaluations are rejected.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "starting_point_results.csv",
        help=(
            "One CSV receiving every setup and start "
            "(default: results/starting_point_results.csv)."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Show live NaDS evaluations.")
    args = parser.parse_args(argv)
    if args.year not in available_years(DATA_DIR):
        parser.error(f"--year {args.year} is not available in {DATA_DIR}")
    if (
        not np.isfinite(args.alpha)
        or args.alpha < 0
        or ExperimentConfig.theta_l * args.alpha >= 1
    ):
        parser.error("--alpha must be non-negative with theta_l * alpha < 1")
    if args.seed_budget <= 0:
        parser.error("--seed-budget must be positive")
    if args.num_starts <= 0:
        parser.error("--num-starts must be positive")
    if not np.isfinite(args.search_seconds) or args.search_seconds <= 0:
        parser.error("--search-seconds must be positive")
    if args.max_iter is not None and args.max_iter < 0:
        parser.error("--max-iter must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ExperimentConfig(
        seed_budget=int(args.seed_budget),
        alpha=float(args.alpha),
        search_seconds_per_year=float(args.search_seconds),
        random_seed=int(args.search_random_seed),
        max_iter=args.max_iter,
    )
    model = load_year_model(args.year, config, data_dir=DATA_DIR)
    starts = generate_starting_points(
        model["node_ids"],
        model["structural_degree"],
        model["node_weights"],
        config.seed_budget,
        args.num_starts,
        args.initialization_seed,
        include_degree_start=not args.random_only,
    )
    spec = _experiment_spec(
        args.year,
        config,
        model,
        num_starts=args.num_starts,
        initialization_seed=args.initialization_seed,
        include_degree_start=not args.random_only,
    )
    experiment_id = _experiment_id(spec)
    results_path = args.output
    results_path.parent.mkdir(parents=True, exist_ok=True)
    runs = (
        load_results(results_path)
        if results_path.exists()
        else pd.DataFrame(columns=RUN_COLUMNS)
    )
    _validate_completed_runs(runs, experiment_id, starts, model, config, spec)
    saved_metadata = runs["starting_point_metadata"].map(
        _parse_starting_point_metadata
    )
    current_indices = [
        index
        for index, metadata in saved_metadata.items()
        if metadata["experiment_id"] == experiment_id
    ]
    completed = {
        int(saved_metadata.loc[index]["run_index"]) for index in current_indices
    }

    for point in starts:
        run_index = int(point["run_index"])
        if run_index in completed:
            print(f"[{run_index + 1}/{args.num_starts}] already saved")
            continue
        initial = np.isin(model["node_ids"], point["seed_ids"]).astype(float)
        print(f"[{run_index + 1}/{args.num_starts}] start_type={point['start_type']}")
        summary, _, _ = run_year(
            args.year,
            config,
            verbose=args.verbose,
            initial_seeds=initial,
            restart_search=False,
            data_dir=DATA_DIR,
        )
        start_ids = _canonical_seed_ids(point["seed_ids"])
        finish_ids = _canonical_seed_ids(summary["finish_list"])
        row_summary = {**summary, "start_list": start_ids, "finish_list": finish_ids}
        row = result_row(row_summary, config)
        row["starting_point_metadata"] = _starting_point_metadata(
            spec,
            experiment_id=experiment_id,
            run_index=run_index,
            start_type=str(point["start_type"]),
        )
        runs.loc[len(runs), RUN_COLUMNS] = [row[column] for column in RUN_COLUMNS]
        sort_metadata = runs["starting_point_metadata"].map(
            _parse_starting_point_metadata
        )
        runs = (
            runs.assign(
                _experiment_id=[value["experiment_id"] for value in sort_metadata],
                _run_index=[value["run_index"] for value in sort_metadata],
            )
            .sort_values(
                ["year", "alpha", "seed_budget", "_experiment_id", "_run_index"],
                kind="stable",
            )
            [RUN_COLUMNS]
            .reset_index(drop=True)
        )
        _atomic_csv(results_path, runs)
        print(
            f"[{run_index + 1}/{args.num_starts}] saved: "
            f"{row['start_spread']:.6g} -> {row['finish_spread']:.6g}"
        )

    _atomic_csv(results_path, runs[RUN_COLUMNS])
    final_metadata = runs["starting_point_metadata"].map(
        _parse_starting_point_metadata
    )
    current = runs.loc[
        [value["experiment_id"] == experiment_id for value in final_metadata]
    ]
    summary = experiment_summary(spec, current)
    print(summary["conclusion"])
    print(
        f"Unique final sets: {summary['unique_final_seed_sets']}/{summary['completed_starts']}; "
        f"spread range: {summary['finish_spread_range']:.6g}"
    )
    print(f"Saved all starting-point experiments to {results_path}")


if __name__ == "__main__":
    main()
