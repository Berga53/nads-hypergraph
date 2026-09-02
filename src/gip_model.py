"""GIP model plus aligned hyperedge and municipality parameter loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_PARAMETER_PATH = PROCESSED_DATA_DIR / "hyperedge_parameters.csv"
DEFAULT_NODE_PARAMETER_PATH = PROCESSED_DATA_DIR / "node_parameters.csv"


@dataclass(frozen=True)
class GIPParameters:
    """Parameters controlling the GIP thresholds and stopping rule."""

    h0: float
    l0: float
    theta_l: float
    theta_h: float
    gamma: float
    eps: float

    def validate(self) -> None:
        if self.h0 < 0 or self.l0 < 0:
            raise ValueError("h0 and l0 must be non-negative")
        if self.theta_l < 0 or self.theta_h < 0:
            raise ValueError("theta_l and theta_h must be non-negative")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must be in [0, 1)")
        if self.eps < 0:
            raise ValueError("eps must be non-negative")


@dataclass
class GIPResult:
    """Complete output of one GIP diffusion simulation."""

    total_spread: float
    spread_history: list[float]
    states: list[np.ndarray]

    @property
    def iterations(self) -> int:
        return len(self.states) - 1


def gip_thresholds(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Remove influence below ``lower`` and cap the remainder at ``upper``."""
    if upper < lower:
        # The original model permits different threshold paths, but a cap below
        # the activation floor makes every active value identical and is almost
        # always a configuration error.
        raise ValueError("The upper GIP threshold cannot be below the lower threshold")
    active_values = np.where(values >= lower, values, 0.0)
    return np.minimum(active_values, upper)


def gip_step(
    incidence: Any,
    edge_weights: np.ndarray,
    state: np.ndarray,
    lower: float,
    upper: float,
    stress_level: float = 0.0,
) -> np.ndarray:
    """Advance one diffusion step on a dense or sparse incidence matrix."""
    edge_pressure = np.asarray(incidence.T @ state).reshape(-1)
    activated_edges = edge_pressure > stress_level
    transmitted_pressure = edge_pressure * activated_edges * edge_weights
    node_influence = np.asarray(incidence @ transmitted_pressure).reshape(-1)
    return gip_thresholds(node_influence, lower, upper)


def gip(
    incidence: Any,
    edge_weights: np.ndarray,
    initial_state: np.ndarray,
    parameters: GIPParameters,
    *,
    node_weights: np.ndarray | None = None,
    stress_level: float = 0.0,
    alpha: float | None = 0.1,
    max_iter: int = 999,
) -> GIPResult:
    """Simulate generalized influence propagation.

    ``alpha`` controls the global threshold scale. Set it explicitly to keep
    experiments comparable while the edge-weight vector expresses relative
    company heterogeneity. If ``None``, the mean edge weight is used.

    ``node_weights`` values each municipality in the spread objective without
    changing the GIP transmission dynamics. With population-derived weights,
    activating a more populous municipality therefore contributes more to the
    objective. If omitted, every node has weight one (the original behavior).
    """
    parameters.validate()
    weights = np.asarray(edge_weights, dtype=float).reshape(-1)
    state = np.asarray(initial_state, dtype=float).reshape(-1)
    if incidence.shape != (len(state), len(weights)):
        raise ValueError(
            "Incidence shape must equal (number of nodes, number of edge weights)"
        )
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("edge_weights must be finite and non-negative")
    if not np.isfinite(state).all() or np.any(state < 0):
        raise ValueError("initial_state must be finite and non-negative")
    if node_weights is None:
        objective_weights = np.ones_like(state)
    else:
        objective_weights = np.asarray(node_weights, dtype=float).reshape(-1)
        if len(objective_weights) != len(state):
            raise ValueError("node_weights must contain one value per node")
        if not np.isfinite(objective_weights).all() or np.any(objective_weights < 0):
            raise ValueError("node_weights must be finite and non-negative")
    if stress_level < 0:
        raise ValueError("stress_level must be non-negative")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    threshold_scale = float(weights.mean()) if alpha is None else float(alpha)
    if threshold_scale < 0:
        raise ValueError("alpha must be non-negative")

    states = [state.copy()]
    # Elementwise reduction avoids spurious BLAS floating-point warnings seen
    # with dot products after sparse matrix operations on some NumPy builds.
    spread_history = [float(np.sum(objective_weights * state))]

    for time_step in range(1, max_iter + 1):
        decayed_norm = np.linalg.norm(
            states[-1] * ((1.0 - parameters.gamma) ** time_step)
        )
        if decayed_norm <= parameters.eps:
            break

        lower = (
            (parameters.theta_l * threshold_scale) ** time_step
        ) * parameters.l0
        upper = (
            parameters.theta_h
            * (parameters.theta_l ** (time_step - 1))
            * (threshold_scale**time_step)
            * parameters.h0
        )
        next_state = gip_step(
            incidence,
            weights,
            states[-1],
            lower,
            upper,
            stress_level,
        )
        if np.array_equal(next_state, states[-1]):
            break

        states.append(next_state)
        spread_history.append(
            spread_history[-1] + float(np.sum(objective_weights * next_state))
        )

    return GIPResult(
        total_spread=spread_history[-1],
        spread_history=spread_history,
        states=states,
    )


def aligned_node_weights(
    node_ids: Iterable[int],
    year: int,
    parameter_path: str | Path = DEFAULT_NODE_PARAMETER_PATH,
    *,
    target_mean: float = 1.0,
    beta: float = 1.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load population-derived weights in the exact incidence-matrix row order.

    ``target_mean=1`` keeps weighted and unweighted spread on a comparable scale.
    ``beta=0`` recovers equal node weights, values between 0 and 1 soften the
    population differences, and ``beta=1`` uses the full relative differences.
    The preparation step supplies an auditable within-year median only where a
    historic municipality cannot be joined to the population table.
    """
    if target_mean <= 0:
        raise ValueError("target_mean must be positive")
    if beta < 0:
        raise ValueError("beta must be non-negative")

    parameters = pd.read_csv(
        parameter_path,
        usecols=[
            "year",
            "municipality_id",
            "municipality_tax_code",
            "municipality_bdap_id",
            "municipality_name",
            "population",
            "population_data_available",
            "node_weight_population",
            "node_weight_basis",
        ],
        dtype={
            "municipality_tax_code": "string",
            "municipality_bdap_id": "string",
        },
    )
    year_parameters = parameters.loc[parameters["year"].eq(int(year))].copy()
    if year_parameters.empty:
        available = sorted(int(value) for value in parameters["year"].unique())
        raise ValueError(f"No node parameters for {year}; available: {available}")
    if year_parameters["municipality_id"].duplicated().any():
        raise ValueError(f"Duplicate node parameters found for {year}")

    year_parameters = year_parameters.set_index("municipality_id")
    node_order = pd.Index([int(node) for node in node_ids], name="municipality_id")
    missing = node_order.difference(year_parameters.index)
    if len(missing):
        raise ValueError(
            f"{len(missing):,} nodes are missing parameters; first IDs: "
            f"{missing[:10].tolist()}"
        )

    audit = year_parameters.reindex(node_order).copy()
    base = audit["node_weight_population"].astype(float)
    if base.isna().any() or (base <= 0).any():
        raise ValueError("Population node weights must be present and positive")
    scaled = base.pow(beta)
    node_weight = target_mean * scaled / scaled.mean()
    weights = node_weight.to_numpy(dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError("Calculated node weights contain non-finite values")

    audit["node_weight"] = node_weight
    audit.attrs.update(
        {"year": int(year), "target_mean": float(target_mean), "beta": float(beta)}
    )
    return weights, audit


def aligned_edge_parameters(
    edge_ids: Iterable[int],
    year: int,
    parameter_path: str | Path = DEFAULT_PARAMETER_PATH,
    *,
    target_mean: float = 1.0,
    beta: float = 1.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load and align one real parameter per hyperedge.

    The base parameter is already adjusted for missing-data confidence by
    ``prepare_data.py``. It is renormalized after filtering/reordering edges so
    the returned vector has ``target_mean`` in the exact HyperNetX edge order.
    ``beta`` controls heterogeneity: 0 gives equal weights, 0.5 softens the
    differences, and 1 uses the full company-size differences.
    """
    if target_mean <= 0:
        raise ValueError("target_mean must be positive")
    if beta < 0:
        raise ValueError("beta must be non-negative")

    parameters = pd.read_csv(
        parameter_path,
        usecols=[
            "year",
            "company_id",
            "company_name",
            "company_size_score_0_100",
            "company_size_data_confidence_0_1",
            "company_size_data_quality",
            "edge_parameter_base_0_1",
            "edge_parameter_basis",
        ],
    )
    year_parameters = parameters.loc[parameters["year"].eq(int(year))].copy()
    if year_parameters.empty:
        available = sorted(int(value) for value in parameters["year"].unique())
        raise ValueError(f"No hyperedge parameters for {year}; available: {available}")
    if year_parameters["company_id"].duplicated().any():
        raise ValueError(f"Duplicate hyperedge parameters found for {year}")

    year_parameters = year_parameters.set_index("company_id")
    edge_order = pd.Index([int(edge) for edge in edge_ids], name="company_id")
    missing = edge_order.difference(year_parameters.index)
    if len(missing):
        raise ValueError(
            f"{len(missing):,} edges are missing parameters; first IDs: "
            f"{missing[:10].tolist()}"
        )

    audit = year_parameters.reindex(edge_order).copy()
    base = audit["edge_parameter_base_0_1"].astype(float)
    if base.isna().any() or (base < 0).any():
        raise ValueError("Base edge parameters must be present and non-negative")
    scaled = base.pow(beta)
    edge_parameter = target_mean * scaled / scaled.mean()
    weights = edge_parameter.to_numpy(dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError("Calculated edge parameters contain non-finite values")

    audit["edge_parameter"] = edge_parameter
    audit.attrs.update(
        {"year": int(year), "target_mean": float(target_mean), "beta": float(beta)}
    )
    return weights, audit
