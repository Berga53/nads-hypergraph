"""Shared-gate GIP without direct self-return, plus economic parameter loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_PARAMETER_PATH = PROCESSED_DATA_DIR / "hyperedge_parameters.csv"
DEFAULT_NODE_PARAMETER_PATH = PROCESSED_DATA_DIR / "node_parameters.csv"


def _nonnegative_scalar(value: float, name: str) -> float:
    if np.ndim(value) != 0 or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return float(value)


def _vector(value: Any, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return array


def _update_count(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class GIPParameters:
    """Bounds, time discount (0 < gamma < 1), and numerical tolerance."""

    h0: float
    l0: float
    theta_l: float
    theta_h: float
    gamma: float
    eps: float

    def validate(self) -> None:
        for name in ("h0", "l0", "theta_l", "theta_h", "eps"):
            _nonnegative_scalar(getattr(self, name), name)
        if np.ndim(self.gamma) != 0 or not np.isfinite(self.gamma) or not 0 < self.gamma < 1:
            raise ValueError("gamma must be in (0, 1)")


@dataclass
class GIPResult:
    """States and cumulative discounted scores, both including t=0."""

    total_spread: float
    spread_history: list[float]
    states: list[np.ndarray]
    stopping_reason: str = "unspecified"
    horizon: int | None = None
    max_iter: int = 999

    @property
    def iterations(self) -> int:
        return len(self.states) - 1


def _nonnegative_difference(total: np.ndarray, own: np.ndarray, operations: Any) -> np.ndarray:
    """Only erase cancellation roundoff, using a local floating-point scale.

    Eight times the operation-count error scale allows for the two sparse
    reductions and products. There is no arbitrary absolute clipping floor.
    A materially negative result indicates invalid arithmetic or an implementation
    error and must never be interpreted as zero influence.
    """
    if not np.isfinite(total).all() or not np.isfinite(own).all():
        raise FloatingPointError("Non-finite influence before self-return subtraction")
    raw = total - own
    scale = np.maximum(np.abs(total), np.abs(own))
    tolerance = (8 * np.finfo(float).eps * operations) * scale
    tolerance += 8 * np.nextafter(0.0, 1.0)
    if np.any(raw < -tolerance):
        raise FloatingPointError("Materially negative influence after self-return subtraction")
    return np.maximum(raw, 0.0)


class PreparedIncidence:
    """Validated private CSR snapshot; reuse its elementwise square across runs.

    Changes to the original B cannot invalidate this snapshot. Construct a new
    instance after changing B. No municipality-by-municipality matrix is built.
    """

    def __init__(self, incidence: Any):
        if sparse.issparse(incidence):
            if incidence.ndim != 2:
                raise ValueError("incidence must be two-dimensional")
            original = incidence.tocoo(copy=True)
            values = original.data
        else:
            original = np.asarray(incidence, dtype=float)
            if original.ndim != 2:
                raise ValueError("incidence must be two-dimensional")
            values = original
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("incidence must be finite and non-negative")
        matrix = sparse.csr_matrix(original, dtype=float, copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if not np.isfinite(matrix.data).all():
            raise ValueError("Summed incidence values must be finite")
        self._matrix = matrix
        # SciPy ** 2 is a matrix power for sparse matrices: never use it here.
        self._squared = matrix.multiply(matrix).tocsr()
        if not np.isfinite(self._squared.data).all():
            raise FloatingPointError("Elementwise incidence square overflowed")
        row_counts = np.diff(matrix.indptr)
        col_counts = np.bincount(matrix.indices, minlength=matrix.shape[1])
        self._operations = row_counts + col_counts.max(initial=0) + 4
        self._edge_operations = 2 * col_counts + 4
        for stored in (self._matrix, self._squared):
            for array in (stored.data, stored.indices, stored.indptr):
                array.flags.writeable = False

    @property
    def shape(self) -> tuple[int, int]:
        return self._matrix.shape

    def _pressure(self, weights: np.ndarray, state: np.ndarray, stress: float):
        pressure = np.asarray(self._matrix.T @ state).reshape(-1)
        if not np.isfinite(pressure).all():
            raise FloatingPointError("Non-finite company pressure")
        # The strict shared gate sees ALL owners and is evaluated before kappa.
        active_weights = weights * (pressure > stress)
        return pressure, active_weights

    def _raw(self, weights: np.ndarray, state: np.ndarray, stress: float) -> np.ndarray:
        pressure, active_weights = self._pressure(weights, state, stress)
        total = np.asarray(self._matrix @ (active_weights * pressure)).reshape(-1)
        own = state * np.asarray(self._squared @ active_weights).reshape(-1)
        return _nonnegative_difference(total, own, self._operations)


def prepare_incidence(incidence: Any) -> PreparedIncidence:
    return incidence if isinstance(incidence, PreparedIncidence) else PreparedIncidence(incidence)


def gip_thresholds(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Remove influence strictly below L and cap influence strictly above U."""
    lower = _nonnegative_scalar(lower, "lower")
    upper = _nonnegative_scalar(upper, "upper")
    if upper < lower:
        raise ValueError("The upper GIP threshold cannot be below the lower threshold")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Threshold inputs must be finite and non-negative")
    return np.minimum(np.where(values >= lower, values, 0.0), upper)


def gip_raw_step(incidence: Any, edge_weights: np.ndarray, state: np.ndarray,
                 stress_level: float = 0.0) -> np.ndarray:
    """Raw shared-gate return, excluding only the recipient's own contribution."""
    network = prepare_incidence(incidence)
    weights = _vector(edge_weights, "edge_weights", network.shape[1])
    state = _vector(state, "state", network.shape[0])
    stress = _nonnegative_scalar(stress_level, "stress_level")
    return network._raw(weights, state, stress)


def gip_step(incidence: Any, edge_weights: np.ndarray, state: np.ndarray,
             lower: float, upper: float, stress_level: float = 0.0) -> np.ndarray:
    """One synchronous update, using the old state for all terms."""
    return gip_thresholds(gip_raw_step(incidence, edge_weights, state, stress_level), lower, upper)


def company_activity(incidence: Any, edge_weights: np.ndarray, state: np.ndarray,
                     stress_level: float = 0.0) -> np.ndarray:
    """Per-company total raw return to OTHER owners, before municipal bounds.

    This diagnostic is unvalued and undiscounted; callers can time-discount it.
    Its sum equals the sum of gip_raw_step, up to roundoff.
    """
    network = prepare_incidence(incidence)
    weights = _vector(edge_weights, "edge_weights", network.shape[1])
    state = _vector(state, "state", network.shape[0])
    stress = _nonnegative_scalar(stress_level, "stress_level")
    pressure, active_weights = network._pressure(weights, state, stress)
    total = np.asarray(network._matrix.sum(axis=0)).reshape(-1) * pressure
    own = np.asarray(network._squared.T @ state).reshape(-1)
    activity = active_weights * _nonnegative_difference(total, own, network._edge_operations)
    if not np.isfinite(activity).all():
        raise FloatingPointError("Non-finite company activity")
    return activity


def gip_bounds(parameters: GIPParameters, alpha: float, step: int) -> tuple[float, float]:
    """Bounds indexed by the next step j >= 1."""
    # Algebraically the specified formulas, using one common growth factor to
    # avoid inf * 0 for long decaying paths (theta_l > 1, alpha < 1).
    growth = parameters.theta_l * alpha
    try:
        factor = growth ** (step - 1)
        lower = growth * parameters.l0 * factor
        upper = parameters.theta_h * alpha * parameters.h0 * factor
    except OverflowError as error:
        raise ValueError("GIP bounds overflowed") from error
    lower = _nonnegative_scalar(lower, "lower bound")
    upper = _nonnegative_scalar(upper, "upper bound")
    if upper < lower:
        raise ValueError("The upper GIP threshold cannot be below the lower threshold")
    return lower, upper


def gip(incidence: Any, edge_weights: np.ndarray, initial_state: np.ndarray,
        parameters: GIPParameters, *, node_weights: np.ndarray | None = None,
        stress_level: float = 0.0, alpha: float | None = 0.1,
        max_iter: int = 999, horizon: int | None = None) -> GIPResult:
    """Evaluate discounted GIP, including seed value exactly once.

    ``horizon=T`` performs exactly T updates and scores T+1 states, including
    T=0; eps and equality never stop it. Otherwise ``max_iter`` is the finite
    J_max, and before update j the test is ||(1-gamma)^(j-1) x_(j-1)||_2 <= eps.
    This numerical truncation is not a bound on omitted score. Gamma and omega
    never enter propagation. No normalization occurs here. ``alpha=None`` keeps
    the historical explicit option of using mean kappa (requires nonempty kappa).
    """
    parameters.validate()
    network = prepare_incidence(incidence)
    weights = _vector(edge_weights, "edge_weights", network.shape[1])
    state = _vector(initial_state, "initial_state", network.shape[0]).copy()
    objective_weights = (np.ones_like(state) if node_weights is None else
                         _vector(node_weights, "node_weights", network.shape[0]))
    stress = _nonnegative_scalar(stress_level, "stress_level")
    max_iter = _update_count(max_iter, "max_iter", 1)
    if horizon is not None:
        horizon = _update_count(horizon, "horizon", 0)
    if alpha is None and not len(weights):
        raise ValueError("alpha=None requires at least one edge weight")
    threshold_scale = _nonnegative_scalar(weights.mean() if alpha is None else alpha, "alpha")
    gip_bounds(parameters, threshold_scale, 1)
    discount = 1.0 - parameters.gamma
    states = [state]
    # Elementwise sum implements dot(omega, x), avoiding platform BLAS warnings.
    spread_history = [float(np.sum(objective_weights * state))]
    if not np.isfinite(spread_history[0]):
        raise FloatingPointError("Non-finite initial score")
    stopping_reason = "fixed_horizon" if horizon is not None else "max_iter"
    updates = max_iter if horizon is None else horizon
    for step in range(1, updates + 1):
        if horizon is None:
            discounted_state = discount ** (step - 1) * state
            # hypot reduction avoids overflow from squaring large finite states.
            if np.hypot.reduce(discounted_state, initial=0.0) <= parameters.eps:
                stopping_reason = "tolerance"
                break
        lower, upper = gip_bounds(parameters, threshold_scale, step)
        state = gip_thresholds(network._raw(weights, state, stress), lower, upper)
        score = spread_history[-1] + float(np.sum(objective_weights * state)) * discount ** step
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite discounted score")
        states.append(state)
        spread_history.append(score)
    return GIPResult(spread_history[-1], spread_history, states,
                     stopping_reason=stopping_reason, horizon=horizon, max_iter=max_iter)


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
