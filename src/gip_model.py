"""Population-aware, shared-gate GIP with vectorized sparse evaluation."""

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


def _node_parameter(value: Any, name: str, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim > 1 or (size is not None and array.ndim == 1 and array.shape != (size,)):
        raise ValueError(f"{name} must be scalar or have shape ({size},)")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return array if size is None else np.broadcast_to(array, (size,))


def _update_count(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True, init=False)
class GIPParameters:
    """Node-wise bounds (scalars broadcast); historical h0 is exactly thesis u0.

    Either h0 or u0 may be supplied, never both. theta_l/theta_h correspond to
    theta_L/theta_H. There is no independent seed intensity a0.
    """

    h0: Any
    l0: Any
    theta_l: Any
    theta_h: Any
    gamma: float
    eps: float

    def __init__(self, h0=None, l0=1.0, theta_l=2.0, theta_h=50.0,
                 gamma=0.1, eps=0.01, *, u0=None):
        if h0 is not None and u0 is not None:
            raise ValueError("h0 is an alias of u0; supply only one")
        values = dict(h0=1.0 if h0 is None and u0 is None else (u0 if h0 is None else h0),
                      l0=l0, theta_l=theta_l, theta_h=theta_h, gamma=gamma, eps=eps)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def u0(self):
        return self.h0

    def validate(self, size: int | None = None, alpha: float | None = None) -> None:
        lower, upper, theta_l, theta_h = [
            _node_parameter(getattr(self, name), name, size)
            for name in ("l0", "h0", "theta_l", "theta_h")
        ]
        try:
            lower, upper, theta_l, theta_h = np.broadcast_arrays(lower, upper, theta_l, theta_h)
        except ValueError as error:
            raise ValueError("Node-wise GIP parameters must have compatible shapes") from error
        if np.any(lower <= 0) or np.any(lower > upper):
            raise ValueError("Require 0 < l0 <= u0 (historical h0)")
        with np.errstate(over="raise", invalid="raise"):
            if np.any(theta_h * upper < theta_l * lower):
                raise ValueError("Require theta_h * u0 >= theta_l * l0")
            if alpha is not None:
                alpha = _nonnegative_scalar(alpha, "alpha")
                if np.any(theta_l * alpha >= 1):
                    raise ValueError("Require theta_l * alpha < 1 for decaying caps")
        if np.ndim(self.gamma) != 0 or not np.isfinite(self.gamma) or not 0 < self.gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        if _nonnegative_scalar(self.eps, "eps") <= 0:
            raise ValueError("eps must be positive")


class IncompleteEvaluationError(RuntimeError):
    """An operational cutoff cannot be used as a converged optimization score."""


@dataclass
class GIPResult:
    """Numerical prefix of the infinite sum, including t=0 exactly once."""

    total_spread: float
    spread_history: list[float]
    states: list[np.ndarray]
    stopping_reason: str
    max_iter: int | None = None

    @property
    def iterations(self) -> int:
        return len(self.states) - 1

    @property
    def steps(self) -> int:
        return self.iterations

    @property
    def score(self) -> float:
        return self.total_spread

    @property
    def final_state(self) -> np.ndarray:
        return self.states[-1]

    @property
    def status(self) -> str:
        return self.stopping_reason

    def require_complete(self) -> GIPResult:
        if self.status != "tolerance_reached":
            raise IncompleteEvaluationError(
                f"Incomplete GIP evaluation: {self.status} after {self.steps} updates "
                f"(max_iter={self.max_iter}); increase/remove the operational guard"
            )
        return self


def _nonnegative_difference(total: np.ndarray, own: np.ndarray, operations: Any) -> np.ndarray:
    """Erase only cancellation within eight operation-count-scaled error units."""
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
    """Private sparse snapshot, incident/owner lists, and lazy co-incidence lists.

    Reuse across seed evaluations. GIP operates directly on the incidence matrix;
    the neighbors/nodes interface supplies NaDS's local-search ordering without
    materializing an n-by-n graph projection.
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
        self._squared = matrix.multiply(matrix).tocsr()  # Elementwise, never B ** 2.
        self._columns = matrix.tocsc()
        if not np.isfinite(self._squared.data).all():
            raise FloatingPointError("Elementwise incidence square overflowed")
        col_counts = np.diff(self._columns.indptr)
        self._operations = np.diff(matrix.indptr) + col_counts.max(initial=0) + 4
        self._edge_operations = 2 * col_counts + 4
        for stored in (self._matrix, self._squared, self._columns):
            for array in (stored.data, stored.indices, stored.indptr):
                array.flags.writeable = False
        self.incident_companies = tuple(matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]
                                        for i in range(matrix.shape[0]))
        self.owners = tuple(self._columns.indices[self._columns.indptr[e]:self._columns.indptr[e+1]]
                            for e in range(matrix.shape[1]))
        self._neighbors: dict[int, tuple[int, ...]] = {}
        self.nodes = range(matrix.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return self._matrix.shape

    def neighbors(self, node: int) -> tuple[int, ...]:
        if node not in self.nodes:
            raise IndexError("Invalid municipality index")
        if node not in self._neighbors:
            others = {int(i) for e in self.incident_companies[node] for i in self.owners[e]}
            others.discard(node)
            self._neighbors[node] = tuple(sorted(others))
        return self._neighbors[node]

    def _pressure(self, weights, state, omega, stress):
        source = omega * state
        pressure = np.asarray(self._matrix.T @ source).reshape(-1)
        if not np.isfinite(source).all() or not np.isfinite(pressure).all():
            raise FloatingPointError("Non-finite company pressure")
        return source, pressure, weights * (pressure > stress)

    def _raw_incidence(self, weights, state, omega, stress):
        source, pressure, active_weights = self._pressure(weights, state, omega, stress)
        total = np.asarray(self._matrix @ (active_weights * pressure)).reshape(-1)
        own = source * np.asarray(self._squared @ active_weights).reshape(-1)
        return _nonnegative_difference(total, own, self._operations)


def prepare_incidence(incidence: Any) -> PreparedIncidence:
    return incidence if isinstance(incidence, PreparedIncidence) else PreparedIncidence(incidence)


def gip_thresholds(values, lower, upper) -> np.ndarray:
    """Componentwise: zero below lower, otherwise cap at upper; equality passes."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Threshold inputs must be a finite non-negative vector")
    lower = _node_parameter(lower, "lower", len(values))
    upper = _node_parameter(upper, "upper", len(values))
    if np.any(upper < lower):
        raise ValueError("The upper GIP threshold cannot be below the lower threshold")
    return np.minimum(np.where(values >= lower, values, 0.0), upper)


def _step_inputs(incidence, edge_weights, state, node_weights, stress_level):
    network = prepare_incidence(incidence)
    weights = _vector(edge_weights, "edge_weights", network.shape[1])
    state = _vector(state, "state", network.shape[0])
    omega = np.ones_like(state) if node_weights is None else _vector(node_weights, "node_weights", len(state))
    stress = _nonnegative_scalar(stress_level, "stress_level")
    return network, weights, state, omega, stress


def gip_raw_step(incidence, edge_weights, state, stress_level=0.0, *, node_weights=None):
    """Production incoming influence from vectorized sparse incidence products."""
    network, weights, state, omega, stress = _step_inputs(incidence, edge_weights, state, node_weights, stress_level)
    return network._raw_incidence(weights, state, omega, stress)


def gip_incidence_raw_step(incidence, edge_weights, state, stress_level=0.0, *, node_weights=None):
    """Explicit name for the same sparse full-incidence production evaluator."""
    return gip_raw_step(
        incidence,
        edge_weights,
        state,
        stress_level,
        node_weights=node_weights,
    )


def gip_source_raw_step(incidence, edge_weights, state, stress_level=0.0, *, node_weights=None):
    """Independent small-case source-neighbor sum, without diagonal subtraction.

    Gates are shared by all recipients. This slow reference never constructs W_H.
    Its first weight index is the source: raw = W_H.T @ q.
    """
    network, weights, state, omega, stress = _step_inputs(incidence, edge_weights, state, node_weights, stress_level)
    source, pressure, active_weights = network._pressure(weights, state, omega, stress)
    raw = np.zeros(network.shape[0])
    for i in np.flatnonzero(state > 0):
        for j in network.neighbors(int(i)):
            shared = set(network.incident_companies[i]).intersection(network.incident_companies[j])
            raw[j] += source[i] * sum(
                network._matrix[i, e] * active_weights[e] * network._matrix[j, e] for e in shared
            )
    if not np.isfinite(raw).all():
        raise FloatingPointError("Non-finite source-neighbor influence")
    return raw


def gip_step(incidence, edge_weights, state, lower, upper, stress_level=0.0, *, node_weights=None):
    """One synchronous update; all inputs use the old state."""
    raw = gip_raw_step(incidence, edge_weights, state, stress_level, node_weights=node_weights)
    return gip_thresholds(raw, lower, upper)


def company_activity(incidence, edge_weights, state, stress_level=0.0, *, node_weights=None):
    """Raw return to other owners, before bounds and recipient valuation/discount.

    Source population enters pressure/transmission. Its sum equals sum(raw).
    """
    network, weights, state, omega, stress = _step_inputs(incidence, edge_weights, state, node_weights, stress_level)
    source, pressure, active_weights = network._pressure(weights, state, omega, stress)
    total = np.asarray(network._matrix.sum(axis=0)).reshape(-1) * pressure
    own = np.asarray(network._squared.T @ source).reshape(-1)
    activity = active_weights * _nonnegative_difference(total, own, network._edge_operations)
    if not np.isfinite(activity).all():
        raise FloatingPointError("Non-finite company activity")
    return activity


def gip_bounds(parameters: GIPParameters, alpha: float, step: int):
    """Original schedules at NEXT step s >= 1; scalars or node-wise arrays.

    The common growth factor avoids inf*0 from separate large powers. At s=1,
    growth**0 = 1 also when theta_l=0, so the first upper cap can be positive.
    """
    step = _update_count(step, "step", 1)
    parameters.validate(alpha=alpha)
    growth = np.asarray(parameters.theta_l) * alpha
    factor = growth ** (step - 1)
    with np.errstate(over="raise", invalid="raise"):
        lower = (growth * factor) * np.asarray(parameters.l0)
        upper = (np.asarray(parameters.theta_h) * alpha) * factor * np.asarray(parameters.u0)
    return lower, upper


def seed_state(seeds, parameters: GIPParameters, *, budget: int | None = None):
    """Initialize q0 = u0*z; validate a binary seed indicator and optional budget."""
    seeds = np.asarray(seeds, dtype=float)
    if seeds.ndim != 1 or not np.isfinite(seeds).all() or np.any((seeds != 0) & (seeds != 1)):
        raise ValueError("seeds must be a finite binary vector")
    parameters.validate(len(seeds))
    if budget is not None and seeds.sum() != _update_count(budget, "budget", 0):
        raise ValueError("Seed count must equal budget")
    return _node_parameter(parameters.u0, "u0", len(seeds)) * seeds


def gip_from_seeds(incidence, edge_weights, seeds, parameters, *, budget=None, **kwargs):
    """Seed-indicator interface to the same evaluator used by all optimizers."""
    return gip(incidence, edge_weights, seed_state(seeds, parameters, budget=budget), parameters, **kwargs)


def gip(incidence, edge_weights, initial_state, parameters: GIPParameters, *,
        node_weights=None, stress_level=0.0, alpha: float | None = 0.1,
        max_iter: int | None = None) -> GIPResult:
    """Approximate the infinite discounted sum until ||discount*q||_2 <= eps.

    initial_state is explicitly q0=u0*z, never an independent amplitude. Prefer
    gip_from_seeds for binary z. No fixed horizon or equal-state stopping exists.
    max_iter is an optional operational guard; its result is incomplete. Alpha
    defaults to the runner's fixed .1; alpha=None explicitly uses mean(kappa).
    Neither choice rescales B. Graph reductions must supply the same graph alpha.
    """
    network, weights, state, omega, stress = _step_inputs(incidence, edge_weights, initial_state, node_weights, stress_level)
    state = state.copy()
    if alpha is None and not len(weights):
        raise ValueError("alpha=None requires at least one edge weight")
    alpha = _nonnegative_scalar(weights.mean() if alpha is None else alpha, "alpha")
    parameters.validate(len(state), alpha)
    u0 = _node_parameter(parameters.u0, "u0", len(state))
    if np.any((state != 0) & (state != u0)):
        raise ValueError("initial_state must equal u0*z with binary z; use gip_from_seeds")
    if max_iter is not None:
        max_iter = _update_count(max_iter, "max_iter", 0)
    # Validate the first cap even when the initial state already meets tolerance.
    gip_bounds(parameters, alpha, 1)
    states = [state]
    scores = [float(np.sum(omega * state))]
    if not np.isfinite(scores[0]):
        raise FloatingPointError("Non-finite initial score")
    step, discount = 0, 1.0
    status = "tolerance_reached"
    while np.hypot.reduce(discount * state, initial=0.0) > parameters.eps:
        if max_iter is not None and step >= max_iter:
            status = "operational_cutoff"
            break
        lower, upper = gip_bounds(parameters, alpha, step + 1)
        next_state = gip_thresholds(
            network._raw_incidence(weights, state, omega, stress),
            lower,
            upper,
        )
        discount *= 1.0 - parameters.gamma
        score = scores[-1] + discount * float(np.sum(omega * next_state))
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite discounted score")
        state = next_state
        step += 1
        states.append(state)
        scores.append(score)
    return GIPResult(scores[-1], scores, states, status, max_iter)


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
