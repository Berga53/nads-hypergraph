"""Versioned content identities for population-aware infinite-discounted GIP."""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.gip_model import (
    GIPParameters, prepare_incidence, gip_bounds, _node_parameter,
    _nonnegative_scalar, _update_count, _vector,
)

MODEL_VERSION = "gip_population_v2"
POPULATION_ROLE = "source_pressure_and_objective"
SCORE_CONVENTION = "infinite_discounted_sum_including_t0"
INITIAL_STATE_CONVENTION = "q0=u0*z;h0=u0"
MODEL_METADATA = {
    "model_version": MODEL_VERSION,
    "population_role": POPULATION_ROLE,
    "score_convention": SCORE_CONVENTION,
    "initial_state_convention": INITIAL_STATE_CONVENTION,
}


def _array_digest(digest, value) -> None:
    array = np.asarray(value, dtype="<f8")
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes(order="C"))


def evaluation_cache_key(incidence, edge_weights, initial_state, parameters: GIPParameters,
                         *, node_weights=None, stress_level=0.0, alpha=0.1,
                         max_iter=None) -> str:
    """Hash seeds/amplitudes, network, parameters, bounds, roles and stopping rule.

    Scalar/node-wise equivalents canonicalize to the same vectors. No scores from
    the earlier score-only population or finite-horizon model can match this key.
    """
    network = prepare_incidence(incidence)
    n = network.shape[0]
    edge_weights = _vector(edge_weights, "edge_weights", network.shape[1])
    initial_state = _vector(initial_state, "initial_state", n)
    omega = np.ones(n) if node_weights is None else _vector(node_weights, "node_weights", n)
    stress_level = _nonnegative_scalar(stress_level, "stress_level")
    if max_iter is not None:
        max_iter = _update_count(max_iter, "max_iter", 0)
    if alpha is None and not len(edge_weights):
        raise ValueError("alpha=None requires at least one edge weight")
    resolved_alpha = _nonnegative_scalar(edge_weights.mean() if alpha is None else alpha, "alpha")
    parameters.validate(n, resolved_alpha)
    if np.any((initial_state != 0) & (initial_state != _node_parameter(parameters.u0, "u0", n))):
        raise ValueError("initial_state must equal u0*z with binary z")
    gip_bounds(parameters, resolved_alpha, 1)
    matrix = network._matrix
    digest = hashlib.sha256()
    settings = {
        **MODEL_METADATA,
        "gamma": float(parameters.gamma), "eps": float(parameters.eps),
        "stress_level": stress_level,
        "alpha": resolved_alpha,
        "alpha_construction": "mean_kappa" if alpha is None else "explicit",
        "max_iter": max_iter,
        "stopping_rule": "norm((1-gamma)^t*q_t,2)<=eps;guard_is_incomplete",
        "shape": matrix.shape,
    }
    digest.update(json.dumps(settings, sort_keys=True, allow_nan=False).encode())
    for array in (matrix.indptr, matrix.indices, matrix.data, edge_weights, initial_state,
                  omega, *(_node_parameter(getattr(parameters, name), name, n)
                           for name in ("l0", "h0", "theta_l", "theta_h"))):
        _array_digest(digest, array)
    return digest.hexdigest()


def input_fingerprint(data_dir: Path, year: int) -> str:
    """Hash all source CSV content; saved checkpoints and notebook caches use it."""
    digest = hashlib.sha256()
    for name in (f"rete_{year}.csv", "hyperedge_parameters.csv", "node_parameters.csv"):
        digest.update(name.encode())
        with (data_dir / name).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
