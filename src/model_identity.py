"""Content-based keys for in-memory caches of the single supported GIP model."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from src.gip_model import (
    GIPParameters,
    prepare_incidence, gip_bounds, _nonnegative_scalar, _update_count, _vector,
)


def _array_digest(digest, value) -> None:
    array = np.asarray(value, dtype="<f8")
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes(order="C"))


def evaluation_cache_key(incidence, edge_weights, initial_state, parameters: GIPParameters,
                         *, node_weights=None, stress_level=0.0, alpha=0.1,
                         max_iter=999, horizon=None, initial_state_convention="explicit-x0") -> str:
    """Identify the complete numerical problem, not a row index or seed count.

    Initial state content distinguishes both S and a0. The convention is also
    explicit so seed-indicator callbacks cannot collide with other interventions.
    Canonical sparse content identifies B without constructing a dense adjacency.
    """
    network = prepare_incidence(incidence)
    parameters.validate()
    edge_weights = _vector(edge_weights, "edge_weights", network.shape[1])
    initial_state = _vector(initial_state, "initial_state", network.shape[0])
    if node_weights is not None:
        node_weights = _vector(node_weights, "node_weights", network.shape[0])
    stress_level = _nonnegative_scalar(stress_level, "stress_level")
    max_iter = _update_count(max_iter, "max_iter", 1)
    if horizon is not None:
        horizon = _update_count(horizon, "horizon", 0)
    if alpha is None and not len(edge_weights):
        raise ValueError("alpha=None requires at least one edge weight")
    resolved_alpha = _nonnegative_scalar(edge_weights.mean() if alpha is None else alpha, "alpha")
    gip_bounds(parameters, resolved_alpha, 1)
    matrix = network._matrix
    digest = hashlib.sha256()
    settings = {
        "initial_state_convention": initial_state_convention,
        "parameters": {name: float(value) for name, value in asdict(parameters).items()},
        "stress_level": float(stress_level),
        "alpha": None if alpha is None else float(alpha),
        "max_iter": int(max_iter), "horizon": None if horizon is None else int(horizon),
        "mode": "fixed" if horizon is not None else "early",
        "shape": matrix.shape,
    }
    digest.update(json.dumps(settings, sort_keys=True, allow_nan=False).encode())
    for array in (matrix.indptr, matrix.indices, matrix.data, edge_weights, initial_state,
                  np.ones(matrix.shape[0]) if node_weights is None else node_weights):
        _array_digest(digest, array)
    return digest.hexdigest()


def input_fingerprint(data_dir: Path, year: int) -> str:
    """Hash source CSV content for notebook cache invalidation; not saved in results."""
    digest = hashlib.sha256()
    for name in (f"rete_{year}.csv", "hyperedge_parameters.csv", "node_parameters.csv"):
        digest.update(name.encode())
        with (data_dir / name).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()

