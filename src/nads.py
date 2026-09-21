"""NaDS search for fixed-budget influence maximization.

This version deliberately contains no multigrid (MG) phase and imposes no
connectivity condition on the seed set.  The diffusion model is supplied as an
objective callback, so it can optimize a graph or hypergraph diffusion model.
"""

from __future__ import annotations

import collections
import itertools
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Deque

import networkx as nx
import numpy as np


SpreadObjective = Callable[[np.ndarray], float]
SeedKey = tuple[int, ...]


def _seedset_from_x(x: np.ndarray) -> set[int]:
    return set(int(index) for index in np.flatnonzero(x))


def _x_from_seedset(n_nodes: int, seedset: set[int]) -> np.ndarray:
    x = np.zeros(n_nodes, dtype=float)
    if seedset:
        x[np.fromiter(seedset, dtype=int)] = 1.0
    return x


def _seed_key(seedset: set[int]) -> SeedKey:
    return tuple(sorted(seedset))


def _time_exceeded(start: float, max_time: float) -> bool:
    return time.monotonic() - start >= max_time


def _print_status(
    verbose: int,
    calls: int,
    start: float,
    max_time: float,
    spread: float,
    *,
    done: bool,
) -> None:
    if not verbose:
        return
    elapsed = time.monotonic() - start
    prefix = "NaDS done" if done else "NaDS search"
    end = "\n" if done else ""
    print(
        f"\r{prefix}: calls={calls}, time={elapsed:.1f}/{max_time:.1f}s, "
        f"spread={spread:.6g}" + " " * 10,
        end=end,
    )


def _graph_swap_neighbors(
    graph: nx.Graph,
    seedset: set[int],
    rng: np.random.Generator,
) -> Iterator[set[int]]:
    """Yield one-for-one exchanges along graph adjacencies.

    Adjacency defines a local-search neighborhood only; it is not a feasibility
    condition on either the current or proposed seed set.
    """
    seed_nodes = np.array(sorted(seedset), dtype=int)
    rng.shuffle(seed_nodes)
    for removed in seed_nodes:
        candidates = np.array(
            sorted(int(node) for node in graph.neighbors(int(removed)) if node not in seedset),
            dtype=int,
        )
        rng.shuffle(candidates)
        for added in candidates:
            neighbor = set(seedset)
            neighbor.remove(int(removed))
            neighbor.add(int(added))
            yield neighbor


def _exchange_neighbors(
    n_nodes: int,
    seedset: set[int],
    max_exchange: int,
    rng: np.random.Generator,
) -> Iterator[set[int]]:
    """Yield exchanges of one through ``max_exchange`` seeds lazily."""
    inside = np.array(sorted(seedset), dtype=int)
    outside = np.array(sorted(set(range(n_nodes)) - seedset), dtype=int)
    rng.shuffle(inside)
    rng.shuffle(outside)

    largest_exchange = min(max_exchange, len(inside), len(outside))
    for exchange_size in range(1, largest_exchange + 1):
        for removed in itertools.combinations(inside.tolist(), exchange_size):
            retained = seedset.difference(removed)
            for added in itertools.combinations(outside.tolist(), exchange_size):
                yield retained.union(added)


def _evaluate_neighbors(
    objective: SpreadObjective,
    neighbors: Iterable[set[int]],
    *,
    n_nodes: int,
    base_spread: float,
    base_x: np.ndarray,
    xi_t: float,
    buffer: Deque[SeedKey],
    calls: int,
    history: list[dict[str, object]],
    start: float,
    max_time: float,
    max_neighbors: int | None,
    verbose: int,
) -> tuple[np.ndarray, float, int, bool, int]:
    """Evaluate a lazy neighborhood and retain its best improving solution."""
    best_x = base_x
    best_spread = base_spread
    evaluated = 0
    stopped = False

    for seedset in neighbors:
        if max_neighbors is not None and evaluated >= max_neighbors:
            break
        if _time_exceeded(start, max_time):
            stopped = True
            break

        key = _seed_key(seedset)
        if key in buffer:
            continue

        candidate_x = _x_from_seedset(n_nodes, seedset)
        candidate_spread = float(objective(candidate_x))
        if not np.isfinite(candidate_spread):
            raise ValueError("The spread objective returned a non-finite value")

        buffer.append(key)
        calls += 1
        evaluated += 1
        _print_status(
            verbose,
            calls,
            start,
            max_time,
            candidate_spread,
            done=False,
        )

        if candidate_spread > best_spread:
            best_x = candidate_x
            best_spread = candidate_spread
            history.append(
                {
                    "call": calls,
                    "elapsed_seconds": time.monotonic() - start,
                    "seed_indices": list(key),
                    "value": best_spread,
                }
            )

            # NaDS restarts its neighborhood after a sufficiently large gain.
            if best_spread > (1.0 + xi_t) * base_spread:
                break

    return best_x, best_spread, calls, stopped, evaluated


def nads(
    objective: SpreadObjective,
    x0: np.ndarray,
    delta: float,
    xi: float,
    d: int,
    max_time: float,
    buffer_dim: int,
    *,
    neighbor_graph: nx.Graph | None = None,
    max_neighbors_per_phase: int | None = None,
    max_iterations: int = 1_000,
    random_seed: int | None = 42,
    verbose: int = 0,
) -> tuple[list[float], list[np.ndarray], list[dict[str, object]]]:
    """Maximize a diffusion objective using fixed-cardinality seed exchanges.

    Parameters
    ----------
    objective:
        Function accepting a binary node-seed vector and returning scalar
        influence spread. The callback is the only diffusion-model dependency.
    x0:
        Initial binary seed vector. Every candidate retains its nonzero count.
    delta:
        Multiplicative reduction of the sufficient-improvement threshold.
    xi:
        Initial relative sufficient-improvement threshold.
    d:
        Maximum symmetric-difference distance for the broader neighborhood.
        ``d=2`` performs one-for-one exchanges; ``d=4`` also tries two-for-two.
    max_time:
        Wall-clock budget in seconds.
    buffer_dim:
        Number of recently evaluated seed sets retained to avoid repeats.
    neighbor_graph:
        Optional graph used only to define the first, local swap neighborhood.
        If omitted, the first phase considers unrestricted one-for-one swaps.
    max_neighbors_per_phase:
        Optional cap on objective evaluations per neighborhood. This is useful
        for large hypergraphs because candidate generation is otherwise exact.
    max_iterations:
        Maximum number of accepted/rejected search iterations.
    random_seed:
        Controls neighborhood ordering; it does not affect the objective.
    verbose:
        Print live progress when nonzero.

    Returns
    -------
    spread_history, seed_history, evaluation_history
        Accepted objective values, their binary seed vectors, and the initial
        evaluation followed by every strict improvement. Each event contains
        its value, elapsed seconds, objective-call count, and seed indices.
    """
    initial_x = np.asarray(x0, dtype=float)
    if initial_x.ndim != 1:
        raise ValueError("x0 must be a one-dimensional seed vector")
    if not np.isfinite(initial_x).all():
        raise ValueError("x0 contains non-finite values")
    if np.any(initial_x < 0):
        raise ValueError("x0 cannot contain negative values")
    if not 0 < delta <= 1:
        raise ValueError("delta must be in (0, 1]")
    if xi < 0:
        raise ValueError("xi must be non-negative")
    if d < 2:
        raise ValueError("d must be at least 2")
    if max_time <= 0:
        raise ValueError("max_time must be positive")
    if buffer_dim <= 0:
        raise ValueError("buffer_dim must be positive")
    if max_neighbors_per_phase is not None and max_neighbors_per_phase <= 0:
        raise ValueError("max_neighbors_per_phase must be positive when provided")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    n_nodes = len(initial_x)
    initial_seedset = _seedset_from_x(initial_x)
    if not initial_seedset:
        raise ValueError("x0 must contain at least one seed")
    if neighbor_graph is not None:
        missing_nodes = initial_seedset.difference(neighbor_graph.nodes)
        if missing_nodes:
            raise ValueError(
                "neighbor_graph is missing seed-vector node indices: "
                f"{sorted(missing_nodes)[:10]}"
            )

    # Candidate vectors are binary even if the caller used another positive
    # marker in x0. Seed intensity belongs inside the objective callback.
    initial_x = _x_from_seedset(n_nodes, initial_seedset)
    start = time.monotonic()
    initial_spread = float(objective(initial_x))
    if not np.isfinite(initial_spread):
        raise ValueError("The spread objective returned a non-finite initial value")

    spread_history = [initial_spread]
    seed_history = [initial_x]
    evaluation_history: list[dict[str, object]] = [
        {
            "call": 1,
            "elapsed_seconds": time.monotonic() - start,
            "seed_indices": sorted(initial_seedset),
            "value": initial_spread,
        }
    ]
    calls = 1
    xi_t = float(xi)
    buffer: Deque[SeedKey] = collections.deque(maxlen=buffer_dim)
    buffer.append(_seed_key(initial_seedset))
    rng = np.random.default_rng(random_seed)

    _print_status(verbose, calls, start, max_time, initial_spread, done=False)

    for _ in range(max_iterations):
        if _time_exceeded(start, max_time):
            break

        current_x = seed_history[-1]
        current_spread = spread_history[-1]
        current_seedset = _seedset_from_x(current_x)

        if neighbor_graph is None:
            local_neighbors = _exchange_neighbors(
                n_nodes,
                current_seedset,
                max_exchange=1,
                rng=rng,
            )
        else:
            local_neighbors = _graph_swap_neighbors(
                neighbor_graph,
                current_seedset,
                rng,
            )

        candidate_x, candidate_spread, calls, stopped, _ = _evaluate_neighbors(
            objective,
            local_neighbors,
            n_nodes=n_nodes,
            base_spread=current_spread,
            base_x=current_x,
            xi_t=xi_t,
            buffer=buffer,
            calls=calls,
            history=evaluation_history,
            start=start,
            max_time=max_time,
            max_neighbors=max_neighbors_per_phase,
            verbose=verbose,
        )
        if candidate_spread > current_spread:
            seed_history.append(candidate_x)
            spread_history.append(candidate_spread)
            if stopped:
                break
            if candidate_spread <= (1.0 + xi_t) * current_spread:
                xi_t *= delta
            continue
        if stopped:
            break

        # Broader d-exchange phase. There is no feasibility filter: every
        # fixed-budget seed set generated here is admissible.
        broad_neighbors = _exchange_neighbors(
            n_nodes,
            current_seedset,
            max_exchange=max(1, d // 2),
            rng=rng,
        )
        candidate_x, candidate_spread, calls, stopped, _ = _evaluate_neighbors(
            objective,
            broad_neighbors,
            n_nodes=n_nodes,
            base_spread=current_spread,
            base_x=current_x,
            xi_t=xi_t,
            buffer=buffer,
            calls=calls,
            history=evaluation_history,
            start=start,
            max_time=max_time,
            max_neighbors=max_neighbors_per_phase,
            verbose=verbose,
        )
        if candidate_spread > current_spread:
            seed_history.append(candidate_x)
            spread_history.append(candidate_spread)
            if stopped:
                break
            if candidate_spread <= (1.0 + xi_t) * current_spread:
                xi_t *= delta
            continue
        if stopped:
            break

        # Neither neighborhood contains an improving solution within the
        # evaluation budget, so the current seed set is the NaDS result.
        break

    _print_status(
        verbose,
        calls,
        start,
        max_time,
        spread_history[-1],
        done=True,
    )
    return spread_history, seed_history, evaluation_history


# Historical aliases for old notebooks; new code should import ``nads``.
nads_td = nads
NaDS_td = nads
