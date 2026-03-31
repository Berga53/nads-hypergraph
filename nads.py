import collections
import itertools
import math
import time
import random
from typing import Callable, Deque, Mapping, Sequence

import networkx as nx
import numpy as np

from .helpers import connected_component_spread, time_exceeded


def _max_comp_size_in_seedset(g: nx.Graph, seedset: set[int]) -> int:
    if len(seedset) == 0:
        return 0
    if len(seedset) == 1:
        return 1
    sub = g.subgraph(seedset)
    return max((len(c) for c in nx.connected_components(sub)), default=0)


def _meets_min_connectedness(g: nx.Graph, seedset: set[int], min_conn_req: int) -> bool:
    return _max_comp_size_in_seedset(g, seedset) >= int(min_conn_req)


def _x_from_seedset(n_nodes: int, seedset: set[int]) -> np.ndarray:
    x = np.zeros(n_nodes)
    x[list(seedset)] = 1.0
    return x


def _seedset_from_x(x: np.ndarray) -> set[int]:
    return set(np.nonzero(x)[0])


def _print_nads_status(
    verbose: int,
    calls: int,
    start: float,
    max_time: float,
    spread: int,
    done: bool = False,
) -> None:
    if not verbose:
        return
    if done:
        print(
            "\r"
            + f"Neighbors search... Done! Calls:{calls}. Time: {round(time.time()-start)}/{max_time} s. "
            + f"Influence spread: {spread}."
            + " " * 20
        )
    else:
        print(
            "\r"
            + f"Neighbors search... Calls:{calls}. Time: {round(time.time()-start)}/{max_time} s. "
            + f"Influence spread: {spread}."
            + " " * 20,
            end="",
        )


def _generate_swap_neighbors(g: nx.Graph, idx: set[int]) -> list[set[int]]:
    neighbors = []
    for elem1 in idx:
        for elem2 in set(g.neighbors(elem1)) - idx:
            temp = idx.copy()
            temp.add(elem2)
            temp.remove(elem1)
            neighbors.append(temp)
    return neighbors


def _generate_d_exchange_neighbors(n_nodes: int, idx: set[int], d: int) -> list[set[int]]:
    neighbors = []
    idx_temp = set(range(n_nodes)) - idx
    for i in range(d // 2):
        for elem1 in itertools.combinations(idx, len(idx) - 1 - i):
            for elem2 in itertools.combinations(idx_temp, i + 1):
                neighbors.append(set(elem1) | set(elem2))
    return neighbors


def _filter_feasible_neighbors(
    g: nx.Graph,
    neighbors: list[set[int]],
    min_conn: int,
) -> list[set[int]]:
    return [seedset for seedset in neighbors if _meets_min_connectedness(g, seedset, min_conn)]


def _evaluate_neighbors(
    g: nx.Graph,
    thetas: Mapping[int, int] | np.ndarray,
    neighbors: list[set[int]],
    s_base: int,
    x_base: np.ndarray,
    n_nodes: int,
    xi_t: float,
    buffer: Deque[tuple[float, ...]],
    calls: int,
    stop: bool,
    history: list[list[float]],
    start: float,
    max_time: float,
    verbose: int,
    target_spread: int,
    early_break: bool = True,
) -> tuple[bool, np.ndarray, int, int, bool]:
    s_temp = s_base
    x_temp = x_base

    for elem in neighbors:
        x_elem = _x_from_seedset(n_nodes, elem)
        key = tuple(x_elem)

        if key in buffer:
            continue
        if time_exceeded(start, max_time):
            stop = True
            break

        s_elem = connected_component_spread(g, x_elem, thetas=thetas, max_t=1000)[0]
        buffer.append(key)
        calls += 1
        _print_nads_status(verbose, calls, start, max_time, s_elem, done=False)

        if s_elem > s_temp:
            history.append([s_elem, time.time() - start, calls])
            x_temp = x_elem
            s_temp = s_elem

            if s_elem == target_spread:
                return True, x_temp, s_temp, calls, stop
            if early_break and s_temp > (1 + xi_t) * s_base:
                break

    return False, x_temp, s_temp, calls, stop


def _make_move_connected_seedset(
    g: nx.Graph,
    seedset: set[int],
    v: int,
    k: int,
) -> set[int] | None:
    v = int(v)
    if v in seedset:
        return None

    try:
        lengths = nx.single_source_shortest_path_length(g, v)
    except Exception:
        return None

    candidates = [(lengths[u], u) for u in seedset if u in lengths]
    if not candidates:
        return None

    _, u_best = min(candidates, key=lambda t: t[0])

    try:
        path = nx.shortest_path(g, v, u_best)
    except nx.NetworkXNoPath:
        return None

    path_nodes = set(int(u) for u in path)
    if len(path_nodes) > k:
        return None

    s_new = set(path_nodes)
    remaining = list(seedset - s_new)

    while len(s_new) < k:
        added = False
        for u in list(remaining):
            for w in g.neighbors(u):
                if int(w) in s_new:
                    s_new.add(int(u))
                    remaining.remove(u)
                    added = True
                    break
            if added:
                break
        if not added:
            break

    if len(s_new) != k:
        return None

    sub = g.subgraph(s_new)
    if not nx.is_connected(sub):
        return None

    return s_new


def _mg_phase(
    g: nx.Graph,
    thetas: Mapping[int, int] | np.ndarray,
    idx: set[int],
    n_nodes: int,
    k: int,
    s_base: int,
    buffer: Deque[tuple[float, ...]],
    calls: int,
    history: list[list[float]],
    start: float,
    max_time: float,
    verbose: int,
    target_spread: int,
    min_conn: int,
    mg_max_depth: int,
    mg_memory_len: int,
) -> tuple[bool, bool, np.ndarray, int, int, bool]:
    improved_mg = False
    stop = False
    x_temp = _x_from_seedset(n_nodes, idx)
    s_temp = s_base

    h = min(max(1, int(mg_max_depth)), k, g.number_of_nodes() - 1)
    mg_memory = collections.deque(maxlen=max(1, int(mg_memory_len)))
    tried_add_nodes = set()

    for u_sel in list(idx):
        if time_exceeded(start, max_time):
            stop = True
            break
        visited = {int(u_sel)}
        frontier = {int(u_sel)}

        for radius in range(1, h + 1):
            if time_exceeded(start, max_time):
                stop = True
                break
            next_frontier = set()
            for u in frontier:
                for v in g.neighbors(u):
                    v = int(v)
                    if v not in visited:
                        next_frontier.add(v)

            if not next_frontier:
                break

            visited |= next_frontier
            frontier = next_frontier

            cand = [v for v in frontier if v not in idx]
            if not cand:
                mg_memory.append([])
                continue

            layer_entries = []
            for v in cand:
                mg_v = sum(1 for w in g.neighbors(v) if thetas[int(w)] <= radius)
                layer_entries.append((int(mg_v), int(v)))

            layer_entries.sort(key=lambda t: (t[0], -t[1]), reverse=True)
            mg_memory.append(layer_entries)

            best_current = None
            for mg_v, v in layer_entries:
                if v not in tried_add_nodes:
                    best_current = (mg_v, v)
                    break

            if best_current is None:
                continue

            best_overall = None
            for entries in mg_memory:
                for mg_v, v in entries:
                    if v in tried_add_nodes:
                        continue
                    if (
                        best_overall is None
                        or mg_v > best_overall[0]
                        or (mg_v == best_overall[0] and v < best_overall[1])
                    ):
                        best_overall = (mg_v, v)

            attempt_list = []
            if best_overall is not None:
                attempt_list.append(best_overall[1])
            if best_current is not None and best_current[1] not in attempt_list:
                attempt_list.append(best_current[1])

            for best_v in attempt_list:
                best_v = int(best_v)
                if best_v in tried_add_nodes:
                    continue
                tried_add_nodes.add(best_v)

                s_new = _make_move_connected_seedset(g, idx, best_v, k)
                if s_new is None:
                    continue
                if not _meets_min_connectedness(g, s_new, min_conn):
                    continue

                x_prov = _x_from_seedset(n_nodes, s_new)
                key = tuple(x_prov)

                if key in buffer:
                    continue
                if time_exceeded(start, max_time):
                    stop = True
                    break

                s_prov = connected_component_spread(g, x_prov, thetas=thetas, max_t=1000)[0]
                buffer.append(key)
                calls += 1
                _print_nads_status(verbose, calls, start, max_time, s_prov, done=False)

                if s_prov > s_temp:
                    history.append([s_prov, time.time() - start, calls])
                    x_temp = x_prov
                    s_temp = s_prov
                    improved_mg = True

                    if s_prov == target_spread:
                        return True, improved_mg, x_temp, s_temp, calls, stop
                    break

            if stop or improved_mg:
                break

        if stop or improved_mg:
            break

    return False, improved_mg, x_temp, s_temp, calls, stop


def NaDS_td(
    g: nx.Graph,
    thetas: Mapping[int, int] | np.ndarray,
    x0: np.ndarray,
    delta: float,
    xi: float,
    d: int,
    min_conn: int,
    mg_max_depth: int,
    mg_memory_len: int,
    max_time: float,
    buffer_dim: int,
    verbose: int = 0,
) -> tuple[list[int], list[np.ndarray], list[list[float]]]:
    n_nodes = len(x0)
    x_hist = [np.array(x0, dtype=float)]

    s_hist = [connected_component_spread(g, x0, thetas=thetas, max_t=1000)[0]]
    r = 0
    xi_t = xi
    target_spread = n_nodes

    k = int(np.count_nonzero(x0))

    stop = False
    buffer = collections.deque(maxlen=buffer_dim)
    calls = 1
    start = time.time()
    history = [[s_hist[-1], time.time() - start, calls]]

    _print_nads_status(verbose, calls, start, max_time, s_hist[-1], done=False)

    if s_hist[-1] == target_spread:
        _print_nads_status(verbose, calls, start, max_time, s_hist[-1], done=True)
        return s_hist, x_hist, history

    while (not stop) and (r < 1000):
        if time_exceeded(start, max_time):
            stop = True
            break

        idx = _seedset_from_x(x_hist[-1])

        found_opt, improved_mg, x_temp, s_temp, calls, mg_stop = _mg_phase(
            g,
            thetas,
            idx,
            n_nodes=n_nodes,
            k=k,
            s_base=s_hist[-1],
            buffer=buffer,
            calls=calls,
            history=history,
            start=start,
            max_time=max_time,
            verbose=verbose,
            target_spread=target_spread,
            min_conn=min_conn,
            mg_max_depth=mg_max_depth,
            mg_memory_len=mg_memory_len,
        )
        if mg_stop:
            stop = True
            break

        if found_opt:
            x_hist.append(x_temp)
            s_hist.append(s_temp)
            _print_nads_status(verbose, calls, start, max_time, s_hist[-1], done=True)
            return s_hist, x_hist, history

        if improved_mg:
            x_hist.append(x_temp)
            s_hist.append(s_temp)
            r += 1

            if s_hist[-1] > (1 + xi_t) * s_hist[-2]:
                continue
            if s_hist[-1] > s_hist[-2]:
                xi_t = xi_t * delta
            continue

        neighbors = _generate_swap_neighbors(g, idx)
        neighbors = _filter_feasible_neighbors(g, neighbors, min_conn=min_conn)
        found_opt, x_temp, s_temp, calls, stop = _evaluate_neighbors(
            g,
            thetas,
            neighbors,
            s_hist[-1],
            x_hist[-1],
            n_nodes=n_nodes,
            xi_t=xi_t,
            buffer=buffer,
            calls=calls,
            stop=stop,
            history=history,
            start=start,
            max_time=max_time,
            verbose=verbose,
            target_spread=target_spread,
            early_break=True,
        )

        if found_opt:
            x_hist.append(x_temp)
            s_hist.append(s_temp)
            _print_nads_status(verbose, calls, start, max_time, s_hist[-1], done=True)
            return s_hist, x_hist, history

        x_hist.append(x_temp)
        s_hist.append(s_temp)
        r += 1

        if s_hist[-1] > (1 + xi_t) * s_hist[-2]:
            continue
        if s_hist[-1] > s_hist[-2]:
            xi_t = xi_t * delta
            continue

        x_hist.pop()
        s_hist.pop()

        idx = _seedset_from_x(x_hist[-1])
        neighbors = _generate_d_exchange_neighbors(n_nodes, idx, d)
        neighbors = _filter_feasible_neighbors(g, neighbors, min_conn=min_conn)
        found_opt, x_temp, s_temp, calls, stop = _evaluate_neighbors(
            g,
            thetas,
            neighbors,
            s_hist[-1],
            x_hist[-1],
            n_nodes=n_nodes,
            xi_t=xi_t,
            buffer=buffer,
            calls=calls,
            stop=stop,
            history=history,
            start=start,
            max_time=max_time,
            verbose=verbose,
            target_spread=target_spread,
            early_break=True,
        )

        if found_opt:
            x_hist.append(x_temp)
            s_hist.append(s_temp)
            _print_nads_status(verbose, calls, start, max_time, s_hist[-1], done=True)
            return s_hist, x_hist, history

        x_hist.append(x_temp)
        s_hist.append(s_temp)
        r += 1

        if s_hist[-1] == s_hist[-2]:
            x_hist.pop()
            s_hist.pop()
            stop = True
        elif s_hist[-1] <= (1 + xi_t) * s_hist[-2]:
            xi_t = xi_t * delta

    history.append([s_hist[-1], time.time() - start, calls])
    _print_nads_status(verbose, calls, start, max_time, s_hist[-1], done=True)
    return s_hist, x_hist, history

def _print_binary_search_status(
    verbose: int,
    k: int,
    best_k: int | None,
    start: float,
    max_time: float,
    done: bool = False,
) -> None:
    if not verbose:
        return
    if done:
        print(
            "\r"
            + f"Binary search... Done! k:{best_k}. Time: {round(time.time()-start)}/{max_time} s. "
            + " " * 20
        )
    else:
        print(
            "\r"
            + f"Binary search... best k:{best_k}. Trying {k}. Time: {round(time.time()-start)}/{max_time} s. "
            + " " * 20,
            end="",
        )

def NaDS_technology_diffusion_binary_search(
    g: nx.Graph,
    thetas: Mapping[int, int] | np.ndarray,
    strategy: Sequence[Callable[..., np.ndarray]],
    delta: float,
    xi: float,
    d: int,
    min_conn: int,
    mg_max_depth: int,
    mg_memory_len: int,
    max_time: float,
    buffer_dim: int,
    verbose: int = 0,
) -> tuple[int | None, np.ndarray | None, float, list[tuple[int, float]]]:
    
    start = time.time()
    n_nodes = g.number_of_nodes()
    tried_k = set()
    best_k = None
    best_solution_x = None
    top_k, bottom_k = n_nodes, 1
    times = {}
    strategy_tried = {}
    temp_x = {}
    history = [(n_nodes, 0.0)]

    while bottom_k <= top_k and time.time() - start < max_time:
        k = (top_k + bottom_k) // 2
        if k in tried_k:
            break
        tried_k.add(k)

        remaining = max_time - (time.time() - start)
        time_single = remaining / max(1, math.ceil(math.log2(top_k - bottom_k + 1)))
        if remaining <= 0:
            break
        
        x = strategy[0](g, n_nodes, k, thetas=thetas, connected=1)

        _print_binary_search_status(verbose, k, best_k, start, max_time, done=False)
        s, final_x, history_nads = NaDS_td(g, thetas, x, delta, xi, d, min_conn, mg_max_depth, mg_memory_len, min(remaining, time_single), buffer_dim, 0)
        spread = s[-1]
        x_last = np.array(final_x[-1], dtype=float)
        times[k] = history_nads[-1][1]
        strategy_tried[k] = -1 if times[k] < 0.9 * time_single else 0
        if strategy_tried[k] == -1:
            temp_x[k] = x_last

        if spread == n_nodes:
            if best_k is None or k < best_k:
                best_k = k
                best_solution_x = x_last.copy()
                history.append((best_k, round(time.time() - start, 4)))
            top_k = k - 1
        else:
            inferred_success_k = k + (n_nodes - spread)
            if best_k is None or inferred_success_k < best_k:
                _, _, active_after = connected_component_spread(g, x_last, thetas, max_t=1000)
                inferred_x = x_last.copy()
                inferred_x[np.where(active_after == 0)[0]] = 1.0
                best_k = inferred_success_k
                best_solution_x = inferred_x
                history.append((best_k, round(time.time() - start, 4)))
            if inferred_success_k <= top_k:
                top_k = inferred_success_k - 1
            bottom_k = k + 1

    random.seed(42)
    while time.time() - start < max_time and best_k is not None and best_k > 1:
        k = best_k - 1
        if strategy_tried.get(k) is None:
            strat = 0
            x = strategy[strat](g, n_nodes, k, thetas=thetas, connected=1)
        elif strategy_tried[k] == -1:
            strat = 0
            x = temp_x[k]
        else:
            strat = min(len(strategy) - 1, strategy_tried[k] + 1)
            x = strategy[strat](g, n_nodes, k, thetas=thetas, connected=1)

        remaining = max_time - (time.time() - start)
        if remaining <= 0:
            break
        
        _print_binary_search_status(verbose, k, best_k, start, max_time, done=False)
        s, final_x, _ = NaDS_td(g, thetas, x, delta, xi, d, min_conn, mg_max_depth, mg_memory_len, min(remaining, time_single), buffer_dim, 0)
        strategy_tried[k] = strat

        if s[-1] == n_nodes:
            best_k = k
            best_solution_x = np.array(final_x[-1], dtype=float)
            history.append((best_k, round(time.time() - start, 4)))
        else:
            continue
    
    history.append((best_k, round(time.time() - start, 4)))
    _print_binary_search_status(verbose, k, best_k, start, max_time, done=True)

    return best_k, best_solution_x, round(float(time.time() - start), 4), history