from typing import Sequence

import numpy as np
import networkx as nx
import hypernetx as hnx


def gip_function(x: np.ndarray, l: float, h: float) -> np.ndarray:
    r = np.multiply(x >= l, x)
    return np.minimum(r, h)

def influence_spread_step(
    H: hnx.Hypergraph,
    I: np.ndarray,
    w: np.ndarray,
    x0: np.ndarray,
    l_t: float,
    h_t: float,
    ) -> np.ndarray:
    
    temp = I.T @ x0

    actives = temp > 0

    s = temp * w * actives

    x_next = I @ s

    x_next = gip_function(x_next, l_t, h_t)

    return x_next

def influence_spread(
    H: hnx.Hypergraph,
    I: np.ndarray,
    w: np.ndarray,
    x0: np.ndarray,
    params: Sequence[float],
    max_t: int = 999,
) -> list[np.ndarray]:
    
    h0, l0, theta_l, theta_h, gamma, eps = params
    x = [x0]
    spread = [np.sum(x0)]

    t = 1

    while np.linalg.norm(x[-1] * ((1 - gamma) ** t)) > eps and t <= max_t:

        l_t = ((theta_l * alpha) ** t) * l0
        h_t = (theta_h * (theta_l ** (t - 1)) * (alpha**t)) * h0

        x_next = influence_spread_step(H, I, w, x[-1], l_t, h_t)
        if np.all(x_next == x[-1]):
            break
        x.append(x_next)
        spread.append(spread[-1] + np.sum(x_next))

    return spread[-1] ,spread, x