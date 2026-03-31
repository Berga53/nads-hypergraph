from typing import Sequence

import numpy as np
import networkx as nx
import hypernetx as hnx


def gip_function(x: np.ndarray, h: float, l: float) -> np.ndarray:
   r = np.multiply(x >= l, x)
   return np.minimum(r, h)


# def _average_edge_weight(g: nx.Graph) -> float:
#   alpha = 0.0
#   for _, _, edge_data in g.edges(data=True):
#     alpha += edge_data["weight"]
#   return alpha / len(g.edges)

def influence_spread_step(
    H: hnx.Hypergraph,
    I: np.ndarray,
    A: np.ndarray,
    x: np.ndarray,
    ) -> np.ndarray:
    

  return 


def influence_evaluation(
  g: hnx.Hypergraph,
  W: np.ndarray,
  x0: np.ndarray,
  params: Sequence[float],
  alpha: float | None = None,
  max_t: int = 999,
) -> tuple[float, list[float], list[np.ndarray]]:
  l0, h0, theta_l, theta_h, gamma, eps = params

  spread = 0.0
  spread_hist = [0.0]
  x_hist = [x0]
  W_arr = np.asarray(W, dtype=float)

  t = 1

  while np.linalg.norm(x_hist[-1] * ((1 - gamma) ** t)) > eps and t <= max_t:
    l_t = ((theta_l * alpha) ** t) * l0
    h_t = (theta_h * (theta_l ** (t - 1)) * (alpha**t)) * h0

    weighted_input = W_arr @ np.asarray(x_hist[-1], dtype=float)
    x_t = gip_function(weighted_input, h_t, l_t)
    spread += float(np.sum(x_t))

    x_hist.append(x_t)
    spread_hist.append(spread)
    t += 1

  return spread, spread_hist, x_hist