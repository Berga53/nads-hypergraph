import itertools
import unittest
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from src.gip_model import (
    GIPParameters, PreparedIncidence, company_activity, gip, gip_raw_step,
    gip_step, gip_thresholds, prepare_incidence, _nonnegative_difference,
)
from src.model_identity import evaluation_cache_key


def reference_raw(B, kappa, x, tau):
    """Independent componentwise sum; deliberately never subtract a diagonal."""
    n, m = B.shape
    return np.array([
        sum(B[v, e] * kappa[e]
            * (sum(B[u, e] * x[u] for u in range(n)) > tau)
            * sum(B[u, e] * x[u] for u in range(n) if u != v)
            for e in range(m))
        for v in range(n)
    ])


PARAMS = GIPParameters(h0=100, l0=0, theta_l=1, theta_h=1, gamma=0.5, eps=0)
EDGE = sparse.csr_matrix([[1.0], [1.0]])


class DynamicsTests(unittest.TestCase):
    def test_sparse_against_explicit_sum(self):
        rng = np.random.default_rng(814)
        examples = [np.array([[2., 1, 0, 0], [3, 0, 2, 0], [0, 1, 4, 0], [0, 0, 0, 0]])]
        for _ in range(15):
            B = rng.uniform(0, 3, size=(5, 7))
            B[rng.random(B.shape) < .55] = 0
            examples.append(B)
        for B in examples:
            x = rng.uniform(0, 4, len(B))
            kappa = rng.uniform(0, 5, B.shape[1]); kappa[0] = 0
            for tau in (0, 1, 7, 100):
                expected = reference_raw(B, kappa, x, tau)
                for representation in (B, sparse.csr_matrix(B), sparse.csc_matrix(B), sparse.csr_array(B)):
                    with self.subTest(shape=B.shape, tau=tau, kind=type(representation)):
                        assert_allclose(gip_raw_step(representation, kappa, x, tau), expected, atol=1e-12)
                        self.assertAlmostEqual(company_activity(representation, kappa, x, tau).sum(), expected.sum(), places=10)

    def test_singleton_sender_and_later_return(self):
        assert_array_equal(gip_raw_step([[3]], [7], [5]), [0])
        run = gip(EDGE, [1], [1, 0], PARAMS, alpha=1, horizon=2)
        assert_array_equal(run.states, [[1, 0], [0, 1], [1, 0]])

    def test_shared_gate_strict_equality_and_kappa(self):
        assert_array_equal(gip_raw_step(EDGE, [1], [1, 1], 1.5), [1, 1])
        assert_array_equal(gip_raw_step(EDGE, [1], [1, 1], 2), [0, 0])
        assert_array_equal(gip_raw_step(EDGE, [1e12], [1, 1], 2), [0, 0])
        assert_array_equal(gip_raw_step(EDGE, [0], [1, 1], 1.5), [0, 0])

    def test_weighted_reference(self):
        assert_array_equal(gip_raw_step(sparse.csr_matrix([[2], [3]]), [7], [4, 5]), [210, 168])

    def test_graph_reduction_full_trajectory_and_score(self):
        B = np.array([[1, 0, 1, 0], [1, 1, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=float)
        kappa = np.array([.6, 1.1, .4, 2])
        adjacency = np.zeros((4, 4))
        for e in range(B.shape[1]):
            u, v = np.flatnonzero(B[:, e]); adjacency[u, v] += kappa[e]; adjacency[v, u] += kappa[e]
        p = replace(PARAMS, h0=3, l0=.3, theta_l=.8, theta_h=1.2, gamma=.2)
        omega = np.array([2, .5, 3, 1]); alpha = .9
        for seed in ([1, 0, 0, 0], [0, 2, 1, 0], [1, 1, 1, 1]):
            run = gip(sparse.csr_matrix(B), kappa, seed, p, node_weights=omega, alpha=alpha, horizon=8)
            states = [np.array(seed, dtype=float)]; scores = [float(omega @ states[0])]
            for j in range(1, 9):
                raw = adjacency @ states[-1]
                lower = (p.theta_l * alpha)**j * p.l0
                upper = p.theta_h * p.theta_l**(j-1) * alpha**j * p.h0
                states.append(np.array([0 if z < lower else min(z, upper) for z in raw]))
                scores.append(scores[-1] + (1-p.gamma)**j * float(omega @ states[-1]))
            assert_allclose(run.states, states, atol=1e-12)
            assert_allclose(run.spread_history, scores, atol=1e-12)

    def test_threshold_boundaries(self):
        assert_array_equal(gip_thresholds([.99, 1, 2, 3], 1, 2), [0, 1, 2, 2])

    def test_prepared_snapshot_and_sparse_duplicates(self):
        B = sparse.coo_matrix(([1., 1., 3.], ([0, 0, 1], [0, 0, 0])), shape=(2, 1))
        original = B.data.copy(); prepared = prepare_incidence(B)
        self.assertIs(prepare_incidence(prepared), prepared)
        self.assertTrue(sparse.issparse(prepared._squared))
        assert_allclose(prepared._squared.data, [4, 9])
        assert_array_equal(B.data, original)
        B.data[:] = 0
        assert_array_equal(gip_raw_step(prepared, [7], [4, 5]), [210, 168])
        with self.assertRaises(ValueError):
            prepared._matrix.data[0] = 5

    def test_no_dense_network_expansion(self):
        # A large node count but only one edge: n-by-n allocation would be huge.
        B = sparse.csr_matrix(([1., 1.], ([0, 99999], [0, 0])), shape=(100000, 1))
        x = np.zeros(100000); x[0] = 1
        raw = gip_raw_step(B, [1], x)
        self.assertEqual(np.count_nonzero(raw), 1)
        self.assertEqual(raw[-1], 1)

    def test_scale_aware_roundoff(self):
        total = np.array([1e-24, 1e24])
        own = np.nextafter(total, np.inf)
        assert_array_equal(_nonnegative_difference(total, own, 8), [0, 0])
        for scale in (1e-24, 1, 1e24):
            with self.assertRaisesRegex(FloatingPointError, 'Materially negative'):
                _nonnegative_difference(np.array([scale]), np.array([scale*1.01]), 8)
        with self.assertRaises(FloatingPointError):
            _nonnegative_difference(np.array([np.inf]), np.array([0]), 8)


class ScoringTests(unittest.TestCase):
    def test_discounted_seed_scoring_and_zero_horizon(self):
        for horizon, score in ((0, 2), (2, 4)):
            run = gip(EDGE, [1], [1, 0], PARAMS, node_weights=[2, 3], alpha=1, horizon=horizon)
            self.assertEqual(run.total_spread, score)
            self.assertEqual(len(run.states), horizon+1)
            self.assertEqual(run.iterations, horizon)
            self.assertEqual(run.stopping_reason, 'fixed_horizon')

    def test_early_tolerance_index(self):
        p = replace(PARAMS, eps=.75)
        run = gip(EDGE, [1], [1, 0], p, alpha=1, max_iter=30)
        self.assertEqual(run.iterations, 1)
        self.assertEqual(run.total_spread, 1.5)
        self.assertEqual(run.stopping_reason, 'tolerance')
        run = gip(EDGE, [1], [1, 0], replace(p, eps=1), alpha=1)
        self.assertEqual(run.iterations, 0)
        self.assertEqual(run.total_spread, 1)

    def test_equal_state_regression_fixed_and_early(self):
        for options in ({'horizon': 3}, {'max_iter': 3}):
            run = gip(EDGE, [1], [1, 1], PARAMS, alpha=1, **options)
            self.assertEqual(run.total_spread, 3.75)
            self.assertEqual(run.iterations, 3)
        self.assertEqual(run.stopping_reason, 'max_iter')

    def test_equal_pair_then_changing_bound(self):
        p = replace(PARAMS, h0=1, theta_l=.5)
        for options in ({'horizon': 3}, {'max_iter': 3}):
            run = gip(EDGE, [1], [1, 1], p, alpha=1, **options)
            assert_array_equal(run.states, [[1, 1], [1, 1], [.5, .5], [.25, .25]])

    def test_fixed_horizon_ignores_tolerance_and_early_limit(self):
        run = gip(EDGE, [1], [1, 0], replace(PARAMS, eps=1e10), horizon=5, max_iter=1, alpha=1)
        self.assertEqual(run.iterations, 5)
        zero = gip(EDGE, [1], [0, 0], PARAMS, horizon=4)
        self.assertEqual(len(zero.states), 5)
        self.assertEqual(zero.total_spread, 0)

    def test_omega_gamma_only_change_fixed_horizon_scores(self):
        baseline = gip(EDGE, [1], [1, 0], PARAMS, horizon=6, alpha=1)
        for omega, gamma in (([3, 9], .5), ([1, 1], .1), ([3, 9], .9)):
            run = gip(EDGE, [1], [1, 0], replace(PARAMS, gamma=gamma), node_weights=omega, horizon=6, alpha=1)
            assert_array_equal(run.states, baseline.states)
            self.assertNotEqual(run.total_spread, baseline.total_spread)

    def test_exhaustive_seed_monotonicity_at_common_horizon(self):
        B = sparse.csr_matrix([[1., .5, 0], [1, 0, 1], [1, 2, 0], [0, 1, 1]])
        seeds = [np.array(s) for s in itertools.product((0., 1.), repeat=4)]
        p = replace(PARAMS, l0=.2, h0=2, theta_l=.7, gamma=.3)
        for tau in (0, 1, 2.5):
            runs = [gip(B, [1, .5, 2], s, p, node_weights=[1, 2, .1, 3], horizon=4, alpha=.8, stress_level=tau) for s in seeds]
            for i, s in enumerate(seeds):
                for j, t in enumerate(seeds):
                    if np.all(s <= t):
                        self.assertLessEqual(runs[i].total_spread, runs[j].total_spread + 1e-12)
                        self.assertTrue(np.all(np.array(runs[i].states) <= np.array(runs[j].states) + 1e-12))

    def test_long_decaying_bounds_remain_finite(self):
        p = replace(PARAMS, theta_l=2, theta_h=50, h0=1)
        run = gip(EDGE, [1], [1, 1], p, alpha=.1, horizon=1100)
        self.assertEqual(run.iterations, 1100)
        self.assertTrue(np.isfinite(run.total_spread))

    def test_empty_edges(self):
        run = gip(sparse.csr_matrix((2, 0)), [], [1, 0], PARAMS, horizon=2)
        assert_array_equal(run.states, [[1, 0], [0, 0], [0, 0]])


class ValidationAndIdentityTests(unittest.TestCase):
    def test_invalid_inputs(self):
        for kwargs in ({'horizon': -1}, {'horizon': 1.5}, {'horizon': True}, {'max_iter': 0},
                       {'max_iter': np.inf}, {'stress_level': np.nan}, {'stress_level': -1},
                       {'alpha': np.inf}, {'alpha': -1}, {'node_weights': [1]},
                       {'node_weights': [1, -1]}, {'node_weights': [1, np.nan]},
                       {'node_weights': [[1, 1]]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gip(EDGE, [1], [1, 0], PARAMS, **kwargs)
        for name in ('h0', 'l0', 'theta_l', 'theta_h', 'gamma', 'eps'):
            for value in (-1, np.nan, np.inf):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    gip(EDGE, [1], [1, 0], replace(PARAMS, **{name: value}))
        for gamma in (0, 1):
            with self.assertRaises(ValueError): gip(EDGE, [1], [1, 0], replace(PARAMS, gamma=gamma))
        for B, weights, state in (([[-1], [1]], [1], [1, 0]), ([[np.inf], [1]], [1], [1, 0]),
                                  (EDGE, [-1], [1, 0]), (EDGE, [np.nan], [1, 0]),
                                  (EDGE, [1, 2], [1, 0]), (EDGE, [1], [[1, 0]]),
                                  (EDGE, [1], [1, -1]), ([1, 1], [1], [1, 0])):
            with self.assertRaises(ValueError): gip(B, weights, state, PARAMS)
        with self.assertRaises(ValueError):
            gip(EDGE, [1], [1, 0], replace(PARAMS, h0=1, l0=2), alpha=1, horizon=0)
        with self.assertRaises(ValueError): gip_step(EDGE, [1], [1, 0], 2, 1)

    def test_cache_identity_covers_entire_problem(self):
        def key(B=EDGE, weights=(1,), x=(1, 0), p=PARAMS, **kwargs):
            return evaluation_cache_key(B, weights, x, p, **kwargs)
        base = key()
        alternatives = [key(x=[0, 1]), key(x=[2, 0]), key(B=[[2], [1]]), key(weights=[2]),
                        key(node_weights=[1, 2]), key(stress_level=1), key(alpha=.2),
                        key(alpha=None), key(horizon=0), key(horizon=2), key(max_iter=10),
                        key(initial_state_convention='x0=a0*seed_indicator')]
        alternatives += [key(p=replace(PARAMS, **{name: value})) for name, value in
                         (('gamma', .2), ('eps', .1), ('h0', 101), ('l0', .1), ('theta_l', .9), ('theta_h', 2))]
        self.assertTrue(all(other != base for other in alternatives))
        self.assertEqual(key(B=EDGE.toarray()), base)
        self.assertEqual(key(node_weights=[1, 1]), base)
        self.assertEqual(key(alpha=1), key(alpha=1.0))
        self.assertEqual(key(p=replace(PARAMS, h0=100.0)), base)
        with self.assertRaises(ValueError):
            key(horizon=1.5)


if __name__ == '__main__':
    unittest.main()
