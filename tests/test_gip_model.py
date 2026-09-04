"""Small independent references for the revised hypergraph model; no data runs."""
import itertools
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from src.gip_model import (
    GIPParameters, IncompleteEvaluationError, company_activity, gip, gip_bounds,
    gip_from_seeds, gip_raw_step, gip_incidence_raw_step, gip_source_raw_step,
    gip_step, gip_thresholds, prepare_incidence, seed_state, _nonnegative_difference,
)
from src.model_identity import MODEL_METADATA, evaluation_cache_key

PARAMS = GIPParameters(u0=1, l0=.1, theta_l=.5, theta_h=1, gamma=.5, eps=1e-8)
EDGE = sparse.csr_matrix([[1.0], [1.0]])


def explicit_raw(B, kappa, omega, q, tau):
    """Independent i != j definition: no diagonal subtraction or shared helper."""
    n, m = B.shape
    gates = [sum(B[i, e] * omega[i] * q[i] for i in range(n)) > tau for e in range(m)]
    return np.array([
        sum(B[j, e] * kappa[e] * gates[e]
            * sum(B[i, e] * omega[i] * q[i] for i in range(n) if i != j)
            for e in range(m))
        for j in range(n)
    ])


def effective_weights(B, kappa, omega, q, tau):
    """Small-case dense reference only; the FIRST index is the source."""
    n, m = B.shape
    gates = (B.T @ (omega * q)) > tau
    return np.array([[0.0 if i == j else omega[i] * sum(
        kappa[e] * B[i, e] * B[j, e] * gates[e] for e in range(m)
    ) for j in range(n)] for i in range(n)])


def finite_prefix(B, weights, seeds, p, omega, alpha, tau, steps):
    """Unit-test helper for common prefixes, never a production objective."""
    states = [seed_state(seeds, p)]
    scores = [float(omega @ states[0])]
    for s in range(1, steps+1):
        lower = (np.asarray(p.theta_l) * alpha)**s * np.asarray(p.l0)
        upper = np.asarray(p.theta_h) * np.asarray(p.theta_l)**(s-1) * alpha**s * np.asarray(p.u0)
        states.append(gip_step(B, weights, states[-1], lower, upper, tau, node_weights=omega))
        scores.append(scores[-1] + (1-p.gamma)**s * float(omega @ states[-1]))
    return np.asarray(states), np.asarray(scores)


class DynamicsTests(unittest.TestCase):
    def test_all_four_weighted_references_and_active_trajectories(self):
        rng = np.random.default_rng(814)
        # Overlaps, singleton, isolated municipality, unused company, zero omega.
        examples = [np.array([[2., 1, 0, 0, 0], [3, 0, 2, 0, 0],
                              [0, 1, 4, 5, 0], [0, 0, 0, 0, 0]])]
        for _ in range(8):
            B = rng.uniform(0, 3, size=(5, 7)); B[rng.random(B.shape) < .55] = 0
            examples.append(B)
        for B in examples:
            omega = rng.uniform(0, 3, len(B)); omega[0] = 0
            q = rng.uniform(0, 4, len(B)); q[1] = 0
            kappa = rng.uniform(0, 5, B.shape[1]); kappa[0] = 0
            for tau in (0, 1, 7, 100):
                expected = explicit_raw(B, kappa, omega, q, tau)
                W = effective_weights(B, kappa, omega, q, tau)
                assert_allclose(W.T @ q, expected, atol=1e-12)
                for representation in (B, sparse.csr_matrix(B), sparse.csc_matrix(B), sparse.csr_array(B)):
                    for evaluator in (gip_raw_step, gip_incidence_raw_step, gip_source_raw_step):
                        assert_allclose(evaluator(representation, kappa, q, tau, node_weights=omega), expected, atol=1e-12)
                    self.assertAlmostEqual(company_activity(representation, kappa, q, tau, node_weights=omega).sum(), expected.sum(), places=10)
                # Exercise the production evaluator, not only the one-step wrapper.
                seeds = rng.integers(0, 2, len(B))
                run = gip_from_seeds(B, kappa, seeds, PARAMS, node_weights=omega, stress_level=tau, alpha=.8)
                for t in range(1, len(run.states)):
                    lower, upper = gip_bounds(PARAMS, .8, t)
                    expected = gip_thresholds(explicit_raw(B, kappa, omega, run.states[t-1], tau), lower, upper)
                    assert_allclose(run.states[t], expected, atol=1e-12)

    def test_active_frontier_deactivation_reactivation_and_supported_persistence(self):
        network = prepare_incidence(EDGE)
        seen = []
        original = network._raw_frontier
        def record(weights, state, omega, stress, active, candidates):
            seen.append((tuple(active), tuple(candidates)))
            return original(weights, state, omega, stress, active, candidates)
        with patch.object(network, '_raw_frontier', side_effect=record):
            run = gip( network, [1], [1, 0], PARAMS, alpha=1)
        assert_array_equal(run.states[:4], [[1, 0], [0, 1], [.5, 0], [0, .25]])
        self.assertEqual(seen[:3], [((0,), (1,)), ((1,), (0,)), ((0,), (1,))])
        self.assertEqual(network.neighbors(0), (1,))
        self.assertIs(network.neighbors(0), network.neighbors(0))
        both = gip(network, [1], [1, 1], PARAMS, alpha=1)
        assert_array_equal(both.states[:2], [[1, 1], [1, 1]])

    def test_fresh_gates_after_frontier_moves(self):
        network = prepare_incidence([[1,0],[1,0],[0,1],[0,1]])
        assert_array_equal(gip_raw_step(network, [1,1], [1,0,0,0]), [0,1,0,0])
        assert_array_equal(gip_raw_step(network, [1,1], [0,0,1,0]), [0,0,0,1])
        assert_array_equal(gip_raw_step(network, [1,1], [0,0,0,0]), [0,0,0,0])
        # An old source incident to a singleton becomes an isolated next state.
        assert_array_equal(gip_from_seeds([[1]], [7], [1], PARAMS).final_state, [0])

    def test_shared_gate_strict_equality_and_kappa(self):
        for evaluator in (gip_raw_step, gip_incidence_raw_step, gip_source_raw_step):
            assert_array_equal(evaluator(EDGE, [1], [1,1], 1.5), [1,1])
            assert_array_equal(evaluator(EDGE, [1], [1,1], 2), [0,0])
            assert_array_equal(evaluator(EDGE, [1e12], [1,1], 2), [0,0])
            assert_array_equal(evaluator([[3]], [7], [5]), [0])

    def test_population_weighted_numerical_reference(self):
        B=np.array([[2.],[3.]])
        omega=np.array([5.,11.]); q=np.array([4.,5.])
        W=effective_weights(B, [7], omega, q, 0)
        assert_array_equal(W, [[0,210],[462,0]])
        for evaluator in (gip_raw_step, gip_incidence_raw_step, gip_source_raw_step):
            assert_array_equal(evaluator(B,[7],q,node_weights=omega),[2310,840])
        assert_array_equal(W.T @ q, [2310,840])

    def test_population_opens_gate_and_values_seed(self):
        assert_array_equal(gip_raw_step(EDGE,[1],[1,0],1.5,node_weights=[1,1]),[0,0])
        assert_array_equal(gip_raw_step(EDGE,[1],[1,0],1.5,node_weights=[2,1]),[0,2])
        closed=gip(EDGE,[1],[1,0],PARAMS,alpha=1,stress_level=1.5)
        opened=gip(EDGE,[1],[1,0],PARAMS,alpha=1,stress_level=1.5,node_weights=[2,1])
        self.assertEqual(closed.score,1)
        self.assertEqual(opened.spread_history[0],2)
        self.assertGreater(opened.score,2)
        self.assertFalse(np.array_equal(closed.states[1],opened.states[1]))

    def test_graph_reduction_ordinary_loopless_gip(self):
        B=np.array([[1,0,1,0],[1,1,0,1],[0,1,1,0],[0,0,0,1]],dtype=float)
        kappa=np.array([.6,1.1,.4,2])
        W=np.zeros((4,4))
        for e in range(B.shape[1]):
            i,j=np.flatnonzero(B[:,e]);W[i,j]+=kappa[e];W[j,i]+=kappa[e]
        p=GIPParameters(u0=[1,2,3,1],l0=[.1,.2,.1,.3],theta_l=.8,theta_h=1.2,gamma=.2,eps=1e-7)
        alpha=.9  # Deliberately identical for graph and hypergraph schedules.
        for seeds in ([1,0,0,0],[0,1,1,0],[1,1,1,1]):
            run=gip_from_seeds(B,kappa,seeds,p,alpha=alpha,node_weights=np.ones(4))
            states=[np.asarray(p.u0)*seeds];scores=[float(np.sum(states[0]))]
            t=0
            while np.linalg.norm((1-p.gamma)**t*states[-1]) > p.eps:
                t+=1
                raw=W.T @ states[-1]
                lo=(p.theta_l*alpha)**t*np.asarray(p.l0)
                hi=p.theta_h*p.theta_l**(t-1)*alpha**t*np.asarray(p.u0)
                states.append(np.array([0 if r<l else min(r,h) for r,l,h in zip(raw,lo,hi)]))
                scores.append(scores[-1]+(1-p.gamma)**t*float(np.sum(states[-1])))
            assert_allclose(run.states,states,atol=1e-12)
            assert_allclose(run.spread_history,scores,atol=1e-12)
            self.assertEqual(run.status,'tolerance_reached')

    def test_node_bounds_initialization_and_zero_theta(self):
        p=GIPParameters(u0=[1,2,3],l0=[.1,.2,.3],theta_l=[0,.5,2],theta_h=[1,2,3],eps=.1)
        assert_array_equal(seed_state([1,0,1],p,budget=2),[1,0,3])
        for s in (1,2,7):
            lo,hi=gip_bounds(p,.2,s)
            assert_allclose(lo,(np.array(p.theta_l)*.2)**s*p.l0)
            assert_allclose(hi,np.array(p.theta_h)*np.array(p.theta_l)**(s-1)*.2**s*p.u0)
        self.assertEqual(gip_bounds(p,.2,1)[1][0],.2)
        self.assertEqual(gip_bounds(p,.2,2)[1][0],0)
        assert_array_equal(gip_thresholds([.99,1,2,3],1,2),[0,1,2,2])
        assert_array_equal(gip_thresholds([1,2],[1,3],[2,4]),[1,0])
        lo,hi=gip_bounds(replace(PARAMS,theta_l=2,theta_h=50),.1,1100)
        self.assertTrue(np.isfinite([lo,hi]).all())

    def test_nodewise_frontier_uses_recipient_bounds(self):
        B = np.array([[1., 0], [1, 1], [0, 1]])
        p = GIPParameters(u0=[1, 2, 3], l0=[.1, .2, .3],
                          theta_l=[0, .5, 2], theta_h=[1, 2, 3], eps=1e-8)
        omega = np.array([1., 2, 0])
        run = gip_from_seeds(B, [1, .5], [1, 0, 1], p, alpha=.2, node_weights=omega)
        for step, state in enumerate(run.states[1:], start=1):
            raw = explicit_raw(B, [1, .5], omega, run.states[step-1], 0)
            lower = (np.array(p.theta_l)*.2)**step * p.l0
            upper = np.array(p.theta_h)*np.array(p.theta_l)**(step-1)*.2**step*p.u0
            expected = np.array([0 if r < l else min(r, h) for r, l, h in zip(raw, lower, upper)])
            assert_allclose(state, expected, atol=1e-15)

    def test_prepared_sparse_snapshot_duplicates_and_large_node_count(self):
        B=sparse.coo_matrix(([1.,1.,3.],([0,0,1],[0,0,0])),shape=(2,1))
        prepared=prepare_incidence(B)
        self.assertIs(prepare_incidence(prepared),prepared)
        self.assertTrue(sparse.issparse(prepared._squared))
        assert_allclose(prepared._squared.data,[4,9])
        B.data[:]=0
        assert_array_equal(gip_raw_step(prepared,[7],[4,5],node_weights=[5,11]),[2310,840])
        with self.assertRaises(ValueError): prepared._matrix.data[0]=5
        B=sparse.csr_matrix(([1.,1.],([0,99999],[0,0])),shape=(100000,1))
        state=np.zeros(100000);state[0]=1
        with patch.object(sparse.csr_matrix,'toarray',side_effect=AssertionError('no dense expansion')):
            raw=gip_raw_step(B,[1],state)
        self.assertEqual(np.count_nonzero(raw),1)
        self.assertEqual(raw[-1],1)

    def test_scale_aware_roundoff_and_invalid_arithmetic(self):
        total=np.array([1e-24,1e24]);own=np.nextafter(total,np.inf)
        assert_array_equal(_nonnegative_difference(total,own,8),[0,0])
        for scale in (1e-24,1,1e24):
            with self.assertRaisesRegex(FloatingPointError,'Materially negative'):
                _nonnegative_difference(np.array([scale]),np.array([scale*1.01]),8)
        with self.assertRaises(FloatingPointError):
            _nonnegative_difference(np.array([np.inf]),np.array([0]),8)
        with np.errstate(over='ignore'), self.assertRaises(FloatingPointError):
            gip_raw_step(EDGE,[1],[1e300,1e300],node_weights=[1e300,1e300])


class ScoringTests(unittest.TestCase):
    def test_infinite_geometric_score_converges(self):
        exact=1/(1-.36);errors=[]
        for eps in (1e-2,1e-5,1e-10):
            p=GIPParameters(u0=1,l0=.1,theta_l=1,theta_h=2,gamma=.1,eps=eps)
            run=gip(EDGE,[.4],[1,0],p,alpha=.4)
            errors.append(abs(run.score-exact))
            self.assertEqual(run.status,'tolerance_reached')
            for t,q in enumerate(run.states):
                expected=np.zeros(2);expected[t%2]=.4**t
                assert_allclose(q,expected,atol=1e-15)
        self.assertTrue(all(a>b for a,b in zip(errors,errors[1:])))
        self.assertLess(errors[-1],1e-10)

    def test_tolerance_current_exponent_seed_once_and_cutoff_boundary(self):
        p=replace(PARAMS,eps=.75)
        run=gip(EDGE,[1],[1,0],p,alpha=1,max_iter=1)
        self.assertEqual(run.steps,1)
        self.assertEqual(run.score,1.5)
        self.assertEqual(run.status,'tolerance_reached')
        self.assertIs(run.require_complete(),run)
        run=gip(EDGE,[1],[1,0],replace(p,eps=1),alpha=1,max_iter=0)
        self.assertEqual(run.steps,0)
        self.assertEqual(run.score,1)
        self.assertEqual(run.status,'tolerance_reached')
        assert_array_equal(run.final_state,[1,0])
        weighted=gip(EDGE,[1],[1,0],replace(p,eps=1),node_weights=[7,0],alpha=1)
        self.assertEqual(weighted.score,7)

    def test_equal_prefix_changes_later_and_operational_cutoff(self):
        run=gip(EDGE,[1],[1,1],PARAMS,alpha=1)
        assert_array_equal(run.states[:4],[[1,1],[1,1],[.5,.5],[.25,.25]])
        self.assertGreater(run.steps,3)
        self.assertEqual(run.status,'tolerance_reached')
        for guard in (0,1,3):
            incomplete=gip(EDGE,[1],[1,1],PARAMS,alpha=1,max_iter=guard)
            self.assertEqual(incomplete.steps,guard)
            self.assertEqual(incomplete.status,'operational_cutoff')
            assert_array_equal(incomplete.states,run.states[:guard+1])
            with self.assertRaisesRegex(IncompleteEvaluationError,'operational_cutoff'):
                incomplete.require_complete()

    def test_gamma_same_prefix_different_stop_and_score(self):
        runs=[gip(EDGE,[1],[1,0],replace(PARAMS,gamma=g,eps=1e-5),alpha=1) for g in (.1,.5,.9)]
        common=min(len(run.states) for run in runs)
        for run in runs[1:]: assert_array_equal(run.states[:common],runs[0].states[:common])
        self.assertEqual(len({run.steps for run in runs}),3)
        self.assertEqual(len({run.score for run in runs}),3)

    def test_zero_seed_empty_edges_and_zero_scale(self):
        zero=gip(EDGE,[1],[0,0],PARAMS)
        self.assertEqual((zero.steps,zero.score),(0,0))
        empty=gip(sparse.csr_matrix((2,0)),[],[1,0],PARAMS)
        assert_array_equal(empty.states,[[1,0],[0,0]])
        self.assertEqual(empty.score,1)
        for alpha in (0,.2):
            run=gip(EDGE,[1],[1,0],replace(PARAMS,theta_l=0),alpha=alpha)
            self.assertEqual(run.final_state.sum(),0)
            self.assertLessEqual(run.steps,2)

    def test_exhaustive_one_step_order_and_partial_sum_order(self):
        B=np.array([[1.,.5,0],[1,0,1],[1,2,0],[0,1,1]])
        omega=np.array([1,2,0,3]);weights=[1,.5,2]
        seeds=[np.array(s) for s in itertools.product((0.,1.),repeat=4)]
        p=GIPParameters(u0=[1,2,1,3],l0=.2,theta_l=.7,theta_h=2,gamma=.3)
        for tau in (0,1,2.5):
            runs=[finite_prefix(B,weights,s,p,omega,.8,tau,4) for s in seeds]
            for i,s in enumerate(seeds):
                for j,t in enumerate(seeds):
                    if np.all(s<=t):
                        self.assertTrue(np.all(runs[i][0] <= runs[j][0]+1e-12))
                        self.assertTrue(np.all(runs[i][1] <= runs[j][1]+1e-12))
            states=[np.array(s) for s in itertools.product((0.,.5,1.),repeat=4)]
            raws=[gip_raw_step(B,weights,q,tau,node_weights=omega) for q in states]
            for i,q in enumerate(states):
                for j,r in enumerate(states):
                    if np.all(q<=r): self.assertTrue(np.all(raws[i]<=raws[j]+1e-12))


class ValidationAndIdentityTests(unittest.TestCase):
    def test_invalid_shapes_bounds_seeds_and_parameters(self):
        for kwargs in ({'max_iter':-1},{'max_iter':1.5},{'max_iter':True},{'stress_level':-1},
                       {'alpha':2},{'alpha':np.inf},{'alpha':-1},{'node_weights':[1]},
                       {'node_weights':[1,-1]},{'node_weights':[1,np.nan]},{'node_weights':[[1,1]]}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                gip(EDGE,[1],[1,0],PARAMS,**kwargs)
        for name in ('h0','l0','theta_l','theta_h','gamma','eps'):
            for value in (-1,np.nan,np.inf):
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                    gip(EDGE,[1],[1,0],replace(PARAMS,**{name:value}))
        for kwargs in ({'l0':0},{'h0':0},{'l0':2},{'eps':0},{'gamma':0},{'gamma':1},
                       {'theta_h':0},{'l0':[.1,.1,.1]},{'h0':[[1,1]]}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                gip(EDGE,[1],[1,0],replace(PARAMS,**kwargs))
        for B,weights,q in (([[-1],[1]],[1],[1,0]),([[np.inf],[1]],[1],[1,0]),
                            (EDGE,[-1],[1,0]),(EDGE,[1,2],[1,0]),(EDGE,[1],[[1,0]]),
                            (EDGE,[1],[2,0]),(EDGE,[1],[1,-1]),([1,1],[1],[1,0])):
            with self.assertRaises(ValueError):gip(B,weights,q,PARAMS)
        for seeds in ([.5,0],[1,2],[1,np.nan],[[1,0]]):
            with self.assertRaises(ValueError):gip_from_seeds(EDGE,[1],seeds,PARAMS)
        with self.assertRaises(ValueError):gip_from_seeds(EDGE,[1],[1,0],PARAMS,budget=2)
        with self.assertRaises(ValueError):GIPParameters(h0=1,u0=2)
        with self.assertRaises(TypeError):gip(EDGE,[1],[1,0],PARAMS,horizon=2)
        with self.assertRaises(ValueError):gip_bounds(PARAMS,1,0)
        with self.assertRaises(ValueError):gip_step(EDGE,[1],[1,0],2,1)
        with self.assertRaises(ValueError):gip(sparse.csr_matrix((2,0)),[],[1,0],PARAMS,alpha=None)

    def test_cache_identity_covers_complete_problem_and_model_version(self):
        def key(B=EDGE,weights=(1,),q=(1,0),p=PARAMS,**kwargs):
            return evaluation_cache_key(B,weights,q,p,**kwargs)
        base=key()
        alternatives=[key(q=[0,1]),key(B=[[2],[1]]),key(weights=[2]),key(node_weights=[1,2]),
                      key(stress_level=1),key(alpha=.2),key(alpha=None),key(max_iter=0),key(max_iter=10),
                      key(p=replace(PARAMS,h0=[1,2])),key(p=replace(PARAMS,h0=2),q=[2,0])]
        alternatives += [key(p=replace(PARAMS,**{name:value})) for name,value in
                         (('gamma',.2),('eps',.1),('l0',.2),('theta_l',.9),('theta_h',2))]
        self.assertTrue(all(other!=base for other in alternatives))
        self.assertEqual(key(B=EDGE.toarray()),base)
        self.assertEqual(key(node_weights=[1,1]),base)
        self.assertEqual(key(alpha=1),key(alpha=1.0))
        self.assertEqual(key(p=replace(PARAMS,h0=[1.,1.])),base)
        for field in MODEL_METADATA:
            with patch.dict(MODEL_METADATA,{field:'old-model-value'}):self.assertNotEqual(key(),base)


if __name__=='__main__':unittest.main()
