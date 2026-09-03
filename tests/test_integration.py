"""Tiny synthetic plumbing checks; never execute a municipal-data experiment."""
import contextlib
import io
import json
import os
import tempfile
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import hypernetx as hnx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from scripts import run_experiment as grid
from src.gip_model import (
    GIPParameters, aligned_edge_parameters, aligned_node_weights,
    company_activity, gip, prepare_incidence,
)
from src.model_identity import evaluation_cache_key, input_fingerprint
from src.nads import nads


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data' / 'processed'; self.data.mkdir(parents=True)
        pd.DataFrame({'CF Comune': [10, 20, 20, 30], 'CF Partecipata': [100, 100, 200, 200],
                      'Quota': [1., 1., 1., 1.]}).to_csv(self.data / 'rete_2023.csv', index=False)
        pd.DataFrame({
            'year': [2023]*2, 'company_id': [100, 200], 'company_name': ['A', 'B'],
            'company_size_score_0_100': [50, 50], 'company_size_data_confidence_0_1': [1, 1],
            'company_size_data_quality': ['complete']*2, 'edge_parameter_base_0_1': [.5, .5],
            'edge_parameter_basis': ['observed']*2,
        }).to_csv(self.data / 'hyperedge_parameters.csv', index=False)
        pd.DataFrame({
            'year': [2023]*3, 'municipality_id': [10, 20, 30], 'municipality_name': ['X','Y','Z'],
            'municipality_tax_code': ['00010','00020','00030'], 'municipality_bdap_id': ['010','020','030'],
            'population': [10,20,30], 'population_data_available': [True]*3,
            'node_weight_population': [10,20,30], 'node_weight_basis': ['observed']*3,
        }).to_csv(self.data / 'node_parameters.csv', index=False)
        self.config = grid.ExperimentConfig(seed_budget=1, horizon=3, search_seconds_per_year=.04,
                                               max_neighbors_per_phase=10, max_search_iterations=4,
                                               alpha=1, l0=0, h0=100, theta_l=1, theta_h=1, gamma=.5)

    def run_year(self, config=None):
        with patch.object(grid, 'DATA_DIR', self.data):
            return grid.run_year(2023, config or self.config, verbose=False)

    def test_runner_notebook_ranking_and_ablation_share_model(self):
        for horizon in (0, 3, None):
            config = replace(self.config, horizon=horizon, max_iter=4)
            summary, selected, seeds = self.run_year(config)
            self.assertNotIn('model_version', summary)
            self.assertNotIn('input_fingerprint', summary)
            self.assertEqual(len(seeds), config.seed_budget)
            self.assertEqual(summary['diffusion_iterations'], horizon if horizon is not None else 4)
            self.assertEqual(summary['stopping_reason'], 'fixed_horizon' if horizon is not None else 'max_iter')
            self.assertGreaterEqual(summary['best_spread'], summary['initial_spread'])
            notebook = json.loads(Path('notebooks/results.ipynb').read_text())
            env = dict(PROJECT_ROOT=self.root, np=np, pd=pd, hnx=hnx, GIPParameters=GIPParameters,
                       ExperimentConfig=grid.ExperimentConfig, load_year_model=grid.load_year_model,
                       aligned_edge_parameters=aligned_edge_parameters, aligned_node_weights=aligned_node_weights,
                       gip=gip, prepare_incidence=prepare_incidence, company_activity=company_activity,
                       evaluation_cache_key=evaluation_cache_key,
                       input_fingerprint=input_fingerprint)
            exec(''.join(notebook['cells'][19]['source']), env)
            row = pd.Series(summary)
            rebuilt = env['propagation_for_row'](0, row)
            self.assertAlmostEqual(rebuilt['propagation'].total_spread, summary['best_spread'])
            # With one seed, the individual ranking uses exactly the same objective.
            self.assertAlmostEqual(selected['individual_weighted_spread'].iloc[0], summary['best_spread'])
            self.assertIs(env['propagation_for_row'](999, row), rebuilt)
            changed = row.copy(); changed['gamma'] = .25
            recomputed = env['propagation_for_row'](0, changed)
            self.assertIsNot(recomputed, rebuilt)
            self.assertEqual(recomputed['parameters'].gamma, .25)
            weights = rebuilt['model']['edge_weights'].copy(); weights[0] = 0
            ablated = gip(rebuilt['model']['incidence'], weights, rebuilt['seed_state'], rebuilt['parameters'],
                          node_weights=rebuilt['model']['node_weights'], alpha=config.alpha,
                          horizon=rebuilt['horizon'], max_iter=rebuilt['max_iter'])
            self.assertEqual(ablated.horizon, config.horizon)
            if horizon is not None:
                self.assertLessEqual(ablated.total_spread, summary['best_spread'] + 1e-12)
            with (self.data / 'rete_2023.csv').open('a') as stream: stream.write('\n')
            reloaded = env['propagation_for_row'](0, row)
            self.assertIsNot(reloaded, rebuilt)
            self.assertAlmostEqual(reloaded['propagation'].total_spread, rebuilt['propagation'].total_spread)

    def test_checkpoint_identity_and_preservation(self):
        summary, _, _ = self.run_year()
        row = {'year': 2023, **grid.parameter_values(self.config),
               'start_spread': summary['initial_spread'], 'finish_spread': summary['best_spread'],
               'start_list': json.dumps(summary['start_list']), 'finish_list': json.dumps(summary['finish_list']),
               **{k: summary[k] for k in ('stopping_reason','diffusion_iterations')}}
        row['alpha'] = .12345678912345678  # Must survive CSV exact-identity lookup.
        results = pd.DataFrame([row]); results['user_note'] = 'preserve me'
        self.assertTrue(grid.instance_mask(results, row).all())
        for field, value in (('gamma',.2),
                             ('horizon',0), ('max_iter',10), ('seed_intensity',2), ('alpha',1+1e-13)):
            changed = {**row, field: value}
            self.assertFalse(grid.instance_mask(results, changed).any())
        retained = grid.upsert_result(results, {**row, 'finish_spread': -99})
        self.assertEqual(retained['finish_spread'].iloc[0], row['finish_spread'])
        path = self.root / 'results.csv'; grid.save_results(path, results)
        loaded = grid.load_results(path)
        self.assertTrue(grid.instance_mask(loaded, row).all())
        self.assertEqual(loaded['user_note'].iloc[0], 'preserve me')
        self.assertFalse({'model_version','input_fingerprint','start_score_key','finish_score_key'} & set(loaded.columns))
        early = {**row, 'horizon': None}
        combined = grid.upsert_result(loaded, early)
        grid.save_results(path, combined)
        self.assertEqual(grid.instance_mask(grid.load_results(path), early).sum(), 1)
        legacy = self.root / 'legacy.csv'
        legacy.write_text('year,start_spread,finish_spread\n2023,1,2\n')
        before = legacy.read_bytes()
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            grid.load_results(legacy)
        self.assertEqual(legacy.read_bytes(), before)

    def test_grid_main_checkpoint_resume_on_synthetic_data(self):
        output = self.root / 'grid'
        args = ['run_experiment', '--years','2023','--alphas','.1','--seed-budgets','1',
                '--search-seconds','.04','--horizon','2','--output-dir',str(output)]
        with patch.object(grid, 'DATA_DIR', self.data), \
             patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
            grid.main()
            before = (output / 'results.csv').read_bytes()
            with patch.object(grid, 'run_year', side_effect=AssertionError('must resume')):
                grid.main()
            self.assertEqual((output / 'results.csv').read_bytes(), before)
        self.assertEqual(grid.load_results(output/'results.csv')['diffusion_iterations'].iloc[0], 2)

    def test_grid_runs_multiple_years(self):
        # Two tiny years exercise the per-year loop through the sole CLI.
        source = self.data/'rete_2023.csv'
        (self.data/'rete_2022.csv').write_bytes(source.read_bytes())
        for filename in ('hyperedge_parameters.csv', 'node_parameters.csv'):
            frame = pd.read_csv(self.data/filename)
            earlier = frame.assign(year=2022)
            pd.concat([frame, earlier], ignore_index=True).to_csv(self.data/filename, index=False)
        output = self.root/'multi_year'
        args = ['run_experiment', '--years','2022','2023','--alphas','.1','--seed-budgets','1',
                '--search-seconds','.04','--horizon','0','--output-dir',str(output)]
        with patch.object(grid, 'DATA_DIR', self.data), patch('sys.argv', args), \
             contextlib.redirect_stdout(io.StringIO()):
            grid.main()
        rows = grid.load_results(output/'results.csv')
        self.assertEqual(rows['year'].tolist(), [2022, 2023])
        self.assertTrue(rows['diffusion_iterations'].eq(0).all())
        self.assertFalse({'model_version','input_fingerprint'} & set(rows.columns))

    def test_standalone_cli_from_another_directory(self):
        completed = subprocess.run(
            [sys.executable, str(grid.PROJECT_ROOT/'scripts'/'run_experiment.py'), '--help'],
            cwd=self.root, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('--years', completed.stdout)
        self.assertIn('--seed-budgets', completed.stdout)
        self.assertEqual(grid.DEFAULT_OUTPUT_DIR, grid.PROJECT_ROOT/'results'/'experiment')

    def test_influence_notebook_synthetic_execution(self):
        notebook = json.loads(Path('notebooks/influence_maximization.ipynb').read_text())
        env = dict(PROJECT_ROOT=self.root, np=np, pd=pd, plt=plt, json=json, os=os,
                   Path=Path, replace=replace, gip=gip, display=lambda *args: None,
                   DATA_DIR=self.data, DEFAULT_OUTPUT_DIR=self.root/'results',
                   **{name: getattr(grid, name) for name in (
                       'ExperimentConfig', 'available_years', 'instance_mask', 'load_results',
                       'load_year_model', 'parameter_values', 'run_year',
                   )})
        with patch.dict(os.environ, {'INFLUENCE_RESULTS_DIR': str(self.root/'results')}):
            exec(''.join(notebook['cells'][2]['source']), env)
        self.assertFalse(env['RUN_SEARCH'])
        self.assertEqual(env['CONFIG'], grid.ExperimentConfig())
        config = replace(self.config, horizon=2, max_iter=1)
        env.update(CONFIG=config, RUN_SEARCH=True)
        with patch.object(grid, 'DATA_DIR', self.data), \
             patch.object(plt, 'show'), \
             contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook['cells'][3:]:
                exec(''.join(cell['source']), env)
            self.assertEqual(env['best_diffusion'].iterations, 2)
            self.assertAlmostEqual(env['best_diffusion'].total_spread, env['finish_score'])
            self.assertEqual(len(env['selected_ids']), 1)
            for table in env['trajectories'].values():
                self.assertEqual(table.iloc[0]['active_now'], config.seed_budget)
                self.assertTrue(table['ever_reached'].diff().dropna().ge(0).all())
                assert_allclose(table['discounted_increment'].cumsum(), table['cumulative_score'])

            row = {'year': 2023, **grid.parameter_values(config),
                   'start_spread': env['start_score'], 'finish_spread': env['finish_score'],
                   'start_list': json.dumps(env['start_ids']), 'finish_list': json.dumps(env['finish_ids']),
                   'stopping_reason': env['best_diffusion'].stopping_reason, 'diffusion_iterations': 2}
            env['RESULTS_PATH'].parent.mkdir()
            grid.save_results(env['RESULTS_PATH'], pd.DataFrame([row]))
            before = env['RESULTS_PATH'].read_bytes()
            env['RUN_SEARCH'] = False
            def unexpected_search(*args, **kwargs):
                raise AssertionError('Saved-result inspection must not optimize')
            env['run_year'] = unexpected_search
            for cell in notebook['cells'][3:]:
                exec(''.join(cell['source']), env)
            self.assertEqual(env['RESULTS_PATH'].read_bytes(), before)
            self.assertAlmostEqual(env['best_diffusion'].total_spread, row['finish_spread'])
            env['saved_results'] = None
            env['CONFIG'] = replace(config, stress_level=2)
            with self.assertRaisesRegex(ValueError, 'Expected one saved result'):
                exec(''.join(notebook['cells'][5]['source']), env)
        plt.close('all')

    def test_nads_keeps_budget_and_strict_ascent_for_shared_objective(self):
        B = prepare_incidence([[1,0],[1,1],[0,1]])
        p = GIPParameters(10,0,1,1,.5,0)
        seen = []
        def objective(seed):
            seen.append(seed.copy())
            return gip(B, [1,1], seed, p, node_weights=[1,2,3], alpha=1, horizon=3).total_spread
        values, seeds, _ = nads(objective, np.array([1.,0,0]), .5, .01, 4, 1, 100,
                               max_neighbors_per_phase=20, max_iterations=10)
        self.assertTrue(all(np.count_nonzero(s)==1 for s in seen+seeds))
        self.assertTrue(all(b>a for a,b in zip(values,values[1:])))
        assert_allclose(values, [objective(s) for s in seeds])


if __name__ == '__main__':
    unittest.main()
