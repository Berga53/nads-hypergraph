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
    company_activity, gip, gip_from_seeds, prepare_incidence, seed_state, IncompleteEvaluationError,
)
from src.model_identity import MODEL_VERSION, MODEL_METADATA, evaluation_cache_key, input_fingerprint
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
        self.config = grid.ExperimentConfig(seed_budget=1, search_seconds_per_year=.08,
                                               max_neighbors_per_phase=10, max_search_iterations=4,
                                               alpha=1, l0=.1, h0=1, theta_l=.5, theta_h=1, gamma=.5, eps=1e-6)

    def run_year(self, config=None):
        with patch.object(grid, 'DATA_DIR', self.data):
            return grid.run_year(2023, config or self.config, verbose=False)

    def result_row(self, summary, config=None):
        config = config or self.config
        return {'year': 2023, **grid.parameter_values(config),
                **{key: summary[key] for key in grid.IDENTITY_COLUMNS + grid.SCORE_KEY_COLUMNS},
                'start_spread': summary['initial_spread'], 'finish_spread': summary['best_spread'],
                'start_list': json.dumps(summary['start_list']), 'finish_list': json.dumps(summary['finish_list']),
                **{key: summary[key] for key in ('stopping_reason', 'diffusion_iterations')}}

    def test_runner_notebook_ranking_and_ablation_share_model(self):
        for amplitude, guard in ((1, None), (2, 100)):
            config = replace(self.config, h0=amplitude, max_iter=guard)
            summary, selected, seeds = self.run_year(config)
            self.assertEqual(summary['model_version'], MODEL_VERSION)
            self.assertEqual(summary['input_fingerprint'], input_fingerprint(self.data, 2023))
            self.assertEqual(len(seeds), config.seed_budget)
            self.assertEqual(summary['stopping_reason'], 'tolerance_reached')
            self.assertGreaterEqual(summary['best_spread'], summary['initial_spread'])
            notebook = json.loads(Path('notebooks/results.ipynb').read_text())
            env = dict(PROJECT_ROOT=self.root, np=np, pd=pd, gip=gip,
                       config_from_row=grid.config_from_row, validate_saved_seeds=grid.validate_saved_seeds,
                       load_year_model=grid.load_year_model, company_activity=company_activity,
                       evaluation_cache_key=evaluation_cache_key, input_fingerprint=input_fingerprint)
            exec(''.join(notebook['cells'][19]['source']), env)
            row = pd.Series(self.result_row(summary, config))
            rebuilt = env['propagation_for_row'](0, row)
            self.assertAlmostEqual(rebuilt['propagation'].total_spread, summary['best_spread'])
            self.assertAlmostEqual(selected['individual_weighted_spread'].iloc[0], summary['best_spread'])
            self.assertEqual(rebuilt['seed_state'].max(), amplitude)
            self.assertIs(env['propagation_for_row'](999, row), rebuilt)
            changed = row.copy(); changed['gamma'] = .25
            with self.assertRaisesRegex(ValueError, 'score key'):
                env['propagation_for_row'](0, changed)
            weights = rebuilt['model']['edge_weights'].copy(); weights[0] = 0
            ablated = gip(rebuilt['model']['incidence'], weights, rebuilt['seed_state'], rebuilt['parameters'],
                          node_weights=rebuilt['model']['node_weights'], alpha=config.alpha,
                          max_iter=guard).require_complete()
            # We compare finite prefixes in the mathematical tests; no blanket
            # ordering assertion for seed-dependent tolerance truncations here.
            self.assertEqual(ablated.status, 'tolerance_reached')
            expected_activity = sum((1-config.gamma)**t * company_activity(
                rebuilt['model']['incidence'], rebuilt['model']['edge_weights'], state,
                config.stress_level, node_weights=rebuilt['model']['node_weights'])
                for t, state in enumerate(rebuilt['propagation'].states[:-1], start=1))
            assert_allclose(rebuilt['metrics']['activity'], expected_activity)
            with (self.data / 'rete_2023.csv').open('a') as stream: stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                env['propagation_for_row'](0, row)

    def test_checkpoint_identity_and_preservation(self):
        summary, _, _ = self.run_year()
        row = self.result_row(summary)
        row['alpha'] = .12345678912345678  # Exact CSV parameter comparison.
        results = pd.DataFrame([row]); results['user_note'] = 'preserve me'
        self.assertTrue(grid.instance_mask(results, row).all())
        for field, value in (('gamma',.2),
                             ('max_iter',10), ('h0',2), ('alpha',1+1e-13),
                             ('model_version','old'), ('input_fingerprint','changed')):
            changed = {**row, field: value}
            self.assertFalse(grid.instance_mask(results, changed).any())
        retained = grid.upsert_result(results, {**row, 'finish_spread': -99})
        self.assertEqual(retained['finish_spread'].iloc[0], row['finish_spread'])
        path = self.root / 'results.csv'; grid.save_results(path, results)
        loaded = grid.load_results(path)
        self.assertTrue(grid.instance_mask(loaded, row).all())
        self.assertEqual(loaded['user_note'].iloc[0], 'preserve me')
        self.assertTrue(set(grid.IDENTITY_COLUMNS + grid.SCORE_KEY_COLUMNS) <= set(loaded.columns))
        early = {**row, 'max_iter': 100}
        combined = grid.upsert_result(loaded, early)
        grid.save_results(path, combined)
        self.assertEqual(grid.instance_mask(grid.load_results(path), early).sum(), 1)
        legacy = self.root / 'legacy.csv'
        legacy.write_text('year,start_spread,finish_spread\n2023,1,2\n')
        before = legacy.read_bytes()
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            grid.load_results(legacy)
        self.assertEqual(legacy.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            grid.save_results(legacy, results)
        self.assertEqual(legacy.read_bytes(), before)
        for field in MODEL_METADATA:
            incompatible = results.copy(); incompatible[field] = 'historical'
            incompatible.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                grid.load_results(path)

    def test_grid_main_checkpoint_resume_on_synthetic_data(self):
        output = self.root / 'grid'
        args = ['run_experiment', '--years','2023','--alphas','.1','--seed-budgets','1',
                '--search-seconds','.08','--max-iter','100','--output-dir',str(output)]
        with patch.object(grid, 'DATA_DIR', self.data), \
             patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
            grid.main()
            before = (output / 'results.csv').read_bytes()
            with patch.object(grid, 'run_year', side_effect=AssertionError('must resume')):
                grid.main()
            self.assertEqual((output / 'results.csv').read_bytes(), before)
        saved = grid.load_results(output/'results.csv')
        self.assertTrue(saved['stopping_reason'].eq('tolerance_reached').all())
        # Changed source data must trigger a fresh instance, preserving prior rows.
        with (self.data/'rete_2023.csv').open('a') as stream: stream.write('\n')
        with patch.object(grid, 'DATA_DIR', self.data), patch('sys.argv', args), \
             contextlib.redirect_stdout(io.StringIO()):
            grid.main()
        self.assertEqual(len(grid.load_results(output/'results.csv')), 2)

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
                '--search-seconds','.08','--max-iter','100','--output-dir',str(output)]
        with patch.object(grid, 'DATA_DIR', self.data), patch('sys.argv', args), \
             contextlib.redirect_stdout(io.StringIO()):
            grid.main()
        rows = grid.load_results(output/'results.csv')
        self.assertEqual(rows['year'].tolist(), [2022, 2023])
        self.assertTrue(rows['diffusion_iterations'].gt(0).all())
        self.assertTrue(rows['model_version'].eq(MODEL_VERSION).all())

    def test_standalone_cli_from_another_directory(self):
        completed = subprocess.run(
            [sys.executable, str(grid.PROJECT_ROOT/'scripts'/'run_experiment.py'), '--help'],
            cwd=self.root, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('--years', completed.stdout)
        self.assertIn('--seed-budgets', completed.stdout)
        self.assertEqual(grid.DEFAULT_OUTPUT_DIR, grid.PROJECT_ROOT/'results')

    def test_influence_notebook_synthetic_execution(self):
        notebook = json.loads(Path('notebooks/influence_maximization.ipynb').read_text())
        env = dict(PROJECT_ROOT=self.root, np=np, pd=pd, plt=plt, json=json, os=os,
                   Path=Path, replace=replace, gip_from_seeds=gip_from_seeds, display=lambda *args: None,
                   DATA_DIR=self.data, DEFAULT_OUTPUT_DIR=self.root/'results',
                   **{name: getattr(grid, name) for name in (
                       'ExperimentConfig', 'available_years', 'instance_mask', 'load_results',
                       'load_year_model', 'parameter_values', 'run_year', 'result_identity', 'validate_saved_seeds',
                   )})
        with patch.dict(os.environ, {'INFLUENCE_RESULTS_DIR': str(self.root/'results')}):
            exec(''.join(notebook['cells'][2]['source']), env)
        self.assertFalse(env['RUN_SEARCH'])
        self.assertEqual(env['CONFIG'], grid.ExperimentConfig())
        config = replace(self.config, h0=2, max_iter=100)
        env.update(CONFIG=config, RUN_SEARCH=True)
        with patch.object(grid, 'DATA_DIR', self.data), \
             patch.object(plt, 'show'), \
             contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook['cells'][3:]:
                exec(''.join(cell['source']), env)
            self.assertEqual(env['best_diffusion'].status, 'tolerance_reached')
            self.assertAlmostEqual(env['best_diffusion'].total_spread, env['finish_score'])
            self.assertEqual(len(env['selected_ids']), 1)
            for table in env['trajectories'].values():
                self.assertEqual(table.iloc[0]['active_now'], config.seed_budget)
                self.assertTrue(table['ever_reached'].diff().dropna().ge(0).all())
                assert_allclose(table['discounted_increment'].cumsum(), table['cumulative_score'])

            row = self.result_row(env['summary'], config)
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
        p = GIPParameters(1,.1,.5,1,.5,1e-6)
        seen = []
        def objective(seed):
            seen.append(seed.copy())
            return gip(B, [1,1], seed, p, node_weights=[1,2,3], alpha=1).require_complete().total_spread
        values, seeds, _ = nads(objective, np.array([1.,0,0]), .5, .01, 4, 1, 100,
                               max_neighbors_per_phase=20, max_iterations=10)
        self.assertTrue(all(np.count_nonzero(s)==1 for s in seen+seeds))
        self.assertTrue(all(b>a for a,b in zip(values,values[1:])))
        assert_allclose(values, [objective(s) for s in seeds])

    def test_cutoff_cannot_enter_search_or_checkpoint(self):
        with self.assertRaisesRegex(IncompleteEvaluationError, 'operational_cutoff'):
            self.run_year(replace(self.config, max_iter=0))
        summary, _, _ = self.run_year()
        row = self.result_row(summary); row['stopping_reason'] = 'operational_cutoff'
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            grid.upsert_result(pd.DataFrame(columns=grid.RESULT_COLUMNS), row)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            grid.save_results(self.root/'cutoff.csv', pd.DataFrame([row]))

    def test_saved_seed_keys_prevent_relabeling(self):
        summary, _, _ = self.run_year()
        row = pd.Series(self.result_row(summary))
        with patch.object(grid, 'DATA_DIR', self.data):
            model = grid.load_year_model(2023, self.config)
        grid.validate_saved_seeds(row, model, self.config)
        bad = row.copy(); bad['finish_score_key'] = 'stale'
        with self.assertRaisesRegex(ValueError, 'score key'):
            grid.validate_saved_seeds(bad, model, self.config)
        bad = row.copy(); bad['finish_list'] = '[10, 10]'
        with self.assertRaisesRegex(ValueError, 'network/budget'):
            grid.validate_saved_seeds(bad, model, self.config)

    def test_plain_csv_layout_and_hidden_metadata(self):
        summary, _, _ = self.run_year()
        row = self.result_row(summary)
        path = self.root/'results.csv'
        grid.save_results(path, pd.DataFrame([row]))
        public = pd.read_csv(path)
        self.assertEqual(list(public.columns), grid.RESULT_COLUMNS)
        self.assertFalse(set(grid.IDENTITY_COLUMNS + grid.SCORE_KEY_COLUMNS) & set(public.columns))
        self.assertEqual(public['seed_intensity'].iloc[0], public['h0'].iloc[0])
        self.assertTrue(public['horizon'].isna().all())
        self.assertTrue(grid.result_metadata_path(path).is_file())
        restored = grid.load_results(path)
        self.assertEqual(restored['finish_score_key'].iloc[0], row['finish_score_key'])
        public.loc[0, 'finish_spread'] += 1
        public.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            grid.load_results(path)

    def test_results_notebook_all_analysis_cells_on_synthetic_grid(self):
        rows = []
        for alpha in (.1, .2):
            for budget in (1, 2):
                config = replace(self.config, alpha=alpha, seed_budget=budget)
                summary, _, _ = self.run_year(config)
                rows.append(self.result_row(summary, config))
        output = self.root/'grid.csv'
        grid.save_results(output, pd.DataFrame(rows))
        notebook = json.loads(Path('notebooks/results.ipynb').read_text())
        env = {}
        exec(''.join(notebook['cells'][1]['source']), env)
        env.update(PROJECT_ROOT=self.root, RESULTS_PATH=output, display=lambda *args: None)
        with patch.object(plt, 'show'), contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook['cells'][2:]:
                if cell['cell_type'] == 'code':
                    exec(''.join(cell['source']), env)
        self.assertEqual(len(env['results']), 4)
        self.assertTrue(env['checks'].all())
        self.assertTrue(env['PROPAGATION_CACHE'])
        # Plots must not silently compare networks with different fingerprints.
        changed = pd.DataFrame(rows)
        changed.loc[0, 'input_fingerprint'] = 'different-network'
        grid.save_results(output, changed)
        with self.assertRaisesRegex(ValueError, 'input_fingerprint'):
            exec(''.join(notebook['cells'][3]['source']), env)
        plt.close('all')

    def test_notebook_sources_compile_and_no_duplicate_diffusion(self):
        for path in Path('notebooks').glob('*.ipynb'):
            notebook = json.loads(path.read_text())
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source'])
                    compile(source, f'{path}:{index}', 'exec')
                    self.assertNotIn('def gip(', source)
                    self.assertNotIn('horizon=', source)
                    self.assertNotIn('seed_intensity', source)



if __name__ == '__main__':
    unittest.main()
