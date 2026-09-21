"""Tiny synthetic plumbing checks; never execute a municipal-data experiment."""
import contextlib
import io
import itertools
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
from scripts import run_starting_point_experiment as multistart
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
        return grid.result_row(summary, config)

    @staticmethod
    def change_group(row, column, field, value):
        changed = dict(row)
        group = json.loads(changed[column])
        group[field] = value
        changed[column] = json.dumps(group, sort_keys=True, separators=(',', ':'))
        return changed

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
            env = dict(PROJECT_ROOT=self.root, np=np, pd=pd, json=json, gip=gip,
                       config_from_row=grid.config_from_row, validate_saved_seeds=grid.validate_saved_seeds,
                       load_year_model=grid.load_year_model, company_activity=company_activity,
                       evaluation_cache_key=evaluation_cache_key, input_fingerprint=input_fingerprint)
            company_functions = next(
                cell for cell in notebook['cells'] if cell.get('id') == 'company-functions'
            )
            exec(''.join(company_functions['source']), env)
            row = pd.Series(self.result_row(summary, config))
            rebuilt = env['propagation_for_row'](0, row)
            self.assertAlmostEqual(rebuilt['propagation'].total_spread, summary['best_spread'])
            self.assertAlmostEqual(selected['individual_weighted_spread'].iloc[0], summary['best_spread'])
            self.assertEqual(rebuilt['seed_state'].max(), amplitude)
            self.assertIs(env['propagation_for_row'](999, row), rebuilt)
            changed = row.copy()
            changed['gip_parameters'] = json.dumps(
                {**json.loads(changed['gip_parameters']), 'gamma': .25},
                sort_keys=True, separators=(',', ':'),
            )
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
        changes = [
            self.change_group(row, 'gip_parameters', 'gamma', .2),
            self.change_group(row, 'gip_parameters', 'max_iter', 10),
            self.change_group(row, 'gip_parameters', 'h0', 2),
            {**row, 'alpha': 1 + 1e-13},
            self.change_group(row, 'model_metadata', 'model_version', 'old'),
            self.change_group(row, 'graph_metadata', 'input_fingerprint', 'changed'),
        ]
        for changed in changes:
            self.assertFalse(grid.instance_mask(results, changed).any())
        retained = grid.upsert_result(results, {**row, 'finish_spread': -99})
        self.assertEqual(retained['finish_spread'].iloc[0], row['finish_spread'])
        path = self.root / 'results.csv'; grid.save_results(path, results)
        loaded = grid.load_results(path)
        self.assertTrue(grid.instance_mask(loaded, row).all())
        self.assertEqual(loaded['user_note'].iloc[0], 'preserve me')
        self.assertTrue(set(grid.IDENTITY_COLUMNS) <= set(loaded.columns))
        early = self.change_group(row, 'gip_parameters', 'max_iter', 100)
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
            incompatible = results.copy()
            metadata = json.loads(incompatible.loc[0, 'model_metadata'])
            metadata[field] = 'historical'
            incompatible.loc[0, 'model_metadata'] = json.dumps(metadata)
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
        self.assertTrue(
            rows['model_metadata']
            .map(lambda value: json.loads(value)['model_version'])
            .eq(MODEL_VERSION)
            .all()
        )

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
            configuration_cell = next(
                cell for cell in notebook['cells'] if cell.get('id') == 'configuration'
            )
            exec(''.join(configuration_cell['source']), env)
        analysis_cells = [
            cell for cell in notebook['cells']
            if cell['cell_type'] == 'code'
            and cell.get('id') not in {'imports', 'configuration'}
        ]
        self.assertTrue(env['RUN_SEARCH'])
        self.assertTrue(env['VERBOSE_NADS'])
        self.assertEqual(env['CONFIG'], grid.ExperimentConfig())
        config = replace(self.config, h0=2, max_iter=100)
        env.update(CONFIG=config, RUN_SEARCH=True)
        with patch.object(grid, 'DATA_DIR', self.data), \
             patch.object(plt, 'show'), \
             contextlib.redirect_stdout(io.StringIO()):
            for cell in analysis_cells:
                exec(''.join(cell['source']), env)
            self.assertEqual(env['summary']['search_restarts'], 1)
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
            for cell in analysis_cells:
                exec(''.join(cell['source']), env)
            self.assertEqual(env['RESULTS_PATH'].read_bytes(), before)
            self.assertAlmostEqual(env['best_diffusion'].total_spread, row['finish_spread'])
            env['saved_results'] = None
            env['CONFIG'] = replace(config, stress_level=2)
            with self.assertRaisesRegex(ValueError, 'Expected one saved result'):
                result_cell = next(
                    cell for cell in notebook['cells'] if cell.get('id') == 'initial-seeds'
                )
                exec(''.join(result_cell['source']), env)
        plt.close('all')

    def test_nads_keeps_budget_and_strict_ascent_for_shared_objective(self):
        B = prepare_incidence([[1,0],[1,1],[0,1]])
        p = GIPParameters(1,.1,.5,1,.5,1e-6)
        seen = []
        def objective(seed):
            seen.append(seed.copy())
            return gip(B, [1,1], seed, p, node_weights=[1,2,3], alpha=1).require_complete().total_spread
        values, seeds, history = nads(
            objective, np.array([1.,0,0]), .5, .01, 4, 1, 100,
            max_neighbors_per_phase=20, max_iterations=10,
        )
        self.assertTrue(all(np.count_nonzero(s)==1 for s in seen+seeds))
        self.assertTrue(all(b>a for a,b in zip(values,values[1:])))
        assert_allclose(values, [objective(s) for s in seeds])
        self.assertEqual(history[0]['seed_indices'], [0])
        self.assertTrue(all(set(event) == {'call', 'elapsed_seconds', 'seed_indices', 'value'}
                            for event in history))
        self.assertTrue(all(right['value'] > left['value']
                            for left, right in zip(history, history[1:])))

    def test_nads_keeps_an_improvement_found_at_the_time_limit(self):
        ticks = iter((0.0, 0.1, 0.2, 0.3, 0.4, 1.0))
        objective = lambda seed: float(np.flatnonzero(seed)[0] + 1)
        with patch('src.nads.time.monotonic', side_effect=lambda: next(ticks)):
            values, seeds, history = nads(
                objective, np.array([1.0, 0.0, 0.0]), 0.5, 0.01, 2, 1.0, 10,
                max_iterations=10, random_seed=0,
            )
        self.assertGreater(values[-1], values[0])
        self.assertEqual(history[-1]['value'], values[-1])
        assert_allclose(seeds[-1], np.eye(3)[int(values[-1] - 1)])

    def test_run_year_accepts_one_external_start_without_restarts(self):
        initial = np.array([0., 0., 1.])
        summary, _, _ = self.run_year_from_initial(initial)
        self.assertEqual(summary['initialization'], 'provided')
        self.assertFalse(summary['restart_search_until_time'])
        self.assertEqual(summary['search_restarts'], 1)
        self.assertEqual(summary['start_list'], [30])
        self.assertEqual(len(summary['nads_history']), 1)
        events = summary['nads_history'][0]['events']
        self.assertEqual(events[0]['seed_list'], [30])
        self.assertAlmostEqual(events[-1]['value'], summary['best_spread'])
        for invalid in (np.array([1., 1., 0.]), np.array([.5, 0., 0.]), np.array([1., 0.])):
            with self.assertRaises(ValueError):
                self.run_year_from_initial(invalid)

    def run_year_from_initial(self, initial):
        return grid.run_year(
            2023,
            self.config,
            verbose=False,
            initial_seeds=initial,
            restart_search=False,
            data_dir=self.data,
        )

    def test_starting_point_experiment_checkpoints_and_resumes(self):
        output = self.root / 'starting_point_results.csv'
        args = ['--year', '2023', '--alpha', '.1', '--seed-budget', '1',
                '--num-starts', '3', '--search-seconds', '.08', '--max-iter', '100',
                '--output', str(output)]
        with patch.object(multistart, 'DATA_DIR', self.data), \
             contextlib.redirect_stdout(io.StringIO()):
            multistart.main(args)
        runs = pd.read_csv(output)
        metadata = runs['starting_point_metadata'].map(json.loads)
        self.assertEqual(metadata.map(lambda value: value['run_index']).tolist(), [0, 1, 2])
        self.assertEqual(runs.columns[:-1].tolist(), grid.RESULT_COLUMNS)
        self.assertEqual(runs.columns[-1], 'starting_point_metadata')
        self.assertEqual(runs['start_list'].nunique(), 3)
        self.assertEqual(
            metadata.map(lambda value: value['start_type']).tolist(),
            ['degree_ranked', 'random', 'random'],
        )
        nads_parameters = runs['nads_parameters'].map(json.loads)
        self.assertTrue(nads_parameters.map(lambda value: value['random_seed']).eq(42).all())
        self.assertTrue(nads_parameters.map(lambda value: value['search_seconds']).eq(.08).all())
        notebook = json.loads(Path('notebooks/results.ipynb').read_text())
        analysis_cell = next(
            cell for cell in notebook['cells'] if cell.get('id') == 'starting-point-load'
        )
        env = dict(
            RESULTS_PATH=self.root / 'results.csv', pd=pd, np=np,
            itertools=itertools, json=json, Markdown=lambda value: value,
            display=lambda *values: None,
        )
        exec(''.join(analysis_cell['source']), env)
        self.assertEqual(len(env['starting_point_summary']), 1)
        self.assertEqual(env['starting_point_summary']['completed_starts'].iloc[0], 3)
        frequency_cell = next(
            cell for cell in notebook['cells'] if cell.get('id') == 'starting-point-frequency'
        )
        env.update(
            ExperimentConfig=grid.ExperimentConfig,
            plt=plt,
            selected_start_runs=env['starting_point_runs'],
            selected_summary=env['starting_point_summary'].iloc[0],
            load_year_model=lambda year, config: grid.load_year_model(
                year, config, data_dir=self.data,
            ),
        )
        with patch.object(plt, 'show'):
            exec(''.join(frequency_cell['source']), env)
        self.assertEqual(env['start_config'].edge_weight_beta, self.config.edge_weight_beta)
        with patch.object(multistart, 'DATA_DIR', self.data), \
             patch.object(multistart, 'run_year', side_effect=AssertionError('must resume')), \
             contextlib.redirect_stdout(io.StringIO()):
            multistart.main(args)

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
        bad = row.copy()
        run_metadata = json.loads(bad['run_metadata'])
        run_metadata['finish_score_key'] = 'stale'
        bad['run_metadata'] = json.dumps(run_metadata)
        with self.assertRaisesRegex(ValueError, 'score key'):
            grid.validate_saved_seeds(bad, model, self.config)
        bad = row.copy(); bad['finish_list'] = '[10, 10]'
        with self.assertRaisesRegex(ValueError, 'network/budget'):
            grid.validate_saved_seeds(bad, model, self.config)

    def test_compact_csv_layout_and_inline_metadata(self):
        summary, _, _ = self.run_year()
        row = self.result_row(summary)
        path = self.root/'results.csv'
        grid.save_results(path, pd.DataFrame([row]))
        public = pd.read_csv(path)
        self.assertEqual(list(public.columns), grid.RESULT_COLUMNS)
        self.assertFalse({'seed_intensity', 'horizon', 'max_iter'} & set(public.columns))
        self.assertEqual(
            list(public.columns[-6:]),
            ['network_parameters', 'gip_parameters', 'nads_parameters',
             'graph_metadata', 'model_metadata', 'run_metadata'],
        )
        self.assertNotIn('max_iter', json.loads(public['gip_parameters'].iloc[0]))
        self.assertIn('start_score_key', json.loads(public['run_metadata'].iloc[0]))
        restored = grid.load_results(path)
        self.assertEqual(restored['run_metadata'].iloc[0], row['run_metadata'])
        public.loc[0, 'finish_spread'] += 1
        public.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'history value'):
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
        metadata = json.loads(changed.loc[0, 'graph_metadata'])
        metadata['input_fingerprint'] = 'different-network'
        changed.loc[0, 'graph_metadata'] = json.dumps(metadata)
        grid.save_results(output, changed)
        with self.assertRaisesRegex(ValueError, 'input_fingerprint'):
            load_cell = next(
                cell for cell in notebook['cells'] if cell.get('id') == 'load-results'
            )
            exec(''.join(load_cell['source']), env)
        plt.close('all')

    def test_notebook_sources_compile_and_no_duplicate_diffusion(self):
        for path in Path('notebooks').glob('*.ipynb'):
            notebook = json.loads(path.read_text())
            for index, cell in enumerate(notebook['cells']):
                source = ''.join(cell['source']).strip()
                if cell['cell_type'] == 'markdown':
                    self.assertTrue(source.startswith('#'))
                    self.assertEqual(len(source.splitlines()), 1)
                    continue
                compile(source, f'{path}:{index}', 'exec')
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
                self.assertNotIn('def gip(', source)
                self.assertNotIn('horizon=', source)
                self.assertNotIn('seed_intensity', source)



if __name__ == '__main__':
    unittest.main()
