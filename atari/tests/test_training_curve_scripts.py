"""Small synthetic checks for the log export and plotting data pipeline."""
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from export_training_returns import load_training_records
from plot_training_returns import aggregate_runs


def record(iteration, values, lengths=None):
    return {'iteration': iteration, 'train_average_return': sum(values) / len(values),
            'train_episode_returns': values,
            'train_episode_lengths': lengths or [10] * len(values)}


class TrainingCurveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_dir(self, name, records):
        run = self.root / name
        (run / 'metrics').mkdir(parents=True)
        (run / 'metrics/training_returns.jsonl').write_text(
            ''.join(json.dumps(value) + '\n' for value in records))
        return run

    def test_raw_episode_means_and_cumulative_actual_frames(self):
        run = self.run_dir('run', [record(0, [1, 3]), record(1, [5], [7])])
        rows, _ = load_training_records(run)
        self.assertEqual(rows[0]['train_average_return'], 2)
        self.assertEqual(rows[1]['cumulative_train_agent_steps'], 27)
        self.assertEqual(rows[1]['cumulative_train_frames'], 108)

    def test_inconsistent_return_is_rejected(self):
        data = record(0, [1, 3])
        data['train_average_return'] = 99
        with self.assertRaisesRegex(ValueError, 'differs from the mean'):
            load_training_records(self.run_dir('run', [data]))

    def test_latest_legacy_snapshot_only_and_explicit_permission(self):
        run = self.root / 'legacy'
        (run / 'logs').mkdir(parents=True)
        for number in [1, 2]:
            log = {'iteration_%d' % i: {'train_average_return': [float(i)],
                                       'train_episode_returns': [float(i)],
                                       'train_episode_lengths': [2]}
                   for i in range(number + 1)}
            (run / ('logs/log_%d' % number)).write_bytes(pickle.dumps(log))
        with self.assertRaisesRegex(ValueError, 'allow-legacy-pickle'):
            load_training_records(run)
        rows, source = load_training_records(run, allow_legacy_pickle=True)
        self.assertEqual(source.name, 'log_2')
        self.assertEqual([row['iteration'] for row in rows], [0, 1, 2])
        self.assertEqual(rows[-1]['cumulative_train_agent_steps'], 6)

    def test_plot_intersection_equal_seed_weights_and_sample_sd(self):
        run_a = self.run_dir('a', [record(0, [0]), record(1, [2, 2]), record(2, [4])])
        run_b = self.run_dir('b', [record(1, [6]), record(2, [8]), record(3, [10])])
        rows, meta = aggregate_runs([run_a, run_b])
        self.assertEqual([row['iteration'] for row in rows], [1, 2])
        self.assertEqual(rows[0]['mean'], 4)
        self.assertAlmostEqual(rows[0]['sample_std'], math.sqrt(8))
        self.assertEqual(meta['inputs'][0]['iterations_not_plotted'], [0])
        self.assertIsNone(meta['smoothing'])

    def test_single_run_has_no_error_band_and_duplicate_input_rejected(self):
        run = self.run_dir('single', [record(0, [1])])
        rows, meta = aggregate_runs([run])
        self.assertEqual(rows[0]['sample_std'], '')
        self.assertIsNone(meta['uncertainty'])
        with self.assertRaisesRegex(ValueError, 'more than once'):
            aggregate_runs([run, run])


if __name__ == '__main__':
    unittest.main()
