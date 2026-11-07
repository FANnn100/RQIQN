"""Regression checks for effective layouts and a real small learning run."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LayoutTests(unittest.TestCase):
    def test_historical_wrapper_effect_and_corrected_layouts(self):
        import gym
        import numpy as np
        from protocols import effective_layout, make_online_configs
        wrapped = gym.make('marinenav_env:marinenav_env-v0', seed=348)
        wrapped.reset_start_and_goal = False
        wrapped.start = np.array([5.0, 5.0])
        wrapped.goal = np.array([45.0, 45.0])
        wrapped.num_cores, wrapped.num_obs = 4, 6
        wrapped.reset()
        actual_first = wrapped.episode_data()
        wrapped.close()
        historical = make_online_configs('historical-online30')
        self.assertEqual(historical['env_0'], actual_first)
        self.assertEqual([(effective_layout(c)['cores'], effective_layout(c)['obstacles'])
                          for c in historical.values()], [(8, 5)] * 30)
        corrected = make_online_configs('corrected-mixed3x10')
        self.assertEqual([(effective_layout(c)['cores'], effective_layout(c)['obstacles'])
                          for c in corrected.values()], [(4, 6)] * 10 + [(6, 8)] * 10 + [(8, 10)] * 10)

    def test_actual_smoke_learns_and_records(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run'
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'train.py'),
                '--output-dir', str(run), '--device', 'cpu', '--smoke'],
                capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            summary = json.loads((run / 'training_summary.json').read_text())
            self.assertEqual(summary['actual_training_actions'], 64)
            self.assertGreater(summary['gradient_updates'], 0)
            self.assertTrue(summary['parameters_changed'])
            records = [json.loads(line) for line in
                       (run / 'metrics/training_episodes.jsonl').read_text().splitlines()]
            self.assertEqual(sum(row['length'] for row in records), 64)
            self.assertTrue((run / 'network_params.pth').exists())
            self.assertTrue(json.loads((run / 'run_manifest.json').read_text())['protocol']['smoke'])
            subprocess.run([sys.executable, '-B', str(ROOT / 'plot.py'), '--runs', str(run),
                            '--output-dir', str(Path(directory) / 'figure')], check=True,
                           capture_output=True, text=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
