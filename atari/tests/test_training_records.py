import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training_records import IterationRecorder


class RecorderTests(unittest.TestCase):
    def test_resume_discards_records_after_restored_checkpoint(self):
        stats = dict(train_episode_returns=[2.0, 4.0], train_episode_lengths=[3, 5],
                     train_average_return=[3.0], train_average_steps_per_second=[10.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training_returns.jsonl'
            first = IterationRecorder(path, seed=0)
            first.record(0, stats, 1.0)
            first.record(1, stats, 1.0)
            recovered = IterationRecorder(path, seed=0, resume_from=1)
            recovered.record(1, stats, 2.0)
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual([row['iteration'] for row in rows], [0, 1])
            self.assertEqual(rows[-1]['cumulative_train_agent_steps'], 16)
            self.assertEqual(rows[-1]['cumulative_train_frames'], 64)
            self.assertEqual(rows[-1]['wall_time_seconds'], 2.0)

    def test_rejects_average_inconsistent_with_episode_records(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = IterationRecorder(Path(directory) / 'records.jsonl', seed=0)
            with self.assertRaises(ValueError):
                recorder.record(0, dict(train_episode_returns=[1.0], train_episode_lengths=[2],
                                       train_average_return=[99.0],
                                       train_average_steps_per_second=[1.0]), 1.0)


if __name__ == '__main__':
    unittest.main()
