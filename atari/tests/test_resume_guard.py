import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    'train_alien', Path(__file__).resolve().parents[1] / 'scripts/train_alien.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ResumeGuardTest(unittest.TestCase):
    def test_failed_checkpoint_restore_preserves_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'checkpoints').mkdir()
            (root / 'checkpoints/sentinel_checkpoint_complete.3').touch()
            records = root / 'training_returns.jsonl'
            records.write_text('existing records\n')
            with self.assertRaisesRegex(RuntimeError, 'restoration failed'):
                MODULE.verify_resume(root, 0)
            self.assertEqual(records.read_text(), 'existing records\n')
            MODULE.verify_resume(root, 4)

    def test_refuses_missing_or_mismatched_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(RuntimeError):
                MODULE.verify_resume(root, 0)
            (root / 'checkpoints').mkdir()
            (root / 'checkpoints/sentinel_checkpoint_complete.0').touch()
            with self.assertRaises(RuntimeError):
                MODULE.verify_resume(root, 2)


if __name__ == '__main__':
    unittest.main()
