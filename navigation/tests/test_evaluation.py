"""Checks for rollout accounting and seed-level, success-conditional aggregation."""

import contextlib
import csv
import io
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

NAV_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NAV_ROOT))

from aggregate import aggregate_episodes, aggregate_runs, main as aggregate_main, summarize_seeds
from evaluation import evaluate_episode


def episode(success=False, reward=0.0, time=10.0, energy=20.0):
    return dict(success=success, collision=not success, timeout=False,
                out_of_boundary=False, discounted_return=reward, raw_return=reward,
                time=time, energy=energy)


class RolloutAccountingTest(unittest.TestCase):
    def test_discounted_return_time_energy_and_policies(self):
        class Robot:
            dt, N = 0.1, 5

            def compute_action_energy_cost(self, action):
                return 2.0

        class Env:
            robot = Robot()
            discount = 0.99

            def reset_with_eval_config(self, config):
                self.index = 0
                return [0.0]

            def step(self, action):
                reward = [-1.0, 2.0, 100.0][self.index]
                self.index += 1
                done = self.index == 3
                return [0.0], reward, done, {"state": "reach goal" if done else "normal"}

        class Agent:
            def act(self, observation, eps, cvar):
                self.eps, self.cvar = eps, cvar
                return 0

            def act_adaptive(self, observation, eps):
                self.eps = eps
                return 0, 0.25

        agent = Agent()
        result = evaluate_episode(agent, Env(), {}, policy="full", max_steps=4)
        self.assertAlmostEqual(result["raw_return"], 101.0)
        self.assertAlmostEqual(result["discounted_return"], -1 + 0.99 * 2 + 0.99 ** 2 * 100)
        self.assertEqual((result["time"], result["energy"], result["length"]), (1.5, 6.0, 3))
        self.assertTrue(result["success"])
        self.assertFalse(result["timeout"])
        self.assertEqual((agent.eps, agent.cvar), (0.0, 1.0))
        adaptive = evaluate_episode(agent, Env(), {}, policy="adaptive", max_steps=2)
        self.assertTrue(adaptive["timeout"])
        self.assertEqual(adaptive["cvars"], [0.25, 0.25])


class AggregationTest(unittest.TestCase):
    def test_table1_percentages_and_seed_conditional_weighting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            for seed, episodes in enumerate((
                    [episode(True, 0.0, 1.0, 2.0)] + [episode(False, 0.0)] * 9,
                    [episode(True, 10.0, 9.0, 18.0)] * 9 + [episode(False, 10.0)])):
                path = root / str(seed)
                path.mkdir()
                paths.append(path)
                settings = dict(protocol="test", layout_sha256="same", num_episodes=10,
                                max_steps=1000, smoke=False, training_seed=seed, policies=["full"])
                (path / "settings.json").write_text(json.dumps(settings))
                with (path / "episodes.jsonl").open("w") as handle:
                    for index, row in enumerate(episodes):
                        row = dict(row, policy="full", episode_index=index)
                        handle.write(json.dumps(row) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                aggregate_main(["--runs"] + [str(path) for path in paths]
                               + ["--output-dir", str(root / "summary")])
            with (root / "summary" / "table1.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["Success (%)"], "50.00 ± 56.57")
            self.assertEqual(rows[0]["Collision (%)"], "50.00 ± 56.57")
            self.assertEqual(rows[0]["Timeout (%)"], "0.00 ± 0.00")
            self.assertEqual(rows[0]["Return (discounted)"], "5.00 ± 7.07")
            self.assertEqual(rows[0]["Time_succ (s)"], "5.00 ± 5.66")
            self.assertEqual(rows[0]["Energy_succ"], "10.00 ± 11.31")

    def test_seed_weighting_and_success_conditional_means(self):
        first = [episode(True, 0.0, 1.0, 2.0)] + [episode(False, 0.0)] * 9
        second = [episode(True, 10.0, 9.0, 18.0)] * 9 + [episode(False, 10.0)]
        per_seed = {"0": aggregate_episodes(first), "1": aggregate_episodes(second)}
        summary = summarize_seeds(per_seed)
        self.assertEqual(summary["discounted_return"]["mean"], 5.0)
        self.assertAlmostEqual(summary["discounted_return"]["std"], statistics.stdev([0.0, 10.0]))
        self.assertEqual(summary["success_rate"]["mean"], 0.5)
        self.assertEqual(summary["time_success"]["mean"], 5.0)
        self.assertEqual(summary["energy_success"]["mean"], 10.0)
        self.assertNotEqual(summary["time_success"]["mean"], (1 + 9 * 9) / 10)

    def test_equal_weight_despite_different_episode_counts(self):
        per_seed = {"0": aggregate_episodes([episode(True, 0.0)]),
                    "1": aggregate_episodes([episode(True, 10.0)] * 9)}
        self.assertEqual(summarize_seeds(per_seed)["discounted_return"]["mean"], 5.0)

    def test_no_success_is_undefined_and_sample_sd_requires_seed_count(self):
        per_seed = {"0": aggregate_episodes([episode(False)]),
                    "1": aggregate_episodes([episode(True, time=4.0)])}
        summary = summarize_seeds(per_seed)
        self.assertIsNone(per_seed["0"]["time_success"])
        self.assertEqual(summary["time_success"], {"mean": 4.0, "std": 0.0, "n_seeds": 1})
        only_failure = summarize_seeds({"0": per_seed["0"]})
        self.assertIsNone(only_failure["energy_success"]["mean"])
        with self.assertRaises(ValueError):
            aggregate_episodes([])

    def test_duplicate_seeds_and_incomplete_runs_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / str(i) for i in range(2)]
            settings = dict(protocol="test", layout_sha256="same", num_episodes=1,
                            max_steps=1000, smoke=False, training_seed=0, policies=["full"])
            for path in paths:
                path.mkdir()
                (path / "settings.json").write_text(json.dumps(settings))
                row = dict(episode(True), policy="full", episode_index=0)
                (path / "episodes.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate training seed"):
                aggregate_runs(paths)
            (paths[0] / "episodes.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                aggregate_runs([paths[0]])


class RealEnvironmentCheckpointTest(unittest.TestCase):
    def test_manifest_seed_identity_check_and_legacy_model_support(self):
        from evaluate import verify_training_seed
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            self.assertTrue(verify_training_seed(model_dir, 1).startswith("unverified"))
            (model_dir / "run_manifest.json").write_text(json.dumps({"protocol": {"seed": 1}}))
            self.assertTrue(verify_training_seed(model_dir, 1).startswith("matched"))
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify_training_seed(model_dir, 0)

    def test_fresh_checkpoint_smoke_evaluation_and_summary(self):
        # Generate a checkpoint inside a temporary directory. No historical
        # weights, ROMs, or stored experiment data are used by this test.
        from evaluate import main as evaluate_main
        from protocols import make_env
        from thirdparty.IQN.rqiqn_agent import RQIQNAgent

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir, output_dir, summary_dir = root / "model", root / "eval", root / "summary"
            model_dir.mkdir()
            env = make_env(seed=0)
            agent = RQIQNAgent(env.get_state_space_dimension(), env.get_action_space_dimension(),
                               seed=100, device="cpu")
            agent.qnetwork_local.save(str(model_dir))
            env.close()
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_main(["--model-dir", str(model_dir), "--output-dir", str(output_dir),
                               "--seed", "0", "--smoke", "--max-steps", "2"])
                aggregate_main(["--runs", str(output_dir), "--output-dir", str(summary_dir)])
            settings = json.loads((output_dir / "settings.json").read_text())
            rows = [json.loads(line) for line in (output_dir / "episodes.jsonl").read_text().splitlines()]
            summary = json.loads((summary_dir / "summary.json").read_text())
            self.assertTrue(settings["smoke"])
            self.assertFalse(settings["boundary_enforced"])
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["policy"] for row in rows}, {"full", "adaptive"})
            self.assertTrue(all(1 <= row["length"] <= 2 for row in rows))
            self.assertTrue(all(layout["obstacles"] == 10 and layout["cores"] == 8
                                and layout["robot_n"] == 5 for layout in settings["effective_layouts"]))
            self.assertEqual(summary["summary"]["full"]["discounted_return"]["n_seeds"], 1)
            self.assertTrue((summary_dir / "per_seed_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
