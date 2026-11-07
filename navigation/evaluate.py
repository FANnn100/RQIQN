"""Evaluate a newly trained RQIQN model on the historical hard-mode protocol."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

NAV_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NAV_ROOT))

from aggregate import aggregate_episodes, write_json
from evaluation import evaluate_episode


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_training_seed(model_dir, requested_seed):
    """Check new training manifests; legacy model-only directories remain valid."""
    path = Path(model_dir) / "run_manifest.json"
    if not path.is_file():
        return "unverified: legacy model directory has no run_manifest.json"
    manifest = json.loads(path.read_text())
    recorded_seed = manifest.get("protocol", {}).get("seed")
    if recorded_seed is None:
        return "unverified: run manifest has no protocol.seed"
    if recorded_seed != requested_seed:
        raise ValueError("--seed {} does not match the model's recorded training seed {}".format(
            requested_seed, recorded_seed))
    return "matched run_manifest.json protocol.seed"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path,
                        help="Run directory containing network_params.pth and constructor_params.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0,
                        help="Training-seed label for aggregation; does not override model constructor seed")
    parser.add_argument("--policies", default="full,adaptive")
    parser.add_argument("--env-seed", type=int, default=348)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy-seed", type=int, default=2,
                        help="Initial agent seed, as in the historical evaluator; model loading then restores constructor seed")
    parser.add_argument("--action-seed", type=int, default=None,
                        help="Optional explicit RNG reset per policy; default preserves constructor-seeded behavior")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Check two generated hard layouts, at most eight actions each; not a paper experiment")
    args = parser.parse_args(argv)
    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    if not policies or len(set(policies)) != len(policies) or any(
            policy not in ("full", "adaptive") for policy in policies):
        parser.error("--policies must be a comma-separated subset of full,adaptive without duplicates")
    if args.num_episodes < 1 or args.max_steps < 1:
        parser.error("--num-episodes and --max-steps must be positive")
    args.policies = policies
    if args.smoke:
        args.num_episodes = min(args.num_episodes, 2)
        args.max_steps = min(args.max_steps, 8)
    return args


def main(argv=None):
    args = parse_args(argv)
    model_dir = args.model_dir.resolve()
    seed_verification = verify_training_seed(model_dir, args.seed)
    settings = {
        "schema_version": 1,
        "protocol": "historical-formal-hard",
        "model_dir": str(model_dir),
        "training_seed": args.seed,
        "training_seed_verification": seed_verification,
        "policies": args.policies,
        "env_seed": args.env_seed,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "device": args.device,
        "policy_initialization_seed": args.policy_seed,
        "action_seed_override": args.action_seed,
        "smoke": args.smoke,
        "obstacles": 10,
        "vortex_cores": 8,
        "robot_n": 5,
        "action_duration_seconds": 0.5,
        "discount": 0.99,
        "boundary_enforced": False,
        "quantile_samples_for_action_selection": 32,
        "exploration_epsilon": 0.0,
        "note": "Policies use sampled quantiles; epsilon=0 does not make quantile sampling deterministic.",
        "sampling": "Reload model per policy; preserve its constructor RNG seed unless explicitly overridden.",
    }
    if args.dry_run:
        print(json.dumps(settings, indent=2, sort_keys=True))
        return
    for name in ("network_params.pth", "constructor_params.json"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(model_dir / name)
    if (args.output_dir / "settings.json").exists() or (args.output_dir / "episodes.jsonl").exists():
        raise FileExistsError("Evaluation output already exists; choose a fresh --output-dir")

    import numpy as np
    import torch
    from protocols import effective_layout, make_env, make_hard_configs
    from thirdparty.IQN.rqiqn_agent import RQIQNAgent

    configs = make_hard_configs(num_episodes=args.num_episodes, env_seed=args.env_seed)
    layout_json = json.dumps(configs, sort_keys=True, separators=(",", ":"))
    settings["layout_sha256"] = hashlib.sha256(layout_json.encode()).hexdigest()
    settings["checkpoint_sha256"] = sha256_file(model_dir / "network_params.pth")
    settings["constructor"] = json.loads((model_dir / "constructor_params.json").read_text())
    settings["effective_layouts"] = [effective_layout(config) for config in configs]
    settings["runtime"] = {
        "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "settings.json", settings)
    write_json(args.output_dir / "eval_configs.json", configs)
    metrics = {}
    with (args.output_dir / "episodes.jsonl").open("w") as handle:
        for policy in args.policies:
            env = make_env(seed=args.env_seed)
            env.set_boundary = False
            agent = RQIQNAgent(env.get_state_space_dimension(), env.get_action_space_dimension(),
                               device=args.device, seed=args.policy_seed)
            agent.load_model(str(model_dir), args.device)
            if args.action_seed is not None:
                random.seed(args.action_seed)
                np.random.seed(args.action_seed)
                torch.manual_seed(args.action_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.action_seed)
            episodes = []
            try:
                for index, config in enumerate(configs):
                    row = evaluate_episode(agent, env, config, policy=policy, max_steps=args.max_steps)
                    row.update(policy=policy, episode_index=index, training_seed=args.seed)
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
                    episodes.append(row)
                    if (index + 1) % 25 == 0 or index + 1 == len(configs):
                        print("{}: {}/{} episodes".format(policy, index + 1, len(configs)), flush=True)
            finally:
                env.close()
            metrics[policy] = aggregate_episodes(episodes)
    write_json(args.output_dir / "per_seed_metrics.json", metrics)
    print("Saved evaluation:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
