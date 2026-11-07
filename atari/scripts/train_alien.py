#!/usr/bin/env python3
"""Train only RQIQN on Alien using the archived experiment's algorithm/config."""
import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def versions():
    result = {'python': sys.version}
    for name in ['numpy', 'jax', 'jaxlib', 'flax', 'optax', 'chex',
                 'orbax-checkpoint', 'tensorstore', 'tensorflow', 'gin-config',
                 'gym', 'ale-py', 'opencv-python', 'scipy', 'msgpack']:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def verify_resume(output, restored_iteration):
    """Refuse a silent fresh start before touching existing metric records."""
    completed = [int(p.name.rsplit('.', 1)[1]) for p in
                 (output / 'checkpoints').glob('sentinel_checkpoint_complete.*')]
    expected = max(completed, default=-1) + 1
    if expected < 1 or restored_iteration != expected:
        raise RuntimeError(
            'Checkpoint restoration failed: expected start iteration {}, got {}. '
            'Existing training records have not been changed.'.format(
                expected, restored_iteration))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--training-steps', type=int, default=250000)
    parser.add_argument('--seed-mode', choices=['agent', 'all'], default='agent',
                        help='agent matches historical explicit seeding; all additionally seeds NumPy, Python and ALE.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true',
                        help='Use a separate small pipeline-test configuration, not paper settings.')
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.iterations < 1 or args.training_steps < 1 or args.seed < 0:
        parser.error('Seed must be nonnegative and training budgets must be positive.')
    if args.resume and args.no_checkpoint:
        parser.error('--resume requires checkpointing.')
    configs = [ROOT / 'configs/alien_rqiqn.gin']
    if args.smoke:
        configs.append(ROOT / 'configs/alien_smoke.gin')
    bindings = ['JaxRQIQNAgent.seed={}'.format(args.seed)]
    if not args.smoke:
        bindings += ['Runner.num_iterations={}'.format(args.iterations),
                     'Runner.training_steps={}'.format(args.training_steps)]
    protocol = dict(game='Alien', agent='RQIQN', seed=args.seed,
                    seed_mode=args.seed_mode, smoke=args.smoke,
                    iterations=1 if args.smoke else args.iterations,
                    training_steps=256 if args.smoke else args.training_steps,
                    checkpointing=not args.no_checkpoint,
                    configs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in configs},
                    bindings=bindings, metric='raw undiscounted training episode return')
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        print('Dry run only; no output directory or training process was created.')
        return
    output = args.output_dir.resolve()
    manifest_path = output / 'run_manifest.json'
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Output directory is nonempty; use a new directory or --resume.')
    if args.resume:
        if not manifest_path.exists():
            parser.error('--resume requires the original run_manifest.json.')
        original = json.loads(manifest_path.read_text())['protocol']
        unchanged = [key for key in protocol if key not in ('iterations', 'bindings')]
        if any(original.get(key) != protocol[key] for key in unchanged):
            parser.error('Resume settings differ from the recorded training protocol.')
        if not list((output / 'checkpoints').glob('sentinel_checkpoint_complete.*')):
            parser.error('No completed checkpoint found to resume.')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'console.log').open('a', buffering=1) as console:
        with contextlib.redirect_stdout(Tee(sys.stdout, console)), contextlib.redirect_stderr(Tee(sys.stderr, console)):
            train(args, configs, bindings, protocol, output, manifest_path)


def train(args, configs, bindings, protocol, output, manifest_path):
    from absl import logging
    import gin
    import numpy as np
    import tensorflow as tf
    from dopamine.discrete_domains import atari_lib, run_experiment
    from training_records import IterationRecorder

    imported = Path(run_experiment.__file__).resolve()
    if ROOT not in imported.parents:
        raise RuntimeError('Dopamine was imported from outside this release: ' + str(imported))
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.disable_v2_behavior()
    if args.seed_mode == 'all':
        random.seed(args.seed)
        np.random.seed(args.seed)
    run_experiment.load_gin_configs([str(p) for p in configs], bindings)

    def environment_factory():
        env = atari_lib.create_atari_environment()
        if args.seed_mode == 'all':
            env.environment.seed(args.seed)
            env.action_space.seed(args.seed)
        return env

    class RecordedTrainRunner(run_experiment.TrainRunner):
        def _run_one_iteration(self, iteration):
            started = time.monotonic()
            stats = super()._run_one_iteration(iteration)
            self.recorder.record(iteration, stats, time.monotonic() - started)
            return stats

        def _checkpoint_experiment(self, iteration):
            if not args.no_checkpoint:
                super()._checkpoint_experiment(iteration)

    runner = RecordedTrainRunner(str(output), run_experiment.create_agent,
                                 create_environment_fn=environment_factory)
    if args.resume:
        verify_resume(output, runner._start_iteration)
    runner.recorder = IterationRecorder(output / 'metrics/training_returns.jsonl',
                                        args.seed, resume_from=runner._start_iteration)
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(ROOT.rglob('*.py'))
                     if 'runs' not in p.relative_to(ROOT).parts}
    manifest = dict(protocol=protocol, package_root=str(ROOT),
                    started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    dependency_versions=versions(), source_hashes=source_hashes,
                    start_iteration=runner._start_iteration,
                    runtime_environment={
                        'gym_id': runner._environment.environment.spec.id,
                        'sticky_action_probability': float(
                            runner._environment.environment.unwrapped.ale.getFloat(
                                'repeat_action_probability')),
                        'frame_skip': 4, 'screen_size': 84,
                        'stack_size': 4, 'terminal_on_life_loss': False},
                    runtime_rqiqn={key: getattr(runner._agent, key) for key in (
                        'loss_type', 'robust_epsilon_init', 'wasserstein_p',
                        'robust_sign', 'robust_decay', 'robust_logistic_k',
                        'robust_logistic_t0', 'state_action_mean_scaling',
                        'state_action_mean_scale_coeff', 'state_action_mean_use_abs',
                        'state_action_mean_use_decay_multiplier', 'num_tau_samples',
                        'num_tau_prime_samples', 'num_quantile_samples')},
                    operative_gin=gin.operative_config_str(),
                    historical_replay_limit='Historical runs recorded agent seeds only; exact historical ALE/NumPy RNG states are unavailable.')
    if args.resume:
        original = json.loads(manifest_path.read_text())
        original.setdefault('resume_history', []).append(manifest)
        manifest = original
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'operative_config.gin').write_text(gin.operative_config_str())
    for p in configs:
        (output / p.name).write_text(p.read_text())
    print('RQIQN Alien:', json.dumps(protocol))
    try:
        runner.run_experiment()
    finally:
        runner._environment.close()


if __name__ == '__main__':
    main()
