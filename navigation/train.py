#!/usr/bin/env python3
"""Train historical navigation RQIQN and record new episodes and evaluations."""
import argparse
import contextlib
import copy
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def append_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(data, allow_nan=False) + '\n')


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


class RecordedTrainingEnv:
    """Observe rewards without changing transitions, resets or random streams."""
    def __init__(self, env, log_path, seed):
        self.env, self.log_path, self.seed = env, log_path, seed
        self.episode = 0
        self.length = 0
        self.raw_return = self.discounted_return = 0.0
        self.start_step = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        observation = self.env.reset()
        self.length = 0
        self.raw_return = self.discounted_return = 0.0
        self.start_step = self.env.total_timesteps
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.raw_return += float(reward)
        self.discounted_return += self.env.discount ** self.length * float(reward)
        self.length += 1
        if done:
            self.record(True, info['state'])
        return observation, reward, done, info

    def record(self, complete, state):
        if not self.length:
            return
        append_jsonl(self.log_path, dict(schema_version=1, seed=self.seed,
                     episode=self.episode, complete=complete,
                     terminal_state=state, length=self.length,
                     raw_return=self.raw_return,
                     discounted_return=self.discounted_return,
                     discount=self.env.discount,
                     start_agent_step=self.start_step,
                     end_agent_step=self.env.total_timesteps,
                     cores=len(self.env.cores), obstacles=len(self.env.obstacles),
                     robot_n=self.env.robot.N))
        self.episode += 1
        self.length = 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/table1_rqiqn.json')
    parser.add_argument('--timesteps', type=int)
    parser.add_argument('--eval-freq', type=int)
    parser.add_argument('--online-protocol', choices=['historical-online30', 'corrected-mixed3x10'])
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='Record metrics without saving newly trained model weights.')
    parser.add_argument('--smoke', action='store_true',
                        help='Run 64 training actions with reduced replay and evaluation budgets.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**32 - 101:
        parser.error('Seed must fit NumPy RandomState after the agent seed offset.')
    config = json.loads(args.config.read_text())
    if args.timesteps is not None:
        config['total_timesteps'] = args.timesteps
    if args.eval_freq is not None:
        config['eval_freq'] = args.eval_freq
    if args.online_protocol:
        config['online_protocol'] = args.online_protocol
    if args.smoke:
        config['total_timesteps'] = 63
        config['eval_freq'] = 16
        config['agent_parameters'].update(BATCH_SIZE=4, BUFFER_SIZE=256,
                                           learning_starts=8,
                                           target_update_interval=16)
    if config['total_timesteps'] < 1 or config['eval_freq'] < 1:
        parser.error('Training and evaluation budgets must be positive.')
    protocol = dict(experiment='historical-navigation-rqiqn-table1',
                    config=config, seed=args.seed,
                    agent_seed=args.seed + config['agent_seed_offset'],
                    device=args.device, smoke=args.smoke,
                    checkpointing=not args.no_checkpoint,
                    requested_loop_limit=config['total_timesteps'],
                    actual_training_actions=config['total_timesteps'] + 1,
                    inclusive_loop_preserved=True,
                    boundary_termination=False,
                    quantile_samples=dict(current=8, target=8, action=32),
                    scale_gradient='detached',
                    update_implementation='upstream navigation IQN/RQIQN',
                    online_policies=['full', 'adaptive'],
                    online_max_steps=8 if args.smoke else 1000,
                    online_episode_count=2 if args.smoke else 30)
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return args, config, protocol, False
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new or empty output directory. Model-only resume is unsupported.')
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir = output
    return args, config, protocol, True


def run_training(args, config, protocol):
    import numpy as np
    import torch
    from evaluation import evaluate_episode
    from protocols import TRAINING_SCHEDULE, effective_layout, make_env, make_online_configs
    from thirdparty.IQN.rqiqn_agent import RQIQNAgent

    output = args.output_dir
    metrics = output / 'metrics'
    metrics.mkdir()
    base_env = make_env(args.seed, schedule=copy.deepcopy(TRAINING_SCHEDULE))
    train_env = RecordedTrainingEnv(base_env, metrics / 'training_episodes.jsonl', args.seed)
    eval_env = make_env(config['online_env_seed'])
    configs = make_online_configs(config['online_protocol'], config['online_env_seed'])
    if args.smoke:
        configs = dict(list(configs.items())[:2])
    protocol['training_schedule'] = TRAINING_SCHEDULE
    protocol['effective_online_layouts'] = [effective_layout(c) for c in configs.values()]
    protocol['online_boundary_termination'] = bool(eval_env.set_boundary)
    write_json(output / 'online_eval_configs.json', configs)

    class RecordedAgent(RQIQNAgent):
        def __init__(self, *positional, **parameters):
            super().__init__(*positional, **parameters)
            self.gradient_updates = 0

        def train(self, experiences):
            loss = super().train(experiences)
            self.gradient_updates += 1
            if self.gradient_updates == 1 or self.gradient_updates % (1 if args.smoke else 100) == 0:
                append_jsonl(metrics / 'losses.jsonl', dict(
                    schema_version=1, seed=args.seed, gradient_update=self.gradient_updates,
                    current_timestep=self.current_timestep,
                    learning_timestep=self.learning_timestep, loss=float(loss),
                    robust_epsilon=self._last_robust_epsilon))
            return loss

        def evaluation(self, env, eval_config, greedy=True, eval_log_path=None):
            policy = 'full' if greedy else 'adaptive'
            episodes = []
            for episode_id, scene in enumerate(eval_config.values()):
                result = evaluate_episode(self, env, scene, policy=policy,
                                           max_steps=protocol['online_max_steps'])
                result.update(schema_version=1, seed=args.seed,
                              timestep=self.current_timestep,
                              learning_timestep=self.learning_timestep,
                              policy=policy, episode=episode_id,
                              protocol=config['online_protocol'], smoke=args.smoke)
                append_jsonl(metrics / 'online_episodes.jsonl', result)
                episodes.append(result)
            successful = [ep for ep in episodes if ep['success']]
            row = dict(schema_version=1, seed=args.seed,
                       timestep=self.current_timestep,
                       learning_timestep=self.learning_timestep,
                       policy=policy, protocol=config['online_protocol'], smoke=args.smoke,
                       episode_count=len(episodes),
                       mean_raw_return=float(np.mean([ep['raw_return'] for ep in episodes])),
                       mean_discounted_return=float(np.mean([ep['discounted_return'] for ep in episodes])),
                       success_rate=float(np.mean([ep['success'] for ep in episodes])),
                       collision_rate=float(np.mean([ep['collision'] for ep in episodes])),
                       timeout_rate=float(np.mean([ep['timeout'] for ep in episodes])),
                       mean_time_success=float(np.mean([ep['time'] for ep in successful])) if successful else None,
                       mean_energy_success=float(np.mean([ep['energy'] for ep in successful])) if successful else None,
                       robust_epsilon=self._last_robust_epsilon)
            append_jsonl(metrics / 'online_evaluations.jsonl', row)
            print('Online evaluation:', json.dumps(row, allow_nan=False))

    agent = RecordedAgent(base_env.get_state_space_dimension(),
                          base_env.get_action_space_dimension(), device=args.device,
                          seed=protocol['agent_seed'], **config['agent_parameters'])
    if args.no_checkpoint:
        agent.qnetwork_local.save = lambda directory: None
    manifest = dict(protocol=protocol,
                    started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(ROOT.rglob('*.py'))
                                   if not any(part in ('runs', '__pycache__') for part in p.relative_to(ROOT).parts)},
                    dependencies={'python': sys.version})
    for distribution in ['numpy', 'scipy', 'torch', 'gym', 'matplotlib']:
        try:
            manifest['dependencies'][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            manifest['dependencies'][distribution] = None
    write_json(output / 'run_manifest.json', manifest)
    write_json(output / 'effective_config.json', config)
    before = [p.detach().clone() for p in agent.qnetwork_local.parameters()] if args.smoke else None
    started = time.monotonic()
    try:
        agent.learn(config['total_timesteps'], train_env, eval_env, configs,
                    config['eval_freq'], str(output), verbose=not args.smoke)
        train_env.record(False, 'training budget exhausted')
        if not args.no_checkpoint:
            agent.qnetwork_local.save(str(output))
        changed = bool(any(not torch.equal(a, b) for a, b in zip(before, agent.qnetwork_local.parameters()))) if before is not None else None
        summary = dict(smoke=args.smoke, actual_training_actions=base_env.total_timesteps,
                       gradient_updates=agent.gradient_updates,
                       parameters_changed=changed,
                       elapsed_seconds=time.monotonic() - started,
                       model_weights_saved=not args.no_checkpoint,
                       resumable_training_checkpoint=False)
        write_json(output / 'training_summary.json', summary)
        print('Training finished:', json.dumps(summary))
    finally:
        base_env.close()
        eval_env.close()


def main():
    args, config, protocol, execute = parse_args()
    if not execute:
        return
    with (args.output_dir / 'console.log').open('a') as console:
        with contextlib.redirect_stdout(Tee(sys.stdout, console)), contextlib.redirect_stderr(Tee(sys.stderr, console)):
            run_training(args, config, protocol)


if __name__ == '__main__':
    main()
