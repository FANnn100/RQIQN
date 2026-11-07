"""Portable iteration records for raw, undiscounted training returns."""
import json
import math
from pathlib import Path
import statistics


def _scalar(value):
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)


class IterationRecorder:
    def __init__(self, path, seed, resume_from=0, frame_skip=4):
        self.path = Path(path)
        self.seed = int(seed)
        self.frame_skip = int(frame_skip)
        existing = []
        if self.path.exists():
            existing = [json.loads(line) for line in self.path.read_text().splitlines()
                        if line.strip()]
        self.records = [row for row in existing if row['iteration'] < resume_from]
        if [row['iteration'] for row in self.records] != list(range(resume_from)):
            raise ValueError('Iteration records do not cover the restored checkpoint history.')
        self._write()

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n'
                                     for row in self.records))
        temporary.replace(self.path)

    def record(self, iteration, statistics_data, wall_time_seconds):
        if iteration != len(self.records):
            raise ValueError('Expected iteration {}, received {}.'.format(len(self.records), iteration))
        returns = [float(x) for x in statistics_data['train_episode_returns']]
        lengths = [int(x) for x in statistics_data['train_episode_lengths']]
        if not returns or len(returns) != len(lengths) or any(x <= 0 for x in lengths):
            raise ValueError('Expected matching nonempty episode returns and lengths.')
        measured = statistics.mean(returns)
        reported = _scalar(statistics_data['train_average_return'])
        if not math.isclose(measured, reported, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError('Episode returns do not reproduce train_average_return.')
        steps = sum(lengths)
        cumulative = steps + (self.records[-1]['cumulative_train_agent_steps']
                              if self.records else 0)
        row = dict(schema_version=1, game='Alien', agent='RQIQN', seed=self.seed,
                   iteration=int(iteration), return_key='train_average_return',
                   train_average_return=reported, train_episode_returns=returns,
                   train_episode_lengths=lengths, num_train_episodes=len(returns),
                   train_average_steps_per_second=_scalar(
                       statistics_data['train_average_steps_per_second']),
                   train_agent_steps=steps, cumulative_train_agent_steps=cumulative,
                   train_frames=steps * self.frame_skip,
                   cumulative_train_frames=cumulative * self.frame_skip,
                   wall_time_seconds=float(wall_time_seconds))
        self.records.append(row)
        self._write()
        return row
