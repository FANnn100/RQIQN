#!/usr/bin/env python3
"""Export one run's raw training returns without smoothing or seed aggregation.

The preferred source is RUN_DIR/metrics/training_returns.jsonl. Legacy cumulative
Dopamine logs/log_N snapshots are read only with --allow-legacy-pickle; only the
highest-numbered snapshot is used. Never load pickle files from untrusted sources.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys


CSV_FIELDS = (
    'iteration', 'train_average_return', 'num_episodes', 'train_agent_steps',
    'cumulative_train_agent_steps', 'train_frames', 'cumulative_train_frames',
    'train_average_steps_per_second', 'wall_time_seconds', 'seed', 'game',
    'agent', 'run_id', 'source_file',
)


def scalar(value):
    """Unwrap scalar fields in Dopamine's legacy list-based statistics."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError('Expected one scalar statistic, got %r' % (value,))
        return scalar(value[0])
    return value.item() if hasattr(value, 'item') else value


def finite_number(value, name):
    value = float(scalar(value))
    if not math.isfinite(value):
        raise ValueError('%s must be finite, got %r' % (name, value))
    return value


def integer(value, name):
    number = finite_number(value, name)
    if number != int(number) or number < 0:
        raise ValueError('%s must be a nonnegative integer' % name)
    return int(number)


def select_source(path, allow_legacy_pickle=False):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if path.suffix.lower() in ('.jsonl', '.csv'):
            return path
        if allow_legacy_pickle:
            return path
        raise ValueError('Legacy pickle input requires --allow-legacy-pickle')
    for candidate in (path / 'metrics/training_returns.jsonl',
                      path / 'training_returns.jsonl'):
        if candidate.is_file():
            return candidate
    log_dir = path / 'logs' if (path / 'logs').is_dir() else path
    candidates = []
    for candidate in log_dir.iterdir():
        match = re.fullmatch(r'log_(\d+)', candidate.name)
        if match and candidate.is_file():
            candidates.append((int(match.group(1)), candidate))
    if candidates:
        if not allow_legacy_pickle:
            raise ValueError('Only legacy pickle logs found in %s; pass '
                             '--allow-legacy-pickle for trusted local logs' % path)
        return max(candidates, key=lambda pair: pair[0])[1]
    raise FileNotFoundError('No training_returns.jsonl or logs/log_N found in %s' % path)


def read_records(path):
    if path.suffix.lower() == '.jsonl':
        records = []
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError('%s:%d: invalid JSON' % (path, line_number)) from error
        return records
    if path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    import pickle
    with path.open('rb') as stream:
        snapshot = pickle.load(stream)
    if not isinstance(snapshot, dict):
        raise ValueError('Legacy log snapshot must be a dictionary')
    records = []
    for key, values in snapshot.items():
        match = re.fullmatch(r'iteration_(\d+)', str(key))
        if not match:
            continue
        record = dict(values)
        record['iteration'] = int(match.group(1))
        records.append(record)
    return records


def normalize_record(record, source_file, run_id, frame_skip):
    if not isinstance(record, dict):
        raise ValueError('Every training record must be a JSON object')
    iteration = integer(record['iteration'], 'iteration')
    if 'train_average_return' not in record:
        raise ValueError('Iteration %d has no train_average_return; evaluation '
                         'returns are not accepted' % iteration)
    average = finite_number(record['train_average_return'], 'train_average_return')
    row = {field: '' for field in CSV_FIELDS}
    row.update(iteration=iteration, train_average_return=average,
               run_id=record.get('run_id') or run_id, source_file=str(source_file))
    returns = record.get('train_episode_returns')
    lengths = record.get('train_episode_lengths')
    if returns is not None:
        returns = [finite_number(value, 'train_episode_returns') for value in returns]
        row['num_episodes'] = len(returns)
        if returns and not math.isclose(statistics.mean(returns), average,
                                       rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError('Iteration %d: recorded training return differs '
                             'from the mean of train_episode_returns' % iteration)
        row['train_episode_returns'] = returns
    elif record.get('num_episodes') not in (None, ''):
        row['num_episodes'] = integer(record['num_episodes'], 'num_episodes')
    if lengths is not None:
        lengths = [integer(value, 'train_episode_lengths') for value in lengths]
        if returns is not None and len(lengths) != len(returns):
            raise ValueError('Iteration %d: episode length/return counts differ' % iteration)
        row['train_episode_lengths'] = lengths
        row['train_agent_steps'] = sum(lengths)
    for key in ('train_agent_steps', 'cumulative_train_agent_steps',
                'train_frames', 'cumulative_train_frames'):
        if record.get(key) not in (None, ''):
            value = integer(record[key], key)
            if key == 'train_agent_steps' and row[key] != '' and row[key] != value:
                raise ValueError('Iteration %d: train_agent_steps differs from '
                                 'the sum of episode lengths' % iteration)
            row[key] = value
    if row['train_agent_steps'] != '':
        expected_frames = row['train_agent_steps'] * frame_skip
        if row['train_frames'] != '' and row['train_frames'] != expected_frames:
            raise ValueError('Iteration %d: train_frames disagrees with frame skip' % iteration)
        row['train_frames'] = expected_frames
    for key in ('train_average_steps_per_second', 'wall_time_seconds'):
        if record.get(key) not in (None, ''):
            row[key] = finite_number(record[key], key)
    for key in ('seed', 'game', 'agent'):
        if record.get(key) not in (None, ''):
            row[key] = scalar(record[key])
    return row


def load_training_records(path, allow_legacy_pickle=False, frame_skip=4):
    if frame_skip < 1:
        raise ValueError('frame_skip must be positive')
    requested = Path(path).expanduser().resolve()
    source = select_source(requested, allow_legacy_pickle)
    run_id = (requested.name if requested.is_dir() else
              source.parent.parent.name if source.parent.name in ('metrics', 'logs')
              else source.stem)
    by_iteration = {}
    repeated = set()
    for record in read_records(source):
        row = normalize_record(record, source, run_id, frame_skip)
        if row['iteration'] in by_iteration:
            repeated.add(row['iteration'])
        by_iteration[row['iteration']] = row
    if repeated:
        print('Warning: repeated iterations in %s; the last record was retained: %s'
              % (source, sorted(repeated)), file=sys.stderr)
    rows = [by_iteration[key] for key in sorted(by_iteration)]
    if not rows:
        raise ValueError('No training iterations found in %s' % source)
    # Cumulative counters may be inferred only for an uninterrupted history from 0.
    running_steps = 0
    uninterrupted = True
    for expected_iteration, row in enumerate(rows):
        uninterrupted &= row['iteration'] == expected_iteration and row['train_agent_steps'] != ''
        if uninterrupted:
            running_steps += row['train_agent_steps']
            if row['cumulative_train_agent_steps'] not in ('', running_steps):
                raise ValueError('Iteration %d: inconsistent cumulative agent steps'
                                 % row['iteration'])
            row['cumulative_train_agent_steps'] = running_steps
        if row['cumulative_train_agent_steps'] != '':
            expected_frames = row['cumulative_train_agent_steps'] * frame_skip
            if row['cumulative_train_frames'] not in ('', expected_frames):
                raise ValueError('Iteration %d: inconsistent cumulative frames'
                                 % row['iteration'])
            row['cumulative_train_frames'] = expected_frames
    return rows, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Run directory, JSONL file, or exported per-run CSV')
    parser.add_argument('--output', type=Path, required=True, help='Output per-iteration CSV')
    parser.add_argument('--episodes-output', type=Path,
                        help='Optional JSONL retaining raw per-episode return/length lists')
    parser.add_argument('--frame-skip', type=int, default=4)
    parser.add_argument('--allow-legacy-pickle', action='store_true')
    args = parser.parse_args()
    rows, source = load_training_records(args.run, args.allow_legacy_pickle, args.frame_skip)
    if args.output.expanduser().resolve() == source:
        raise SystemExit('Output must differ from the input training record')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    if args.episodes_output:
        if args.episodes_output.expanduser().resolve() in (source, args.output.expanduser().resolve()):
            raise SystemExit('Episode output must differ from input and CSV output')
        args.episodes_output.parent.mkdir(parents=True, exist_ok=True)
        with args.episodes_output.open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    print('Source:', source)
    print('Metric: raw undiscounted training episode return; no evaluation returns')
    print('Iterations:', len(rows), '(%d--%d)' % (rows[0]['iteration'], rows[-1]['iteration']))
    print('CSV:', args.output)


if __name__ == '__main__':
    main()
