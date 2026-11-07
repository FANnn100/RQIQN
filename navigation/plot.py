#!/usr/bin/env python3
"""Plot online navigation evaluation curves with equal weight per training seed."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def load_runs(paths):
    runs, signatures, seeds = [], [], []
    for path in paths:
        manifest = json.loads((path / 'run_manifest.json').read_text())
        protocol = manifest['protocol']
        signature = {key: protocol[key] for key in (
            'config', 'smoke', 'quantile_samples', 'scale_gradient',
            'update_implementation', 'online_max_steps', 'online_episode_count')}
        signatures.append(signature)
        seeds.append(protocol['seed'])
        records = [json.loads(line) for line in
                   (path / 'metrics/online_evaluations.jsonl').read_text().splitlines() if line.strip()]
        indexed = {}
        for row in records:
            key = row['policy'], row['timestep']
            if key in indexed:
                raise ValueError('Duplicate online evaluation in ' + str(path))
            indexed[key] = row
        runs.append(indexed)
    if len(set(seeds)) != len(seeds):
        raise ValueError('Each run must have a distinct training seed.')
    if any(signature != signatures[0] for signature in signatures):
        raise ValueError('Training protocols differ; plot them in separate figures.')
    return runs, signatures[0], seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--metric', default='mean_discounted_return', choices=[
        'mean_discounted_return', 'mean_raw_return', 'success_rate',
        'collision_rate', 'timeout_rate'])
    args = parser.parse_args()
    runs, protocol, seeds = load_runs(args.runs)
    common = set.intersection(*(set(run) for run in runs))
    if not common:
        parser.error('No evaluation steps are shared by all runs.')
    rows = []
    for policy, timestep in sorted(common):
        values = [float(run[(policy, timestep)][args.metric]) for run in runs]
        rows.append(dict(policy=policy, timestep=timestep, mean=statistics.mean(values),
                         sample_std=statistics.stdev(values) if len(values) > 1 else None,
                         seed_count=len(values)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'curve.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / 'curve.json').write_text(json.dumps(dict(
        metric=args.metric, aggregation='Equal seed weight; sample standard deviation (ddof=1).',
        seeds=seeds, smoke=protocol['smoke'], protocol=protocol, points=rows), indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for policy in sorted({row['policy'] for row in rows}):
        points = [row for row in rows if row['policy'] == policy]
        x = np.array([row['timestep'] for row in points])
        y = np.array([row['mean'] for row in points])
        line, = ax.plot(x, y, label='RQIQN ({})'.format(policy))
        if len(seeds) > 1:
            std = np.array([row['sample_std'] for row in points])
            ax.fill_between(x, y - std, y + std, color=line.get_color(), alpha=0.18)
    labels = dict(mean_discounted_return='Discounted online evaluation return',
                  mean_raw_return='Undiscounted online evaluation return',
                  success_rate='Success rate', collision_rate='Collision rate',
                  timeout_rate='Timeout rate')
    ax.set_xlabel('Training action index (historical counter)')
    ax.set_ylabel(labels[args.metric])
    ax.set_title('Pipeline smoke test' if protocol['smoke'] else
                 protocol['config']['online_protocol'])
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / 'curve.pdf')
    fig.savefig(args.output_dir / 'curve.png', dpi=200)
    plt.close(fig)
    print('Saved curve.csv, curve.json, curve.pdf and curve.png to', args.output_dir)


if __name__ == '__main__':
    main()
