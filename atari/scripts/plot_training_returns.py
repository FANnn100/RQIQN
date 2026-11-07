#!/usr/bin/env python3
"""Plot RQIQN training returns on common iterations without smoothing.

Inputs are run directories, training JSONL files, or per-run CSVs exported by
export_training_returns.py. Every run receives equal weight, irrespective of
its episode count. A single run has no uncertainty band; multiple runs use the
sample standard deviation across run-level means, not a confidence interval.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

from export_training_returns import load_training_records


def aggregate_runs(inputs, allow_legacy_pickle=False):
    series, sources, labels = [], [], []
    for value in inputs:
        rows, source = load_training_records(value, allow_legacy_pickle)
        if source in sources:
            raise ValueError('The same run was supplied more than once: %s' % source)
        series.append({row['iteration']: row['train_average_return'] for row in rows})
        sources.append(source)
        labels.append(rows[0]['run_id'])
    if not series:
        raise ValueError('At least one training run is required')
    common = sorted(set.intersection(*(set(run) for run in series)))
    if not common:
        raise ValueError('Input runs have no common iteration')
    aligned = []
    for iteration in common:
        values = [run[iteration] for run in series]
        aligned.append({'iteration': iteration,
                        **{'run_%d' % index: value for index, value in enumerate(values)},
                        'mean': statistics.mean(values),
                        'sample_std': statistics.stdev(values) if len(values) > 1 else '',
                        'n_runs': len(values)})
    metadata = {
        'metric': 'train_average_return',
        'metric_definition': 'Raw undiscounted training episode return, averaged within each iteration',
        'aggregation': 'Equal-weight mean of run-level iteration averages',
        'alignment': 'Intersection of completed iteration indices; no interpolation',
        'smoothing': None,
        'uncertainty': 'sample standard deviation across runs (ddof=1)' if len(series) > 1 else None,
        'n_runs': len(series),
        'common_iteration_count': len(common),
        'inputs': [{'source': str(source), 'run_id': label,
                    'iteration_count': len(run),
                    'iterations_not_plotted': sorted(set(run) - set(common))}
                   for source, label, run in zip(sources, labels, series)],
    }
    return aligned, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output-prefix', type=Path, required=True,
                        help='Outputs PREFIX.pdf, PREFIX.png, PREFIX.csv, PREFIX.json')
    parser.add_argument('--label', default='RQIQN')
    parser.add_argument('--title', default='Alien')
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--allow-legacy-pickle', action='store_true')
    args = parser.parse_args()
    aligned, metadata = aggregate_runs(args.runs, args.allow_legacy_pickle)
    prefix = args.output_prefix.expanduser()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {extension: Path(str(prefix) + '.' + extension)
             for extension in ('pdf', 'png', 'csv', 'json')}
    sources = {Path(value['source']).resolve() for value in metadata['inputs']}
    if any(path.resolve() in sources for path in paths.values()):
        raise SystemExit('Plot output must not replace an input training record')
    for source in metadata['inputs']:
        if source['iterations_not_plotted']:
            print('Only common iterations plotted; omitted %d iterations from %s'
                  % (len(source['iterations_not_plotted']), source['run_id']), file=sys.stderr)
    with paths['csv'].open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aligned[0]))
        writer.writeheader()
        writer.writerows(aligned)
    metadata.update(label=args.label, title=args.title,
                    plotted_values_csv=str(paths['csv']))
    paths['json'].write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n',
                             encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    iterations = [row['iteration'] for row in aligned]
    means = [row['mean'] for row in aligned]
    count = metadata['n_runs']
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    if count > 1:
        axis.fill_between(iterations,
                          [row['mean'] - row['sample_std'] for row in aligned],
                          [row['mean'] + row['sample_std'] for row in aligned],
                          color='#7151a1', alpha=.2)
        label = '%s: mean ± sample SD (n=%d)' % (args.label, count)
    else:
        label = '%s (n=1)' % args.label
    axis.plot(iterations, means, color='#7151a1', linewidth=1.7, label=label)
    axis.set(xlabel='Iteration', ylabel='Training episode return', title=args.title)
    axis.grid(alpha=.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(paths['pdf'])
    fig.savefig(paths['png'], dpi=args.dpi)
    plt.close(fig)
    print('Training returns only; no smoothing, interpolation, or baseline curves.')
    print('Runs:', count, 'Common iterations:', len(aligned))
    for extension, path in paths.items():
        print(extension.upper() + ':', path)


if __name__ == '__main__':
    main()
