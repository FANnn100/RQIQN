# RQIQN on Atari Alien

This component provides **RQIQN training on Alien**, with the experiment configuration, per-episode recording, CSV export, and learning-curve plotting. RQIQN adds a quantile correction to the current network output during distributional TD learning. The implementation uses the IQN architecture and uniform replay from Dopamine.

The release contains source code and configurations. Run the commands below to generate training records and curves; historical checkpoints, experiment statistics, and Atari ROMs are not distributed.

## Method

For each sampled state and action, RQIQN applies a Wasserstein correction to the predicted quantiles before computing the quantile regression loss. The Alien configuration uses Wasserstein order 2, check loss, a reverse-logistic time coefficient, and a scale given by the absolute mean of the selected action's predicted quantiles. The scale participates in differentiation.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the correction, gradient handling, return definition, and checkpoint behavior.

## Repository layout

```text
atari/
├── configs/                     # Alien experiment and smoke configurations
├── dopamine/                    # Required Dopamine and RQIQN modules
├── scripts/
│   ├── train_alien.py            # Training, resume, and manifest generation
│   ├── export_training_returns.py
│   └── plot_training_returns.py
├── tests/                       # Recording, resume guard, and curve tests
├── training_records.py          # Per-iteration JSONL recorder
├── core_source_manifest.json    # Source provenance and file hashes
├── ENVIRONMENT.md
└── IMPLEMENTATION.md
```

## Setup

Use **Python 3.8** for training. From the repository root:

```bash
cd atari
python -m pip install -r requirements-train-cpu.txt
python -m pip check
```

Provide an Alien ROM that ALE-Py can discover. The dependency file specifies a proposed CPU installation; a fresh installation has not been validated. The existing training environment, optional CUDA setup, and ROM check are documented in [ENVIRONMENT.md](ENVIRONMENT.md).

Plotting can run in a separate environment without Atari or the training dependencies:

```bash
python -m pip install -r requirements-plot.txt
```

## Configuration

The experiment settings are in [configs/alien_rqiqn.gin](configs/alien_rqiqn.gin).

| Setting | Value |
| --- | --- |
| Environment | `AlienNoFrameskip-v0`, sticky-action probability 0.25 |
| Observation | Grayscale 84 × 84, stack size 4, frame skip 4 |
| Episode termination | Game over or 27,000 agent steps; no life-loss termination |
| Training budget | 200 iterations × at least 250,000 agent steps, nominally 200M frames |
| Replay / minibatch | Uniform replay, capacity 1,000,000 / batch size 32 |
| Discount / update horizon | 0.99 / 3 |
| Replay warmup / update period | 20,000 / 4 agent steps |
| Target update period | 8,000 agent steps |
| Quantile samples | Current network 64, target network 64, action selection 32 |
| Optimizer | Adam, learning rate 5e-5, epsilon 3.125e-4 |
| Exploration | Final training epsilon 0.01, decay period 250,000 agent steps |
| RQIQN correction | Check loss, Wasserstein order 2, positive correction, initial radius coefficient 1 |
| Time coefficient | Reverse-logistic, k=1.2e-6, midpoint 3,750,000 agent steps |
| State-action scale | Absolute mean of selected-action quantiles, multiplier 1 |

Each iteration completes its final episode, so actual interaction counts can exceed the nominal budget. Records retain the actual counts. Returns are raw, undiscounted **training episode returns**, computed before reward clipping; this configuration runs a training-only runner.

## Training

Start one independent run per seed:

```bash
python scripts/train_alien.py --output-dir runs/alien_seed0 --seed 0
python scripts/train_alien.py --output-dir runs/alien_seed1 --seed 1
python scripts/train_alien.py --output-dir runs/alien_seed2 --seed 2
```

The default `--seed-mode agent` matches the original explicit agent/JAX seeding. Use `--seed-mode all` to also seed Python, NumPy, ALE, and the action space. GPU selection can be supplied with `CUDA_VISIBLE_DEVICES`.

Inspect the protocol without loading Atari or starting training:

```bash
python scripts/train_alien.py --output-dir runs/alien_seed0 --seed 0 --dry-run
```

Resume a run with its original settings:

```bash
python scripts/train_alien.py --output-dir runs/alien_seed0 --seed 0 --resume
```

Checkpoints are enabled by default. Add `--no-checkpoint` when resume support is unnecessary. See [IMPLEMENTATION.md](IMPLEMENTATION.md) for resume checks and RNG limitations.

## Recording and plotting

Training writes the following files to the selected run directory:

| Output | Contents |
| --- | --- |
| `metrics/training_returns.jsonl` | One row per iteration: raw episode returns and lengths, iteration mean, actual and cumulative interaction counts |
| `run_manifest.json` | Protocol, seeds, configuration and source hashes, dependency versions, runtime RQIQN and environment settings |
| `operative_config.gin` | Effective Gin configuration |
| `console.log` | Console output |
| `logs/log_*` | Dopamine's cumulative legacy logger snapshots |
| `checkpoints/` | Checkpoints generated during training |

Export one run and optionally retain episode details:

```bash
python scripts/export_training_returns.py runs/alien_seed0 \
  --output curves/seed0.csv \
  --episodes-output curves/seed0_episodes.jsonl
```

Plot independent runs:

```bash
python scripts/plot_training_returns.py \
  runs/alien_seed0 runs/alien_seed1 runs/alien_seed2 \
  --output-prefix curves/alien_rqiqn
```

The plotter also accepts one run, JSONL files, or exported CSV files. It produces PDF, PNG, the exact plotted CSV, and a JSON description of sources and aggregation. Runs align on their common completed iterations. The line is the equally weighted run mean and shading is the sample standard deviation across runs (ddof=1), with the run count reported. A single run has no uncertainty band. No smoothing, interpolation, or missing-iteration filling is applied.

New runs use JSONL directly. To read trusted local legacy `logs/log_*` files, explicitly add `--allow-legacy-pickle`; only the latest cumulative snapshot is loaded.

## Pipeline checks

The smoke configuration reduces replay capacity, warmup, quantile counts, and the interaction budget to exercise training and recording. Its output is not a paper experiment.

```bash
python scripts/train_alien.py --output-dir runs/smoke \
  --seed 0 --seed-mode all --smoke --no-checkpoint
python scripts/export_training_returns.py runs/smoke --output curves/smoke.csv
python scripts/plot_training_returns.py runs/smoke --output-prefix curves/smoke
python -m unittest discover -s tests
```

## Citation

If this implementation is useful in your research, please cite:

```bibtex
@inproceedings{zhang2026rqiqn,
  title={Quantile Geometry Regularization for Distributional Reinforcement Learning},
  author={Zhang, Zhaofan and Yang, Minghao and Chen, Rufeng and Xie, Sihong and Xiong, Hui},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Acknowledgements

This implementation builds on [Dopamine](https://github.com/google/dopamine) and its Implicit Quantile Network implementation. Original copyright notices, the Apache-2.0 [LICENSE](LICENSE), and [AUTHORS](AUTHORS) are retained. Source provenance and the small compatibility changes are documented in [IMPLEMENTATION.md](IMPLEMENTATION.md) and [core_source_manifest.json](core_source_manifest.json).

## License

Original contributions by The RQIQN Authors, including experiment scripts and supporting tools, are released under the [MIT License](../LICENSE). Dopamine-derived files retain their existing copyright notices and [Apache-2.0 license](LICENSE).
