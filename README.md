# RQIQN: Quantile Geometry Regularization for Distributional Reinforcement Learning

[Navigation](navigation/README.md) | [Atari example](atari/README.md) | [Citation](#citation)

Zhaofan Zhang, Minghao Yang, Rufeng Chen, Sihong Xie, Hui Xiong

NeurIPS 2026

RQIQN augments implicit quantile networks with a bounded, fraction-dependent correction motivated by Wasserstein distributionally robust quantile estimation. This repository provides two experimental components: risk-sensitive surface-vessel navigation in PyTorch and an Atari example in the Arcade Learning Environment using JAX/Dopamine.

## Method

The implementation follows three steps:

1. Sample current and target quantile fractions to construct distributional temporal-difference errors.
2. Apply a bounded type-2 Wasserstein correction, scaled by a reverse-logistic time schedule and the current state-action quantile mean.
3. Optimize the corrected quantile objective while retaining the implicit quantile representation.

The components preserve the implementations used in the respective experiments. Each component documents its training settings, gradient handling, evaluation layouts, and metric definitions.

## Repository layout

```text
RQIQN/
├── atari/                     # Atari example: training, recording, and plotting
│   ├── configs/               # Formal and smoke-test Gin configurations
│   ├── dopamine/              # Required Dopamine implementation
│   ├── scripts/               # Train, export CSV, and plot learning curves
│   └── tests/
└── navigation/                # Surface-vessel navigation
    ├── configs/               # Training and evaluation protocols
    ├── marinenav_env/         # Vessel dynamics, perception, and flow fields
    ├── thirdparty/IQN/        # RQIQN and required IQN implementation
    ├── train.py               # Train and record new experiments
    ├── evaluate.py            # Evaluate newly trained policies
    ├── aggregate.py           # Aggregate independent training seeds
    ├── plot.py                # Plot recorded learning curves
    └── tests/
```

## Setup

Use separate Python environments for the PyTorch navigation component and the JAX/TensorFlow Atari example. Each component documents its tested environment and installation requirements:

- [Navigation setup](navigation/README.md)
- [Atari example setup](atari/ENVIRONMENT.md)

The repository contains source code and configurations. Existing experiment statistics, trained weights, and Atari ROMs are not distributed. Training creates new records and model files in the output directory you select.

## Navigation

Train a navigation policy using the historical main-result configuration:

```bash
python navigation/train.py --output-dir runs/navigation_seed0 --seed 0 --device cpu
```

Evaluate it on the shared hard-mode layouts with full-support and adaptive CVaR policies:

```bash
python navigation/evaluate.py \
  --model-dir runs/navigation_seed0 \
  --output-dir outputs/navigation_seed0 \
  --seed 0 --policies full,adaptive
```

Repeat training and evaluation for seeds 0 through 4, then aggregate the five independent seeds:

```bash
python navigation/aggregate.py \
  --runs outputs/navigation_seed0 outputs/navigation_seed1 \
         outputs/navigation_seed2 outputs/navigation_seed3 \
         outputs/navigation_seed4 \
  --output-dir outputs/navigation_summary
```

The formal test protocol uses 500 layouts per seed, with 10 obstacles and 8 vortex cores. It is separate from the fixed 30-scene online learning evaluation. See the [navigation documentation](navigation/README.md) for the control periods, metric definitions, online scene protocols, and curve plotting.

## Atari example

Train one of the three independent RQIQN runs:

```bash
cd atari
python scripts/train_alien.py --output-dir ../runs/alien_seed0 --seed 0
```

After running seeds 0, 1, and 2, plot their recorded training returns:

```bash
python scripts/plot_training_returns.py \
  ../runs/alien_seed0 ../runs/alien_seed1 ../runs/alien_seed2 \
  --output-prefix ../outputs/alien_rqiqn
```

The default budget is approximately 200M Atari frames per run under sticky actions with probability 0.25. This example covers one game from the paper's Atari experiments. See the [Atari example documentation](atari/README.md) for CSV export, resuming, seeding, and smoke tests.

## Reproducibility

Each component records the resolved configuration, source provenance, dependency versions, and experiment seed. Learning records retain the episode returns needed to recompute the plotted averages. Multi-seed summaries give each independent training seed equal weight and report sample standard deviation.

The release has been checked with short real-environment runs and recording/plotting tests. These checks establish that the pipeline executes; they are not new full-budget experiments or evidence that a new seed will reproduce an individual published score exactly.

## Citation

```bibtex
@article{zhang2026quantile,
  title={Quantile Geometry Regularization for Distributional Reinforcement Learning},
  author={Zhang, Zhaofan and Yang, Minghao and Chen, Rufeng and Xie, Sihong and Xiong, Hui},
  journal={arXiv preprint arXiv:2605.08182},
  year={2026}
}
```

## Acknowledgements

Our implementation is based on the following open-source projects. We thank their authors for making their code publicly available and for their outstanding work:

1. [Dopamine](https://github.com/google/dopamine).
2. [Robust Unmanned Surface Vehicle Navigation with Distributional Reinforcement Learning](https://github.com/RobustFieldAutonomyLab/Distributional_RL_Navigation) by Xi Lin, John McConnell, and Brendan Englot.

## License

Original contributions by The RQIQN Authors, including RQIQN-specific implementations and experiment scripts, are released under the [MIT License](LICENSE).

Third-party material and files derived from it retain their existing copyright notices and applicable license terms. Dopamine-derived code retains the [Apache-2.0 license](atari/LICENSE), while the upstream navigation environment and IQN implementation retain their [MIT license](navigation/LICENSE).
