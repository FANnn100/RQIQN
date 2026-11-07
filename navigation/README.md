# Navigation RQIQN

## Environment

The implementation was validated with Python 3.8, PyTorch 2.4.0, Gym 0.21.0,
NumPy 1.24.4, SciPy 1.10.1 and Matplotlib 3.7.5; versions are listed in
`requirements.txt`.

```bash
python -m pip install 'pip==23.0.1' 'setuptools==65.5.0' 'wheel==0.38.4'
python -m pip install --no-build-isolation -r navigation/requirements.txt
python -m pip check
```

Gym 0.21 contains legacy optional-dependency metadata that newer build tools
can reject, so the installer versions are specified explicitly. Scene
generation in this release configures the simulator directly; it does not
depend on Gym wrapper attribute shadowing.

## Training

```bash
python navigation/train.py --output-dir runs/navigation/seed_0 --seed 0 --device cpu --dry-run
python navigation/train.py --output-dir runs/navigation/seed_0 --seed 0 --device cuda:0
```

Each new run writes:

- `run_manifest.json` and `effective_config.json`: protocol, effective layouts,
  dependency versions and source hashes.
- `metrics/training_episodes.jsonl`: raw and discounted returns, lengths and
  terminal outcomes; an unfinished final episode is marked `complete=false`.
- `metrics/online_episodes.jsonl` and `online_evaluations.jsonl`: individual
  monitoring episodes and their aggregates for each policy and training step.
- `metrics/losses.jsonl`: sampled update losses and robust schedule values.
- `network_params.pth` and `constructor_params.json`: newly trained model
  weights in the source format. These are model files, not complete resumable
  training checkpoints. `--no-checkpoint` disables writing them.

## Evaluation

```bash
python navigation/evaluate.py --model-dir runs/navigation/seed_0 --output-dir runs/navigation_eval/seed_0 --seed 0 --policies full,adaptive --dry-run
python navigation/evaluate.py --model-dir runs/navigation/seed_0 --output-dir runs/navigation_eval/seed_0 --seed 0 --policies full,adaptive --device cpu
python navigation/aggregate.py --runs runs/navigation_eval/seed_0 runs/navigation_eval/seed_1 runs/navigation_eval/seed_2 runs/navigation_eval/seed_3 runs/navigation_eval/seed_4 --output-dir runs/navigation_summary
```

## Curves and validation

```bash
python navigation/plot.py --runs runs/navigation/seed_0 runs/navigation/seed_1 runs/navigation/seed_2 runs/navigation/seed_3 runs/navigation/seed_4 --output-dir runs/navigation_curves
python navigation/train.py --output-dir /tmp/rqiqn_navigation_smoke --seed 0 --device cpu --smoke
python -B -m unittest discover -s navigation/tests -v
```

## Source attribution

The simulator and IQN implementation derive from
[Distributional_RL_Navigation](https://github.com/RobustFieldAutonomyLab/Distributional_RL_Navigation)
by Robust Field Autonomy Lab (MIT license, retained in `LICENSE`). The source
project credits [BY571/IQN](https://github.com/BY571/IQN) for the IQN base.

Please cite Xi Lin, John McConnell and Brendan Englot, “Robust Unmanned Surface
Vehicle Navigation with Distributional Reinforcement Learning,” IROS 2023,
pp. 6185–6191, doi:10.1109/IROS55552.2023.10342389, for the navigation simulator
and original distributional navigation implementation, together with the
RQIQN paper for its method and experiments.
