# Navigation RQIQN

This directory reproduces the historical RQIQN implementation used for the
navigation results in camera-ready Table 1. It includes training, fresh model
evaluation, per-seed aggregation and learning-curve generation. Historical
checkpoints, statistics and trajectories are not distributed.

## Environment

The implementation was validated with Python 3.8, PyTorch 2.4.0, Gym 0.21.0,
NumPy 1.24.4, SciPy 1.10.1 and Matplotlib 3.7.5; versions are listed in
`requirements.txt`. These are the available validation environment versions;
the original training environment was not fully recorded. Execute the commands
below from the repository root in an environment with these dependencies.

For a separate Python 3.8 environment, the following is an installation
candidate; a clean environment installation has not been tested:

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

Use training seeds 0–4 for the five-run protocol. `--output-dir` must be new or
empty. The frozen configuration is `configs/table1_rqiqn.json`. It uses a
3,000,000 loop limit, Adam with learning rate 1e-4, batch size 32, replay capacity
1,000,000, discount 0.99, one-step targets, updates every four learning steps,
10,000 initial exploration steps and target copies every 10,000 learning steps.
The source inclusive training loop is preserved and executes 3,000,001 actions.
The environment seed is the training seed; the agent seed is the training seed
plus 100.

The curriculum changes after 0, 1,000,000 and 2,000,000 environment actions to
4/6, 6/8 and 8/10 vortex-core/obstacle counts, with minimum start-to-goal
distances of 30, 35 and 40 m. The changes take effect at episode resets. The
robot integrates ten 0.1 s simulation steps per training action.

The robust adjustment uses p=2, the check loss, an initial scale of 1, and the
reverse logistic schedule with k=1.2e-6 and center 590,000 learning steps.
Its scale also multiplies the absolute sampled chosen-action quantile mean.
The mean is detached from the gradient. The implementation uses 8 current and
8 target quantile samples, 32 action-selection samples, and gradient clipping
at 0.5. The training updates retain the upstream navigation implementation.
CVaR is fixed to 1 during training.

The default online monitoring protocol, `historical-online30`, recreates the
effective archived configuration: 30 fixed scenes generated with seed 348,
each with 8 vortex cores and 5 obstacles, random start/goal positions and
initial heading/speed, ten integration steps per action, and no boundary
termination. Earlier training code attempted to assign three difficulties on
a Gym OrderEnforcing wrapper; those writes did not reach the simulator. The
explicit generator recreates the resulting scenes without relying on that
wrapper behavior. The archived Table 1 training run's saved configuration was
checked against this protocol. Full and adaptive CVaR policies are evaluated
in that order every 10,000 learning steps, starting at learning step 0.

`--online-protocol corrected-mixed3x10` instead generates ten scenes each with
4/6, 6/8 and 8/10 cores/obstacles, fixed (5,5) m to (45,45) m endpoints and
random initial heading/speed. This is a separately labelled new experiment;
its monitoring returns do not recreate the historical online curves.

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

Online monitoring consumes the same model and Python random streams as the
source implementation. Its two policies are run sequentially without reseeding.
The JSONL recorder adds no random draws and does not modify training updates.

## Final evaluation

```bash
python navigation/evaluate.py --model-dir runs/navigation/seed_0 --output-dir runs/navigation_eval/seed_0 --seed 0 --policies full,adaptive --dry-run
python navigation/evaluate.py --model-dir runs/navigation/seed_0 --output-dir runs/navigation_eval/seed_0 --seed 0 --policies full,adaptive --device cpu
python navigation/aggregate.py --runs runs/navigation_eval/seed_0 runs/navigation_eval/seed_1 runs/navigation_eval/seed_2 runs/navigation_eval/seed_3 runs/navigation_eval/seed_4 --output-dir runs/navigation_summary
```

Final evaluation uses 500 shared hard scenes generated with seed 348, 8 vortex
cores and 10 obstacles, obstacle radii uniform in [1,3] m, fixed endpoints
(5,5) m and (45,45) m, fixed initial heading pi/4 and zero initial speed, and
five 0.1 s integration steps per action. Each episode is capped at 1,000
actions. Full CVaR uses alpha=1; adaptive CVaR uses the original nearest-sonar
reflection distance rule, clipped above at alpha=1. Exploration epsilon is 0.
The source Table 1 consumer omitted restoration of the boundary flag, so
boundary termination is explicitly disabled in this historical protocol.

Table 1 return is the discounted sum of rewards with gamma=0.99. Time and
energy are averaged over successful episodes; the source energy metric is the
sum of normalized absolute acceleration and angular-rate commands. Aggregate
results give each training seed equal weight and report sample standard
deviation across seed-level means (ddof=1). A fresh model load is used for each
standalone evaluation policy, matching the separate historical evaluations.
The initial evaluator seed defaults to 2; loading the model then restores
the network constructor's seed, which governs subsequent quantile draws.
These values are recorded separately. When a training manifest is available,
the evaluator checks that `--seed` matches its training-seed label.

Aggregation writes `per_seed_metrics.csv`, `summary.csv`, `summary.json`, and
`table1.csv`. The Table 1 file reports the six paper metrics as mean ± sample
standard deviation, with success/collision/timeout rates converted to percent.
The general CSV and JSON summaries retain rates in [0,1].

## Curves and validation

```bash
python navigation/plot.py --runs runs/navigation/seed_0 runs/navigation/seed_1 runs/navigation/seed_2 runs/navigation/seed_3 runs/navigation/seed_4 --output-dir runs/navigation_curves
python navigation/train.py --output-dir /tmp/rqiqn_navigation_smoke --seed 0 --device cpu --smoke
python -B -m unittest discover -s navigation/tests -v
```

Plots use online evaluation means, equal seed weights and sample-standard-
deviation bands, without smoothing. PDF/PNG figures and the exact CSV/JSON
points are saved. The default metric is discounted return; raw return and
success/collision/timeout rates can be selected with `--metric`. Different
training or evaluation protocols must be plotted separately.

The smoke configuration performs 64 real training actions with a smaller
replay, batch size and online evaluation budget. Tests check actual parameter
updates, recording and plotting, effective historical and corrected layout
counts, and evaluation aggregation. Smoke artifacts are marked and are not
paper results. Tests use temporary directories; no smoke outputs are shipped.

## Source attribution

The simulator and IQN implementation derive from
[Distributional_RL_Navigation](https://github.com/RobustFieldAutonomyLab/Distributional_RL_Navigation)
by Robust Field Autonomy Lab (MIT license, retained in `LICENSE`). The source
project credits [BY571/IQN](https://github.com/BY571/IQN) for the IQN base.
RQIQN extensions are copied from the local `USV_RQIQN` workspace. Original
file hashes and the source commit are in `source_manifest.json`; unchanged
algorithm files are identified separately from the reduced package registries.

Please cite Xi Lin, John McConnell and Brendan Englot, “Robust Unmanned Surface
Vehicle Navigation with Distributional Reinforcement Learning,” IROS 2023,
pp. 6185–6191, doi:10.1109/IROS55552.2023.10342389, for the navigation simulator
and original distributional navigation implementation, together with the
RQIQN paper for its method and experiments.

## License

Original RQIQN extensions and reproduction scripts contributed by The RQIQN Authors are released under the [MIT License](../LICENSE). The upstream simulator and IQN implementation retain their [MIT license](LICENSE), copyright (c) 2023 Robust Field Autonomy Lab.
