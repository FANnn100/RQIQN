# Experiment protocols and implementation details

## Paper version and release scope

The release was checked against the revised camera-ready paper,
*Quantile Geometry Regularization for Distributional Reinforcement Learning*,
in `Camera_Ready___RQIQN___NeurIPS_2026.pdf`. The checked PDF has SHA-256:

```text
a2b06a6a1d79f0b53af73e605cd8368ee8486824d2aa932384b2891e68358547
```

This repository covers navigation in Section 4.1 and Alien from Section 4.2.
It provides the experiment implementations, fixed configurations, training,
evaluation, recording, and plotting tools. Navigation retains the upstream
navigation implementation. The chain experiments and the other eight Atari
games are outside this release.

## Navigation: main-result configuration

The navigation configuration corresponds to the `RQIQN_dynamic_1chu5` run
family used for Table 1. The original local episode records were independently
checked: all 30 reported mean/standard-deviation entries match recomputation.
The original records and weights are not distributed.

| Setting | Value |
| --- | --- |
| Training seeds | 0, 1, 2, 3, 4 |
| Training loop limit | 3,000,000; the source inclusive loop executes 3,000,001 actions |
| Replay / batch | 1,000,000 / 32 |
| Discount / update horizon | 0.99 / 1 |
| Optimizer | Adam, learning rate 1e-4 |
| Initial replay collection | 10,000 interactions |
| Current / target TD fractions | 8 / 8 |
| Action-selection fractions | 32 |
| RQIQN loss / Wasserstein order | Check / 2 |
| Initial time coefficient | 1 |
| Reverse-logistic slope / midpoint | 1.2e-6 / 590,000 learning steps |
| Gradient norm clipping | 0.5 |

The revised main text reports the state-action mean scaling, and revised
Appendix Table 3 reports the active navigation sample counts, 8/8/32.

Training uses ten 0.1 s simulator integration steps per action. The curriculum
changes at episode resets after 0, 1M, and 2M training interactions:

| Stage | Vortex cores | Obstacles | Minimum start-goal distance |
| --- | --- | --- | --- |
| 0 to 1M | 4 | 6 | 30 m |
| 1M to 2M | 6 | 8 | 35 m |
| 2M onward | 8 | 10 | 40 m |

## Navigation: formal evaluation

Main-text Table 1 uses the following protocol:

| Setting | Value |
| --- | --- |
| Trained seeds | 5 |
| Episodes per seed and policy | 500 |
| Shared layout seed | 348 |
| Obstacles / vortex cores | 10 / 8 |
| Obstacle radii | Uniform in [1, 3] m |
| Start / goal | (5, 5) / (45, 45) m |
| Initial heading / forward speed | pi/4 / 0 |
| Integration steps per action | 5, giving 0.5 s per decision |
| Action cap | 1,000 per episode |
| Workspace boundary termination | Disabled |
| Exploration epsilon | 0 |
| Policies | Full quantile support and adaptive CVaR |

All 500 generated formal scene dictionaries match the archived evaluation
configuration exactly. Their canonical JSON hash is recorded by the evaluator,
so independent training seeds can be checked against the same layouts.

Return is the reward sum discounted by 0.99 per decision step. Completion time
and energy are first averaged over successful episodes within each seed.
Across seeds, each trained policy receives equal weight and uncertainty is
reported as sample standard deviation, ddof=1. The aggregation script exports
the six paper metrics in `table1.csv`, converting rates to percentages.

Each evaluation policy starts from a fresh model load. The model constructor's
saved seed affects quantile sampling; the evaluator records it separately from
its initial agent seed and any explicit action-seed override. Epsilon zero
disables exploration but does not remove sampled-quantile randomness.

## Navigation: online monitoring

The fixed 30-scene monitoring set used during training is separate from the
500-scene hard-mode test. Its archived effective protocol consists of 30 scenes
with 5 obstacles and 8 cores, randomized start/goal positions and initial vessel
states, a 1.0 s decision period, and disabled boundary termination.

Earlier attribute assignments on Gym's `OrderEnforcing` wrapper did not reach
the simulator. The released `historical-online30` generator recreates the
effective configuration directly, and all 30 generated dictionaries match the
training archive. It does not depend on wrapper attribute assignment.

The optional `corrected-mixed3x10` protocol generates ten scenes each with
4/6, 6/8, and 8/10 cores/obstacles and fixed endpoints. It is a separately
labelled experiment; its returns are not the historical online curve. Online
monitoring results and formal test results should be identified separately.

## State-action scaling and gradients

For both navigation and Alien, the time factor is

```text
epsilon_t = epsilon_0 / (1 + exp(k * (t - t0)))
```

The correction amplitude additionally multiplies the absolute mean of the
selected current-action quantiles, with scale coefficient 1:

```text
epsilon_effective(t, s, a)
    = epsilon_t * abs(mean_i Z_theta(s, a, tau_i))
```

Navigation detaches the mean factor before computing the correction. Alien
allows gradients through it. These task-specific differentiation rules are
preserved in the source. Navigation counts learning steps after the initial
10,000 interactions; Alien uses the training-interaction counter including
replay warm-up.

For p=2, the helper's `(2*tau-1)` correction is added to the current prediction.
In the target-minus-prediction residual it is equivalent to the paper's
target-side `(1-2*tau)` expression.

## Simulator representation and metrics

The 26-dimensional navigation observation contains two vessel-velocity
components, two relative-goal coordinates, and eleven frontal sonar-hit
coordinate pairs. Adaptive CVaR derives obstacle proximity from those points.
The simulator uses idealized ray intersections without separately added
Gaussian observation or action noise.

The nominal 50 m by 50 m domain constrains scene placement. Boundary termination
is explicitly disabled in the released historical protocols. The vortex
placement parameter `v_rel_max=1` is used for placement constraints; the
superposed velocity field is not globally clipped to 1 m/s. The recorded energy
metric is normalized control effort, accumulated from absolute acceleration
and turn-rate commands, rather than physical joules.

## Alien

Alien uses 200 iterations of at least 250,000 agent steps, nominally 200M
frames, with sticky-action probability 0.25. Its current/target/action quantile
counts are 64/64/32. The reverse-logistic settings are initial coefficient 1,
k=1.2e-6, and midpoint 3,750,000 agent training steps.

Rewards supplied for learning are clipped to [-1,1]. Recorded curves use raw,
undiscounted training episode returns from the training-only runner. Actual
interaction counts are retained because iterations finish their final episode.
See [the Alien implementation notes](../atari/IMPLEMENTATION.md) for replay,
seeding, checkpoint restoration, and aggregation details.

## Validation scope

The release passes 19 tests covering real short navigation training with
parameter updates, evaluation with newly generated weights, return accounting,
layout generation, seed aggregation, recording, and plotting. Alien was also
checked with real short training, checkpoint save/restore, and curve export.
The English relocation preserves executable source behavior, as checked by
source hashes and executable-AST comparison.

These checks are pipeline validation, not new full-budget experiments. No
historical weights, statistics, trajectories, or ROMs are included. Fresh
dependency installation has not been validated; tested environments and
installation candidates are documented in each component.
