# Implementation details

## RQIQN correction and gradients

The Alien configuration perturbs the current network's quantiles for the sampled replay action. For Wasserstein order 2, the correction is

$$
\Delta(\tau;s,a,t)=\frac{\epsilon_{\mathrm{eff}}(s,a,t)}{2}
\frac{2\tau-1}{\sqrt{\tau^2+(1-\tau)^2}},
$$

with

$$
\epsilon_t=\frac{1}{1+\exp(k(t-t_0))},\qquad
\epsilon_{\mathrm{eff}}(s,a,t)=\epsilon_t
\left|\frac{1}{N}\sum_{i=1}^{N}Z_\theta(s,a;\tau_i)\right|.
$$

Here, `t` is the agent's training interaction counter, `k=1.2e-6`, `t0=3,750,000`, and `N=64`. There is no robust-schedule warmup in the experiment configuration. The implementation clips the exponent to [-60, 60] for numerical stability. It adds the correction with positive sign and uses check loss. This is both a time-dependent schedule and a state-action-dependent amplitude; describing the entire radius as only time dependent would omit the enabled state-action scale.

The selected-action mean is computed inside the differentiated loss. Its scale is not detached, so gradients can flow through the correction amplitude as well as the predicted quantiles. Bellman target quantiles and the sign indicator in the quantile loss use stop-gradient. Changing either treatment would change the optimization behavior.

The superclass constructor may log its own default argument `loss_type: huber`. The RQIQN update uses the separate subclass attribute `loss_type='check'`; `run_manifest.json` records that runtime value under `runtime_rqiqn.loss_type`.

## Environment and measured quantity

The environment is `AlienNoFrameskip-v0`, with sticky-action probability 0.25, frame skip 4, grayscale 84 × 84 observations, and four-frame stacking. An episode ends at game over or 27,000 agent interaction steps. Life loss does not terminate an episode. The original runner adds no random no-op start procedure.

Rewards passed to the agent and replay are clipped to [-1, 1]. The runner separately accumulates the original rewards for raw, undiscounted episode returns. `TrainRunner` runs only the training phase. The retained Gin evaluation-step and evaluation-epsilon settings do not cause a separate evaluation phase. Curves from these scripts must be identified as training returns.

Iterations contain at least 250,000 agent interaction steps because the runner checks the budget after completing an episode. The nominal total is 50M agent steps, corresponding to 200M frames at frame skip 4. Each record stores actual and cumulative agent steps. Its frame fields are the conventional `agent_steps * 4` conversion; partial frame skips at terminal transitions are not measured as separate ALE frame counts.

## Recording and aggregation

`metrics/training_returns.jsonl` contains one record per zero-based iteration. The recorder validates the mean of `train_episode_returns` against `train_average_return`. `wall_time_seconds` is the duration of that iteration, not cumulative elapsed time. Step totals use the recorded episode lengths.

CSV export prefers JSONL. Legacy pickle loading requires `--allow-legacy-pickle`, and the latest cumulative logger snapshot is read once to avoid counting earlier snapshots repeatedly. Use that option only for trusted local files.

For multiple independent runs, plotting uses the intersection of completed iteration indices and equally weights each run. It plots the mean of per-run iteration means and the sample standard deviation across runs (ddof=1). This standard deviation is not a confidence interval. A single run receives no uncertainty band. The plotted CSV and metadata JSON preserve the exact data and aggregation used by the figure.

## Seeds and checkpoint restoration

The default `--seed-mode agent` reproduces the original explicit agent/JAX seed setting. The optional `all` mode also sets Python, NumPy, ALE, and action-space seeds and records that choice. Historical runs did not preserve complete ALE and NumPy RNG states, so matching an agent seed does not promise exact historical trajectories or scores.

Resume requires a completed checkpoint and the original run manifest. Seed, seed mode, configuration hashes, checkpoint policy, and per-iteration budget must match; the requested number of iterations may increase. Restoration failure raises an error before existing metric records are changed. Successful restoration removes metric iterations at or beyond the restored start iteration before recording new iterations. The original manifest is retained and resume information is appended to `resume_history`.

The original Dopamine checkpoint restores parameters, optimizer state, replay, and training counters. It does not preserve all JAX, NumPy, and ALE RNG state, so resumed trajectories need not match an uninterrupted run bit for bit. `--no-checkpoint` runs cannot be resumed.

## Source provenance

The included `dopamine/` modules form the minimal training dependency closure extracted from `Dopamine_dynamic`. [core_source_manifest.json](core_source_manifest.json) records the source Git HEAD, original file hashes, packaged file hashes, and reasons for including or changing files. The original workspace contained uncommitted RQIQN changes; the Git HEAD alone therefore does not identify the complete implementation. Use the per-file hashes and shipped source together.

The original packaging made two compatibility changes: the experiment factory restricts agent creation to `jax_rqiqn` and removes unrelated imports; JAX DQN checkpoint restoration recognizes ordinary Flax Linen parameter dictionaries while retaining the saved parameter and optimizer PyTree types. The latter prevents current checkpoints from being mistaken for legacy parameter formats. It does not change the training loss, gradients, or network updates.

This English-language release additionally translates comments and docstrings in `atari_lib.py`, `checkpointer.py`, and `run_experiment.py`. Python ASTs, after excluding docstrings, are verified against the previous source package. RQIQN update code, networks, and replay algorithms retain their executable behavior. IQN/DQN parent classes and shared modules remain as inheritance and import dependencies.

Of the 31 vendored Python files, 27 remain byte-identical to their source-project counterparts. Four differ: the two compatibility files above, plus `atari_lib.py` and `checkpointer.py` for translation; `run_experiment.py` includes both the factory restriction and translated text. The manifest distinguishes the original source hashes from the packaged hashes.

The training launcher, JSONL recorder, export and plotting scripts, and tests are supplementary experiment tools. Original Dopamine copyright headers, [LICENSE](LICENSE), and [AUTHORS](AUTHORS) are retained. No historical records, checkpoint weights, or ROMs are included.

## Validation scope

The source package was checked with real Alien short training, replay sampling, gradient updates, metric recording, checkpoint saving and restoration, and CSV/PDF/PNG generation. The English-language relocation is checked with executable-AST comparison, the nine recording/resume/curve tests, a dry-run, and a scan for Chinese text. These checks do not constitute a new full-budget experiment or a fresh dependency installation.
