# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Robust Quantile IQN (RQIQN) agent.

Extends IQN with distributionally robust quantile regression perturbation.
Based on the Wasserstein DRO quantile regression theory, a location adjustment
delta_s is added to the current network's quantile outputs before computing
the quantile Huber loss.

For p = inf (q = 1):  delta_s(tau) = epsilon * (2*tau - 1)
For p = 2   (q = 2):  delta_s(tau) = (epsilon/2) * (2*tau - 1) / sqrt(tau^2 + (1-tau)^2)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math

from dopamine.jax import networks
from dopamine.jax.agents.implicit_quantile import implicit_quantile_agent
from dopamine.metrics import statistics_instance
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf


def robust_delta_s(quantiles, robust_epsilon, wasserstein_p):
  """Compute the robust location adjustment delta_s for each quantile.

  Args:
    quantiles: jnp array of shape (batch_size, num_tau_samples, 1),
      quantile levels tau in [0, 1].
    robust_epsilon: float or jnp array broadcastable to quantiles, the current
      perturbation magnitude.
    wasserstein_p: float, the Wasserstein order (2.0 or inf).

  Returns:
    delta_s: jnp array of same shape as quantiles.
  """
  tau = quantiles
  sign_term = 2.0 * tau - 1.0

  # p = inf (q = 1): delta_s = epsilon * (2*tau - 1)
  delta_s_inf = robust_epsilon * sign_term

  # p = 2 (q = 2): delta_s = (epsilon/2) * (2*tau - 1) / sqrt(tau^2 + (1-tau)^2)
  denom = jnp.sqrt(tau ** 2 + (1.0 - tau) ** 2)
  denom = jnp.maximum(denom, 1e-8)  # avoid division by zero
  delta_s_2 = (robust_epsilon / 2.0) * sign_term / denom

  # Branch on wasserstein_p (static, so this is resolved at trace time)
  delta_s = jnp.where(wasserstein_p == 2.0, delta_s_2, delta_s_inf)
  return delta_s


@functools.partial(
    jax.vmap,
    in_axes=(None, None, None, 0, 0, 0, None, None, None, None, 0),
    out_axes=(0, 0))
def target_quantile_values(network_def, online_params, target_params,
                           next_states, rewards, terminals,
                           num_tau_prime_samples, num_quantile_samples,
                           cumulative_gamma, double_dqn, rng):
  """Build the target for return values at given quantiles.

  Identical to the IQN version - target values are not perturbed.
  """
  rewards = jnp.tile(rewards, [num_tau_prime_samples])
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  gamma_with_terminal = jnp.tile(gamma_with_terminal, [num_tau_prime_samples])
  rng, rng1, rng2 = jax.random.split(rng, num=3)
  if double_dqn:
    outputs_action = network_def.apply(online_params,
                                       next_states,
                                       num_quantiles=num_quantile_samples,
                                       rng=rng1)
  else:
    outputs_action = network_def.apply(target_params,
                                       next_states,
                                       num_quantiles=num_quantile_samples,
                                       rng=rng1)
  target_quantile_values_action = outputs_action.quantile_values
  target_q_values = jnp.squeeze(
      jnp.mean(target_quantile_values_action, axis=0))
  next_qt_argmax = jnp.argmax(target_q_values)
  next_qt_argmax = jnp.tile(next_qt_argmax, [num_tau_prime_samples])
  next_state_target_outputs = network_def.apply(
      target_params,
      next_states,
      num_quantiles=num_tau_prime_samples,
      rng=rng2)
  target_quantile_vals = (
      jax.vmap(lambda x, y: x[y])(next_state_target_outputs.quantile_values,
                                  next_qt_argmax))
  target_quantile_vals = rewards + gamma_with_terminal * target_quantile_vals
  return rng, jax.lax.stop_gradient(target_quantile_vals[:, None])


@functools.partial(
    jax.jit,
    static_argnums=(0,   # network_def
                    3,   # optimizer
                    10,  # num_tau_samples
                    11,  # num_tau_prime_samples
                    12,  # num_quantile_samples
                    13,  # cumulative_gamma
                    14,  # double_dqn
                    15,  # kappa
                    17,  # wasserstein_p
                    18,  # use_check_loss
                    19,  # use_state_action_mean_scaling
                    20,  # state_action_mean_use_abs
                    21)) # state_action_mean_use_decay_multiplier
def train_rqiqn(network_def,          # 0  static
                online_params,         # 1
                target_params,         # 2
                optimizer,             # 3  static
                optimizer_state,       # 4
                states,                # 5
                actions,               # 6
                next_states,           # 7
                rewards,               # 8
                terminals,             # 9
                num_tau_samples,        # 10 static
                num_tau_prime_samples,  # 11 static
                num_quantile_samples,  # 12 static
                cumulative_gamma,      # 13 static
                double_dqn,            # 14 static
                kappa,                 # 15 static
                robust_epsilon,        # 16 dynamic (decays each step)
                wasserstein_p,         # 17 static
                use_check_loss,        # 18 static
                use_state_action_mean_scaling,  # 19 static
                state_action_mean_use_abs,      # 20 static
                state_action_mean_use_decay_multiplier,  # 21 static
                state_action_mean_scale_coeff,  # 22 dynamic
                robust_epsilon_init,            # 23 dynamic
                robust_sign,                    # 24 dynamic
                rng):                           # 25
  """Run a training step with robust quantile perturbation.

  Args:
    use_check_loss: bool (static). If True, use pure Check Loss (l_tau(u) =
      |tau - I(u<0)| * |u|) which matches the DR-QR theory exactly. If False
      (default), use Quantile Huber Loss (same as original IQN).
  """
  batch_size = states.shape[0]

  def loss_fn(params, rng_input, target_quantile_vals):
    def online(state, key):
      return network_def.apply(params, state, num_quantiles=num_tau_samples,
                               rng=key)

    batched_rng = jnp.stack(jax.random.split(rng_input, num=batch_size))
    model_output = jax.vmap(online)(states, batched_rng)
    quantile_values = model_output.quantile_values
    quantiles = model_output.quantiles
    chosen_action_quantile_values = jax.vmap(lambda x, y: x[:, y][:, None])(
        quantile_values, actions)

    robust_epsilon_for_delta = robust_epsilon
    if use_state_action_mean_scaling:
      # Optional state-conditioned scaling:
      # default legacy behavior:
      #   epsilon(s, a) = epsilon_t * coeff * mean_tau(Z_tau(s, a)).
      # optional new behavior:
      #   epsilon(s, a) = epsilon_init * coeff * mean_tau(Z_tau(s, a))
      #                   * decay_multiplier_t,
      # where decay_multiplier_t = epsilon_t / epsilon_init.
      state_action_mean = jnp.mean(
          chosen_action_quantile_values, axis=1, keepdims=True)
      if state_action_mean_use_abs:
        state_action_mean = jnp.abs(state_action_mean)
      if state_action_mean_use_decay_multiplier:
        eps0_mag = jnp.maximum(jnp.abs(robust_epsilon_init), 1e-8)
        decay_multiplier = robust_epsilon / eps0_mag
        base_state_epsilon = (
            eps0_mag * state_action_mean_scale_coeff * state_action_mean)
        robust_epsilon_for_delta = (
            decay_multiplier * base_state_epsilon)
      else:
        robust_epsilon_for_delta = (
            robust_epsilon * state_action_mean_scale_coeff * state_action_mean)

    # === RQIQN: Add robust perturbation to current network output ===
    delta_s = robust_delta_s(
        quantiles, robust_epsilon_for_delta, wasserstein_p)
    chosen_action_quantile_values = (
        chosen_action_quantile_values + robust_sign * delta_s)
    # ================================================================

    # Shape of bellman_errors:
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    bellman_errors = (target_quantile_vals[:, :, None, :] -
                      chosen_action_quantile_values[:, None, :, :])

    # Tile quantiles for loss computation.
    quantiles_tiled = jnp.tile(
        quantiles[:, None, :, :],
        [1, num_tau_prime_samples, 1, 1]).astype(jnp.float32)
    below_zero = jax.lax.stop_gradient(
        (bellman_errors < 0).astype(jnp.float32))

    if use_check_loss:
      # Pure Check Loss: l_tau(u) = |tau - I(u<0)| * |u|
      # Matches the DR-QR theory exactly but may be less stable for training.
      quantile_loss = jnp.abs(quantiles_tiled - below_zero) * \
          jnp.abs(bellman_errors)
    else:
      # Quantile Huber Loss (default, same as original IQN).
      huber_loss_case_one = (
          (jnp.abs(bellman_errors) <= kappa).astype(jnp.float32) *
          0.5 * bellman_errors ** 2)
      huber_loss_case_two = (
          (jnp.abs(bellman_errors) > kappa).astype(jnp.float32) *
          kappa * (jnp.abs(bellman_errors) - 0.5 * kappa))
      huber_loss = huber_loss_case_one + huber_loss_case_two
      quantile_loss = jnp.abs(quantiles_tiled - below_zero) * huber_loss / kappa

    loss = jnp.sum(quantile_loss, axis=2)
    loss = jnp.mean(loss, axis=1)
    return jnp.mean(loss)

  rng, target_rng = jax.random.split(rng, num=2)
  batched_target_rng = jnp.stack(jax.random.split(target_rng, num=batch_size))
  _, target_quantile_vals = target_quantile_values(
      network_def,
      online_params,
      target_params,
      next_states,
      rewards,
      terminals,
      num_tau_prime_samples,
      num_quantile_samples,
      cumulative_gamma,
      double_dqn,
      batched_target_rng)
  grad_fn = jax.value_and_grad(loss_fn)
  rng, rng_input = jax.random.split(rng)
  loss, grad = grad_fn(online_params, rng_input, target_quantile_vals)
  updates, optimizer_state = optimizer.update(grad, optimizer_state,
                                              params=online_params)
  online_params = optax.apply_updates(online_params, updates)
  return rng, optimizer_state, online_params, loss


@gin.register
def float_inf():
  """Returns float('inf') for use in gin configs."""
  return float('inf')


@gin.configurable
class JaxRQIQNAgent(implicit_quantile_agent.JaxImplicitQuantileAgent):
  """IQN agent with distributionally robust quantile perturbation (RQIQN).

  Adds a location adjustment delta_s to the current network's quantile
  outputs, derived from Wasserstein DRO theory. The perturbation magnitude
  can use sqrt/linear/logistic/constant schedules and can optionally be scaled
  by each sampled state's chosen-action quantile mean.
  """

  def __init__(self,
               num_actions,
               robust_epsilon_init=0.1,
               wasserstein_p=float('inf'),
               robust_sign=1.0,
               robust_decay='sqrt',
               robust_decay_period=1000000,
               robust_decay_warmup_iterations=0,
               robust_decay_iteration_steps=250000,
               robust_logistic_k=5e-4,
               robust_logistic_t0=-1.0,
               loss_type='huber',
               state_action_mean_scaling=False,
               state_action_mean_scale_coeff=1.0,
               state_action_mean_use_abs=True,
               state_action_mean_use_decay_multiplier=False,
               **kwargs):
    """Initializes the RQIQN agent.

    Args:
      num_actions: int, number of actions the agent can take.
      robust_epsilon_init: float, initial perturbation magnitude epsilon_0.
      wasserstein_p: float, Wasserstein order p. Supported: 2.0 or float('inf').
      robust_sign: float, sign factor applied to robust delta_s. Use +1.0 for
        original direction, -1.0 to reverse direction.
      robust_decay: str, epsilon decay schedule. 'sqrt', 'linear', 'logistic',
        or 'none'. The logistic mode is a reverse-logistic decay:
        epsilon_t = epsilon_0 / (1 + exp(k * (t - t0))).
      robust_decay_period: int, decay period for linear schedule and the
        time-scale reference used by logistic mode.
      robust_decay_warmup_iterations: int, number of full iterations to keep
        epsilon fixed at epsilon_0 before applying robust_decay.
      robust_decay_iteration_steps: int, training steps per iteration, used to
        convert warmup iterations to warmup steps.
      robust_logistic_k: float, steepness for logistic decay.
      robust_logistic_t0: float, center t0 for logistic decay; if < 0, auto-set
        to 0.7 * robust_decay_period.
      loss_type: str, 'huber' for Quantile Huber Loss (default, same as IQN),
        or 'check' for pure Check Loss (matches DR-QR theory exactly).
      state_action_mean_scaling: bool, if True, scale epsilon by each sampled
        state's chosen-action quantile mean.
      state_action_mean_scale_coeff: float, multiplier applied before the
        state-action quantile mean.
      state_action_mean_use_abs: bool, whether to use absolute value of the
        state-action quantile mean when scaling epsilon.
      state_action_mean_use_decay_multiplier: bool, if True, interpret
        robust_epsilon as a time decay multiplier by normalizing with
        robust_epsilon_init while keeping epsilon_init as the base magnitude,
        i.e. epsilon(s,a)=epsilon_init*coeff*mean_tau(Z_tau(s,a))*
        (epsilon_t/epsilon_init).
      **kwargs: Additional arguments passed to JaxImplicitQuantileAgent.
    """
    self.robust_epsilon_init = robust_epsilon_init
    self.wasserstein_p = wasserstein_p
    self.robust_sign = robust_sign
    self.robust_decay = robust_decay
    self.robust_decay_period = robust_decay_period
    self.robust_decay_warmup_iterations = robust_decay_warmup_iterations
    self.robust_decay_iteration_steps = robust_decay_iteration_steps
    self.robust_logistic_k = robust_logistic_k
    self.robust_logistic_t0 = robust_logistic_t0
    self.loss_type = loss_type
    self.state_action_mean_scaling = state_action_mean_scaling
    self.state_action_mean_scale_coeff = state_action_mean_scale_coeff
    self.state_action_mean_use_abs = state_action_mean_use_abs
    self.state_action_mean_use_decay_multiplier = (
        state_action_mean_use_decay_multiplier)
    super(JaxRQIQNAgent, self).__init__(
        num_actions=num_actions, **kwargs)

  def _robust_decay_warmup_steps(self):
    """Convert iteration-level robust warmup to agent-step warmup."""
    warmup_iters = max(int(self.robust_decay_warmup_iterations), 0)
    iter_steps = max(int(self.robust_decay_iteration_steps), 1)
    return warmup_iters * iter_steps

  def _compute_robust_epsilon(self):
    """Compute the current robust epsilon based on decay schedule."""
    t = max(self.training_steps, 0)
    warmup_steps = self._robust_decay_warmup_steps()
    if t < warmup_steps:
      return self.robust_epsilon_init
    t = t - warmup_steps

    if self.robust_decay == 'sqrt':
      return self.robust_epsilon_init / math.sqrt(t + 1)
    elif self.robust_decay == 'linear':
      period = max(float(self.robust_decay_period), 1.0)
      return self.robust_epsilon_init * max(
          0.0, 1.0 - t / period)
    elif self.robust_decay == 'logistic':
      period = max(float(self.robust_decay_period), 1.0)
      t0 = self.robust_logistic_t0
      if t0 < 0:
        t0 = 0.7 * period
      x = self.robust_logistic_k * (float(t) - float(t0))
      x = max(-60.0, min(60.0, x))  # Numerical stability for exp.
      return self.robust_epsilon_init / (1.0 + math.exp(x))
    elif self.robust_decay == 'none':
      return self.robust_epsilon_init
    else:
      raise ValueError(
          'Unknown robust_decay schedule: {}'.format(self.robust_decay))

  def _train_step(self):
    """Runs a single training step with robust quantile perturbation."""
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()

        robust_epsilon = self._compute_robust_epsilon()

        self._rng, self.optimizer_state, self.online_params, loss = \
            train_rqiqn(
                self.network_def,
                self.online_params,
                self.target_network_params,
                self.optimizer,
                self.optimizer_state,
                self.preprocess_fn(self.replay_elements['state']),
                self.replay_elements['action'],
                self.preprocess_fn(self.replay_elements['next_state']),
                self.replay_elements['reward'],
                self.replay_elements['terminal'],
                self.num_tau_samples,
                self.num_tau_prime_samples,
                self.num_quantile_samples,
                self.cumulative_gamma,
                self.double_dqn,
                self.kappa,
                robust_epsilon,
                self.wasserstein_p,
                self.loss_type == 'check',
                self.state_action_mean_scaling,
                self.state_action_mean_use_abs,
                self.state_action_mean_use_decay_multiplier,
                self.state_action_mean_scale_coeff,
                self.robust_epsilon_init,
                self.robust_sign,
                self._rng)

        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          with self.summary_writer.as_default():
            tf.summary.scalar('QuantileLoss', loss, step=self.training_steps)
            tf.summary.scalar('RobustEpsilon', robust_epsilon,
                              step=self.training_steps)
          self.summary_writer.flush()
          if hasattr(self, 'collector_dispatcher'):
            self.collector_dispatcher.write(
                [statistics_instance.StatisticsInstance(
                    'Loss', loss.to_py(), step=self.training_steps),
                 statistics_instance.StatisticsInstance(
                    'RobustEpsilon', robust_epsilon,
                    step=self.training_steps),
                 ],
                collector_allowlist=self._collector_allowlist)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1
