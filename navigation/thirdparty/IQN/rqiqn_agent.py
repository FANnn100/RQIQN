"""Robust Quantile IQN (RQIQN) agent.

PyTorch port of the JAX Dopamine RQIQN algorithm. Extends IQN with a
distributionally robust location adjustment `delta_s(tau)` added to the
current network's chosen-action quantile outputs before computing the
quantile loss.

For p = inf (q = 1):  delta_s(tau) = epsilon * (2*tau - 1)
For p = 2   (q = 2):  delta_s(tau) = (epsilon/2) * (2*tau - 1) / sqrt(tau^2 + (1-tau)^2)
"""

import math
import os
import random
import numpy as np
import torch

from thirdparty.IQN.agent import IQNAgent, calculate_huber_loss


def robust_delta_s(taus, epsilon, wasserstein_p):
    """Compute the robust location adjustment delta_s for each quantile.

    Args:
        taus: Tensor of shape (B, N, 1), quantile levels tau in [0, 1].
        epsilon: Scalar or tensor broadcastable to taus, perturbation magnitude.
        wasserstein_p: "inf" / float("inf") or 2 / 2.0.

    Returns:
        Tensor with the same shape as taus.
    """
    two_tau_m1 = 2.0 * taus - 1.0
    if wasserstein_p == "inf" or (isinstance(wasserstein_p, float) and
                                  math.isinf(wasserstein_p)):
        return epsilon * two_tau_m1
    if float(wasserstein_p) == 2.0:
        denom = torch.sqrt(taus.pow(2) + (1.0 - taus).pow(2)).clamp(min=1e-8)
        return 0.5 * epsilon * two_tau_m1 / denom
    raise ValueError(f"Unsupported wasserstein_p={wasserstein_p!r}")


def compute_robust_epsilon(t, eps0, schedule, period, warmup, k, t0):
    """Scalar robust epsilon given the current learning timestep.

    Verbatim port of JaxRQIQNAgent._compute_robust_epsilon:
    during warmup returns eps0 (NOT 0).
    """
    t = max(int(t), 0)
    if t < warmup:
        return eps0
    t = t - warmup
    if schedule == "none":
        return eps0
    if schedule == "sqrt":
        return eps0 / math.sqrt(t + 1)
    if schedule == "linear":
        p = max(float(period), 1.0)
        return eps0 * max(0.0, 1.0 - t / p)
    if schedule == "logistic":
        p = max(float(period), 1.0)
        center = t0 if t0 >= 0 else 0.7 * p
        x = k * (float(t) - float(center))
        x = max(-60.0, min(60.0, x))
        return eps0 / (1.0 + math.exp(x))
    raise ValueError(f"Unknown schedule={schedule!r}")


class RQIQNAgent(IQNAgent):
    """IQN agent with distributionally robust quantile perturbation (RQIQN).

    Adds a location adjustment delta_s to the current network's quantile
    outputs, derived from Wasserstein DRO theory. The perturbation magnitude
    can use sqrt/linear/logistic/none schedules and can optionally be scaled
    by each sampled state's chosen-action quantile mean.
    """

    def __init__(self, *args,
                 robust_epsilon_init=0.1,
                 wasserstein_p="inf",
                 robust_sign=1.0,
                 robust_decay="sqrt",
                 robust_decay_period=1_000_000,
                 robust_decay_warmup_timesteps=0,
                 robust_logistic_k=5e-4,
                 robust_logistic_t0=-1.0,
                 loss_type="huber",
                 state_action_mean_scaling=False,
                 state_action_mean_scale_coeff=1.0,
                 state_action_mean_use_abs=True,
                 state_action_mean_use_decay_multiplier=False,
                 disable_cvar_in_training=True,
                 **kwargs):
        # Destination codebase has no iterations*iteration_steps concept,
        # so warmup is expressed directly in learning timesteps.
        super().__init__(*args, **kwargs)
        self.robust_epsilon_init = float(robust_epsilon_init)
        self.wasserstein_p = wasserstein_p
        self.robust_sign = float(robust_sign)
        self.robust_decay = robust_decay
        self.robust_decay_period = int(robust_decay_period)
        self.robust_decay_warmup_timesteps = int(robust_decay_warmup_timesteps)
        self.robust_logistic_k = float(robust_logistic_k)
        self.robust_logistic_t0 = float(robust_logistic_t0)
        self.loss_type = loss_type
        self.state_action_mean_scaling = bool(state_action_mean_scaling)
        self.state_action_mean_scale_coeff = float(state_action_mean_scale_coeff)
        self.state_action_mean_use_abs = bool(state_action_mean_use_abs)
        self.state_action_mean_use_decay_multiplier = bool(
            state_action_mean_use_decay_multiplier)
        self.disable_cvar_in_training = bool(disable_cvar_in_training)

        # Stash the most recent epsilon for logging.
        self._last_robust_epsilon = self.robust_epsilon_init

    def _compute_robust_epsilon(self):
        return compute_robust_epsilon(
            self.learning_timestep,
            self.robust_epsilon_init,
            self.robust_decay,
            self.robust_decay_period,
            self.robust_decay_warmup_timesteps,
            self.robust_logistic_k,
            self.robust_logistic_t0,
        )

    def _sample_cvar(self):
        if self.disable_cvar_in_training:
            return 1.0
        cvar = np.random.uniform(0.0, 1.0)
        if cvar == 0.0:
            cvar = 1e-6
        return cvar

    def learn(self,
              total_timesteps,
              train_env,
              eval_env,
              eval_config,
              eval_freq,
              eval_log_path,
              verbose=True):
        """Same as IQNAgent.learn, but clamps CVaR to 1.0 during training
        when `disable_cvar_in_training=True`."""

        state = train_env.reset()
        cvar = self._sample_cvar()

        ep_reward = 0.0
        ep_length = 0
        ep_num = 0

        while self.current_timestep <= total_timesteps:
            eps = self.linear_eps(total_timesteps)
            action = self.act(state, eps, cvar)
            next_state, reward, done, info = train_env.step(action)

            ep_reward += train_env.discount ** ep_length * reward
            ep_length += 1

            self.memory.add(state, action, reward, next_state, done)
            state = next_state

            if self.current_timestep >= self.learning_starts:
                if self.learning_timestep % self.UPDATE_EVERY == 0:
                    if len(self.memory) > self.BATCH_SIZE:
                        experiences = self.memory.sample()
                        self.train(experiences)

                if self.learning_timestep % self.target_update_interval == 0:
                    self.soft_update(self.qnetwork_local, self.qnetwork_target)

                if self.learning_timestep % eval_freq == 0:
                    self.evaluation(eval_env, eval_config=eval_config,
                                    eval_log_path=eval_log_path)
                    self.evaluation(eval_env, eval_config=eval_config,
                                    greedy=False, eval_log_path=eval_log_path)
                    self.qnetwork_local.save(eval_log_path)

                self.learning_timestep += 1

            if done:
                ep_num += 1

                if verbose:
                    print("======== training info ========")
                    print("current ep_length: ", ep_length)
                    print("current ep_reward: ", ep_reward)
                    print("current ep_result: ", info["state"])
                    print("episodes_num: ", ep_num)
                    print("exploration_rate: ", eps)
                    print("current_timesteps: ", self.current_timestep)
                    print("total_timesteps: ", total_timesteps)
                    print("current_cvar: ", cvar)
                    print("robust_epsilon: ", self._last_robust_epsilon)
                    print("======== training info ========\n")

                ep_reward = 0.0
                ep_length = 0
                state = train_env.reset()
                cvar = self._sample_cvar()

            self.current_timestep += 1

    def train(self, experiences):
        """Update value parameters with a robust quantile perturbation."""
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next, _ = self.qnetwork_target(next_states)
        Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1)
        Q_targets = rewards.unsqueeze(-1) + (
            self.GAMMA ** self.n_step * Q_targets_next *
            (1. - dones.unsqueeze(-1)))

        Q_expected, taus = self.qnetwork_local(states)
        Q_expected = Q_expected.gather(
            2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, 8, 1))

        # === RQIQN robust perturbation ===
        eps_t = self._compute_robust_epsilon()
        self._last_robust_epsilon = float(eps_t)

        if self.state_action_mean_scaling:
            mean_sa = Q_expected.detach().mean(dim=1, keepdim=True)
            if self.state_action_mean_use_abs:
                mean_sa = mean_sa.abs()
            if self.state_action_mean_use_decay_multiplier:
                eps0_mag = max(abs(self.robust_epsilon_init), 1e-8)
                decay_mult = eps_t / eps0_mag
                epsilon_for_delta = (decay_mult * eps0_mag *
                                     self.state_action_mean_scale_coeff *
                                     mean_sa)
            else:
                epsilon_for_delta = (eps_t *
                                     self.state_action_mean_scale_coeff *
                                     mean_sa)
        else:
            epsilon_for_delta = eps_t

        delta_s = robust_delta_s(taus, epsilon_for_delta, self.wasserstein_p)
        assert delta_s.shape == Q_expected.shape, (
            f"delta_s shape {tuple(delta_s.shape)} != Q_expected shape "
            f"{tuple(Q_expected.shape)}")
        Q_expected = Q_expected + self.robust_sign * delta_s
        # =================================

        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, 8, 8), "wrong td error shape"

        if self.loss_type == "check":
            loss_el = (taus - (td_error.detach() < 0).float()).abs() * \
                td_error.abs()
        else:
            huber = calculate_huber_loss(td_error, 1.0)
            loss_el = (taus - (td_error.detach() < 0).float()).abs() * \
                huber / 1.0

        loss = loss_el.sum(dim=1).mean(dim=1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 0.5)
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def evaluation(self, eval_env, eval_config, greedy=True, eval_log_path=None):
        """Run parent evaluation and append a CSV row per call.

        CSV path: ``<eval_log_path>/eval_log.csv``. Header is written on the
        first row. Each evaluation call (greedy or adaptive) appends one row.
        """
        super().evaluation(eval_env, eval_config=eval_config, greedy=greedy,
                           eval_log_path=eval_log_path)
        if eval_log_path is None:
            return

        policy = "greedy" if greedy else "adaptive"
        rewards = self.eval_rewards[policy][-1]
        successes = np.array(self.eval_successes[policy][-1])
        collisions = np.array(self.eval_collisions[policy][-1])
        toolongs = np.array(self.eval_toolongs[policy][-1])
        times = np.array(self.eval_times[policy][-1])
        energies = np.array(self.eval_energies[policy][-1])
        timestep = self.eval_timesteps[policy][-1]

        succ_idx = np.where(successes == 1)[0]
        avg_r = float(np.mean(rewards))
        success_rate = float(successes.sum() / len(successes))
        collision_rate = float(collisions.sum() / len(collisions))
        toolong_rate = float(toolongs.sum() / len(toolongs))
        if len(succ_idx) > 0:
            avg_t = float(np.mean(times[succ_idx]))
            avg_e = float(np.mean(energies[succ_idx]))
        else:
            avg_t = float("nan")
            avg_e = float("nan")

        csv_path = os.path.join(eval_log_path, "eval_log.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if write_header:
                f.write("timestep,policy,avg_reward,success_rate,"
                        "collision_rate,toolong_rate,avg_time,avg_energy,"
                        "robust_epsilon\n")
            f.write(f"{timestep},{policy},{avg_r:.6f},"
                    f"{success_rate:.6f},{collision_rate:.6f},"
                    f"{toolong_rate:.6f},{avg_t:.6f},{avg_e:.6f},"
                    f"{self._last_robust_epsilon:.6e}\n")
