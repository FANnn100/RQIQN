"""Shared rollout logic for the historical navigation evaluation protocol."""


def evaluate_episode(agent, env, config, policy="full", max_steps=1000):
    """Evaluate one saved layout without changing its dynamics or agent RNG.

    ``full`` fixes CVaR to one; ``adaptive`` uses the original sonar-distance
    rule. Both disable epsilon exploration. Quantile samples remain stochastic.
    Boundary handling is determined by ``env``; this function never enables it.
    """
    if policy not in ("full", "adaptive"):
        raise ValueError("policy must be full or adaptive")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    observation = env.reset_with_eval_config(config)
    done = False
    length = 0
    raw_return = 0.0
    discounted_return = 0.0
    energy = 0.0
    actions = []
    cvars = []
    info = {"state": "not finished"}
    while not done and length < max_steps:
        if policy == "adaptive":
            action, cvar = agent.act_adaptive(observation, eps=0.0)
        else:
            action = agent.act(observation, eps=0.0, cvar=1.0)
            cvar = 1.0
        action = int(action)
        observation, reward, done, info = env.step(action)
        raw_return += float(reward)
        discounted_return += float(env.discount ** length * reward)
        length += 1
        energy += float(env.robot.compute_action_energy_cost(action))
        actions.append(action)
        cvars.append(float(cvar))
    terminal_state = info["state"]
    return {
        "raw_return": raw_return,
        "discounted_return": discounted_return,
        "success": terminal_state == "reach goal",
        "collision": terminal_state == "collision",
        "timeout": terminal_state == "too long episode" or length >= max_steps,
        "out_of_boundary": terminal_state == "out of boundary",
        "time": float(env.robot.dt * env.robot.N * length),
        "energy": energy,
        "length": length,
        "terminal_state": terminal_state,
        "actions": actions,
        "cvars": cvars,
    }
