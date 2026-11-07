"""Explicit historical and corrected navigation scene-generation protocols."""
import contextlib
import io
import numpy as np


ONLINE_PROTOCOLS = ('historical-online30', 'corrected-mixed3x10')
TRAINING_SCHEDULE = dict(timesteps=[0, 1000000, 2000000],
                         num_cores=[4, 6, 8], num_obstacles=[6, 8, 10],
                         min_start_goal_dis=[30.0, 35.0, 40.0])


def make_env(seed=0, schedule=None):
    from marinenav_env.envs.marinenav_env import MarineNavEnv
    return MarineNavEnv(seed=seed, schedule=schedule)


def effective_layout(config):
    return dict(cores=len(config['env']['cores']['positions']),
                obstacles=len(config['env']['obstacles']['positions']),
                robot_n=config['robot']['N'],
                start=config['env']['start'], goal=config['env']['goal'])


def make_online_configs(protocol='historical-online30', env_seed=348):
    """Recreate effective layouts without depending on wrapper attribute writes.

    The archived Gym 0.21 OrderEnforcing wrapper shadowed writes to num_obs,
    num_cores, start, goal and reset_start_and_goal. Historical mode explicitly
    recreates the resulting raw-environment defaults instead of the intended
    three-difficulty configuration. Corrected mode is a distinct new experiment.
    """
    if protocol not in ONLINE_PROTOCOLS:
        raise ValueError('Unknown online evaluation protocol: ' + protocol)
    env = make_env(env_seed)
    configs = {}
    try:
        for index in range(30):
            if protocol == 'corrected-mixed3x10':
                group = index // 10
                env.num_cores = [4, 6, 8][group]
                env.num_obs = [6, 8, 10][group]
                env.reset_start_and_goal = False
                env.start = np.array([5.0, 5.0])
                env.goal = np.array([45.0, 45.0])
            env.reset()
            config = env.episode_data()
            expected = (8, 5) if protocol == 'historical-online30' else (
                [4, 6, 8][index // 10], [6, 8, 10][index // 10])
            actual = effective_layout(config)
            if (actual['cores'], actual['obstacles']) != expected:
                raise RuntimeError('Scene placement did not reach the requested counts.')
            configs['env_{}'.format(index)] = config
    finally:
        env.close()
    return configs


def make_hard_configs(num_episodes=500, env_seed=348):
    """Recreate the shared hard-layout generator used for camera-ready Table 1.

    Generation uses set_boundary=True; episode_data does not serialize this
    flag. Historical consumers therefore run with set_boundary=False. The
    evaluator must set that effective consumer flag explicitly.
    """
    if num_episodes < 1:
        raise ValueError('The episode count must be positive.')
    env = make_env(env_seed)
    env.reset_start_and_goal = False
    env.random_reset_state = False
    env.set_boundary = True
    env.obs_r_range = [1, 3]
    env.start = np.array([5.0, 5.0])
    env.goal = np.array([45.0, 45.0])
    env.robot.N = 5
    env.num_obs, env.num_cores = 10, 8
    try:
        configs = []
        for _ in range(num_episodes):
            env.reset()
            configs.append(env.episode_data())
        return configs
    finally:
        env.close()
