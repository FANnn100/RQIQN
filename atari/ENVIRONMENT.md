# Environment and dependencies

Training targets Python 3.8 and the legacy Dopamine/JAX stack below. The existing environment has passed imports, short Alien training with gradient updates, metric recording, checkpoint saving, and checkpoint restoration. These checks do not establish a fresh dependency installation or reproduce a complete 200M-frame experiment.

## Validated existing training environment

| Component | Version |
| --- | --- |
| Python | 3.8.20 |
| NumPy | 1.24.3 |
| SciPy | 1.10.1 |
| JAX | 0.4.13 |
| jaxlib | 0.4.13+cuda12.cudnn89 |
| Flax | 0.7.2 |
| Optax / Chex | 0.1.7 / 0.1.7 |
| Orbax checkpoint / TensorStore | 0.2.3 / 0.1.45 |
| TensorFlow | 2.13.1 |
| ml-dtypes / typing-extensions | 0.2.0 / 4.5.0 |
| gin-config / absl-py | 0.5.0 / 2.3.1 |
| Gym / ALE-Py | 0.26.2 / 0.8.1 |
| OpenCV (`opencv-python`) | 4.13.0.92 |

The environment creates `AlienNoFrameskip-v0` with an `OrderEnforcing` wrapper and an underlying `AtariEnv`. The measured ALE `repeat_action_probability` is 0.25. `reset()` returns two fields and `step()` returns five; the included Atari preprocessing handles these formats. The underlying environment provides `seed()` in this version.

## Proposed fresh CPU installation

Create a Python 3.8 environment, then run:

```bash
python -m pip install -r requirements-train-cpu.txt
python -m pip check
```

This installation has not been tested in a fresh environment. The requirements are explicit version selections based on the working stack, rather than a complete dependency lockfile. Transitive dependencies are still resolved by pip.

The CPU requirements use `jaxlib==0.4.13` and `tensorflow==2.13.1`. To request CPU execution explicitly:

```bash
export CUDA_VISIBLE_DEVICES=''
export JAX_PLATFORMS=cpu
```

The proposed CPU configuration uses `opencv-python==4.8.1.78` with NumPy 1.24.3; that fresh combination has not been validated. The existing environment instead has OpenCV 4.13.0.92. Its package metadata requires NumPy <2 for Python <3.9, but NumPy >=2 for Python >=3.9. Consequently, the existing Python 3.8 stack should not be transferred directly to newer Python versions or upgraded to NumPy 2 alongside TensorFlow 2.13.1.

The included dependency closure does not require pandas, TensorFlow Probability, or tf-slim. ROM installation utilities are not training dependencies.

## Optional legacy CUDA wheel

The validated existing environment uses `jaxlib==0.4.13+cuda12.cudnn89`. On a machine with compatible CUDA/cuDNN libraries and drivers, the legacy wheel can replace CPU jaxlib after installing the training requirements:

```bash
python -m pip install --no-deps 'jaxlib==0.4.13+cuda12.cudnn89' \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip check
```

This command has not been run in a fresh environment. Wheel availability and runtime compatibility must be checked on the target machine. Installing a current `jax[cuda12]` distribution can change the required JAX/Flax/Optax combination. TensorFlow is used for logging and shared infrastructure; TensorFlow GPU execution is unnecessary.

## Alien ROM

Provide an Alien ROM that ALE-Py 0.8.1 can discover. No ROM is included. Check a locally installed ROM without downloading or importing files:

```bash
python - <<'PY'
import gym
from ale_py import roms

print('Alien ROM:', roms.Alien)
env = gym.make('AlienNoFrameskip-v0')
try:
    env.reset()
    transition = env.step(0)
    print('step fields:', len(transition))
    print('sticky-action probability:',
          env.unwrapped.ale.getFloat('repeat_action_probability'))
finally:
    env.close()
PY
```

## Plotting

Plotting can use a separate environment with no ROM or training dependencies:

```bash
python -m pip install -r requirements-plot.txt
```

The plotting requirements select Python 3.8-compatible NumPy 1.24.3 and Matplotlib 3.7.5. Fresh installation of these requirements has not been validated. Actual CSV/PDF/PNG generation was checked separately with Python 3.11.11, NumPy 1.26.4, and Matplotlib 3.8.0.

## Smoke configuration

Use `--smoke` for a short pipeline check. It reduces both replay capacity and replay warmup, so gradient updates occur within the small interaction budget. Merely shortening the normal training budget can finish before the default 20,000-step replay warmup. Smoke settings are not the paper configuration; run manifests identify them explicitly.
