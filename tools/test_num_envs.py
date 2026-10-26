"""One-off: try to train a handful of steps at a given num_envs and see if it OOMs.
Each candidate should be run in its own process (XLA doesn't release GPU memory
within a process between attempts).
Usage: .venv/bin/python tools/test_num_envs.py <num_envs>
"""
import functools
import sys

from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from playground.dcubot import joystick, randomize

num_envs = int(sys.argv[1])

env = joystick.Joystick(task="flat_terrain_train")
eval_env = joystick.Joystick(task="flat_terrain_train")

ppo_params = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
params = dict(ppo_params)
params["num_envs"] = num_envs
params["num_evals"] = 1
params["num_timesteps"] = num_envs * params["unroll_length"] * 4  # a few batches, not a real run
network_factory = functools.partial(ppo_networks.make_ppo_networks, **params.pop("network_factory"))

print(f"trying num_envs={num_envs}, num_timesteps={params['num_timesteps']}", flush=True)
try:
    ppo.train(
        environment=env,
        eval_env=eval_env,
        randomization_fn=randomize.domain_randomize,
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=lambda s, m: print(f"  step {s}", flush=True),
        **params,
    )
    print(f"num_envs={num_envs}: OK", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"num_envs={num_envs}: FAIL ({type(e).__name__})", flush=True)
