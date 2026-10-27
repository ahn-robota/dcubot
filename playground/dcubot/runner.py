"""Runs training and evaluation loop for DCUBOT."""

import argparse
import functools

from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from playground.dcubot import randomize
from playground.common.runner import BaseRunner
from playground.dcubot import joystick

# The borrowed "BerkeleyHumanoidJoystickFlatTerrain" PPO config defaults to
# num_envs=8192, which needs more VRAM than some machines have (e.g. this
# laptop's RTX 4050, 6GB). --num_envs lets each machine pick its own value;
# 1024 is the default (safe for a 6GB GPU). A GPU with more VRAM (e.g. a
# lab RTX 3080, 10GB) can likely raise this for better throughput.
DCUBOT_NUM_ENVS = 1024
# Number of evaluations (and progress prints) over the whole run. Each eval costs
# roughly 5 min on this machine, so 30 gives a print every ~11 min at 20M steps.
# NOTE: brax's internal training-iteration granularity is a fixed chunk of
# num_envs*unroll_length*num_updates_per_batch steps (163,840 here) -- it rounds
# steps-per-eval up to at least that one chunk. If num_timesteps/num_evals is
# smaller than that chunk (true for any short curriculum phase well under ~5M
# steps at num_evals=30), training does NOT stop near num_timesteps -- it keeps
# running until it's done num_evals chunks total, massively overshooting
# (confirmed: a 1,000,000-step phase ran past 1,310,720 and was still climbing
# toward 30*163,840=4,915,200 before being killed). Short phases must pass a
# smaller --num_evals.
DCUBOT_NUM_EVALS = 30


class DcubotRunner(BaseRunner):

    def train(self) -> None:
        self.ppo_params = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )
        self.ppo_training_params = dict(self.ppo_params)
        self.ppo_training_params["num_envs"] = self.num_envs
        self.ppo_training_params["num_evals"] = self.num_evals

        if "network_factory" in self.ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **self.ppo_params.network_factory
            )
            del self.ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks
        self.ppo_training_params["num_timesteps"] = self.num_timesteps
        print(f"PPO params: {self.ppo_training_params}")

        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=self.policy_params_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )

        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )

    def __init__(self, args):
        super().__init__(args)
        available_envs = {
            "joystick": (joystick, joystick.Joystick),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]

        self.env_config = self.env_file[0].default_config()
        self.env = self.env_file[1](task=args.task)
        self.eval_env = self.env_file[1](task=args.task)
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )  # 0: state 1: privileged_state
        self.restore_checkpoint_path = args.restore_checkpoint_path
        self.num_evals = args.num_evals
        self.num_envs = args.num_envs
        print(f"Observation size: {self.obs_size}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Duck Mini Runner Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save the checkpoints",
    )
    # parser.add_argument("--num_timesteps", type=int, default=300000000)
    parser.add_argument("--num_timesteps", type=int, default=150000000)
    parser.add_argument("--env", type=str, default="joystick", help="env")
    parser.add_argument("--task", type=str, default="flat_terrain", help="Task to run")
    parser.add_argument(
        "--restore_checkpoint_path",
        type=str,
        default=None,
        help="Resume training from this checkpoint",
    )
    parser.add_argument(
        "--num_evals",
        type=int,
        default=DCUBOT_NUM_EVALS,
        help="Lower this for short runs (e.g. curriculum phases) -- see the note "
        "on DCUBOT_NUM_EVALS: training won't stop near num_timesteps if "
        "num_timesteps/num_evals is smaller than brax's ~163,840-step chunk.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=DCUBOT_NUM_ENVS,
        help="Parallel env count -- scale to the GPU's VRAM (default 1024, "
        "safe for a 6GB GPU; raise for a GPU with more VRAM).",
    )
    # parser.add_argument(
    #     "--debug", action="store_true", help="Run in debug mode with minimal parameters"
    # )
    args = parser.parse_args()

    runner = DcubotRunner(args)

    runner.train()


if __name__ == "__main__":
    main()
