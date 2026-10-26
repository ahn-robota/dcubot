"""Runs training and evaluation loop for DCUBOT."""

import argparse
import functools

from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from playground.dcubot import randomize
from playground.common.runner import BaseRunner
from playground.dcubot import joystick, standing

# The borrowed "BerkeleyHumanoidJoystickFlatTerrain" PPO config defaults to
# num_envs=8192, which needs more VRAM than this machine's GPU has (RTX 4050
# laptop, 6GB). Scaled down to fit.
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
        self.ppo_training_params["num_envs"] = DCUBOT_NUM_ENVS
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
            "standing": (standing, standing.Standing),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]

        config_overrides = {}
        if getattr(args, "stand_still_scale", None) is not None:
            config_overrides["reward_config.scales.stand_still"] = args.stand_still_scale
        if getattr(args, "push_magnitude_scale", None) is not None:
            base_min, base_max = self.env_file[0].default_config().push_config.magnitude_range
            scale = args.push_magnitude_scale
            config_overrides["push_config.magnitude_range"] = [base_min * scale, base_max * scale]
        if getattr(args, "fallen_init_prob", None) is not None:
            config_overrides["fallen_init_config.prob"] = args.fallen_init_prob

        self.env_config = self.env_file[0].default_config()
        self.env = self.env_file[1](task=args.task, config_overrides=config_overrides or None)
        self.eval_env = self.env_file[1](task=args.task, config_overrides=config_overrides or None)
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )  # 0: state 1: privileged_state
        self.restore_checkpoint_path = args.restore_checkpoint_path
        self.num_evals = args.num_evals
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
        "--stand_still_scale",
        type=float,
        default=None,
        help="Override standing.py's reward_config.scales.stand_still (for the "
        "piecewise stand_still curriculum: run successive phases with this "
        "raised step by step, e.g. -0.05 -> -0.15 -> -0.25 -> -0.3)",
    )
    parser.add_argument(
        "--push_magnitude_scale",
        type=float,
        default=None,
        help="Scale push_config.magnitude_range (both ends) by this factor "
        "(0.0-1.0), for a piecewise push-force curriculum: run successive "
        "phases raising this 0.0 -> 1.0 (e.g. 0.0, 0.25, 0.5, 0.75, 1.0), "
        "each resuming from the previous phase's checkpoint via "
        "--restore_checkpoint_path. Same mechanism as --stand_still_scale -- "
        "brax's ppo.train() here is one opaque call with no hook to vary env "
        "config continuously mid-run, so this piecewise approximation is "
        "used instead of the Disney papers' continuous per-iteration ramp.",
    )
    parser.add_argument(
        "--fallen_init_prob",
        type=float,
        default=None,
        help="Override joystick.py's fallen_init_config.prob -- fraction of "
        "standing-mode episode resets that start already tilted/moving "
        "(roll/pitch, 2m/s/0.5rad/s ranges from standing.py's USE_FALLEN_INIT), "
        "instead of the default near-upright reset. Walking-mode episodes are "
        "never affected. Same piecewise-phase mechanism as "
        "--push_magnitude_scale/--stand_still_scale: run successive phases "
        "raising this step by step (e.g. 0.05 -> 0.10 -> 0.15 -> 0.20), each "
        "resuming from the previous phase's checkpoint via "
        "--restore_checkpoint_path.",
    )
    # parser.add_argument(
    #     "--debug", action="store_true", help="Run in debug mode with minimal parameters"
    # )
    args = parser.parse_args()

    runner = DcubotRunner(args)

    runner.train()


if __name__ == "__main__":
    main()
