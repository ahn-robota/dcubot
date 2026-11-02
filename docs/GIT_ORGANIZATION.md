# Git organization (2026-11-03)

Two public GitHub repos, both owned by `ahn-robota`:

| Repo | What it is | Path on each PC |
|---|---|---|
| [`ahn-robota/dcubot`](https://github.com/ahn-robota/dcubot) | This repo. The RL training pipeline: robot model (MJCF), reward/obs/config, scripts, docs. What the lab PC actually trains with. | `~/dcubot/DCUBOT/rl_training` |
| [`ahn-robota/dcubot-reference-motion-generator`](https://github.com/ahn-robota/dcubot-reference-motion-generator) | Placo-based reference-motion generator (forked from Open_Duck_reference_motion_generator). Where walking gaits get generated from a robot URDF, before being fit into the `.pkl` files this repo trains on. | `~/dcubot/DCUBOT/reference_motion_generator` |

Both public. Everything needed to actually reproduce the pipeline is
tracked; large, purely-regenerable intermediate data is gitignored in
each repo instead of committed:

- `dcubot` (this repo): `checkpoints/` (training output, not source)
- `dcubot-reference-motion-generator`: `recordings_omnidirectional*/`
  (raw Placo sweep recordings, 100MB+ each, regenerable from
  `auto_gait.json` + the robot's own URDF via `auto_waddle.py --sweep`
  -- see `docs/CHANGES_FROM_ORIGINAL.md` "Final URDF remodel" for the
  exact command).

## Why two repos, not one

The reference-motion generator is a separate tool with its own
dependencies (Placo, meshcat) and its own purpose (producing `.pkl`
files, not training). It was historically developed outside the `dcubot`
repo's directory tree entirely (sibling folder, never git-tracked) and
is kept separate rather than merged in, matching that existing structure.
The `dcubot` repo only needs the *output* of the generator (the `.pkl`
files, already committed under `playground/dcubot/data/`), not the
generator itself, to train.

## Syncing with the lab PC

Both PCs work at the same paths (`~/dcubot/DCUBOT/rl_training` and
`~/dcubot/DCUBOT/reference_motion_generator`), same convention as
before. To pull this work onto the lab PC:

```bash
cd ~/dcubot/DCUBOT/rl_training
git pull

# only needed if the lab PC will also regenerate reference motions --
# normally it won't, since the .pkl this repo needs is already committed
cd ~/dcubot/DCUBOT/reference_motion_generator
git clone https://github.com/ahn-robota/dcubot-reference-motion-generator.git .
```

`open_duck_reference_motion_generator/robots/dcubot/dcubot.urdf`'s mesh
paths are relative (`assets/*.stl`), not hardcoded to this laptop's
`/home/ahn/...` path, so the clone works as-is on another machine without
editing. A few older one-off debug scripts under `scripts_dcubot/` still
have this laptop's absolute path baked in (not a secret, just not
portable) -- harmless unless someone runs those specific scripts
elsewhere.
