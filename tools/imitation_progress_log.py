"""Log DCUBOT training progress (any env -- joystick, standing, ...), one row per evaluation.

Reads the TensorBoard events written by the runner and writes a CSV (MATLAB readtable-friendly):
    <dir>/progress_log.csv
With --watch it keeps running and prints one briefing line for each new evaluation.

Columns are discovered automatically from whatever eval/* scalars that run's
reward_config.scales actually produced (so it works the same for joystick's
imitation/tracking_lin_vel/feet_gait/... and standing's orientation/
stand_still/head_pos/... without hand-editing a fixed list per env).

Usage:
    .venv/bin/python tools/imitation_progress_log.py [--dir checkpoints_dcubot_imitation] [--watch]
"""
import argparse
import csv
import glob
import os
import time

from tensorboard.backend.event_processing import event_accumulator

TOTAL_STEPS = 20_000_000

# Tag -> fixed column name, for the ones every run has under a non-reward-term path.
FIXED_COLUMNS = {
    "eval/walltime": "walltime_s",
    "eval/episode_reward": "eval_reward",
    "eval/episode_reward_std": "eval_reward_std",
    "eval/avg_episode_length": "ep_len",
    "eval/sps": "sps",
}


def _column_name(tag):
    if tag in FIXED_COLUMNS:
        return FIXED_COLUMNS[tag]
    if tag.startswith("eval/episode_reward/") and not tag.endswith("_std"):
        return "r_" + tag.rsplit("/", 1)[-1]
    if tag.startswith("eval/episode_cost/") and not tag.endswith("_std"):
        return "c_" + tag.rsplit("/", 1)[-1]
    return None  # other tags (images, _std variants, etc.) are skipped


def _latest_tfevents_dir(log_dir):
    """--restore_checkpoint_path resumes reset the step counter to 0 and open a
    new tfevents file each time, left alongside earlier ones in the same dir.
    EventAccumulator always re-scans the whole directory it's pointed at (even
    when handed a single file's path, it silently falls back to the parent
    dir), so just passing the latest file's path does NOT isolate it -- it
    still merges in the older files and resurfaces their (pre-resume) values
    at colliding step numbers. Symlinking only the latest file into its own
    throwaway subdirectory is what actually isolates it."""
    files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not files:
        return log_dir
    latest = max(files, key=os.path.getmtime)
    iso_dir = os.path.join(log_dir, ".latest_only")
    os.makedirs(iso_dir, exist_ok=True)
    for f in os.listdir(iso_dir):
        os.remove(os.path.join(iso_dir, f))
    os.symlink(os.path.abspath(latest), os.path.join(iso_dir, os.path.basename(latest)))
    return iso_dir


def read_rows(log_dir, step_offset=0):
    """One dict per eval step, from every eval/* scalar this run actually logged."""
    acc = event_accumulator.EventAccumulator(_latest_tfevents_dir(log_dir), size_guidance={"scalars": 0})
    acc.Reload()
    tag_to_col = {}
    for tag in acc.Tags()["scalars"]:
        col = _column_name(tag)
        if col is not None:
            tag_to_col[tag] = col
    by_step = {}
    for tag, col in tag_to_col.items():
        for ev in acc.Scalars(tag):
            step = ev.step + step_offset
            by_step.setdefault(step, {"step": step})[col] = ev.value
    # Keep a stable, readable column order: fixed ones first, then r_*, then c_*, alphabetical within each group.
    cols = list(FIXED_COLUMNS.values())
    extra = sorted(c for c in set(tag_to_col.values()) if c not in cols)
    cols += sorted(c for c in extra if c.startswith("r_")) + sorted(c for c in extra if c.startswith("c_"))
    return [by_step[s] for s in sorted(by_step)], cols


def write_csv(rows, cols, path):
    all_cols = ["step"] + [c for c in cols if c != "step"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in all_cols})


def briefing(row, prev, total_steps):
    step = row["step"]
    pct = 100.0 * step / total_steps
    delta = ""
    if prev is not None:
        delta = f" | reward {row.get('eval_reward', float('nan')) - prev.get('eval_reward', float('nan')):+.1f}"
    # Show the single biggest reward term too (whatever it's called in this env), for a quick eyeball check.
    top = max(
        ((c, v) for c, v in row.items() if c.startswith("r_") and isinstance(v, float)),
        key=lambda cv: cv[1], default=(None, None),
    )
    top_txt = f" | top reward: {top[0]}={top[1]:.2f}" if top[0] else ""
    return (
        f"[{time.strftime('%H:%M:%S')}] step {step:,} ({pct:.1f}%) | "
        f"eval_reward {row.get('eval_reward', float('nan')):.1f} "
        f"| ep_len {row.get('ep_len', float('nan')):.0f} "
        f"| sps {row.get('sps', float('nan')):.0f}{delta}{top_txt}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="checkpoints_dcubot_imitation")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--total_steps", type=int, default=TOTAL_STEPS, help="for the (%%) in the briefing line only")
    ap.add_argument("--step_offset", type=int, default=0,
                     help="absolute step the currently-running process was --restore_checkpoint_path'd from "
                          "(its own step counter restarts at 0 on resume, so this is needed to show the real total)")
    args = ap.parse_args()

    out = os.path.join(args.dir, "progress_log.csv")
    seen = set()
    prev = None
    while True:
        if glob.glob(os.path.join(args.dir, "events.out.tfevents*")):
            rows, cols = read_rows(args.dir, args.step_offset)
            write_csv(rows, cols, out)
            for r in rows:
                if r["step"] not in seen:
                    seen.add(r["step"])
                    print(briefing(r, prev, args.total_steps), flush=True)
                    prev = r
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
