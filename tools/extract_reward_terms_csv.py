"""Metric 1 for the presentation: reward-term breakdown (long format) +
reward/episode-length curve (wide format), read straight from TensorBoard
event files. Unlike imitation_progress_log.py (which always isolates the
single *latest* tfevents file in a dir, for live monitoring of one run),
this takes an explicit list of (tfevents_file, step_offset) pairs so it can
stitch together a run that was resumed multiple times (each resume opens a
new tfevents file and resets its own step counter to 0).

Usage:
    .venv/bin/python tools/extract_reward_terms_csv.py \
        --files <tfevents1> 0 <tfevents2> 983040 ... \
        --out_long <long.csv> --out_wide <wide.csv>
"""
import argparse
import csv
import os
import tempfile

from tensorboard.backend.event_processing import event_accumulator


def _column_name(tag):
    if tag == "eval/episode_reward":
        return "eval_reward"
    if tag == "eval/avg_episode_length":
        return "ep_len"
    if tag.startswith("eval/episode_reward/") and not tag.endswith("_std"):
        return "r_" + tag.rsplit("/", 1)[-1]
    if tag.startswith("eval/episode_cost/") and not tag.endswith("_std"):
        return "c_" + tag.rsplit("/", 1)[-1]
    return None


def read_one_file_isolated(path):
    """Isolate a single tfevents file (EventAccumulator always re-scans the
    whole parent directory otherwise -- see imitation_progress_log.py's note
    on the same issue). The scratch dir's name must NOT contain "tfevents"
    anywhere -- TB's own directory scanner matches on that substring, so a
    dir named *.../events.out.tfevents...-isolated/ gets treated as another
    candidate event file itself and fails with "Is a directory"."""
    iso_dir = tempfile.mkdtemp(prefix="single_event_")
    link = os.path.join(iso_dir, os.path.basename(path))
    os.symlink(os.path.abspath(path), link)
    acc = event_accumulator.EventAccumulator(iso_dir, size_guidance={"scalars": 0})
    acc.Reload()
    by_step = {}
    for tag in acc.Tags()["scalars"]:
        col = _column_name(tag)
        if col is None:
            continue
        for ev in acc.Scalars(tag):
            by_step.setdefault(ev.step, {"step": ev.step})[col] = ev.value
    return [by_step[s] for s in sorted(by_step)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                     help="alternating <tfevents_path> <step_offset> pairs")
    ap.add_argument("--out_long", required=True)
    ap.add_argument("--out_wide", required=True)
    args = ap.parse_args()

    assert len(args.files) % 2 == 0, "--files must be pairs of <path> <offset>"
    pairs = [(args.files[i], int(args.files[i + 1])) for i in range(0, len(args.files), 2)]

    all_rows = []
    for path, offset in pairs:
        rows = read_one_file_isolated(path)
        for r in rows:
            r = dict(r)
            r["step"] = r["step"] + offset
            all_rows.append(r)
    all_rows.sort(key=lambda r: r["step"])
    # dedupe on step (resumed-run boundaries repeat step 0 of the next phase
    # at the same absolute step as the previous phase's last checkpoint)
    dedup = {}
    for r in all_rows:
        dedup[r["step"]] = r
    all_rows = [dedup[s] for s in sorted(dedup)]

    # wide: step, eval_reward, ep_len
    os.makedirs(os.path.dirname(args.out_wide), exist_ok=True)
    with open(args.out_wide, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "eval_reward", "ep_len"])
        for r in all_rows:
            w.writerow([r["step"], r.get("eval_reward", ""), r.get("ep_len", "")])
    print(f"saved {args.out_wide} ({len(all_rows)} rows)")

    # long: iteration, term_name, value (every r_*/c_* term)
    with open(args.out_long, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "term_name", "value"])
        for r in all_rows:
            for k, v in r.items():
                if k in ("step", "eval_reward", "ep_len"):
                    continue
                w.writerow([r["step"], k, v])
    print(f"saved {args.out_long}")


if __name__ == "__main__":
    main()
