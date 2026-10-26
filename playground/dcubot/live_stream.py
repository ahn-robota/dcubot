"""Headless live view of the dcubot policy, streamed to a browser (MJPEG).

Runs the newest checkpoint from --watch_dir in a loop under training-like
conditions (random pushes every 5-10 s, sensor noise, action delay), resets
when the robot falls, and hot-swaps to each new checkpoint as training saves it.
Open http://localhost:8000 . No display needed; renders in software (mesa).

  MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
  .venv/bin/python -m playground.dcubot.live_stream --watch_dir checkpoints_dcubot_gait
"""
import argparse
import collections
import glob
import io
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from playground.common.onnx_infer import OnnxInfer
from playground.dcubot import debug_vis, joystick
from playground.dcubot.mujoco_infer import MjInfer

PAGE = b"""<html><body style="background:#111;color:#ddd;font-family:sans-serif;text-align:center">
<h3>dcubot live policy (latest checkpoint, training-like pushes/noise)</h3>
<img src="/stream" style="width:90%;max-width:960px"><p>blue arrow = command, cyan = actual velocity</p></body></html>"""

_frame = {"jpg": None}
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/frame.jpg"):
            with _lock:
                jpg = _frame["jpg"]
            self.send_response(200 if jpg else 503)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(jpg or b"")
        elif self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        jpg = _frame["jpg"]
                    if jpg:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                        self.wfile.write(jpg + b"\r\n")
                    time.sleep(1 / 20)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)


def newest(watch_dir):
    files = glob.glob(os.path.join(watch_dir, "*.onnx"))
    return max(files, key=os.path.getmtime) if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch_dir", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model_path", default="playground/dcubot/xmls/scene_flat_terrain_train.xml")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--cmd", default="random", help="'random' (like training) or 'forward'")
    ap.add_argument("--no_push", action="store_true")
    a = ap.parse_args()

    threading.Thread(target=lambda: ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever(), daemon=True).start()
    print(f"open http://localhost:{a.port}")

    while (path := newest(a.watch_dir)) is None:
        print("waiting for the first checkpoint ..."); time.sleep(10)

    cfg = joystick.default_config()
    ns, pc = cfg.noise_config, cfg.push_config
    inf = MjInfer(a.model_path, path, False)
    inf.debug_vis = True
    m, d = inf.model, inf.data
    torso = m.body("base_link").id
    W, H = 480, 360
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    r = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = torso
    cam.distance, cam.azimuth, cam.elevation = 1.3, 125, -22

    rng = np.random.default_rng()
    joint_noise = np.zeros(inf.num_dofs)
    for i, n in enumerate(inf.actuator_names):
        joint_noise[i] = ns.scales.hip_pos if "hip" in n and "head" not in n else ns.scales.knee_pos if "knee" in n else ns.scales.ankle_pos if "ankle" in n else 0.0

    def new_episode():
        mujoco.mj_resetDataKeyframe(m, d, 0)
        inf.last_action[:] = inf.last_last_action[:] = inf.last_last_last_action[:] = 0
        inf.motor_targets = inf.default_actuator.copy()
        inf.prev_motor_targets = inf.default_actuator.copy()
        inf.gait_step = 0
        act_buf.clear()
        if a.cmd == "forward":
            inf.commands = [0.15, 0, 0, 0, 0, 0]
        else:
            inf.commands = [rng.uniform(*cfg.lin_vel_x), rng.uniform(*cfg.lin_vel_y), rng.uniform(*cfg.ang_vel_yaw), 0, 0, 0]

    act_buf = collections.deque(maxlen=8)
    ep, falls, ep_steps, survive = 0, 0, 0, collections.deque(maxlen=10)
    next_push = rng.uniform(*pc.interval_range)
    new_episode()
    dt_policy = inf.sim_dt * inf.decimation
    render_every = max(1, round(1 / (a.fps * dt_policy)))
    status, hold, cnt, t0 = "", 0, 0, time.time()
    loaded = os.path.basename(path)

    while True:
        t_start = time.time()
        # hot-swap to the newest checkpoint (between episodes only)
        if hold == 0 and cnt % 100 == 0:
            p = newest(a.watch_dir)
            if p and os.path.basename(p) != loaded:
                try:
                    inf.policy = OnnxInfer(p, awd=True); loaded = os.path.basename(p)
                    new_episode(); ep_steps = 0; print("[reloaded]", loaded)
                except Exception as e:
                    print("waiting for", p, e)
        if hold > 0:
            hold -= 1
            if hold == 0:
                new_episode(); ep_steps = 0; ep += 1
        else:
            for _ in range(inf.decimation):
                mujoco.mj_step(m, d)
            obs = inf.get_obs(d, inf.commands)
            obs[0:3] += rng.uniform(-1, 1, 3) * ns.level * ns.scales.gyro
            obs[3:6] += rng.uniform(-1, 1, 3) * ns.level * ns.scales.accelerometer
            obs[12:25] += rng.uniform(-1, 1, 13) * ns.level * joint_noise
            act = inf.policy.infer(obs)
            act_buf.append(act.copy())
            delay = rng.integers(ns.action_min_delay, ns.action_max_delay + 1)
            act_d = act_buf[max(0, len(act_buf) - 1 - delay)]
            inf.last_last_last_action = inf.last_last_action.copy()
            inf.last_last_action = inf.last_action.copy()
            inf.last_action = act.copy()
            inf.motor_targets = inf.default_actuator + act_d * inf.action_scale
            inf.motor_targets = np.clip(inf.motor_targets, inf.prev_motor_targets - inf.max_motor_velocity * dt_policy,
                                        inf.prev_motor_targets + inf.max_motor_velocity * dt_policy)
            inf.prev_motor_targets = inf.motor_targets.copy()
            d.ctrl = inf.motor_targets.copy()
            ep_steps += 1
            if not a.no_push and d.time >= next_push:
                ang = rng.uniform(0, 2 * np.pi); mag = rng.uniform(*pc.magnitude_range)
                d.qvel[0] += mag * np.cos(ang); d.qvel[1] += mag * np.sin(ang)
                next_push = d.time + rng.uniform(*pc.interval_range)
            up = d.site_xmat[inf._imu_id].reshape(3, 3)[2, 2]
            if up < 0.3:
                falls += 1; survive.append(ep_steps * dt_policy); hold = int(1.0 / dt_policy); status = "FALLEN"
            elif ep_steps >= 1000:
                survive.append(ep_steps * dt_policy); hold = 1; status = "20s OK"
            else:
                status = "standing/walking"

        cnt += 1
        if cnt % render_every == 0:
            r.update_scene(d, camera=cam)
            debug_vis.draw(r.scene, d, inf._imu_id, inf.commands, inf.get_gyro(d)[2])
            img = Image.fromarray(r.render())
            g = ImageDraw.Draw(img)
            lv = debug_vis.local_velocity(d, inf._imu_id)
            avg = f"{np.mean(survive):.1f}s" if survive else "-"
            lines = [f"ckpt {loaded}", f"episode {ep}  t={ep_steps*dt_policy:.1f}s  {status}",
                     f"falls {falls}  avg survive(last10) {avg}",
                     f"cmd(fwd,left)=({inf.commands[0]:+.2f},{inf.commands[1]:+.2f}) actual=({lv[0]:+.2f},{lv[1]:+.2f})"]
            for i, t in enumerate(lines):
                g.text((6, 6 + 13 * i), t, fill=(255, 60, 60) if status == "FALLEN" and i == 1 else (20, 20, 20))
            buf = io.BytesIO(); img.save(buf, "JPEG", quality=80)
            with _lock:
                _frame["jpg"] = buf.getvalue()
        sleep = dt_policy - (time.time() - t_start)
        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    main()
