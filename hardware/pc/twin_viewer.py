"""PC-side viewer/controller for the dcubot digital twin bridge.

--ping runs a no-GUI connection test (Hz / RTT / packet loss in both
directions), then exits. --mock skips --pi-host: it starts the exact same
mock Bridge from hardware/pi/bridge_server.py in-process on loopback, so
this can be tested with no Raspberry Pi and no second terminal.

With neither flag, opens the MuJoCo viewer in Mirror mode: the sim robot's
tilt/yaw and joint angles follow the real robot's telemetry. Command mode
(Control-panel sliders, ghost, torque ramp) lands in a later step -- see
docs/DIGITAL_TWIN.md.
"""
import argparse
import math
import socket
import sys
import time
from pathlib import Path
from threading import Event, Lock, Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import COMMAND_HZ, DEFAULT_PORT, SeqLossTracker, make_command, pack, unpack  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from playground.dcubot import policy_constants  # noqa: E402

DEFAULT_MODEL = "playground/dcubot/xmls/scene_flat_terrain.xml"


def start_local_mock_bridge(port):
    """Runs hardware/pi/bridge_server.py's mock Bridge in a background
    thread of this same process, bound to loopback. Reuses that one
    MockSensors implementation instead of duplicating fake data here."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi"))
    from bridge_server import Bridge, build_arg_parser

    args = build_arg_parser().parse_args(["--port", str(port), "--mock"])
    Thread(target=lambda: Bridge(args).run(), daemon=True).start()
    time.sleep(0.2)  # let it bind before we start sending


def fmt_pct(x):
    return "warming up" if x is None else f"{x:.1%}"


class BridgeLink:
    """The PC<->bridge UDP link: sends command/heartbeat packets at
    COMMAND_HZ in a background thread and keeps the latest received
    telemetry plus RTT/loss stats. Shared by --ping and the viewer so this
    isn't implemented twice.

    torque_request/estop/joint_target are plain attributes the caller
    (key_callback, slider code, ...) writes directly; the send thread picks
    up whatever is current on its own schedule.
    """

    def __init__(self, pi_host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)
        self.addr = (pi_host, port)

        self.torque_request = False
        self.estop = False
        self.joint_target = {}
        self.velocity_command = [0.0, 0.0, 0.0]  # [vx, vy, wz] -- hardware/pi/policy_runner.py only

        self.seq = 0
        self.lock = Lock()
        self.latest_telem = None
        self.telem_loss = SeqLossTracker()
        self.cmd_loss = SeqLossTracker()  # PC->Pi, inferred from last_cmd_seq_recv
        self.recv_count = 0
        self.rtts = []

        self._stop = Event()
        Thread(target=self._send_loop, daemon=True).start()
        Thread(target=self._recv_loop, daemon=True).start()

    def _send_loop(self):
        period = 1.0 / COMMAND_HZ
        while not self._stop.is_set():
            self.seq += 1
            cmd = make_command(
                seq=self.seq,
                t_pc=time.time(),
                torque_request=self.torque_request,
                estop=self.estop,
                joint_target=dict(self.joint_target),
                velocity_command=list(self.velocity_command),
            )
            self.sock.sendto(pack(cmd), self.addr)
            time.sleep(period)

    def _recv_loop(self):
        # Own thread so arrival is timestamped the instant recvfrom()
        # returns, decoupled from COMMAND_HZ -- see the RTT-measurement fix
        # in docs/DIGITAL_TWIN.md / git history for why this matters.
        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            t_arrival = time.time()
            telem = unpack(data)
            echo_t_pc = telem.get("echo_cmd_t_pc")
            hold_ms = telem.get("hold_ms")
            with self.lock:
                self.latest_telem = telem
                self.telem_loss.observe(telem.get("seq"))
                self.cmd_loss.observe(telem.get("last_cmd_seq_recv"))
                self.recv_count += 1
                if echo_t_pc is not None and hold_ms is not None:
                    self.rtts.append((t_arrival - echo_t_pc) * 1000.0 - hold_ms)

    def snapshot(self):
        with self.lock:
            recent = self.rtts[-50:]
            return {
                "telem": self.latest_telem,
                "recv_count": self.recv_count,
                "pi_pc_loss": self.telem_loss.loss_rate(),
                "pc_pi_loss": self.cmd_loss.loss_rate(),
                "rtt_ms": sum(recent) / len(recent) if recent else float("nan"),
            }

    def stop(self):
        self._stop.set()


def run_ping(pi_host, port, duration_s):
    link = BridgeLink(pi_host, port)
    t_start = time.time()
    last_print = t_start

    print(f"ping: sending to {pi_host}:{port} at {COMMAND_HZ:.0f} Hz for {duration_s:.0f}s")

    while time.time() - t_start < duration_s:
        if time.time() - last_print > 1.0:
            snap = link.snapshot()
            print(
                f"sent={link.seq} recv={snap['recv_count']} "
                f"pi->pc_loss={fmt_pct(snap['pi_pc_loss'])} pc->pi_loss={fmt_pct(snap['pc_pi_loss'])} "
                f"rtt_ms={snap['rtt_ms']:.2f}"
            )
            last_print = time.time()
        time.sleep(0.05)

    link.stop()
    print("--- ping done ---")
    if link.recv_count == 0:
        print("WARNING: received 0 telemetry packets -- check --pi-host/--port and that "
              "bridge_server.py is running.")


# ---- Mirror/Command viewer -------------------------------------------------
# Imports mujoco lazily (only reached for the GUI path) so --ping stays
# usable with nothing but the standard library.

def _quat_conj(q):
    import numpy as np
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def _quat_mul(a, b):
    import mujoco
    import numpy as np
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


def _twist_about_z(quat_wxyz):
    """The quat's rotation-about-world-Z (yaw-only) component, as a unit
    quaternion (w,0,0,z). Falls back to identity if degenerate (looking
    straight up/down, where yaw isn't well-defined from this alone)."""
    import numpy as np
    w, x, y, z = quat_wxyz
    norm = math.hypot(w, z)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.array([w / norm, 0.0, 0.0, z / norm])


class YawZero:
    """Mirror-mode body orientation: roll/pitch exactly as measured
    (absolute, from gravity), yaw relative to the first frame received
    this session -- the BNO055 has no fixed heading reference, so raw yaw
    is an arbitrary angle picked at its own power-on, not the room's.
    See docs/DIGITAL_TWIN.md step 4."""

    def __init__(self):
        self.twist0 = None

    def reset(self):
        """Re-zero relative yaw to whatever the next frame reports (the Y
        key in the viewer) -- useful after a magnetometer-free IMU's yaw
        has drifted, or just to re-align without restarting the viewer."""
        self.twist0 = None

    def apply(self, quat_wxyz):
        twist = _twist_about_z(quat_wxyz)
        if self.twist0 is None:
            self.twist0 = twist
        swing = _quat_mul(_quat_conj(twist), quat_wxyz)  # current absolute tilt
        twist_rel = _quat_mul(twist, _quat_conj(self.twist0))  # yaw since first frame
        return _quat_mul(twist_rel, swing)


def _hinge_qpos_addrs(model):
    """{joint_name: qpos index} for every 1-dof hinge joint -- looked up by
    name so this doesn't depend on the model's own joint declaration
    order matching hardware/dcubot_hw_config.json's order."""
    import mujoco
    addrs = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            addrs[model.joint(j).name] = model.jnt_qposadr[j]
    return addrs


FOOT_ARROW_LEN = 0.15  # m -- deliberately large, meant to be unmissable
FOOT_ARROW_RADIUS = 0.018


def _add_foot_markers(scn, site_pos, contact):
    """Small sphere at each foot site (red=no contact, green=pressed) plus
    a big up-arrow when pressed, added to the viewer's own user_scn so
    they render without touching the MJCF. Caller resets scn.ngeom=0 for
    the frame -- this only appends, since the ghost (Command mode) also
    appends into the same scene."""
    import mujoco
    import numpy as np

    identity = np.eye(3).flatten()
    for pos, pressed in zip(site_pos, contact):
        color = [0.1, 0.9, 0.1, 1.0] if pressed else [0.7, 0.1, 0.1, 0.6]
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.015, 0, 0],
            pos=pos,
            mat=identity,
            rgba=color,
        )
        scn.ngeom += 1
        if pressed:
            # Arrow's local +Z is its length; pos is its center, so sit it
            # on top of the foot and let it stick straight up.
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=[FOOT_ARROW_RADIUS, FOOT_ARROW_RADIUS, FOOT_ARROW_LEN],
                pos=pos + np.array([0.0, 0.0, FOOT_ARROW_LEN / 2 + 0.02]),
                mat=identity,
                rgba=[0.1, 0.9, 0.1, 1.0],
            )
            scn.ngeom += 1


def _add_ghost(scn, model, d_ghost, scene_option, pert, alpha=0.35):
    """Appends the real robot's actual pose (d_ghost, already mj_forward'd)
    into the viewer's user_scn as translucent geoms, on top of whatever the
    viewer's own main scene is already showing (the commanded pose). Reuses
    the multi-instance compositing technique from scripts/render_checkpoints.py
    -- mjv_addGeoms, not mjv_updateScene (that would replace the viewer's own
    main scene, which we don't own here)."""
    import mujoco

    before = scn.ngeom
    mujoco.mjv_addGeoms(model, d_ghost, scene_option, pert, mujoco.mjtCatBit.mjCAT_ALL, scn)
    for i in range(before, scn.ngeom):
        scn.geoms[i].rgba[3] = alpha


def run_viewer(pi_host, port, model_path):
    import mujoco
    import mujoco.viewer

    link = BridgeLink(pi_host, port)
    yaw_zero = YawZero()

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)  # also sets d.ctrl to the home pose
    home_z = float(d.qpos[2])
    joint_qpos_adr = _hinge_qpos_addrs(m)
    left_site = m.site("left_foot").id
    right_site = m.site("right_foot").id

    # Command mode: a second MjData for the same model, driven purely from
    # telemetry (the ghost), rendered translucent into the primary d's
    # scene. Only used while mode == "command".
    d_ghost = mujoco.MjData(m)
    scene_option = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    ctrl_idx_by_name = {m.actuator(i).name: i for i in range(m.nu)}

    mode = ["mirror"]  # list so key_callback (closure) can mutate it

    def key_callback(keycode):
        if keycode == 32:  # Space
            link.estop = not link.estop
            print(f"[estop {'ON' if link.estop else 'off'}]")
        elif keycode == ord("T"):
            link.torque_request = not link.torque_request
            print(f"[T] torque request -> {'ON' if link.torque_request else 'off'}")
        elif keycode == ord("M"):
            mode[0] = "command" if mode[0] == "mirror" else "mirror"
            print(f"[M] mode -> {mode[0]}")
        elif keycode == ord("Y"):
            yaw_zero.reset()
            print("[Y] relative yaw re-zeroed to the current heading")
        elif keycode in (265, 264, 263, 262, ord("Q"), ord("E"), ord("S")):
            # Live velocity command for hardware/pi/policy_runner.py (ignored
            # by bridge_server.py's Command mode) -- same one-key-per-axis
            # feel as playground/dcubot/mujoco_infer.py's keyboard viewer, so
            # muscle memory from sim testing carries over directly. Each
            # press sets exactly one axis to the trained range's extreme and
            # zeros the other two; S (or any other unlisted key) stops.
            vx = vy = wz = 0.0
            if keycode == 265:  # up
                vx = policy_constants.LIN_VEL_X_RANGE[1]
            elif keycode == 264:  # down
                vx = policy_constants.LIN_VEL_X_RANGE[0]
            elif keycode == 263:  # left
                vy = policy_constants.LIN_VEL_Y_RANGE[1]
            elif keycode == 262:  # right
                vy = policy_constants.LIN_VEL_Y_RANGE[0]
            elif keycode == ord("Q"):
                wz = policy_constants.ANG_VEL_YAW_RANGE[1]
            elif keycode == ord("E"):
                wz = policy_constants.ANG_VEL_YAW_RANGE[0]
            link.velocity_command = [vx, vy, wz]
            print(f"[velocity] -> {[round(v, 3) for v in link.velocity_command]}")

    print(f"viewer: connecting to {pi_host}:{port}")
    print("keys: Space = e-stop, T = torque request on/off, M = Mirror<->Command, Y = re-zero yaw")
    print(
        "      up/down = forward/back, left/right = strafe, Q/E = turn, S = stop "
        "(velocity_command, for hardware/pi/policy_runner.py only)"
    )
    print("Command mode: move the Control-panel actuator sliders (right side) to set joint targets.")

    last_print = time.time()
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        while viewer.is_running():
            snap = link.snapshot()
            telem = snap["telem"]
            viewer.user_scn.ngeom = 0

            if mode[0] == "command":
                # Primary/opaque robot = commanded pose: teleport straight
                # from the Control-panel sliders' own d.ctrl, base fixed in
                # the air (same treatment as Mirror -- this is a target
                # reference pose, not a measurement).
                for name, idx in ctrl_idx_by_name.items():
                    adr = joint_qpos_adr.get(name)
                    if adr is not None:
                        d.qpos[adr] = d.ctrl[idx]
                d.qpos[0:2] = 0.0
                d.qpos[2] = home_z
                d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                mujoco.mj_forward(m, d)
                link.joint_target = {name: float(d.ctrl[idx]) for name, idx in ctrl_idx_by_name.items()}

                if telem is not None:
                    for name, val in telem["joint_pos"].items():
                        adr = joint_qpos_adr.get(name)
                        if adr is not None:
                            d_ghost.qpos[adr] = val
                    d_ghost.qpos[0:2] = 0.0
                    d_ghost.qpos[2] = home_z
                    d_ghost.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                    mujoco.mj_forward(m, d_ghost)
                    _add_ghost(viewer.user_scn, m, d_ghost, scene_option, pert)
            elif telem is not None:
                # Mirror mode: primary/opaque robot = measured pose. Echo
                # the measured joints back as joint_target so torque
                # staying on in this mode (e.g. the user pressed T out of
                # curiosity) commands no motion -- target == current.
                final_quat = yaw_zero.apply(telem["imu_quat_wxyz"])
                d.qpos[0:2] = 0.0
                d.qpos[2] = home_z
                d.qpos[3:7] = final_quat
                for name, val in telem["joint_pos"].items():
                    adr = joint_qpos_adr.get(name)
                    if adr is not None:
                        d.qpos[adr] = val
                mujoco.mj_forward(m, d)
                link.joint_target = dict(telem["joint_pos"])

            if telem is not None:
                contact = telem.get("foot_contact", {})
                _add_foot_markers(
                    viewer.user_scn,
                    [d.site_xpos[left_site].copy(), d.site_xpos[right_site].copy()],
                    [contact.get("left", False), contact.get("right", False)],
                )

            viewer.sync()

            if time.time() - last_print > 0.5:
                if telem is not None:
                    print(
                        f"[{mode[0]}] bridge loop_hz={telem.get('loop_hz')} servo_read_ms={telem.get('servo_read_ms')} "
                        f"fault={telem.get('fault')} torque_enabled={telem.get('torque_enabled')} | "
                        f"link recv={snap['recv_count']} pi->pc_loss={fmt_pct(snap['pi_pc_loss'])} "
                        f"pc->pi_loss={fmt_pct(snap['pc_pi_loss'])} rtt_ms={snap['rtt_ms']:.2f}"
                    )
                else:
                    print("waiting for telemetry ...")
                last_print = time.time()

            time.sleep(0.01)

    link.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi-host", help="bridge_server.py's address (not needed with --mock)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--mock", action="store_true", help="start a local mock bridge instead of --pi-host")
    parser.add_argument("--ping", action="store_true", help="no-GUI connection test: Hz/RTT/loss, then exit")
    parser.add_argument("--ping-seconds", type=float, default=10.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.mock:
        start_local_mock_bridge(args.port)
        pi_host = "127.0.0.1"
    else:
        if not args.pi_host:
            parser.error("--pi-host is required unless --mock is set")
        pi_host = args.pi_host

    if args.ping:
        run_ping(pi_host, args.port, args.ping_seconds)
        return

    run_viewer(pi_host, args.port, args.model)


if __name__ == "__main__":
    main()
