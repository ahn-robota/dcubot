"""PC-side viewer/controller for the dcubot digital twin bridge.

--ping runs a no-GUI connection test (Hz / RTT / packet loss in both
directions), then exits. --mock skips --pi-host: it starts the exact same
mock Bridge from hardware/pi/bridge_server.py in-process on loopback, so
this can be tested with no Raspberry Pi and no second terminal.

The Mirror/Command 3D viewer (MuJoCo window, Control-panel sliders, ghost,
foot-switch arrows) lands in a later step -- see docs/DIGITAL_TWIN.md.
"""
import argparse
import socket
import sys
import time
from pathlib import Path
from threading import Event, Lock, Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import COMMAND_HZ, DEFAULT_PORT, SeqLossTracker, make_command, pack, unpack  # noqa: E402


def start_local_mock_bridge(port):
    """Runs hardware/pi/bridge_server.py's mock Bridge in a background
    thread of this same process, bound to loopback. Reuses that one
    MockSensors implementation instead of duplicating fake data here."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi"))
    from bridge_server import Bridge, build_arg_parser

    args = build_arg_parser().parse_args(["--port", str(port), "--mock"])
    Thread(target=lambda: Bridge(args).run(), daemon=True).start()
    time.sleep(0.2)  # let it bind before we start sending


class _PingState:
    def __init__(self):
        self.lock = Lock()
        self.telem_loss = SeqLossTracker()
        self.cmd_loss = SeqLossTracker()  # PC->Pi, inferred from last_cmd_seq_recv
        self.recv_count = 0
        self.rtts = []


def _ping_recv_loop(sock, state, stop_event):
    """Timestamps each telemetry packet's arrival the instant recvfrom()
    returns, in its own thread -- decoupled from the 20Hz send tick, so
    that cadence doesn't add to the measured RTT."""
    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(4096)
        except socket.timeout:
            continue
        except OSError:
            return
        t_arrival = time.time()
        telem = unpack(data)
        echo_t_pc = telem.get("echo_cmd_t_pc")
        hold_ms = telem.get("hold_ms")
        with state.lock:
            state.telem_loss.observe(telem.get("seq"))
            state.cmd_loss.observe(telem.get("last_cmd_seq_recv"))
            state.recv_count += 1
            if echo_t_pc is not None and hold_ms is not None:
                rtt_ms = (t_arrival - echo_t_pc) * 1000.0 - hold_ms
                state.rtts.append(rtt_ms)


def run_ping(pi_host, port, duration_s):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.5)
    state = _PingState()
    stop_event = Event()
    Thread(target=_ping_recv_loop, args=(sock, state, stop_event), daemon=True).start()

    seq = 0
    t_start = time.time()
    last_print = t_start

    print(f"ping: sending to {pi_host}:{port} at {COMMAND_HZ:.0f} Hz for {duration_s:.0f}s")

    while time.time() - t_start < duration_s:
        seq += 1
        cmd = make_command(seq=seq, t_pc=time.time(), torque_request=False, estop=False, joint_target={})
        sock.sendto(pack(cmd), (pi_host, port))

        if time.time() - last_print > 1.0:
            with state.lock:
                recent_rtts = state.rtts[-50:]
                recv_count = state.recv_count
                pi_pc_loss = state.telem_loss.loss_rate()
                pc_pi_loss = state.cmd_loss.loss_rate()
            avg_rtt = sum(recent_rtts) / len(recent_rtts) if recent_rtts else float("nan")
            fmt_pct = lambda x: "warming up" if x is None else f"{x:.1%}"  # noqa: E731
            print(
                f"sent={seq} recv={recv_count} "
                f"pi->pc_loss={fmt_pct(pi_pc_loss)} pc->pi_loss={fmt_pct(pc_pi_loss)} "
                f"rtt_ms={avg_rtt:.2f}"
            )
            last_print = time.time()

        time.sleep(1.0 / COMMAND_HZ)

    stop_event.set()
    print("--- ping done ---")
    if state.recv_count == 0:
        print("WARNING: received 0 telemetry packets -- check --pi-host/--port and that "
              "bridge_server.py is running.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi-host", help="bridge_server.py's address (not needed with --mock)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--mock", action="store_true", help="start a local mock bridge instead of --pi-host")
    parser.add_argument("--ping", action="store_true", help="no-GUI connection test: Hz/RTT/loss, then exit")
    parser.add_argument("--ping-seconds", type=float, default=10.0)
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

    raise NotImplementedError(
        "Mirror/Command 3D viewer lands in a later step -- see "
        "docs/DIGITAL_TWIN.md. Use --ping for now."
    )


if __name__ == "__main__":
    main()
