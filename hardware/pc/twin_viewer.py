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
from threading import Thread

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


def run_ping(pi_host, port, duration_s):
    # Telemetry arrives at 50Hz but we only send/tick at COMMAND_HZ (20Hz) --
    # a blocking single recvfrom() per tick would let the socket's receive
    # buffer back up, so every read would return an increasingly stale
    # packet (RTT would grow without bound). Drain everything queued each
    # tick instead, non-blocking, and only use the most recent for RTT.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    telem_loss = SeqLossTracker()
    cmd_loss = SeqLossTracker()  # PC->Pi, inferred from last_cmd_seq_recv
    seq = 0
    recv_count = 0
    rtts = []
    t_start = time.time()
    last_print = t_start

    print(f"ping: sending to {pi_host}:{port} at {COMMAND_HZ:.0f} Hz for {duration_s:.0f}s")

    while time.time() - t_start < duration_s:
        seq += 1
        cmd = make_command(seq=seq, t_pc=time.time(), torque_request=False, estop=False, joint_target={})
        sock.sendto(pack(cmd), (pi_host, port))

        latest_telem = None
        while True:
            try:
                data, _ = sock.recvfrom(4096)
            except BlockingIOError:
                break
            telem = unpack(data)
            telem_loss.observe(telem.get("seq"))
            cmd_loss.observe(telem.get("last_cmd_seq_recv"))
            recv_count += 1
            latest_telem = telem
        if latest_telem is not None:
            echo_t_pc = latest_telem.get("echo_cmd_t_pc")
            if echo_t_pc is not None:
                rtts.append((time.time() - echo_t_pc) * 1000.0)

        if time.time() - last_print > 1.0:
            recent_rtts = rtts[-50:]
            avg_rtt = sum(recent_rtts) / len(recent_rtts) if recent_rtts else float("nan")
            print(
                f"sent={seq} recv={recv_count} "
                f"pi->pc_loss={telem_loss.loss_rate():.1%} "
                f"pc->pi_loss={cmd_loss.loss_rate():.1%} "
                f"rtt_ms={avg_rtt:.1f}"
            )
            last_print = time.time()

        time.sleep(1.0 / COMMAND_HZ)

    print("--- ping done ---")
    if recv_count == 0:
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
