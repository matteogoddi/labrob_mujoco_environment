"""Exercise the real DDS client against the test peer inside docker --network none."""
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import time

assert {path.name for path in Path("/sys/class/net").iterdir()} == {"lo"}
controller, fixture = sys.argv[1:]
drop = Path("/tmp/step1-drop")
wrong = Path("/tmp/step1-wrong-mode")


def scenario(test):
    drop.unlink(missing_ok=True)
    wrong.unlink(missing_ok=True)
    delay = Path("/tmp/step1-delay")
    compete = Path("/tmp/step1-compete")
    delay.unlink(missing_ok=True)
    compete.unlink(missing_ok=True)
    with open("/tmp/step1-peer.log", "w") as log:
        peer = subprocess.Popen([fixture], stdout=log, stderr=log)
        pid, fd = pty.fork()
        if pid == 0:
            os.execl(controller, controller, "lo")
        output = bytearray()

        def wait_for(marker, timeout=6):
            end = time.monotonic() + timeout
            while marker.encode() not in output:
                if time.monotonic() >= end:
                    raise AssertionError((marker, output.decode(errors="replace")))
                if select.select([fd], [], [], 0.1)[0]:
                    try:
                        output.extend(os.read(fd, 8192))
                    except OSError as error:
                        raise AssertionError((marker, output.decode(errors="replace"), Path(log.name).read_text())) from error
            del output[:output.index(marker.encode()) + len(marker)]

        def key(value):
            os.write(fd, value.encode())

        try:
            wait_for("Motion service: ai")
            key("1")
            time.sleep(0.15)
            assert "ON" not in Path(log.name).read_text()
            if test == "wrong":
                wrong.touch()
                time.sleep(0.15)
                key("d")
                wait_for("Need one fresh Mode 5")
                key("x")
            else:
                key("d")
                wait_for("y confirms")
                if test == "pending":
                    delay.touch()
                key("y")
                if test == "pending":
                    time.sleep(.05)
                    key("x")
                else:
                    wait_for("Arms idle")
                    time.sleep(0.15)
                    assert "ON" not in Path(log.name).read_text()
                    key("1")
                    wait_for("Step 1 moving")
                    if test == "restore":
                        wait_for("measured at target", 8)
                        key("r")
                        wait_for("y confirms")
                        wait_for("Confirmation expired", 12)
                        key("r")
                        wait_for("y confirms")
                        time.sleep(0.15)
                        key("y")
                        wait_for("Prior service restored")
                    elif test == "compete":
                        compete.touch()
                        wait_for("Feedback or control ownership lost")
                    elif test == "drop":
                        drop.touch()
                        wait_for("Feedback or control ownership lost")
                    else:
                        time.sleep(.1)
                        key(" ")
            wait_for("Test stopped")
            _, status = os.waitpid(pid, 0)
            pid = 0
            assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == int(test in ("compete", "drop"))
            time.sleep(0.15)
            events = [line for line in Path(log.name).read_text().splitlines()
                      if line in {"RELEASE", "RESTORE", "ON", "OFF"}]
            expected = [] if test == "wrong" else ["RELEASE", "ON", "OFF"]
            if test == "pending":
                expected = ["RELEASE"]
            if test == "compete":
                expected = ["RELEASE", "ON"]
            if test == "restore":
                expected.append("RESTORE")
                line = next(v for v in Path(log.name).read_text().splitlines() if v.startswith("DELTA "))
                offsets = list(map(float, line.split()[1:]))
                target = [-.0081, .22, .0701, .2183, -.0081, -.22, -.0701, .2183]
                assert len(offsets) == 8 and all(abs(a-b) < 1e-4 for a, b in zip(offsets, target))
            assert events == expected, (test, events, expected)
            print(test, "passed", flush=True)
        finally:
            if pid:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            os.close(fd)
            peer.terminate()
            peer.wait(timeout=3)


for case in ("wrong", "pending", "stop", "drop", "compete", "restore"):
    scenario(case)
