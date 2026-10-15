"""`aerosplat up` supervision: a worker that dies is restarted, stop() ends them all."""

import sys
import time

from aerosplat.server import up


def test_supervisor_restarts_a_dead_worker(monkeypatch):
    spawned = []

    class FakeProc:
        def __init__(self, code):
            self.returncode = code
            self._code = code

        def poll(self):
            return self._code

    def fake_spawn(self, resource):
        # First process dies at once (exit 1), the restarted one keeps running.
        proc = FakeProc(1 if not spawned else None)
        spawned.append(proc)
        self.procs[resource] = proc
        self.started[resource] = time.monotonic()

    monkeypatch.setattr(up.WorkerSupervisor, "_spawn", fake_spawn)
    monkeypatch.setattr(up, "RESTART_DELAYS_S", (0.0,))
    monkeypatch.setattr(up, "POLL_S", 0.02)
    killed = []
    monkeypatch.setattr(up, "kill_tree", lambda p: killed.append(p))

    sup = up.WorkerSupervisor(["gpu"])
    sup.start_all()
    deadline = time.time() + 5
    while time.time() < deadline and len(spawned) < 2:
        time.sleep(0.02)
    sup.stop()
    sup.join(timeout=2)
    assert len(spawned) == 2 and sup.failures["gpu"] == 1
    assert killed == [spawned[1]]          # stop() kills the live worker only


def test_supervisor_real_process_stops(tmp_path, monkeypatch):
    # A real child that just sleeps stands in for the worker module.
    def spawn(self, resource):
        import subprocess

        self.procs[resource] = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                                **up.group_kwargs())
        self.started[resource] = time.monotonic()

    monkeypatch.setattr(up.WorkerSupervisor, "_spawn", spawn)
    sup = up.WorkerSupervisor(["cpu"])
    sup.start_all()
    proc = sup.procs["cpu"]
    assert proc.poll() is None
    sup.stop()
    assert proc.poll() is not None


def test_render_service_is_supervised_like_a_worker(monkeypatch):
    spawned = []

    class FakeProc:
        returncode = None

        def poll(self):
            return None

    def fake_popen(cmd, **kw):
        spawned.append(cmd)
        return FakeProc()

    monkeypatch.setattr(up.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(up, "kill_tree", lambda p: None)
    sup = up.WorkerSupervisor(["cpu", up.RENDER], {up.RENDER: ["render-cmd", "--port", "9"]})
    sup.start_all()
    sup.stop()
    sup.join(timeout=2)
    assert ["render-cmd", "--port", "9"] in spawned
    assert any(c[-2:] == ["--resource", "cpu"] for c in spawned)       # plain workers keep their command


def test_render_command_uses_the_gsplat_env_script_only_when_needed(monkeypatch, tmp_path):
    env_bat = tmp_path / "tools" / "gsplat_env.bat"
    env_bat.parent.mkdir()
    env_bat.write_text("@echo off\n")
    monkeypatch.setattr(up, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(up.os, "name", "nt")
    monkeypatch.setitem(sys.modules, "gsplat", type(sys)("gsplat"))     # a gsplat without csrc: JIT build
    cmd = up.render_service_command(8123)
    assert cmd[:3] == ["cmd.exe", "/c", str(env_bat)]
    assert cmd[-2:] == ["--port", "8123"] and "aerosplat.render.service" in cmd
    csrc = type(sys)("gsplat.csrc")
    monkeypatch.setitem(sys.modules, "gsplat.csrc", csrc)               # prebuilt: no toolchain wrapper
    monkeypatch.setattr(sys.modules["gsplat"], "csrc", csrc, raising=False)
    assert up.render_service_command(8123)[0] == sys.executable
    monkeypatch.setattr(up.os, "name", "posix")
    assert up.render_service_command(8123)[0] == sys.executable
