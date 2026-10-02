"""Bounded process groups with log, RSS, artifact, and wall-time supervision."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable


POLL_SECONDS = 0.1


def _tree_rss(pid: int) -> int | None:
    if not sys.platform.startswith("linux"):
        return None
    parents: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            child = int(entry.name)
            try:
                stat_line = (entry / "stat").read_text()
                close = stat_line.rfind(")")
                fields = stat_line[close + 2:].split()
                parents.setdefault(int(fields[1]), []).append(child)
                rss[child] = int((entry / "statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                continue
        owned = {pid}
        pending = [pid]
        while pending:
            pending_pid = pending.pop()
            for child in parents.get(pending_pid, []):
                if child not in owned:
                    owned.add(child)
                    pending.append(child)
        if pid not in rss:
            return None
        return sum(rss.get(item, 0) for item in owned)
    except OSError:
        return None


def _group_has_live_process(group: int) -> bool:
    if sys.platform.startswith("linux"):
        try:
            for entry in Path("/proc").iterdir():
                if not entry.name.isdecimal():
                    continue
                try:
                    stat_line = (entry / "stat").read_text()
                    fields = stat_line[stat_line.rfind(")") + 2:].split()
                    if int(fields[2]) == group and fields[0] not in {"Z", "X"}:
                        return True
                except (OSError, ValueError, IndexError):
                    continue
            return False
        except OSError:
            return True
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _signal_group(group: int, sig: int) -> None:
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def _terminate_group(group: int, process: subprocess.Popen | None, grace: float = 0.05) -> dict:
    """Escalate the owned process group even if its leader exited first."""
    try:
        _signal_group(group, signal.SIGTERM)
        if process is not None:
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        until = time.monotonic() + grace
        while _group_has_live_process(group) and time.monotonic() < until:
            time.sleep(0.01)
        if _group_has_live_process(group):
            _signal_group(group, signal.SIGKILL)
        if process is not None:
            try:
                process.wait(timeout=max(grace, 0.1))
            except subprocess.TimeoutExpired:
                return {"success": False, "reaped": False, "error": "leader did not reap after SIGKILL"}
        until = time.monotonic() + max(grace, 0.1)
        while _group_has_live_process(group) and time.monotonic() < until:
            time.sleep(0.01)
        alive = _group_has_live_process(group)
        return {"success": not alive, "reaped": process is None or process.poll() is not None,
                "error": "owned process group remains after SIGKILL" if alive else None}
    except Exception as exc:
        return {"success": False, "reaped": process is None or process.poll() is not None,
                "error": str(exc)}


class OwnedProcess:
    """Supervise one new session; a background watchdog remains active during caller I/O."""

    def __init__(self, argv: list[str], timeout_seconds: float, rss_limit_bytes: int,
                 log_limit_bytes: int, cleanup_reserve_seconds: float | None = None,
                 artifact_size: Callable[[], int] | None = None,
                 artifact_limit_bytes: int | None = None, merge_stderr: bool = False,
                 require_running: bool = False):
        self.started = time.monotonic()
        self.timeout_seconds = timeout_seconds
        self.deadline = self.started + timeout_seconds
        if cleanup_reserve_seconds is None:
            cleanup_reserve_seconds = min(0.4, max(0.03, timeout_seconds * 0.05))
        self.run_deadline = max(self.started, self.deadline - cleanup_reserve_seconds)
        self.rss_limit_bytes = rss_limit_bytes
        self.log_limit_bytes = log_limit_bytes
        self.artifact_size = artifact_size
        self.artifact_limit_bytes = artifact_limit_bytes
        self.require_running = require_running
        self._lock = threading.RLock()
        self._terminate_lock = threading.Lock()
        self._cleanup_result: dict | None = None
        self._status: str | None = None
        self._closed = False
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "HOME": os.environ.get("HOME", "/nonexistent"),
               "LANG": "C", "LC_ALL": "C", "HF_HUB_OFFLINE": "1"}
        self.process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            start_new_session=True, close_fds=True, env=env, bufsize=0,
        )
        self.group = self.process.pid
        self.logs = {"stdout": bytearray(), "stderr": bytearray()}
        self._log_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._start_drain(self.process.stdout, "stdout")
        if not merge_stderr:
            self._start_drain(self.process.stderr, "stderr")
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()

    def _start_drain(self, stream, label: str):
        thread = threading.Thread(target=self._drain, args=(stream, label), daemon=True)
        thread.start()
        self._threads.append(thread)

    def _drain(self, stream, label: str):
        selector = selectors.DefaultSelector()
        try:
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)
            while True:
                events = selector.select(0.1)
                if not events and self.process.poll() is not None:
                    return
                for key, _mask in events:
                    try:
                        chunk = os.read(key.fileobj.fileno(), 8192)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        return
                    with self._lock:
                        buf = self.logs[label]
                        room = self.log_limit_bytes - len(buf)
                        buf.extend(chunk[:max(0, room)])
                        if len(chunk) > room:
                            self._log_event.set()
        finally:
            selector.close()

    def _watch(self):
        while not self._closed:
            if self._status is not None:
                self._terminate_owned()
                return
            if time.monotonic() >= self.run_deadline:
                self._status = "timeout"
            elif self._log_event.is_set():
                self._status = "log_limit"
            elif self.process.poll() is not None:
                if self.require_running:
                    self._status = "exit_error"
                else:
                    return
            else:
                current_rss = _tree_rss(self.process.pid)
                if current_rss is not None and current_rss > self.rss_limit_bytes:
                    self._status = "rss_limit"
                elif self.artifact_size is not None and self.artifact_limit_bytes is not None:
                    try:
                        if self.artifact_size() > self.artifact_limit_bytes:
                            self._status = "artifact_limit"
                    except Exception:
                        self._status = "artifact_watch_error"
            if self._status is not None:
                self._terminate_owned()
                return
            time.sleep(POLL_SECONDS)

    def check(self) -> str | None:
        if self._status is None and time.monotonic() >= self.run_deadline:
            self._status = "timeout"
        if self._status is None and self.require_running and self.process.poll() is not None:
            self._status = "exit_error"
        return self._status

    def run_monitored(self, operation: Callable[[], object]):
        """Run potentially blocking I/O while this object's watchdog remains active."""
        outcome: dict[str, object] = {}
        completed = threading.Event()

        def invoke():
            try:
                outcome["value"] = operation()
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                completed.set()

        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        while not completed.wait(POLL_SECONDS):
            if self._status is not None:
                return self._status
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    def close(self) -> dict:
        if self._closed:
            return {"success": False, "reaped": False, "error": "process supervisor already closed"}
        self._closed = True
        self._watchdog.join(timeout=0.25)
        cleanup = self._terminate_owned()
        for thread in self._threads:
            thread.join(timeout=0.05)
        for stream in (self.process.stdout, self.process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        # A short-lived child can exit between watchdog polls. Check its final
        # artifact footprint after all owned writers have been stopped.
        if cleanup["success"] and self.artifact_size is not None and self.artifact_limit_bytes is not None:
            try:
                if self.artifact_size() > self.artifact_limit_bytes:
                    self._status = "artifact_limit"
            except Exception:
                self._status = "artifact_watch_error"
        return cleanup

    def _terminate_owned(self) -> dict:
        with self._terminate_lock:
            if self._cleanup_result is None:
                self._cleanup_result = _terminate_group(self.group, self.process)
            return self._cleanup_result

    @property
    def status(self) -> str | None:
        return self._status

    @property
    def stdout(self) -> str:
        return self.logs["stdout"].decode("utf-8", "replace")

    @property
    def stderr(self) -> str:
        return self.logs["stderr"].decode("utf-8", "replace")

    @property
    def rss_scope(self) -> str:
        return "linux_owned_process_tree_sampled_100ms" if sys.platform.startswith("linux") else "unavailable_non_linux"


def run_child(argv: list[str], timeout_seconds: float, rss_limit_bytes: int,
              log_limit_bytes: int, sample_interval: float = POLL_SECONDS,
              input_bytes: bytes | None = None, artifact_size: Callable[[], int] | None = None,
              artifact_limit_bytes: int | None = None) -> dict:
    """Run an argv without a shell; continuously drain and bound both output channels."""
    started = time.monotonic()
    if input_bytes is not None:
        # Current caller payloads are capped well below the pipe capacity.
        command_input = input_bytes
    else:
        command_input = None
    try:
        owned = OwnedProcess(argv, timeout_seconds, rss_limit_bytes, log_limit_bytes,
                             artifact_size=artifact_size, artifact_limit_bytes=artifact_limit_bytes)
    except (OSError, ValueError) as exc:
        return {"status": "startup_error", "stdout": "", "stderr": "", "returncode": None,
                "elapsed_seconds": None, "rss_scope": "linux_owned_process_tree_sampled_100ms",
                "cleanup": {"success": True, "reaped": True, "error": None}, "error": str(exc)}
    if command_input is not None:
        # Child stdin is intentionally closed for safe, deadlock-free bounded execution.
        cleanup = owned.close()
        return {"status": "input_unsupported", "stdout": owned.stdout, "stderr": owned.stderr,
                "returncode": owned.process.returncode, "elapsed_seconds": time.monotonic()-started,
                "rss_scope": owned.rss_scope, "cleanup": cleanup, "error": "stdin input is not supported"}
    while owned.process.poll() is None and owned.check() is None:
        time.sleep(min(sample_interval, POLL_SECONDS))
    # close() performs bounded TERM/KILL escalation and reaping; do not wait
    # through the wall deadline for a child that has already tripped a watchdog.
    cleanup = owned.close()
    status = owned.status
    if status is None and owned.process.returncode != 0:
        status = "exit_error"
    if status is None:
        status = "complete"
    if status == "complete" and owned._log_event.is_set():
        status = "log_limit"
    if not cleanup["success"]:
        status = "cleanup_error"
    return {"status": status, "stdout": owned.stdout, "stderr": owned.stderr,
            "returncode": owned.process.returncode, "elapsed_seconds": time.monotonic()-started,
            "rss_scope": owned.rss_scope, "cleanup": cleanup,
            "error": None if status == "complete" else status}


def stop_child(process, process_group_id: int | None = None,
               timeout_seconds: float = 1.0) -> dict:
    """Close only an OwnedProcess handle, optionally checking its group ID."""
    if not isinstance(process, OwnedProcess):
        return {"cleanup": {"success": False, "reaped": False,
                            "error": "an owned process handle is required"}}
    if process_group_id is not None and process_group_id != process.group:
        return {"cleanup": {"success": False, "reaped": False,
                            "error": "process group does not belong to the supplied handle"}}
    return {"cleanup": process.close()}
