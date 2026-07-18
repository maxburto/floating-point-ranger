"""Owner attribution — no anonymous GPU jobs.

Given a host PID, resolve WHO owns it, so any GPU process (even one that never went through
the lease API) is traceable to a real service/unit/script. Resolution order, most
specific first:

  1. docker container name (if the pid is inside a container) — via the cgroup path;
  2. systemd unit (system or user) — from /proc/<pid>/cgroup;
  3. login user (owner uid → name);
  4. cmdline basename.

Returns a compact dict the Ranger stores on the lease and logs for flagged/unattributed jobs.
Pure stdlib + /proc + a best-effort `docker inspect`; degrades gracefully.
"""
from __future__ import annotations

import contextlib
import os
import pwd
import re
import subprocess

_DOCKER_RE = re.compile(r"docker[-/]([0-9a-f]{12,64})")
_SCOPE_RE = re.compile(r"/([A-Za-z0-9@._-]+\.(?:service|scope))")
_docker_name: dict[str, str] = {}  # container-id -> name (cache)


def _cgroup(pid: int) -> str:
    with contextlib.suppress(OSError):
        with open(f"/proc/{pid}/cgroup") as f:
            return f.read()
    return ""


def _uid_name(pid: int) -> str | None:
    with contextlib.suppress(OSError, KeyError):
        uid = os.stat(f"/proc/{pid}").st_uid
        return pwd.getpwuid(uid).pw_name
    return None


def _cmdline(pid: int) -> str:
    with contextlib.suppress(OSError):
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv0 = f.read().split(b"\0", 1)[0].decode(errors="replace").split(" ", 1)[0]
        return os.path.basename(argv0)
    return "?"


def _docker_container(cid: str) -> str | None:
    if cid in _docker_name:
        return _docker_name[cid]
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        out = subprocess.run(["docker", "inspect", "-f", "{{.Name}}", cid[:12]],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        if out:
            name = out.lstrip("/")
            _docker_name[cid] = name
            return name
    return None


def attribute(pid: int | None) -> dict:
    """Best-effort owner of a host pid. Always returns a dict with an `owner` string."""
    if not pid:
        return {"owner": "unknown", "kind": "none"}
    cg = _cgroup(pid)
    m = _DOCKER_RE.search(cg)
    if m:
        name = _docker_container(m.group(1))
        if name:
            return {"owner": f"docker:{name}", "kind": "docker",
                    "container": name, "pid": pid, "cmd": _cmdline(pid)}
    units = _SCOPE_RE.findall(cg)
    unit = next((u for u in units if not u.startswith(("docker-", "session-", "user@"))), None) \
        or (units[-1] if units else None)
    if unit:
        return {"owner": f"unit:{unit}", "kind": "systemd", "unit": unit,
                "pid": pid, "cmd": _cmdline(pid), "user": _uid_name(pid)}
    user = _uid_name(pid)
    cmd = _cmdline(pid)
    return {"owner": f"user:{user}:{cmd}" if user else f"cmd:{cmd}", "kind": "process",
            "pid": pid, "cmd": cmd, "user": user}
