"""Rogue-grab watch — detect GPU compute processes that bypassed the manager.

Every interval: list compute processes on managed GPUs (NVML), subtract what the manager
knows about (lease-registered PIDs, its own model units' cgroups) and what config
explicitly allowlists (cmdline patterns for not-yet-migrated consumers and always-on
serving). Anything left above the VRAM crumb threshold is a ROGUE: alert once per PID via
ntfy (basic auth from env NTFY_USER/NTFY_PASSWORD; journal-only when unset) and surface it
on /v1/gpu/status. THE PROCESS IS NEVER TOUCHED — this is posture, not enforcement
(enforcement = DeviceAllow default-deny on service units, Phase 3).
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time

import requests

from .admission import Leases
from .config import Config

_alerted: dict[int, float] = {}   # pid -> first-seen
_current: list[dict] = []         # last tick's rogues (for the status endpoint)


def current() -> list[dict]:
    return _current


def _cmdline(pid: int) -> str:
    with contextlib.suppress(OSError):
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    return ""


def _notify(cfg: Config, title: str, body: str) -> None:
    url, topic = os.environ.get("NTFY_URL", ""), os.environ.get("NTFY_TOPIC", "")
    user, pw = os.environ.get("NTFY_USER", ""), os.environ.get("NTFY_PASSWORD", "")
    print(f"[rogue-watch] {title}: {body}", file=sys.stderr, flush=True)  # journal record
    if not (url and topic and user and pw):
        return
    with contextlib.suppress(Exception):
        requests.post(f"{url.rstrip('/')}/{topic}", data=body,
                      headers={"Title": title, "Priority": "default", "Tags": "gpu"},
                      auth=(user, pw), timeout=10)


def _model_unit_pids(cfg: Config) -> set[int]:
    """MainPIDs of the manager's own model units — never rogues."""
    import subprocess
    pids = set()
    for m in cfg.models:
        try:
            out = subprocess.run(["systemctl", "show", "-p", "MainPID", "--value", m.unit],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            if out and out != "0":
                pids.add(int(out))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
    return pids


def _tick(cfg: Config, leases: Leases) -> None:
    global _current
    from . import owners, probes  # local import keeps module import cheap for tests
    rw = cfg.rogue_watch
    snap = probes.nvml_snapshot([g.uuid for g in cfg.gpus])
    known_pids = {h["pid"] for h in leases.active() if h.get("pid")} | _model_unit_pids(cfg)
    rogues = []
    for g in cfg.gpus:
        for p in snap.get(g.uuid, {}).get("processes", []):
            if p["kind"] != "compute":
                continue  # graphics clients = the interactive side we protect, never rogues
            if (p["used_mib"] or 0) < rw.min_vram_mib:
                continue
            if p["pid"] in known_pids:
                continue
            cmd = _cmdline(p["pid"])
            if any(pat in cmd for pat in rw.allow_patterns):
                continue
            # no anonymous jobs: resolve WHO owns this unmanaged process (unit/docker/user/cmd)
            owner = owners.attribute(p["pid"]).get("owner", "unknown")
            rogues.append({"gpu": g.name, "gpu_uuid": g.uuid, "pid": p["pid"],
                           "name": p["name"], "used_mib": p["used_mib"], "owner": owner,
                           "cmdline": cmd[:200]})
    _current = rogues
    for r in rogues:
        if r["pid"] in _alerted:
            continue
        _alerted[r["pid"]] = time.time()
        _notify(cfg, f"[gpu-manager] unmanaged GPU process on {r['gpu']}",
                f"pid {r['pid']} ({r['name']}, {r['used_mib']} MiB, owner={r.get('owner', '?')}) is "
                f"using {r['gpu']} without a lease. Not touched (never-preempt). cmdline: {r['cmdline']}")
    # forget exited pids so a recycled pid can alert again
    live = {r["pid"] for r in rogues}
    for pid in [p for p in _alerted if p not in live and not os.path.exists(f"/proc/{p}")]:
        _alerted.pop(pid, None)


async def run(cfg: Config, leases: Leases) -> None:
    while True:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(_tick, cfg, leases)
        await asyncio.sleep(cfg.rogue_watch.interval_s)
