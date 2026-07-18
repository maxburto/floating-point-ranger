"""Stall-watch — the refined never-preempt reaper.

A granted lease whose process does no GPU work for `stall_window_s` is only SUSPECTED stalled.
Before anything is touched we ask the owner (the lease's registered `health_url`) "is this a
normal dead-spot or actually stalled?". Vouched-alive → keep (reset the clock). Stalled / no
answer / no health_url → FLAG it in the journal with resolved owner attribution, and — only when
`enforce` is on — evict it (stop its unit, or SIGTERM the pid) and release the lease.

Hard guardrails, always: never a model-manager lease (models idle-drain themselves), never a
lease on an interactive-role GPU (the desktop is never a target), never a working or vouched job.
NO operator alerts — this module never touches the notify path; flagged jobs live in the
log/dashboard for periodic review. Feature-flagged: `stall_watch.enabled=false` → pure idle-drain.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import time

import requests

from . import mps, owners, probes, util
from .admission import Leases
from .config import Config

_idle_since: dict[str, float] = {}   # lease_id -> when it first went idle
_logged: set[str] = set()            # lease_ids already logged STALLED (flag-only de-dupe)
_flagged: list[dict] = []            # last tick's flagged jobs (for /v1/gpu/status + dashboard)


def current() -> list[dict]:
    return _flagged


def _log(msg: str) -> None:
    # journal only — never a push notification (flag-in-log is the contract; alert fatigue kills alerting)
    print(f"[stall-watch] {msg}", file=sys.stderr, flush=True)


def _health(health_url, lease_id, label, timeout):
    """None = no health_url (no vouch possible); True = owner vouched alive; False = stalled/negative
    (a non-200, timeout, connection error, or unparseable body all read as 'no vouch')."""
    if not health_url:
        return None
    try:
        r = requests.get(health_url, params={"lease": lease_id, "label": label}, timeout=timeout)
        if r.status_code == 200:
            return bool(r.json().get("alive"))
    except Exception:  # noqa: BLE001
        return False
    return False


_MANAGER_UNIT = "gpu-manager.service"


def _pid_starttime(pid) -> str | None:
    """/proc/<pid>/stat field 22 (starttime) — identifies THIS incarnation of a pid, so a
    recycled pid (same number, different process) is detectable before we ever signal it."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _evictable_unit(info: dict) -> str | None:
    """The systemd unit to `systemctl stop` — ONLY a concrete app `*.service`. NEVER a shared
    `session-*`/`user@*`/`*.slice` scope (stopping those kills a whole login session or a UID's
    entire process tree — owners.attribute() can fall back to one) and never the manager itself.
    Anything else → None, and the caller falls back to SIGTERM-ing just the pid."""
    if info.get("kind") != "systemd":
        return None
    unit = info.get("unit") or ""
    if not unit.endswith(".service") or unit == _MANAGER_UNIT:
        return None
    if unit.startswith(("session-", "user@", "user-")) or ".slice" in unit:
        return None
    return unit


def _evict(lease: dict, leases: Leases, pid_st0: str | None) -> str:
    """Stop a confirmed zombie: its owning app-unit if safely resolvable, else SIGTERM the pid
    (never SIGKILL first, never pid<=2, never our own pid, never a session/user/slice scope).
    Guards against a pid recycled during the health-ping window — the start-time must still
    match what we sampled — so we never signal an unrelated process. Always releases the lease."""
    pid = lease.get("pid")
    if pid and _pid_starttime(pid) != pid_st0:  # recycled or gone during the health window
        leases.release(lease["id"])
        return "pid gone/recycled — lease released only, process untouched"
    how = "no-op"
    unit = _evictable_unit(owners.attribute(pid))
    if unit:
        rc = subprocess.run(["systemctl", "stop", unit],
                            capture_output=True, text=True, timeout=30).returncode
        how = f"stop {unit} rc={rc}"
    elif pid and pid > 2 and pid != os.getpid() and os.path.exists(f"/proc/{pid}"):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)
        how = f"SIGTERM {pid}"
    leases.release(lease["id"])
    return how


def _tick(cfg: Config, leases: Leases) -> None:
    global _flagged
    sw = cfg.stall_watch
    snap = probes.nvml_snapshot([g.uuid for g in cfg.gpus])
    interactive = {g.uuid for g in cfg.gpus if g.role == "interactive"}
    # A remote card's processes live in ANOTHER guest, so their pid numbers mean nothing here:
    # attributing or signalling one would hit an unrelated LOCAL process. Stall-watch is
    # local-only; cross-node cards are reported (amd.py), never reaped.
    local_uuids = {g.uuid for g in cfg.gpus if g.probe != "remote-amd"}
    active = leases.active()
    activity = util.sample_activity(snap, active, sw.active_util_pct)
    now = time.time()
    live_ids: set[str] = set()
    flagged: list[dict] = []
    for h in active:
        lid = h["id"]
        # eligibility: a local pid, not a manager-owned model, not on the interactive card
        if (not h.get("pid") or h.get("initiator") == "model-manager"
                or h["gpu_uuid"] in interactive or h["gpu_uuid"] not in local_uuids):
            continue
        live_ids.add(lid)
        if activity.get(lid):
            _idle_since.pop(lid, None)
            _logged.discard(lid)
            continue
        started = _idle_since.setdefault(lid, now)
        idle_s = now - started
        if idle_s < sw.stall_window_s:
            continue
        pid_st0 = _pid_starttime(h.get("pid"))  # capture pid identity BEFORE the blocking health-ping
        vouch = _health(h.get("health_url"), lid, h["label"], sw.health_timeout_s)
        if vouch is True:
            _idle_since[lid] = now  # grace: re-check a full window from now
            _logged.discard(lid)
            _log(f"lease {lid[:8]} ({h['label']}) idle {int(idle_s)}s — owner vouched alive, kept")
            continue
        owner = h.get("owner") or owners.attribute(h.get("pid")).get("owner")
        vouch_s = "no" if vouch is False else "none"
        if sw.enforce:
            action = "evicted: " + _evict(h, leases, pid_st0)
            # only for an actual MPS client: concurrency genuinely active AND this lease was
            # co-executing (model units and exclusive leases are not MPS clients)
            if mps.concurrency_active(cfg, h["gpu_uuid"]) and not h.get("exclusive"):
                # NVIDIA: killing an MPS client without draining its GPU work can leave the MPS
                # server in an undefined state. Surface it rather than silently assuming it's
                # fine — full handling is on the B2 checklist, before `enforce` is ever flipped.
                _log(f"WARNING: evicted an MPS client on {h['gpu_uuid']} — the MPS server may "
                     f"need a bounce (systemctl restart nvidia-mps) if later jobs misbehave")
            _idle_since.pop(lid, None)
            _logged.discard(lid)
            _log(f"STALLED lease {lid[:8]} owner={owner} pid={h['pid']} label={h['label']} "
                 f"idle={int(idle_s)}s vouch={vouch_s} — {action}")
        else:
            action = "would-evict (enforce off)"
            if lid not in _logged:  # de-dupe the flag-only log; the dashboard shows it every tick
                _log(f"STALLED lease {lid[:8]} owner={owner} pid={h['pid']} label={h['label']} "
                     f"idle={int(idle_s)}s vouch={vouch_s} — {action}")
                _logged.add(lid)
        flagged.append({"lease_id": lid, "label": h["label"], "owner": owner, "pid": h["pid"],
                        "gpu_uuid": h["gpu_uuid"], "idle_s": int(idle_s), "vouch": vouch_s,
                        "action": action, "at": now})
    for lid in [k for k in _idle_since if k not in live_ids]:  # forget gone leases
        _idle_since.pop(lid, None)
        _logged.discard(lid)
    _flagged = flagged


async def run(cfg: Config, leases: Leases) -> None:
    while True:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(_tick, cfg, leases)
        await asyncio.sleep(cfg.stall_watch.interval_s)
