"""Model residency — staged, admission-gated systemd model servers.

A "model" is a systemd unit the manager alone starts/stops (e.g. a llama.cpp
`llama-server` unit), declared in config with its GPU target and VRAM floor.

ensure(name) is the single consumer verb:
  resident            unit is active (refreshes the idle clock)
  loading             admission passed and the unit was just started
  deferred:<reasons>  admission failed — the caller queues/retries; NOTHING was disturbed

Admission reuses the lease registry: a model holds a NON-exclusive lease (VRAM
reservation) for as long as its unit runs, so batch admission math sees resident models
and vice versa. Idle models are drained (unit stopped + lease released) after
`idle_drain_s` without an ensure — stopping OUR OWN model unit is not preemption; foreign
processes are never touched. A nightly pre-warm window is a timer POSTing ensure.
"""
from __future__ import annotations

import asyncio
import subprocess
import threading
import time

from .admission import Leases
from .config import Config, ModelCfg

_state: dict[str, dict] = {}  # name -> {last_ensure: float, lease_id: str|None}
_state_lock = threading.Lock()  # ensure() (threadpool) and drain_loop (worker) both mutate _state


def _systemctl(*args: str) -> tuple[int, str]:
    r = subprocess.run(["systemctl", *args], capture_output=True, text=True, timeout=30)
    return r.returncode, (r.stdout + r.stderr).strip()


def _unit_active(unit: str) -> str:
    rc, out = _systemctl("is-active", unit)
    return out.splitlines()[0] if out else ("active" if rc == 0 else "inactive")


def _unit_mainpid(unit: str) -> int | None:
    rc, out = _systemctl("show", "-p", "MainPID", "--value", unit)
    try:
        pid = int(out.strip())
        return pid or None
    except ValueError:
        return None


def _find(cfg: Config, name: str) -> ModelCfg | None:
    return next((m for m in cfg.models if m.name == name), None)


def _reattach(m: ModelCfg, st: dict, leases: Leases) -> None:
    """Re-bind a running model's VRAM reservation: adopt the orphan granted lease if one
    survives, else request a fresh non-exclusive one. Caller holds _state_lock."""
    matches = [h for h in leases.active(m.gpu_uuid)
               if h["initiator"] == "model-manager" and h["label"] == m.name]
    if matches:
        st["lease_id"] = matches[0]["id"]
        for extra in matches[1:]:
            leases.release(extra["id"])  # release duplicate orphans from prior restarts
        return
    # vram_mib=0 on reattach: the running model's memory is ALREADY allocated, so NVML
    # free reflects it — a full floor here would double-count and be wrongly denied.
    res = leases.request(gpu=m.gpu_uuid, initiator="model-manager", label=m.name,
                         vram_mib=0, exclusive=False, ttl_s=m.idle_drain_s + 300)
    if res.get("granted"):
        st["lease_id"] = res["lease_id"]
    else:
        import sys
        print(f"[models] WARNING: resident {m.name} has NO reservation "
              f"(reattach denied: {res.get('reasons')})", file=sys.stderr, flush=True)


def ensure(cfg: Config, leases: Leases, name: str) -> dict:
    m = _find(cfg, name)
    if m is None:
        return {"state": "unknown-model", "models": [x.name for x in cfg.models]}
    with _state_lock:
        st = _state.setdefault(name, {"last_ensure": 0.0, "lease_id": None})
        st["last_ensure"] = time.time()
        unit_state = _unit_active(m.unit)
        if unit_state == "active":
            if st.get("lease_id") is None:
                # resident-but-unreserved (post-restart ensure raced the adoption tick):
                # reattach the orphan reservation, or take a fresh one, so batch admission
                # keeps seeing this model's VRAM floor.
                _reattach(m, st, leases)
            return {"state": "resident", "unit": m.unit, "port": m.port}
        if unit_state == "activating":
            return {"state": "loading", "unit": m.unit, "port": m.port}
        # admission: non-exclusive VRAM reservation on the target card, then start the unit
        res = leases.request(gpu=m.gpu_uuid, initiator="model-manager", label=name,
                             vram_mib=m.vram_mib, exclusive=False, ttl_s=m.idle_drain_s + 300)
        if not res["granted"]:
            return {"state": "deferred", "reasons": res["reasons"], "retry_in_s": res.get("retry_in_s")}
        rc, out = _systemctl("start", m.unit)
        if rc != 0:
            leases.release(res["lease_id"])
            return {"state": "error", "detail": out[-300:]}
        st["lease_id"] = res["lease_id"]
        return {"state": "loading", "unit": m.unit, "port": m.port}


def status(cfg: Config) -> list[dict]:
    out = []
    for m in cfg.models:
        st = _state.get(m.name, {})
        out.append({"name": m.name, "unit": m.unit, "unit_state": _unit_active(m.unit),
                    "gpu_uuid": m.gpu_uuid, "vram_mib": m.vram_mib, "port": m.port,
                    "idle_s": round(time.time() - st["last_ensure"]) if st.get("last_ensure") else None})
    return out


def _drain_tick(cfg: Config, leases: Leases) -> None:
    now = time.time()
    for m in cfg.models:
        with _state_lock:  # per-model critical section: a concurrent ensure() can't lose its lease
            st = _state.get(m.name)
            if _unit_active(m.unit) != "active":
                # unit not running (manual stop / crash / drained) — it must NOT keep a
                # reservation, or its floor blocks the card (and its own restart) for a model
                # that isn't there. Reconcile: release any stale model-manager lease for it.
                for h in leases.active(m.gpu_uuid):
                    if h["initiator"] == "model-manager" and h["label"] == m.name:
                        leases.release(h["id"])
                if st:
                    st["lease_id"] = None
                continue
            if st is None:
                # manager restarted under a running model: ADOPT it (fresh idle clock +
                # re-attach its granted reservation) instead of instantly draining it
                _state[m.name] = st = {"last_ensure": now, "lease_id": None}
                _reattach(m, st, leases)
                continue
            last = st["last_ensure"] if st else 0.0
            if now - last > m.idle_drain_s:
                _systemctl("stop", m.unit)
                if st and st.get("lease_id"):
                    leases.release(st["lease_id"])
                    st["lease_id"] = None
            elif st and st.get("lease_id"):
                # keep the reservation alive AND bound to the unit's pid — the pid lets
                # admission net the model's realized allocation out of its reserved floor
                leases.heartbeat(st["lease_id"], pid=_unit_mainpid(m.unit))


async def drain_loop(cfg: Config, leases: Leases) -> None:
    while True:
        await asyncio.to_thread(_drain_tick, cfg, leases)
        await asyncio.sleep(60)
