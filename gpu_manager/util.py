"""Per-lease activity sampling — the coarse "is this job doing GPU work?" signal (Phase B).

On these consumer GeForce cards `nvmlDeviceGetProcessUtilization` is unreliable: it raises
NOT_FOUND for the WHOLE call when the card is idle (so a stalled 0%-util process is simply
ABSENT from the per-process samples, not reported as "<5%"), and its device scoping leaks pids
across handles. We therefore treat "absent from proc_util" as 0% util and lean on three signals,
ANY of which marks a lease ACTIVE — erring toward "alive" keeps the never-preempt invariant safe:

  1. the lease's pid is in proc_util with smUtil >= threshold      (definitely computing)
  2. it is the sole NON-MODEL compute consumer on its card AND the card's device-wide util
     (reliable) is >= threshold                                     (attribute card activity)
  3. its resident VRAM changed materially since the previous tick   (allocation churn = working)

Off-host (no NVML) every lease reads inactive, which just means the stall loop falls through to
the authoritative owner health-ping. The device-wide util comes from probes.nvml_snapshot's
`util_gpu`; per-process smUtil from its `proc_util` (NOT_FOUND-tolerant → {}).

Also holds `pid_alive()` — the process-liveness primitive that lets the reaper tell a
DEAD-holder lease from a live-but-idle one.
"""
from __future__ import annotations

import os

_last_vram: dict[int, int] = {}  # pid -> used_mib last tick (module-level churn memory)
_VRAM_CHURN_MIB = 50


def pid_alive(pid) -> bool:
    """Does this pid still exist?

    This is the distinction the whole zombie-reaping path turns on: a lease whose holder is
    GONE has no work behind it, so expiring it is bookkeeping and the never-preempt invariant
    does not apply. A lease whose holder is ALIVE but idle is the judgment call that
    `stall_watch.enforce` gates.

    Errs toward ALIVE by construction: a recycled pid (same number, unrelated process) reads
    as alive, and that lease simply falls back to its full `ttl_s`. The only possible error is
    reaping LATER than ideal — never expiring a live job's reservation early. That asymmetry
    is why no pid-starttime bookkeeping is needed here.

    A falsy pid returns False, but that is NOT the same as "pid gone": callers must handle
    "no pid was ever registered" separately, since nothing can be concluded about such a lease.
    """
    if not pid:
        return False
    return os.path.exists(f"/proc/{pid}")


def sample_activity(snap: dict, active_leases: list[dict], active_util_pct: int = 5) -> dict[str, bool]:
    """Return {lease_id: is_active} for every granted lease. `snap` is a probes.nvml_snapshot."""
    # a pid computing on ANY card counts as alive (device scoping is unreliable on GeForce)
    proc_util: dict[int, int] = {}
    for s in snap.values():
        for pid, sm in (s.get("proc_util") or {}).items():
            proc_util[pid] = max(proc_util.get(pid, 0), sm)
    model_pids = {h["pid"] for h in active_leases
                  if h.get("initiator") == "model-manager" and h.get("pid")}
    out: dict[str, bool] = {}
    seen: set[int] = set()
    for h in active_leases:
        pid = h.get("pid")
        out[h["id"]] = _pid_active(pid, h.get("gpu_uuid"), snap, proc_util, model_pids, active_util_pct)
        if pid:
            seen.add(pid)
    for pid in [p for p in _last_vram if p not in seen]:  # forget churn memory for gone pids
        _last_vram.pop(pid, None)
    return out


def _pid_active(pid, gpu_uuid, snap, proc_util, model_pids, thr) -> bool:
    if not pid:
        return False
    if proc_util.get(pid, 0) >= thr:                                    # signal 1
        return True
    s = snap.get(gpu_uuid, {})
    compute = [p for p in s.get("processes", []) if p.get("kind") == "compute"]
    non_model = [p for p in compute if p["pid"] not in model_pids]
    if (s.get("util_gpu") or 0) >= thr and len(non_model) == 1 \
            and non_model[0]["pid"] == pid:                             # signal 2
        return True
    used = next((p.get("used_mib") for p in compute if p["pid"] == pid), None)  # signal 3
    active = False
    if used is not None:
        prev = _last_vram.get(pid)
        if prev is not None and abs(used - prev) >= _VRAM_CHURN_MIB:
            active = True
        _last_vram[pid] = used
    return active
