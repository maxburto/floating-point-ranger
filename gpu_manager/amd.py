"""Remote AMD card state — the Ranger's cross-node view of an AMD card in another guest.

The card lives in a different guest, so the Ranger cannot probe it directly (the cross-VM GPU
rule: a GPU lives in ONE guest). A thin read-only agent inside that guest
(`agent/fpr_amd_agent.py`) serves its sysfs + DRM-fdinfo state; we poll it
here and shape the result exactly like `probes.nvml_snapshot()` so every caller stays uniform.

**FAIL-OPEN is the contract.** If the agent is unreachable or its data goes stale, the card is
marked offline and AMD work simply keeps running unmanaged. A dead reporter must NEVER block
admission — least of all on the NVIDIA cards, which are governed independently.

This module reports; it never dispatches. "Routing to AMD" means recording/validating that a
job's home is a service resident in the card's guest — the Ranger never moves a process
between guests.
"""
from __future__ import annotations

import asyncio
import contextlib
import sys
import time

import requests

from .config import Config, GpuCfg

_snap: dict[str, dict] = {}                    # gpu_uuid -> last poll result
_prev_engine: dict[str, dict[str, int]] = {}   # gpu_uuid -> {client_id: cumulative compute ns}

STALE_AFTER_S = 120  # a snapshot older than this is not trustworthy for admission


def snapshot(gpu_uuid: str) -> dict | None:
    """Last good state, or None when the agent is unreachable / never answered / stale."""
    s = _snap.get(gpu_uuid)
    if not s or not s.get("online"):
        return None
    if time.time() - (s.get("checked_at") or 0) > STALE_AFTER_S:
        return None
    return s


def probe_snapshot(g: GpuCfg) -> dict | None:
    """Shape the remote state like probes.nvml_snapshot()[uuid] so admission/status callers
    need no special-casing beyond picking the source."""
    rs = snapshot(g.uuid)
    if not rs:
        return None
    card = rs.get("card") or {}
    # A reachable agent can still report a card it cannot read (renderD128 renumbered, amdgpu
    # reloaded, card unbound) — it answers ok:true with all-None values. That is NOT usable for
    # admission: hand it back as offline rather than letting None reach the VRAM arithmetic.
    if not card.get("present") or any(card.get(k) is None for k in
                                      ("vram_total_mib", "vram_used_mib", "vram_free_mib")):
        return None
    procs = [{"pid": c.get("pid"), "name": c.get("comm", "?"), "kind": "compute",
              "used_mib": c.get("vram_mib"), "util_pct": c.get("compute_util_pct"),
              "client_id": c.get("client_id")}
             for c in rs.get("clients", [])]
    return {
        "mem_total_mib": card.get("vram_total_mib"),
        "mem_used_mib": card.get("vram_used_mib"),
        "mem_free_mib": card.get("vram_free_mib"),
        "processes": procs,
        # busy_percent is NOT reliable on this card (reads 0 with a model resident) — carried
        # for display only; never use it as an activity signal.
        "util_gpu": None,
        "proc_util": {},
        "remote": True,
        "sampled_at": rs.get("sampled_at"),
    }


def _poll_one(g: GpuCfg, timeout: float) -> None:
    data = None
    if g.agent_url:
        try:
            r = requests.get(f"{g.agent_url.rstrip('/')}/v1/amd/state", timeout=timeout)
            if r.status_code == 200:
                data = r.json()
        except Exception:  # noqa: BLE001 — unreachable/timeout/parse all mean "offline"
            data = None
    prev = _snap.get(g.uuid)
    if not data or not data.get("ok"):
        if prev is None or prev.get("online"):
            print(f"[amd] {g.name}: agent unreachable — marking OFFLINE; AMD work continues "
                  f"unmanaged (fail-open)", file=sys.stderr, flush=True)
        _snap[g.uuid] = {**(prev or {}), "online": False, "checked_at": time.time()}
        # Drop the engine baseline: on recovery a delta measured against a pre-outage counter
        # would be spread over the whole outage and read as a meaningless (often 100%) spike.
        _prev_engine.pop(g.uuid, None)
        return
    # Guard against pointing at the wrong reporter: the configured uuid embeds the card's PCI
    # address, so a mismatch means this agent is serving a DIFFERENT card and its state must not
    # be attributed here.
    pdev = (data.get("card") or {}).get("pdev")
    if pdev and not g.uuid.endswith(pdev):
        if prev is None or prev.get("online"):
            print(f"[amd] {g.name}: agent reports pdev {pdev} which does not match configured "
                  f"uuid {g.uuid} — marking OFFLINE (wrong agent_url?)", file=sys.stderr, flush=True)
        _snap[g.uuid] = {**(prev or {}), "online": False, "checked_at": time.time()}
        _prev_engine.pop(g.uuid, None)
        return
    now = data.get("sampled_at") or time.time()
    clients = data.get("clients", [])
    # per-client compute utilization from cumulative engine-ns deltas (the only usable
    # activity signal on this card — see the agent's module docstring)
    prev_eng = _prev_engine.get(g.uuid, {})
    dt = now - (prev.get("sampled_at") or 0) if prev and prev.get("sampled_at") else 0
    cur_eng: dict[str, int] = {}
    for c in clients:
        ns = int((c.get("engine_ns") or {}).get("compute", 0))
        cid = c.get("client_id")
        cur_eng[cid] = ns
        if dt > 0 and cid in prev_eng:
            delta = max(0, ns - prev_eng[cid])
            c["compute_util_pct"] = round(min(100.0, delta / (dt * 1e9) * 100), 1)
        else:
            c["compute_util_pct"] = None
    _prev_engine[g.uuid] = cur_eng
    if prev is not None and not prev.get("online"):
        print(f"[amd] {g.name}: agent back ONLINE", file=sys.stderr, flush=True)
    _snap[g.uuid] = {"card": data.get("card", {}), "clients": clients, "sampled_at": now,
                     "online": True, "checked_at": time.time()}


def _tick(cfg: Config) -> None:
    for g in cfg.gpus:
        if g.probe == "remote-amd":
            _poll_one(g, cfg.amd_poll_timeout_s)


async def run(cfg: Config) -> None:
    while True:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(_tick, cfg)
        await asyncio.sleep(cfg.amd_poll_interval_s)
