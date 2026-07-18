"""MPS state — read-only view of NVIDIA Multi-Process Service on the batch card.

MPS lets several CUDA processes share one GPU context so batch jobs **co-execute** instead of
serialising on the exclusive lease lane. This module only *reports* whether it is up; the
concurrency policy itself lives in `admission.py` (the exclusive->non-exclusive downgrade).

Facts this is built around, measured live (Ampere RTX 3060, driver 595.71.05):
  * MPS works on this consumer GeForce — a control daemon pinned to GPU0 spawned a server and
    `get_client_list` showed two CUDA clients attached and co-resident.
  * Compute mode stays **DEFAULT**, never EXCLUSIVE_PROCESS: EXCLUSIVE_PROCESS would let ONLY
    the MPS server touch the card, and our docker model units (speaches/kokoro/qwen) have their
    own /tmp so they never see the MPS pipe — they would be locked off GPU0 entirely. In DEFAULT
    mode, MPS clients use MPS and everything else keeps its own context, side by side.
  * Servers are per-UID (a root control daemon spawns one server per client UID), so
    co-execution happens within a UID — which covers the common case of two `gpu-lease` jobs.
  * MPS enables co-execution; it does NOT create capacity. VRAM remains the binding constraint
    on a card that also hosts resident models, and the VRAM-floor admission math still rules.
"""
from __future__ import annotations

import os
import sys

from .config import Config

_warned: set[str] = set()


def _cfg(cfg: Config):
    return getattr(cfg, "mps", None)


def _valid_card(cfg: Config) -> bool:
    """The configured MPS card must exist in `gpus:` and must NOT be the interactive one.

    `mps.gpu_uuid` is duplicated in the unit's CUDA_VISIBLE_DEVICES and the gpus list, held
    together only by convention — a typo pointing it at the desktop card would downgrade
    INTERACTIVE leases to non-exclusive, which is the one invariant we never bend. Refuse and
    warn once rather than act on a bad uuid."""
    m = _cfg(cfg)
    if not m or not m.gpu_uuid:
        return False
    card = next((g for g in cfg.gpus if g.uuid == m.gpu_uuid), None)
    problem = None
    if card is None:
        problem = "is not present in gpus:"
    elif card.role == "interactive":
        problem = "is the INTERACTIVE card — MPS must never govern the desktop"
    if problem:
        if m.gpu_uuid not in _warned:
            _warned.add(m.gpu_uuid)
            print(f"[mps] REFUSING to enable MPS: configured gpu_uuid {m.gpu_uuid} {problem}",
                  file=sys.stderr, flush=True)
        return False
    return True


def is_mps_card(cfg: Config, gpu_uuid: str) -> bool:
    m = _cfg(cfg)
    return bool(m and m.enabled and m.gpu_uuid == gpu_uuid and _valid_card(cfg))


def server_up(cfg: Config) -> bool:
    """Is the MPS control daemon ACTUALLY running?

    Deliberately NOT "does the pipe directory exist". An unclean stop leaves that directory
    behind, and trusting it is a live-verified false positive: batch leases kept being
    downgraded to non-exclusive while no MPS was running at all — i.e. the exclusive lane was
    gone with nothing replacing it. So read the daemon's own pid file and confirm the process
    is alive AND really is the control daemon. One small file read + one /proc read; cheap
    enough for the status endpoint's polling rate."""
    m = _cfg(cfg)
    if not m or not m.enabled:
        return False
    try:
        with open(os.path.join(m.pipe_dir, "nvidia-cuda-mps-control.pid")) as f:
            pid = int(f.read().strip())
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"nvidia-cuda-mps-control" in f.read()
    except (OSError, ValueError):
        return False


def concurrency_active(cfg: Config, gpu_uuid: str) -> bool:
    """Should batch work co-execute on this card RIGHT NOW? Requires the card to be the MPS
    card, concurrency not disabled by `batch_exclusive`, AND the daemon actually running.

    The server_up check is the important one: if MPS is configured but its daemon is dead, we
    must NOT keep downgrading leases to non-exclusive — that would drop the exclusive-lane
    protection while delivering no MPS. A dead daemon therefore fails SAFE, back to the
    pre-Phase-A exclusive-serial behaviour."""
    m = _cfg(cfg)
    return bool(is_mps_card(cfg, gpu_uuid) and m and not m.batch_exclusive and server_up(cfg))


def card_state(cfg: Config, gpu_uuid: str) -> dict | None:
    """Per-card MPS block for /v1/gpu/status; None for cards MPS does not apply to."""
    m = _cfg(cfg)
    if not m or not m.enabled or m.gpu_uuid != gpu_uuid:
        return None
    return {"enabled": True, "server_up": server_up(cfg),
            # when batch_exclusive is on, MPS is running but batch still serialises (rollback
            # without stopping the daemon)
            "concurrent_batch": not m.batch_exclusive}
