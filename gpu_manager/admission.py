"""Lease registry + check-and-defer admission.

The arbitration core. Consumers ask for a lease on a GPU (by UUID or role) declaring a
VRAM floor; the manager GRANTS or DEFERS. ADMISSION never preempts, evicts, or signals any
running process. (Refined never-preempt invariant: the ONE place a job can be
evicted is the separate stall-watch reaper in `stall.py`, and only a job confirmed <5% util
for the stall window whose owner does NOT vouch it alive — never a working/vouched job, never
the interactive desktop, and only when `stall_watch.enforce` is on.) Decision inputs at grant time:

  1. interactive hold — an interactive-role GPU with an active hold marker admits nothing;
  2. real free VRAM (NVML, live) minus the reservations of already-granted leases whose
     processes may not have allocated yet — both must cover the request plus a safety margin;
  3. per-GPU serial lanes ("exclusive" leases) — at most one exclusive lease per GPU.

Leases are rows in a WAL SQLite DB (survives restarts). Holders heartbeat; the reaper expires a
lease on either of two clocks — the short `leases.zombie_ttl_s` once its registered pid is
provably GONE, or its own `ttl_s` otherwise (see `reap()`). Bookkeeping only: the process itself
is never touched. Everything is observable via /v1/gpu/status and the dashboard.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
import uuid as uuidlib

from . import amd, mps, owners, probes, util
from .config import Config

MARGIN_MIB = 512  # safety margin between "free" and "grantable"
# Capabilities that describe a card but are NEVER a lease lane (see request()).
NON_LEASABLE_CAPS = frozenset({"vaapi"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lease (
    id          TEXT PRIMARY KEY,
    gpu_uuid    TEXT NOT NULL,
    initiator   TEXT NOT NULL,
    label       TEXT NOT NULL,
    vram_mib    INTEGER NOT NULL DEFAULT 0,
    exclusive   INTEGER NOT NULL DEFAULT 1,
    state       TEXT NOT NULL DEFAULT 'granted',   -- granted | released | expired
    pid         INTEGER,
    created_at  REAL NOT NULL,
    heartbeat   REAL NOT NULL,
    ttl_s       INTEGER NOT NULL DEFAULT 900,
    released_at REAL
);
CREATE INDEX IF NOT EXISTS ix_lease_state ON lease(state, gpu_uuid);
-- belt-and-braces: the DB itself enforces "at most one exclusive lease per GPU"
CREATE UNIQUE INDEX IF NOT EXISTS ux_lease_excl ON lease(gpu_uuid)
    WHERE state='granted' AND exclusive=1;
"""

# Slurm-lite job metadata added to the lease row. Migrated as ADD COLUMNs so an
# existing state.db upgrades in place. owner=UID resolvable to a real service;
# capability=cuda|vulkan|vaapi|cpu (routing); needs_model=staged dependency; job_type=batch|oneshot;
# health_url=owner endpoint the stall reaper pings (Phase B).
_META_COLUMNS = (
    ("owner", "TEXT"), ("capability", "TEXT"), ("needs_model", "TEXT"),
    ("job_type", "TEXT DEFAULT 'oneshot'"), ("health_url", "TEXT"),
)


class Leases:
    """All methods serialize on one lock: endpoints run in FastAPI's threadpool and the
    background loops (drain, rogue, reaper) run in worker threads, all sharing this one
    SQLite connection — the lock makes check-and-insert admission atomic (no TOCTOU
    double-grant of the exclusive lane / VRAM over-commit) and keeps commits from
    interleaving across threads."""

    def __init__(self, cfg: Config, db_path: str | None = None):
        self.cfg = cfg
        self._lock = threading.RLock()
        p = db_path or os.environ.get("GPU_MANAGER_STATE", "/opt/gpu-manager/state.db")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        self.conn = sqlite3.connect(p, timeout=10, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        for col, decl in _META_COLUMNS:  # in-place migration (harmless if already present)
            try:
                self.conn.execute(f"ALTER TABLE lease ADD COLUMN {col} {decl}")
            except sqlite3.OperationalError:
                pass
        self.conn.commit()

    # -- queries ------------------------------------------------------------
    def active(self, gpu_uuid: str | None = None) -> list[dict]:
        q = "SELECT * FROM lease WHERE state='granted'"
        args: tuple = ()
        if gpu_uuid:
            q += " AND gpu_uuid=?"
            args = (gpu_uuid,)
        with self._lock:
            return [dict(r) for r in self.conn.execute(q + " ORDER BY created_at", args)]

    def reap(self) -> int:
        """Expire granted leases whose holder is gone or whose heartbeat has gone stale.

        Bookkeeping only — a lease whose registered pid is still ALIVE is never expired: a
        stalled-but-running job outranks its own missed heartbeats, and dropping its row would
        also drop its VRAM floor out of the admission math (over-admission on a card MPS
        deliberately packs tighter). This guard used to be keyed on `exclusive`, which silently
        stopped protecting EVERY batch job the moment Phase A made batch leases non-exclusive —
        precisely the class of job it was written for.

        TWO clocks, because "the holder is dead" and "the holder stopped talking" are different
        facts and only one of them is a judgment call:

          pid-gone  — the lease registered a pid and that pid no longer exists. Nothing is
                      running behind the reservation, so it expires after the much shorter
                      `leases.zombie_ttl_s` rather than waiting out a CLIENT-chosen `ttl_s`
                      that has no server-side ceiling. Not preemption: there is no process
                      left to preempt, so this is deliberately NOT gated on
                      `stall_watch.enforce`.
          ttl       — the pre-existing path, unchanged. Covers leases that never registered a
                      pid (about whose holder nothing can be concluded) and is the fallback
                      whenever the short clock is disabled or has not yet elapsed.

        Every expiry is logged with its reason. Before this, expiry was completely silent, so
        "did the reaper ever run?" was unanswerable from the journal — which is exactly how a
        lease released manually 5 minutes early got recorded as "never reaped" (FPR #1).
        """
        now = time.time()
        zttl = self.cfg.leases.zombie_ttl_s
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, pid, initiator, label, vram_mib, gpu_uuid, heartbeat, ttl_s "
                "FROM lease WHERE state='granted'").fetchall()
            doomed: list[tuple] = []
            for r in rows:
                stale_s = now - r["heartbeat"]
                # A pid we can still see = a live holder. Never expired, on either clock.
                if r["pid"] and util.pid_alive(r["pid"]):
                    continue
                if r["pid"] and zttl > 0 and stale_s > zttl:
                    doomed.append((r, "pid-gone"))
                elif stale_s > r["ttl_s"]:
                    doomed.append((r, "ttl"))
            for r, reason in doomed:
                self.conn.execute(
                    "UPDATE lease SET state='expired', released_at=? WHERE id=?", (now, r["id"]))
                print(f"[reaper] expired lease {r['id'][:8]} reason={reason} "
                      f"initiator={r['initiator']} label={r['label']} pid={r['pid']} "
                      f"vram_mib={r['vram_mib']} stale={int(now - r['heartbeat'])}s "
                      f"ttl_s={r['ttl_s']} gpu={r['gpu_uuid']}", file=sys.stderr, flush=True)
            self.conn.commit()
        return len(doomed)

    # -- admission ----------------------------------------------------------
    def _resolve_gpu(self, gpu: str, capability: str | None = None, vram_mib: int = 0) -> list:
        """A UUID, or a role name → candidate GpuCfgs. For a ROLE ask carrying a capability we
        narrow to the cards that offer it, which is what implements the ROUTING POLICY: a
        `vulkan` batch job resolves to the AMD card rather than a scarce CUDA card. An explicit
        UUID ask is still honoured as-is (and capability-checked below)."""
        by_uuid = [g for g in self.cfg.gpus if g.uuid == gpu]
        if by_uuid:
            return by_uuid
        cands = [g for g in self.cfg.gpus if g.role == gpu]
        if capability:
            # EXCLUSIVE, never best-effort: a "no match → fall back to all cards" widening is
            # how a job lands on a card that cannot run it.
            cands = [g for g in cands if capability in g.caps]
        else:
            # No capability declared = a legacy caller. `gpu-lease` defaults to `--gpu batch`
            # and sends none, then pins CUDA_VISIBLE_DEVICES to whatever card we return — so
            # handing it a REMOTE AMD card silently gives its CUDA job zero devices (and
            # strands a lease on a card in another guest). Legacy callers only ever get local
            # cards.
            cands = [g for g in cands if g.probe != "remote-amd"]
        # Bounded spillover: a role ask may ALSO be offered interactive cards that opt in
        # (batch_spillover_max_mib > 0) — appended LAST, so batch-role cards are always
        # preferred and the interactive card only takes overflow. Only for BOUNDED asks
        # (0 < vram_mib <= cap): a job that won't declare its footprint never gets the
        # desktop card. Interactive priority itself is the hold machinery, which gates
        # admission on these cards exactly as it always has.
        if gpu != "interactive":
            spill = [g for g in self.cfg.gpus
                     if g.role == "interactive" and g.probe != "remote-amd"
                     and 0 < vram_mib <= g.batch_spillover_max_mib
                     and (capability in g.caps if capability else True)]
            cands = cands + [g for g in spill if g not in cands]
        return cands

    def request(self, gpu: str, initiator: str, label: str, vram_mib: int = 0,
                exclusive: bool = True, ttl_s: int = 900, pid: int | None = None,
                owner: str | None = None, capability: str | None = None,
                needs_model: str | None = None, job_type: str = "oneshot",
                health_url: str | None = None) -> dict:
        """Try to grant a lease. Returns {granted: bool, ...} — deferral carries reasons.
        owner/capability/needs_model/job_type/health_url = Slurm-lite job metadata:
        owner is auto-resolved from the pid when the caller doesn't supply one."""
        self.reap()
        if capability in NON_LEASABLE_CAPS:
            # Fixed-function transcode (Jellyfin/VAAPI) runs inline and is NEVER gated: it is
            # not measurable on the AMD card (no dec/enc engine counters) and must never be
            # blocked. Advertised for routing/visibility, refused as a lease lane.
            return {"granted": False, "retry_in_s": None, "reasons": [
                f"capability '{capability}' is never leased — fixed-function transcode runs "
                f"inline and is never gated"]}
        candidates = self._resolve_gpu(gpu, capability, vram_mib)
        if not candidates:
            why = (f"no card in role '{gpu}' offers capability '{capability}'" if capability
                   else f"unknown gpu or role: {gpu}")
            return {"granted": False, "reasons": [why], "retry_in_s": None}
        # NVML snapshot outside the lock (slow); the check-and-insert below is atomic under it.
        local_uuids = [g.uuid for g in candidates if g.probe != "remote-amd"]
        snap = probes.nvml_snapshot(local_uuids) if local_uuids else {}
        for g in candidates:  # remote cards come from their reporter agent, never NVML
            if g.probe == "remote-amd":
                rs = amd.probe_snapshot(g)
                if rs is not None:
                    snap[g.uuid] = rs
        reasons = []
        with self._lock:
            for g in candidates:
                why = []
                # capability gate: never hand a job a card that can't run it (a CUDA job must
                # never land on the AMD card, and per the ROUTING POLICY Vulkan work stays off
                # the reserved CUDA cards). Callers that declare no capability are unaffected.
                if capability and capability not in g.caps:
                    why.append(f"capability '{capability}' not available on {g.name} "
                               f"(offers: {'/'.join(g.caps)})")
                # Holds gate EVERY card, not just interactive-role ones. The runbook has always
                # documented "pause batch on a card -> nothing new admits", but this condition
                # used to be `g.role == "interactive"`, so a hold set on the batch card was
                # silently ignored. Interactive priority depends on this.
                # A hold means "pause BATCH on this card" — it must NOT deny `model-manager`.
                # Denying it would (a) silently break every gated model consumer (STT/TTS/OCR), since
                # models.ensure() maps a deferral to "not resident", and (b) deny _reattach()'s
                # vram_mib=0 request for an ALREADY-RESIDENT model after a manager restart,
                # dropping that model's VRAM floor out of the admission math while it still
                # holds real VRAM — which over-admits batch. "Restart gpu-manager" is step 2 of
                # the documented rollback, so that path is reachable in normal operation.
                hold_markers = probes.holds(self.cfg.hold_dir, g.uuid)
                if hold_markers and initiator != "model-manager":
                    why.append("hold active" + (" (interactive card)" if g.role == "interactive"
                                                else f" on {g.name}"))
                # MPS concurrency: on the MPS card, batch work CO-EXECUTES rather than taking the
                # serial lane, so an exclusive ask is downgraded server-side (no client change
                # needed — `gpu-lease` keeps its CLI). VRAM-floor admission below is still the
                # real gate; MPS enables sharing, it does not create capacity. Model-manager
                # leases are already non-exclusive. `mps.batch_exclusive` disables the downgrade.
                eff_exclusive = exclusive
                if (exclusive and initiator != "model-manager"
                        and mps.concurrency_active(self.cfg, g.uuid)):
                    eff_exclusive = False
                holders = self.active(g.uuid)
                # Same process+initiator re-requesting = its previous release was lost (e.g. the
                # manager restarted at release time). Reclaim it, or a long-lived pid keeps the
                # orphan alive forever (reap() never expires an alive-pid lease) and the consumer
                # deadlocks against itself. This covers NON-exclusive leases too since Phase A:
                # otherwise a downgraded batch lease would leak a duplicate row whose VRAM floor
                # double-counts, permanently now that reap() protects any live pid.
                orphan = next((h for h in holders if pid and h["pid"] == pid
                               and h["initiator"] == initiator), None)
                if orphan is not None:
                    self.conn.execute(
                        "UPDATE lease SET state='released', released_at=? WHERE id=?",
                        (time.time(), orphan["id"]))
                    self.conn.commit()
                    holders = self.active(g.uuid)
                excl = next((h for h in holders if h["exclusive"]), None)
                if excl is not None:
                    # an exclusive holder = a running batch job's serial lane: admit NOTHING new
                    why.append(f"exclusive lease held by {excl['initiator']}:{excl['label']}")
                elif eff_exclusive:
                    # An EXCLUSIVE ask means "I need this card to myself", so it must also defer
                    # to NON-exclusive co-tenants. Without this, flipping the batch_exclusive
                    # rollback lever (or losing the MPS daemon) while jobs co-execute would grant
                    # a fresh "exclusive" lease alongside them: the operator believes serial
                    # protection is restored while N jobs run and one thinks it owns the card.
                    # Resident models are exempt — they hold non-exclusive reservations by design
                    # and never blocked an exclusive batch job before Phase A.
                    co = [h for h in holders if h["initiator"] != "model-manager"]
                    if co:
                        why.append("exclusive ask blocked by "
                                   + ", ".join(f"{h['initiator']}:{h['label']}" for h in co[:3])
                                   + (f" (+{len(co) - 3} more)" if len(co) > 3 else ""))
                s = snap.get(g.uuid)
                # A snapshot missing its VRAM numbers is NOT usable for admission — treat it as
                # offline rather than doing arithmetic on None (which would 500 the lease API
                # and, since a remote card can share a role with local ones, take NVIDIA
                # admission down with it).
                if s is None or s.get("mem_free_mib") is None or s.get("mem_used_mib") is None:
                    why.append("gpu offline / reporter agent unreachable or reporting no VRAM "
                               "(AMD work itself keeps running, unmanaged)"
                               if g.probe == "remote-amd" else "gpu offline / not visible to NVML")
                elif vram_mib:
                    # `free` already reflects VRAM that resident holders have ALLOCATED, so
                    # reserve only the holders' UNREALIZED floor (promised but not yet
                    # consumed). Attribute realized usage in AGGREGATE (holders' summed floor
                    # vs the card's used VRAM) rather than per-pid: a containerized model
                    # (docker) uses an in-container pid that NVML reports but the lease's
                    # MainPID is the `docker run` host pid, so per-pid netting wrongly counts
                    # its full floor twice and blocks a half-free card. Accurate on a
                    # models-only card where holders dominate `used`.
                    sum_floors = sum(h["vram_mib"] for h in holders)
                    unrealized = max(0, sum_floors - s["mem_used_mib"])
                    grantable = s["mem_free_mib"] - unrealized - MARGIN_MIB
                    if grantable < vram_mib:
                        why.append(f"insufficient VRAM: {max(grantable, 0)} MiB grantable < {vram_mib} MiB requested")
                if not why:
                    lid = str(uuidlib.uuid4())
                    now = time.time()
                    # no anonymous jobs: resolve an owner from the pid if the caller gave none
                    owner_final = owner or (owners.attribute(pid)["owner"] if pid else initiator)
                    try:
                        self.conn.execute(
                            "INSERT INTO lease (id, gpu_uuid, initiator, label, vram_mib, exclusive,"
                            " pid, created_at, heartbeat, ttl_s, owner, capability, needs_model,"
                            " job_type, health_url) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (lid, g.uuid, initiator, label, vram_mib, int(eff_exclusive), pid, now,
                             now, ttl_s, owner_final, capability, needs_model, job_type, health_url))
                        self.conn.commit()
                        if eff_exclusive != exclusive:
                            print(f"[mps] {g.name}: {initiator}:{label} admitted NON-exclusive "
                                  f"(MPS concurrency); VRAM floor still gates the card",
                                  file=sys.stderr, flush=True)
                    except sqlite3.IntegrityError:  # ux_lease_excl: lost a cross-process race
                        self.conn.rollback()
                        reasons.append(f"{g.name}: exclusive lease raced away")
                        continue
                    return {"granted": True, "lease_id": lid, "gpu_uuid": g.uuid, "gpu_name": g.name,
                            "ttl_s": ttl_s}
                reasons.append(f"{g.name}: " + "; ".join(why))
        return {"granted": False, "reasons": reasons, "retry_in_s": 30}

    def heartbeat(self, lease_id: str, pid: int | None = None) -> bool:
        """Refresh a lease; optionally (re)bind its pid — the gpu-lease shim uses this to
        register the actual CHILD process so rogue-watch recognizes it."""
        with self._lock:
            if pid:
                cur = self.conn.execute(
                    "UPDATE lease SET heartbeat=?, pid=? WHERE id=? AND state='granted'",
                    (time.time(), pid, lease_id))
            else:
                cur = self.conn.execute(
                    "UPDATE lease SET heartbeat=? WHERE id=? AND state='granted'",
                    (time.time(), lease_id))
            self.conn.commit()
        return cur.rowcount == 1

    def release(self, lease_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute(
                "UPDATE lease SET state='released', released_at=? WHERE id=? AND state='granted'",
                (time.time(), lease_id))
            self.conn.commit()
        return cur.rowcount == 1
