"""Dead-holder ("zombie") lease reaping — the short clock that runs independent of enforce.

The situation these pin, from the live incident (FPR #1): a service holding a 5000 MiB lease on
an 8 GB card was restarted, its pid died, and the reservation stayed on the books for the full
CLIENT-chosen `ttl_s` (900 s) — refusing a 1 GiB request the whole time while nvidia-smi showed
the card empty. The card was starved by a lease with nothing behind it.

The distinction the fix turns on, and what each half is allowed to do:

  * holder pid GONE          -> expire on the short `leases.zombie_ttl_s`. There is no process
                                to preempt, so this is bookkeeping and is NOT gated on
                                `stall_watch.enforce`.
  * holder pid ALIVE, idle   -> never touched here at any age. That is the judgment call
                                `stall_watch.enforce` gates, and it has real false positives
                                (see the #2028 observation review), so the reaper must not
                                quietly become a second path to it.

`_DEAD_PID` is above /proc/sys/kernel/pid_max on any normal Linux, so it can never be live.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

OPTIONAL_DEPS_AVAILABLE = True
OPTIONAL_DEPS_REASON = ""
try:
    sys.path.insert(0, str(REPO))
    from gpu_manager import probes, util
    from gpu_manager.admission import Leases
    from gpu_manager.config import Config, GpuCfg, LeaseCfg
except Exception as exc:  # pragma: no cover
    OPTIONAL_DEPS_AVAILABLE = False
    OPTIONAL_DEPS_REASON = f"gpu_manager deps unavailable: {exc}"

BATCH = "GPU-BATCH"
_DEAD_PID = 2147480000


@unittest.skipUnless(OPTIONAL_DEPS_AVAILABLE, OPTIONAL_DEPS_REASON)
class TestZombieReaping(unittest.TestCase):
    def setUp(self):
        probes.nvml_snapshot = lambda uuids: {
            u: {"mem_total_mib": 8192, "mem_used_mib": 500, "mem_free_mib": 7692,
                "processes": [], "util_gpu": 0, "proc_util": {}} for u in uuids}

    def _leases(self, zombie_ttl_s=300):
        cfg = Config(
            hold_dir=tempfile.mkdtemp(),
            leases=LeaseCfg(zombie_ttl_s=zombie_ttl_s),
            gpus=[GpuCfg(uuid=BATCH, name="RTX 3060 12GB", role="batch", host="host-a")],
        )
        return Leases(cfg, db_path=os.path.join(tempfile.mkdtemp(), "state.db"))

    def _age_heartbeat(self, leases, lease_id, seconds):
        """Backdate the heartbeat, which is how a lease goes stale without a real wait."""
        with leases._lock:
            leases.conn.execute("UPDATE lease SET heartbeat=? WHERE id=?",
                                (time.time() - seconds, lease_id))
            leases.conn.commit()

    def _ids(self, leases):
        return [h["id"] for h in leases.active(BATCH)]

    # -- the incident ------------------------------------------------------

    def test_dead_holder_is_reaped_on_the_short_clock_not_the_client_ttl(self):
        """The incident shape: a long client ttl_s must not keep a dead holder's VRAM booked."""
        L = self._leases(zombie_ttl_s=300)
        r = L.request(gpu=BATCH, initiator="photogrammetry-nearlive", label="phone-live-preview",
                      vram_mib=5000, exclusive=False, ttl_s=900, pid=_DEAD_PID)
        self.assertTrue(r["granted"], r)
        self._age_heartbeat(L, r["lease_id"], 400)  # past zombie_ttl_s, well inside ttl_s
        L.reap()
        self.assertNotIn(r["lease_id"], self._ids(L),
                         "a lease whose holder is gone must not wait out a client-chosen ttl_s")

    def test_reaping_the_zombie_frees_the_card_for_a_real_request(self):
        """The cost that made this worth fixing was the refusal, not the stale row."""
        L = self._leases(zombie_ttl_s=300)
        z = L.request(gpu=BATCH, initiator="nearlive", label="phantom", vram_mib=5000,
                      exclusive=False, ttl_s=900, pid=_DEAD_PID)
        blocked = L.request(gpu=BATCH, initiator="mx", label="smoke", vram_mib=3000,
                            exclusive=False, ttl_s=300, pid=os.getpid())
        self.assertFalse(blocked["granted"], "precondition: the phantom must actually block")
        self._age_heartbeat(L, z["lease_id"], 400)
        L.reap()
        after = L.request(gpu=BATCH, initiator="mx", label="smoke", vram_mib=3000,
                          exclusive=False, ttl_s=300, pid=os.getpid())
        self.assertTrue(after["granted"], f"card must be grantable once the phantom is gone: {after}")

    # -- never-preempt stays intact ----------------------------------------

    def test_live_holder_is_never_reaped_however_stale(self):
        """The whole never-preempt invariant in one assertion: a RUNNING job that has stopped
        heartbeating keeps its reservation. Evicting this case is what `enforce` gates."""
        L = self._leases(zombie_ttl_s=1)
        r = L.request(gpu=BATCH, initiator="batch", label="wedged-but-running", vram_mib=100,
                      exclusive=True, ttl_s=900, pid=os.getpid())
        self._age_heartbeat(L, r["lease_id"], 100_000)  # far past BOTH clocks
        L.reap()
        self.assertIn(r["lease_id"], self._ids(L),
                      "a live pid must survive both the zombie clock and its own ttl")

    def test_lease_without_a_pid_is_not_accelerated(self):
        """No pid = nothing can be concluded about the holder, so the short clock must not
        apply — otherwise every pid-less lease silently gets a much shorter life than it asked
        for."""
        L = self._leases(zombie_ttl_s=300)
        r = L.request(gpu=BATCH, initiator="batch", label="no-pid", vram_mib=100,
                      exclusive=True, ttl_s=900, pid=None)
        self._age_heartbeat(L, r["lease_id"], 400)  # past zombie_ttl_s, inside ttl_s
        L.reap()
        self.assertIn(r["lease_id"], self._ids(L), "a pid-less lease keeps its full ttl_s")
        self._age_heartbeat(L, r["lease_id"], 1000)  # now past ttl_s
        L.reap()
        self.assertNotIn(r["lease_id"], self._ids(L), "ttl expiry itself must still work")

    def test_model_manager_lease_is_exempt_from_the_short_clock(self):
        """Model residency has its own lifecycle and must not be second-guessed here — the same
        exemption stall.py makes.

        The concrete trap: `models._drain_tick` heartbeats these leases with the systemd unit's
        MainPID, but a DOCKER model's compute runs under an in-container pid, so the recorded
        pid can be dead while the model is still resident holding real VRAM. `ensure()` reserves
        the model's FULL floor while it loads, before that allocation exists — dropping it early
        would admit batch work straight into memory the model is about to claim.
        """
        L = self._leases(zombie_ttl_s=300)
        r = L.request(gpu=BATCH, initiator="model-manager", label="qwen3-vl-8b", vram_mib=7000,
                      exclusive=False, ttl_s=2100, pid=_DEAD_PID)
        self._age_heartbeat(L, r["lease_id"], 400)  # past zombie_ttl_s, well inside ttl_s
        L.reap()
        self.assertIn(r["lease_id"], self._ids(L),
                      "a model reservation must keep its full ttl_s even with a dead unit pid")
        self._age_heartbeat(L, r["lease_id"], 2200)  # now past its own ttl_s
        L.reap()
        self.assertNotIn(r["lease_id"], self._ids(L),
                         "the ordinary ttl path must still apply to model leases")

    # -- the off switch ----------------------------------------------------

    def test_zombie_ttl_zero_restores_pure_ttl_behaviour(self):
        """The rollback lever: 0 disables the short clock, leaving exactly the old behaviour."""
        L = self._leases(zombie_ttl_s=0)
        r = L.request(gpu=BATCH, initiator="batch", label="ghost", vram_mib=100,
                      exclusive=True, ttl_s=900, pid=_DEAD_PID)
        self._age_heartbeat(L, r["lease_id"], 400)
        L.reap()
        self.assertIn(r["lease_id"], self._ids(L), "with the short clock off, ttl_s rules")
        self._age_heartbeat(L, r["lease_id"], 1000)
        L.reap()
        self.assertNotIn(r["lease_id"], self._ids(L))

    # -- observability -----------------------------------------------------

    def test_every_expiry_is_logged_with_its_reason(self):
        """Expiry used to be entirely silent, which is why "did the reaper run?" was
        unanswerable from the journal and the incident got written up as "never reaped"."""
        L = self._leases(zombie_ttl_s=300)
        zombie = L.request(gpu=BATCH, initiator="nearlive", label="phantom", vram_mib=100,
                           exclusive=False, ttl_s=900, pid=_DEAD_PID)
        stale = L.request(gpu=BATCH, initiator="batch", label="no-pid", vram_mib=100,
                          exclusive=False, ttl_s=60, pid=None)
        self._age_heartbeat(L, zombie["lease_id"], 400)
        self._age_heartbeat(L, stale["lease_id"], 400)
        buf = io.StringIO()
        with redirect_stderr(buf):
            self.assertEqual(L.reap(), 2)
        out = buf.getvalue()
        self.assertIn("reason=pid-gone", out)
        self.assertIn("reason=ttl", out)
        self.assertIn(zombie["lease_id"][:8], out)
        self.assertIn(stale["lease_id"][:8], out)

    def test_pid_alive_errs_toward_alive(self):
        """The asymmetry the reaper depends on: the only way pid_alive() can be wrong is by
        reporting a recycled pid as alive, which merely DELAYS reaping. It must never report a
        running process as gone, which would expire a working job's reservation."""
        self.assertTrue(util.pid_alive(os.getpid()))
        self.assertFalse(util.pid_alive(_DEAD_PID))
        self.assertFalse(util.pid_alive(None), "a falsy pid is 'unknown', handled by the caller")


if __name__ == "__main__":
    unittest.main()
