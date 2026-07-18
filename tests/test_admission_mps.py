"""Floating-Point Ranger Phase A — MPS batch concurrency + interactive priority.

Pins the two behaviours that make this phase safe:
  * batch jobs CO-EXECUTE on the MPS card (an exclusive ask is downgraded server-side), but only
    while MPS is genuinely running — a configured-but-dead daemon must fail SAFE back to
    exclusive-serial rather than silently dropping the exclusive-lane protection;
  * a hold gates admission on ANY card. This used to be `g.role == "interactive"`, so a hold on
    the batch card was silently ignored even though the runbook documented the opposite.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GPUM = REPO

OPTIONAL_DEPS_AVAILABLE = True
OPTIONAL_DEPS_REASON = ""
try:
    sys.path.insert(0, str(GPUM))
    from gpu_manager import mps as mps_mod
    from gpu_manager import probes
    from gpu_manager.admission import Leases
    from gpu_manager.config import Config, GpuCfg, MpsCfg
except Exception as exc:  # pragma: no cover
    OPTIONAL_DEPS_AVAILABLE = False
    OPTIONAL_DEPS_REASON = f"gpu_manager deps unavailable: {exc}"

BATCH = "GPU-BATCH"
INTER = "GPU-INTER"


@unittest.skipUnless(OPTIONAL_DEPS_AVAILABLE, OPTIONAL_DEPS_REASON)
class TestMpsConcurrency(unittest.TestCase):
    def setUp(self):
        # NVML is absent off-host; give every local card plenty of free VRAM so admission
        # decisions under test are about exclusivity/holds, not memory.
        probes.nvml_snapshot = lambda uuids: {
            u: {"mem_total_mib": 12288, "mem_used_mib": 500, "mem_free_mib": 11788,
                "processes": [], "util_gpu": 0, "proc_util": {}} for u in uuids}
        self.tmp = tempfile.mkdtemp()
        self.hold_dir = os.path.join(self.tmp, "holds")
        self.pipe_dir = os.path.join(self.tmp, "nvidia-mps")
        os.makedirs(self.hold_dir, exist_ok=True)
        self._real_server_up = mps_mod.server_up
        mps_mod.server_up = lambda cfg: False  # daemon down unless a test says otherwise

    def tearDown(self):
        mps_mod.server_up = self._real_server_up

    def _mps_up(self):
        """Daemon running. server_up() itself verifies the daemon's pid against /proc, which
        cannot be faked here — it gets its own direct test in TestServerUpLiveness."""
        mps_mod.server_up = lambda cfg: True

    def _leases(self, **mps_kw):
        mps = MpsCfg(**{"enabled": True, "gpu_uuid": BATCH, "pipe_dir": self.pipe_dir, **mps_kw})
        cfg = Config(hold_dir=self.hold_dir, mps=mps, gpus=[
            GpuCfg(uuid=BATCH, name="RTX 3060 12GB", role="batch", host="host-a"),
            GpuCfg(uuid=INTER, name="RTX 3060 Ti", role="interactive", host="host-a"),
        ])
        return Leases(cfg, db_path=os.path.join(tempfile.mkdtemp(), "state.db"))

    def _ask(self, leases, label, gpu=BATCH, exclusive=True, initiator="batch"):
        return leases.request(gpu=gpu, initiator=initiator, label=label,
                              vram_mib=100, exclusive=exclusive)

    def test_two_batch_jobs_coexecute_when_mps_is_up(self):
        self._mps_up()
        L = self._leases()
        a = self._ask(L, "job-a")
        b = self._ask(L, "job-b")
        self.assertTrue(a["granted"], a)
        self.assertTrue(b["granted"], f"second batch job must co-execute under MPS: {b}")
        # both leases are recorded NON-exclusive despite asking exclusive
        self.assertEqual([h["exclusive"] for h in L.active(BATCH)], [0, 0])

    def test_dead_mps_daemon_fails_safe_to_exclusive_serial(self):
        """Configured but NOT running: must NOT downgrade — otherwise we would drop the
        exclusive lane while delivering no MPS at all."""
        L = self._leases()  # setUp leaves server_up False
        a = self._ask(L, "job-a")
        b = self._ask(L, "job-b")
        self.assertTrue(a["granted"])
        self.assertFalse(b["granted"], "with MPS down the serial lane must still hold")
        self.assertIn("exclusive lease held", b["reasons"][0])

    def test_batch_exclusive_flag_restores_serial_behaviour(self):
        self._mps_up()
        L = self._leases(batch_exclusive=True)
        self.assertTrue(self._ask(L, "job-a")["granted"])
        b = self._ask(L, "job-b")
        self.assertFalse(b["granted"], "batch_exclusive=true is the rollback lever")
        self.assertIn("exclusive lease held", b["reasons"][0])

    def test_mps_disabled_restores_serial_behaviour(self):
        self._mps_up()
        L = self._leases(enabled=False)
        self.assertTrue(self._ask(L, "job-a")["granted"])
        self.assertFalse(self._ask(L, "job-b")["granted"])

    def test_non_mps_card_is_unaffected(self):
        self._mps_up()
        L = self._leases()
        a = self._ask(L, "job-a", gpu=INTER)
        self.assertTrue(a["granted"])
        self.assertEqual([h["exclusive"] for h in L.active(INTER)], [1], "still exclusive")
        self.assertFalse(self._ask(L, "job-b", gpu=INTER)["granted"])

    def test_model_manager_leases_are_not_downgraded(self):
        self._mps_up()
        L = self._leases()
        r = L.request(gpu=BATCH, initiator="model-manager", label="qwen",
                      vram_mib=100, exclusive=True)
        self.assertTrue(r["granted"])
        self.assertEqual([h["exclusive"] for h in L.active(BATCH)], [1])


@unittest.skipUnless(OPTIONAL_DEPS_AVAILABLE, OPTIONAL_DEPS_REASON)
class TestHoldGating(TestMpsConcurrency):
    def _hold(self, uuid, reason="operator"):
        open(os.path.join(self.hold_dir, f"hold-{uuid}.{reason}"), "w").close()

    def test_hold_on_the_BATCH_card_now_defers_admission(self):
        """The bug this phase fixes: holds used to be honoured only on interactive-role cards,
        so `POST /v1/hold/<batch uuid>` was silently ignored despite the documented behaviour."""
        self._mps_up()
        L = self._leases()
        self._hold(BATCH)
        r = self._ask(L, "job-a")
        self.assertFalse(r["granted"], "a hold on the batch card must block new admission")
        self.assertIn("hold active", r["reasons"][0])

    def test_hold_on_the_interactive_card_still_defers(self):
        L = self._leases()
        self._hold(INTER, "nx")
        r = self._ask(L, "job-a", gpu=INTER)
        self.assertFalse(r["granted"])
        self.assertIn("hold active", r["reasons"][0])

    def test_running_jobs_are_untouched_by_a_hold(self):
        """Never-preempt: a hold blocks NEW admission only; existing leases stay granted."""
        self._mps_up()
        L = self._leases()
        a = self._ask(L, "running-job")
        self.assertTrue(a["granted"])
        self._hold(BATCH)
        self.assertFalse(self._ask(L, "new-job")["granted"])
        still = [h["id"] for h in L.active(BATCH)]
        self.assertIn(a["lease_id"], still, "the running job must NOT be disturbed by a hold")


@unittest.skipUnless(OPTIONAL_DEPS_AVAILABLE, OPTIONAL_DEPS_REASON)
class TestRollbackSafety(TestMpsConcurrency):
    """Findings from the Phase A pre-merge review: the rollback path itself was unsafe."""

    def test_exclusive_ask_defers_to_running_co_tenants(self):
        """Flipping the rollback lever while jobs co-execute must NOT hand out an 'exclusive'
        lease alongside them — the operator would believe serial protection was restored while
        N jobs keep running and one thinks it owns the card."""
        self._mps_up()
        L = self._leases()
        self.assertTrue(self._ask(L, "co-1")["granted"])
        self.assertTrue(self._ask(L, "co-2")["granted"])
        L.cfg.mps.batch_exclusive = True          # the documented hot rollback
        r = self._ask(L, "post-rollback")
        self.assertFalse(r["granted"], "exclusive ask must defer to running co-tenants")
        self.assertIn("exclusive ask blocked by", r["reasons"][0])

    def test_resident_models_do_not_block_an_exclusive_ask(self):
        """Models hold non-exclusive reservations by design and never blocked exclusive batch
        work before Phase A — that must stay true."""
        L = self._leases()  # MPS down => no downgrade, plain exclusive path
        L.request(gpu=BATCH, initiator="model-manager", label="qwen",
                  vram_mib=100, exclusive=False)
        self.assertTrue(self._ask(L, "batch-job")["granted"])

    def test_a_hold_never_starves_model_manager(self):
        """A hold means 'pause BATCH'. Denying model-manager would break gated STT/TTS/OCR
        consumers via ensure(), and deny _reattach()'s vram_mib=0 request for an
        already-resident model — dropping its VRAM floor from the admission math while it
        still holds real VRAM."""
        L = self._leases()
        open(os.path.join(self.hold_dir, f"hold-{BATCH}.operator"), "w").close()
        self.assertFalse(self._ask(L, "batch-blocked")["granted"], "batch must be paused")
        ensure_like = L.request(gpu=BATCH, initiator="model-manager", label="kokoro",
                                vram_mib=1500, exclusive=False)
        self.assertTrue(ensure_like["granted"], "models must still start under a hold")
        reattach_like = L.request(gpu=BATCH, initiator="model-manager", label="qwen",
                                  vram_mib=0, exclusive=False)
        self.assertTrue(reattach_like["granted"], "resident-model reattach must not be denied")

    def test_reap_protects_a_live_pid_non_exclusive_lease(self):
        """The alive-pid guard used to be keyed on `exclusive`, so Phase A silently removed it
        from every batch job. An expired-but-running lease loses its VRAM floor -> over-admission."""
        self._mps_up()
        L = self._leases()
        r = L.request(gpu=BATCH, initiator="batch", label="long-job", vram_mib=100,
                      exclusive=True, ttl_s=0, pid=os.getpid())   # downgraded, ttl already due
        self.assertTrue(r["granted"])
        time.sleep(0.02)
        L.reap()
        self.assertIn(r["lease_id"], [h["id"] for h in L.active(BATCH)],
                      "a lease whose process is still alive must never be expired")

    def test_reap_still_expires_a_dead_pid_lease(self):
        self._mps_up()
        L = self._leases()
        r = L.request(gpu=BATCH, initiator="batch", label="ghost", vram_mib=100,
                      exclusive=True, ttl_s=0, pid=2147480000)  # cannot exist
        time.sleep(0.02)
        L.reap()
        self.assertNotIn(r["lease_id"], [h["id"] for h in L.active(BATCH)])

    def test_mps_pointed_at_the_interactive_card_is_refused(self):
        """A typo in mps.gpu_uuid must never downgrade INTERACTIVE leases."""
        self._mps_up()
        L = self._leases(gpu_uuid=INTER)
        self.assertFalse(mps_mod.concurrency_active(L.cfg, INTER))
        self.assertTrue(self._ask(L, "desktop-job", gpu=INTER)["granted"])
        self.assertEqual([h["exclusive"] for h in L.active(INTER)], [1], "must stay exclusive")

    def test_mps_uuid_absent_from_gpus_is_refused(self):
        self._mps_up()
        L = self._leases(gpu_uuid="GPU-DOES-NOT-EXIST")
        self.assertFalse(mps_mod.concurrency_active(L.cfg, BATCH))


@unittest.skipUnless(OPTIONAL_DEPS_AVAILABLE, OPTIONAL_DEPS_REASON)
class TestServerUpLiveness(unittest.TestCase):
    """server_up() must prove the daemon is ALIVE, not merely that its directory exists.

    Regression: an unclean stop left /tmp/nvidia-mps behind, the old directory-existence check
    kept reporting MPS as up, and batch leases stayed non-exclusive with no MPS running — the
    exclusive lane gone with nothing replacing it. Caught by the Phase A rollback verification.
    """

    def _cfg(self, pipe_dir):
        return Config(mps=MpsCfg(enabled=True, gpu_uuid=BATCH, pipe_dir=pipe_dir))

    def test_missing_pipe_dir_reads_as_down(self):
        self.assertFalse(mps_mod.server_up(self._cfg(os.path.join(tempfile.mkdtemp(), "nope"))))

    def test_stale_pipe_dir_with_no_pid_file_reads_as_down(self):
        d = tempfile.mkdtemp()
        open(os.path.join(d, "control"), "w").close()  # looks alive to the OLD check
        self.assertFalse(mps_mod.server_up(self._cfg(d)), "directory presence is not liveness")

    def test_pid_file_pointing_at_a_non_daemon_process_reads_as_down(self):
        """A recycled/wrong pid must not count: we verify the cmdline really is the daemon."""
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "nvidia-cuda-mps-control.pid"), "w") as f:
            f.write(str(os.getpid()))  # alive, but it is the test runner, not the daemon
        self.assertFalse(mps_mod.server_up(self._cfg(d)))

    def test_disabled_is_always_down(self):
        cfg = Config(mps=MpsCfg(enabled=False, gpu_uuid=BATCH))
        self.assertFalse(mps_mod.server_up(cfg))


if __name__ == "__main__":
    unittest.main()
