"""Bounded batch spillover onto an interactive card.

Policy: all cards should be utilized. A role-ask batch job that DECLARES a bounded VRAM
floor (0 < vram_mib <= the card's batch_spillover_max_mib) may be offered an opted-in
interactive card as a LAST-RESORT candidate. The guards that keep interactive priority
intact are the ones that already exist: batch-role cards are always tried first, hold
markers (presence or manual) block new admission on the interactive card, the exclusive
serial lane bounds concurrency there, and nothing running is ever preempted.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GPUM = REPO

OPTIONAL_DEPS_AVAILABLE = True
OPTIONAL_DEPS_REASON = ""
try:
    sys.path.insert(0, str(GPUM))
    from gpu_manager import probes
    from gpu_manager.admission import Leases
    from gpu_manager.config import Config, GpuCfg
except Exception as exc:  # pragma: no cover
    OPTIONAL_DEPS_AVAILABLE = False
    OPTIONAL_DEPS_REASON = f"gpu_manager deps unavailable: {exc}"

BATCH = "GPU-BATCH"
INTER = "GPU-INTER"


@unittest.skipUnless(OPTIONAL_DEPS_AVAILABLE, OPTIONAL_DEPS_REASON)
class TestSpillover(unittest.TestCase):
    def setUp(self):
        probes.nvml_snapshot = lambda uuids: {
            u: {"mem_total_mib": 12288, "mem_used_mib": 500, "mem_free_mib": 11788,
                "processes": [], "util_gpu": 0, "proc_util": {}} for u in uuids}
        self.tmp = tempfile.mkdtemp()
        self.hold_dir = os.path.join(self.tmp, "holds")
        os.makedirs(self.hold_dir, exist_ok=True)

    def _leases(self, spill_cap=4096):
        cfg = Config(hold_dir=self.hold_dir, gpus=[
            GpuCfg(uuid=BATCH, name="RTX 3060 12GB", role="batch", host="host-a"),
            GpuCfg(uuid=INTER, name="RTX 3060 Ti", role="interactive", host="host-a",
                   batch_spillover_max_mib=spill_cap),
        ])
        return Leases(cfg, db_path=os.path.join(tempfile.mkdtemp(), "state.db"))

    def _fill_batch_card(self, leases):
        r = leases.request(gpu=BATCH, initiator="t", label="hog", vram_mib=100)
        self.assertTrue(r["granted"])

    def test_batch_card_is_always_preferred(self):
        L = self._leases()
        r = L.request(gpu="batch", initiator="t", label="job", vram_mib=1000)
        self.assertEqual(r["gpu_uuid"], BATCH, "spillover must be last resort, not first")

    def test_bounded_job_spills_when_batch_card_is_busy(self):
        L = self._leases()
        self._fill_batch_card(L)
        r = L.request(gpu="batch", initiator="t", label="small", vram_mib=1000)
        self.assertTrue(r["granted"], r)
        self.assertEqual(r["gpu_uuid"], INTER)

    def test_unbounded_job_never_spills(self):
        """vram_mib=0 (the gpu-lease default) = an undeclared footprint — it must never
        be offered the desktop card, even with the batch card busy."""
        L = self._leases()
        self._fill_batch_card(L)
        r = L.request(gpu="batch", initiator="t", label="unbounded", vram_mib=0)
        self.assertFalse(r["granted"])
        self.assertNotIn(INTER, [h["gpu_uuid"] for h in L.active()])

    def test_job_above_the_cap_never_spills(self):
        L = self._leases(spill_cap=4096)
        self._fill_batch_card(L)
        r = L.request(gpu="batch", initiator="t", label="big", vram_mib=6000)
        self.assertFalse(r["granted"])

    def test_hold_blocks_spillover(self):
        """The interactive-priority guard: a presence/manual hold on the interactive card
        defers spillover admission exactly like any other admission."""
        L = self._leases()
        self._fill_batch_card(L)
        open(os.path.join(self.hold_dir, f"hold-{INTER}.nx"), "w").close()
        r = L.request(gpu="batch", initiator="t", label="small", vram_mib=1000)
        self.assertFalse(r["granted"], "hold must block spillover")

    def test_spillover_off_by_default(self):
        L = self._leases(spill_cap=0)
        self._fill_batch_card(L)
        r = L.request(gpu="batch", initiator="t", label="small", vram_mib=1000)
        self.assertFalse(r["granted"], "cap 0 = spillover disabled")

    def test_wrong_capability_never_spills(self):
        """A vulkan ask must not land on the CUDA-only interactive card."""
        L = self._leases()
        r = L.request(gpu="batch", initiator="t", label="vk", vram_mib=1000,
                      capability="vulkan")
        self.assertFalse(r["granted"])

    def test_explicit_interactive_ask_unchanged(self):
        L = self._leases()
        r = L.request(gpu="interactive", initiator="t", label="desktop-job", vram_mib=100)
        self.assertTrue(r["granted"])
        self.assertEqual(r["gpu_uuid"], INTER)

    def test_running_spill_job_survives_a_new_hold(self):
        """Never-preempt: a hold set AFTER a spillover grant blocks new admission only."""
        L = self._leases()
        self._fill_batch_card(L)
        r = L.request(gpu="batch", initiator="t", label="small", vram_mib=1000)
        self.assertTrue(r["granted"])
        open(os.path.join(self.hold_dir, f"hold-{INTER}.nx"), "w").close()
        self.assertIn(r["lease_id"], [h["id"] for h in L.active(INTER)])
        self.assertFalse(L.request(gpu="batch", initiator="t", label="next",
                                   vram_mib=1000)["granted"])


if __name__ == "__main__":
    unittest.main()
