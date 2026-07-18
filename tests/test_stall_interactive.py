"""Provenance-aware stall watch on interactive cards.

Policy: interactive-card leases are examined when `watch_interactive` is on, so
agent-launched jobs there (spillover, direct API callers) get the normal stall pathway —
but a lease whose provenance matches `desktop_exempt_patterns` (a wrapped desktop
application) is left alone even when it looks stalled. Off by default = legacy behavior.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

OPTIONAL_DEPS_AVAILABLE = True
OPTIONAL_DEPS_REASON = ""
try:
    sys.path.insert(0, str(REPO))
    from gpu_manager import owners, probes, stall
    from gpu_manager.admission import Leases
    from gpu_manager.config import Config, GpuCfg, StallWatchCfg
except Exception as exc:  # pragma: no cover
    OPTIONAL_DEPS_AVAILABLE = False
    OPTIONAL_DEPS_REASON = f"gpu_manager deps unavailable: {exc}"

BATCH = "GPU-BATCH"
INTER = "GPU-INTER"


@unittest.skipUnless(OPTIONAL_DEPS_AVAILABLE, OPTIONAL_DEPS_REASON)
class TestStallInteractive(unittest.TestCase):
    def setUp(self):
        # every card idle: no proc_util samples, zero device util -> all leases read idle
        probes.nvml_snapshot = lambda uuids: {
            u: {"mem_total_mib": 8192, "mem_used_mib": 500, "mem_free_mib": 7692,
                "processes": [], "util_gpu": 0, "proc_util": {}} for u in [BATCH, INTER]}
        self._attr = owners.attribute
        self.tmp = tempfile.mkdtemp()
        stall._idle_since.clear()
        stall._logged.clear()
        stall._exempt_logged.clear()
        stall._flagged = []

    def tearDown(self):
        owners.attribute = self._attr

    def _cfg(self, watch_interactive=True, patterns=()):
        return Config(hold_dir=os.path.join(self.tmp, "holds"),
                      stall_watch=StallWatchCfg(enabled=True, enforce=False,
                                                stall_window_s=0,  # instantly "stalled"
                                                watch_interactive=watch_interactive,
                                                desktop_exempt_patterns=list(patterns)),
                      gpus=[GpuCfg(uuid=BATCH, name="B", role="batch", host="h"),
                            GpuCfg(uuid=INTER, name="I", role="interactive", host="h",
                                   batch_spillover_max_mib=4096)])

    def _lease_on(self, leases, uuid, label):
        r = leases.request(gpu=uuid, initiator="t", label=label, vram_mib=100,
                           exclusive=False, pid=os.getpid())
        self.assertTrue(r["granted"], r)
        return r["lease_id"]

    def _tick_twice(self, cfg, leases):
        # tick 1 starts the idle clock; window=0 so tick 2 flags
        stall._tick(cfg, leases)
        stall._tick(cfg, leases)
        return {f["lease_id"] for f in stall.current()}

    def test_interactive_lease_flagged_when_watching(self):
        cfg = self._cfg(watch_interactive=True)
        L = Leases(cfg, db_path=os.path.join(tempfile.mkdtemp(), "s.db"))
        owners.attribute = lambda pid: {"owner": "unit:codex-job.service",
                                        "kind": "systemd", "cmd": "python3"}
        lid = self._lease_on(L, INTER, "codex-job")
        self.assertIn(lid, self._tick_twice(cfg, L))

    def test_interactive_lease_skipped_when_off(self):
        cfg = self._cfg(watch_interactive=False)
        L = Leases(cfg, db_path=os.path.join(tempfile.mkdtemp(), "s.db"))
        lid = self._lease_on(L, INTER, "legacy")
        self.assertNotIn(lid, self._tick_twice(cfg, L))

    def test_desktop_provenance_is_exempt(self):
        cfg = self._cfg(patterns=["blender", "gimp"])
        L = Leases(cfg, db_path=os.path.join(tempfile.mkdtemp(), "s.db"))
        owners.attribute = lambda pid: {"owner": "unit:app-blender-1234.scope",
                                        "kind": "systemd", "cmd": "blender"}
        lid = self._lease_on(L, INTER, "wrapped-blender")
        self.assertNotIn(lid, self._tick_twice(cfg, L))

    def test_agent_job_not_matching_patterns_is_flagged(self):
        cfg = self._cfg(patterns=["blender", "gimp"])
        L = Leases(cfg, db_path=os.path.join(tempfile.mkdtemp(), "s.db"))
        owners.attribute = lambda pid: {"owner": "unit:app-codex-desktop-9.scope",
                                        "kind": "systemd", "cmd": "python3"}
        lid = self._lease_on(L, INTER, "codex-render")
        self.assertIn(lid, self._tick_twice(cfg, L))

    def test_batch_card_behavior_unchanged(self):
        cfg = self._cfg(patterns=["blender"])
        L = Leases(cfg, db_path=os.path.join(tempfile.mkdtemp(), "s.db"))
        owners.attribute = lambda pid: {"owner": "unit:blender-farm.service",
                                        "kind": "systemd", "cmd": "python3"}
        lid = self._lease_on(L, BATCH, "batch-job")
        # exempt patterns apply ONLY on interactive cards; batch leases still flag
        self.assertIn(lid, self._tick_twice(cfg, L))


if __name__ == "__main__":
    unittest.main()
