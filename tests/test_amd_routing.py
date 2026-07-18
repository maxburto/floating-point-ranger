"""Floating-Point Ranger — remote AMD reporter + capability routing.

Regression net for two blockers an adversarial pre-merge review found by execution,
plus the fdinfo parser, which is pure and cheap to pin:

  * BLOCKER 1 — a legacy caller that declares NO capability (every `gpu-lease` invocation:
    it defaults to `--gpu batch` and sends none) could be granted the REMOTE AMD card once
    the local card filled up. `gpu-lease` then pins CUDA_VISIBLE_DEVICES to an AMD uuid, so
    the CUDA job sees zero devices and dies holding a lease on a card in another guest.
  * BLOCKER 2 — a reachable agent that cannot read the card still answers ok:true with
    all-None VRAM; that None reached the admission arithmetic and 500'd the lease API. Because
    a remote card shares the `batch` role with local ones, it could take NVIDIA admission down.

Both are invariant violations ("a CUDA job must never land on the AMD card"; "a dead/degraded
agent must never block admission"), so they get explicit tests.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GPUM = REPO

# The agent is stdlib-only and runs on a different host; load it by path.
_spec = importlib.util.spec_from_file_location("fpr_amd_agent", GPUM / "agent" / "fpr_amd_agent.py")
agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent)

OPTIONAL_DEPS_AVAILABLE = True
OPTIONAL_DEPS_REASON = ""
try:
    sys.path.insert(0, str(GPUM))
    from gpu_manager import amd
    from gpu_manager.admission import Leases
    from gpu_manager.config import Config, GpuCfg
except Exception as exc:  # pragma: no cover - yaml/requests absent
    OPTIONAL_DEPS_AVAILABLE = False
    OPTIONAL_DEPS_REASON = f"gpu_manager deps unavailable: {exc}"

# Verbatim from a live Radeon Pro WX5100 (llama-server client) — Polaris/gfx803.
REAL_FDINFO = """pos:\t0
flags:\t02100002
drm-driver:\tamdgpu
drm-client-id:\t276
drm-pdev:\t0000:43:00.0
drm-total-gtt:\t578424 KiB
drm-resident-gtt:\t578424 KiB
drm-total-vram:\t653568 KiB
drm-resident-vram:\t653568 KiB
drm-memory-vram:\t653568 KiB
drm-engine-compute:\t259324103 ns
"""

AMD_UUID = "AMD-WX5100-0000:43:00.0"


class TestFdinfoParser(unittest.TestCase):
    def test_parses_real_card_output(self):
        k = agent.parse_fdinfo(REAL_FDINFO)
        self.assertEqual(k["drm-driver"], "amdgpu")
        self.assertEqual(k["drm-client-id"], "276")
        self.assertEqual(agent._to_mib(k["drm-resident-vram"]), 638)
        # drm-memory-* is a deprecated amdgpu alias for drm-resident-* and must agree
        self.assertEqual(k["drm-memory-vram"], k["drm-resident-vram"])

    def test_unit_suffixes(self):
        self.assertEqual(agent._to_mib("653568 KiB"), 638)
        self.assertEqual(agent._to_mib("100 MiB"), 100)
        self.assertEqual(agent._to_mib("2 GiB"), 2048)
        self.assertEqual(agent._to_mib("2097152"), 2)  # bare = bytes
        self.assertEqual(agent._to_mib("garbage"), 0)  # tolerant, never raises
        self.assertEqual(agent._to_mib(""), 0)

    def test_engine_extraction_survives_empty_value_and_skips_capacity(self):
        """An empty drm-engine value used to raise IndexError and 500 the WHOLE endpoint, and
        drm-engine-capacity-<class> was mis-parsed as a bogus 'capacity-compute' engine."""
        keys = agent.parse_fdinfo(
            "drm-driver:\tamdgpu\ndrm-engine-compute:\t123 ns\n"
            "drm-engine-capacity-compute:\t4\ndrm-engine-broken:\t\n")
        engines = {k[len("drm-engine-"):]: (v.split() or ["0"])[0]
                   for k, v in keys.items()
                   if k.startswith("drm-engine-") and not k.startswith("drm-engine-capacity-")}
        self.assertEqual(engines.get("compute"), "123")
        self.assertNotIn("capacity-compute", engines)
        self.assertEqual(engines.get("broken"), "0")


@unittest.skipUnless(OPTIONAL_DEPS_AVAILABLE, OPTIONAL_DEPS_REASON)
class TestCapabilityRouting(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(gpus=[
            GpuCfg(uuid="GPU-LOCAL", name="RTX 3060 12GB", role="batch", host="host-a"),
            GpuCfg(uuid=AMD_UUID, name="WX5100", role="batch", host="host-b",
                   probe="remote-amd", agent_url="http://127.0.0.1:1",
                   capabilities=["vulkan", "vaapi"]),
        ])
        self.leases = Leases(self.cfg, db_path=os.path.join(tempfile.mkdtemp(), "state.db"))
        amd._snap.clear()

    def _online(self, **card):
        base = {"present": True, "pdev": "0000:43:00.0", "vram_total_mib": 8192,
                "vram_used_mib": 640, "vram_free_mib": 7552}
        base.update(card)
        amd._snap[AMD_UUID] = {"online": True, "checked_at": time.time(),
                               "sampled_at": time.time(), "card": base, "clients": []}

    def test_blocker1_legacy_caller_never_gets_the_remote_amd_card(self):
        """A caller declaring no capability (every gpu-lease invocation) must only ever be
        offered LOCAL cards, even when the remote card is online and the local one is busy."""
        self._online()
        for g in self.leases._resolve_gpu("batch", None):
            self.assertNotEqual(g.probe, "remote-amd", "legacy ask resolved to a remote card")
        res = self.leases.request(gpu="batch", initiator="legacy", label="blender")
        self.assertNotEqual(res.get("gpu_uuid"), AMD_UUID)

    def test_blocker2_degraded_agent_defers_cleanly_instead_of_raising(self):
        """A reachable agent reporting a card it cannot read must degrade to offline, not put
        None into the VRAM arithmetic (which 500'd the lease API and could break NVIDIA too)."""
        for bad in ({"present": False}, {"vram_used_mib": None}, {"vram_free_mib": None}):
            with self.subTest(bad=bad):
                self._online(**bad)
                self.assertIsNone(amd.probe_snapshot(self.cfg.gpus[1]))
                res = self.leases.request(gpu=AMD_UUID, initiator="t", label="vk",
                                          capability="vulkan", vram_mib=100)  # must not raise
                self.assertFalse(res["granted"])

    def test_cuda_is_refused_on_the_amd_card(self):
        self._online()
        res = self.leases.request(gpu=AMD_UUID, initiator="t", label="cuda", capability="cuda")
        self.assertFalse(res["granted"])
        self.assertIn("capability 'cuda' not available", res["reasons"][0])

    def test_vulkan_role_ask_routes_to_the_amd_card(self):
        self._online()
        res = self.leases.request(gpu="batch", initiator="t", label="vk",
                                  capability="vulkan", vram_mib=100, exclusive=False)
        self.assertTrue(res["granted"])
        self.assertEqual(res["gpu_uuid"], AMD_UUID)

    def test_vaapi_is_never_leasable(self):
        """Fixed-function transcode runs inline and must never be gated by a lease."""
        self._online()
        res = self.leases.request(gpu=AMD_UUID, initiator="t", label="transcode",
                                  capability="vaapi")
        self.assertFalse(res["granted"])
        self.assertIn("never leased", res["reasons"][0])

    def test_capability_with_no_matching_card_gives_an_honest_reason(self):
        res = self.leases.request(gpu="batch", initiator="t", label="x", capability="rocm")
        self.assertFalse(res["granted"])
        self.assertIn("no card in role", res["reasons"][0])


if __name__ == "__main__":
    unittest.main()
