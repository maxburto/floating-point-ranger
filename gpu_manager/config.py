"""Config loading for gpu-manager.

One YAML file describes the machine: which GPUs exist (by stable UUID), what role each
plays (interactive GPUs admit batch work only when no hold is active), which legacy
flock lease files to surface, and which external job queues to merge into /v1/gpu/queue.
Env var GPU_MANAGER_CONFIG points at the file (default /opt/gpu-manager/config.yaml).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml


@dataclass
class GpuCfg:
    uuid: str
    name: str
    role: str = "batch"  # batch | interactive
    host: str = "local"
    probe: str = "nvml"           # nvml (local NVIDIA) | remote-amd (remote reporter agent)
    agent_url: str | None = None  # remote-amd: base URL of the card's reporter agent
    capabilities: list[str] = field(default_factory=list)  # cuda | vulkan | vaapi

    @property
    def caps(self) -> list[str]:
        """What this card is allowed to run. Defaults encode the ROUTING POLICY:
        the AMD card offers vulkan+vaapi, and the CUDA cards are RESERVED for
        CUDA-only work — so an NVIDIA card does NOT advertise vulkan unless a config says so
        explicitly. That is what keeps Vulkan work off the scarce CUDA cards."""
        if self.capabilities:
            return self.capabilities
        return ["vulkan", "vaapi"] if self.probe == "remote-amd" else ["cuda"]


@dataclass
class LockCfg:
    path: str
    gpu_uuid: str | None = None
    label: str = "lease"


@dataclass
class SourceCfg:
    kind: str  # sqlite-photogrammetry | exapp-jobs
    initiator: str
    enabled: bool = True
    options: dict = field(default_factory=dict)


@dataclass
class PresenceCfg:
    """A hold provider: sets/clears an interactive-hold marker from a liveness probe."""
    kind: str  # nomachine
    gpu_uuid: str
    command: str = "/usr/NX/bin/nxserver --list"
    interval_s: int = 30
    cooldown_s: int = 90
    enabled: bool = True


@dataclass
class ModelCfg:
    """A model server the manager alone may start/stop (unit allowlist by construction)."""
    name: str
    unit: str
    gpu_uuid: str
    vram_mib: int
    port: int | None = None
    idle_drain_s: int = 1800
    prewarm: bool = False  # included in the nightly pre-warm window (gpu-manager-prewarm.timer)
    gate_port: int | None = None  # wake-on-use TCP gate (gates.py): consumers hit this port
    health_path: str | None = None  # gate readiness probe (docker-proxy binds before the app listens)


@dataclass
class RogueWatchCfg:
    enabled: bool = False
    interval_s: int = 30
    min_vram_mib: int = 64
    allow_patterns: list[str] = field(default_factory=list)


@dataclass
class StallWatchCfg:
    """Refined never-preempt reaper. `enabled` starts the watch loop;
    `enforce` gates actual eviction (default off = flag-only: log stalled jobs with owner
    attribution, evict nothing — the first-week observation mode)."""
    enabled: bool = False
    enforce: bool = False
    interval_s: int = 60
    stall_window_s: int = 1500   # 25 min near-idle before a job is even suspected stalled
    active_util_pct: int = 5     # smUtil / device-wide util below this = "no GPU work this tick"
    health_timeout_s: int = 5    # owner health-ping GET timeout


@dataclass
class MpsCfg:
    """NVIDIA MPS concurrency on ONE batch card. TWO independent off-switches:
    `enabled` says MPS is in play at all, and `batch_exclusive` keeps the old exclusive-serial
    behaviour even while the daemon runs — either one alone restores pre-Phase-A behaviour
    without touching the host. `gpu_uuid` MUST match CUDA_VISIBLE_DEVICES in nvidia-mps.service.
    Compute mode is never changed (see mps.py for why EXCLUSIVE_PROCESS would break us)."""
    enabled: bool = False
    gpu_uuid: str | None = None
    batch_exclusive: bool = False   # true = MPS up but batch still serialises (rollback lever)
    # /run, not /tmp: server_up() trusts the daemon's pid file here, and world-writable /tmp
    # would let any local user plant one and re-enable downgrades with no MPS running.
    pipe_dir: str = "/run/nvidia-mps"


@dataclass
class Config:
    bind: str = "0.0.0.0"
    port: int = 8768
    hold_dir: str = "/run/gpu-manager"
    mps: MpsCfg = field(default_factory=MpsCfg)
    amd_poll_interval_s: int = 15   # how often to poll remote-amd reporter agents
    amd_poll_timeout_s: float = 3.0  # short: a slow agent must never stall admission
    rogue_watch: RogueWatchCfg = field(default_factory=RogueWatchCfg)
    stall_watch: StallWatchCfg = field(default_factory=StallWatchCfg)
    gpus: list[GpuCfg] = field(default_factory=list)
    locks: list[LockCfg] = field(default_factory=list)
    sources: list[SourceCfg] = field(default_factory=list)
    presence: list[PresenceCfg] = field(default_factory=list)
    models: list[ModelCfg] = field(default_factory=list)


def load(path: str | None = None) -> Config:
    p = path or os.environ.get("GPU_MANAGER_CONFIG", "/opt/gpu-manager/config.yaml")
    with open(p) as f:
        raw = yaml.safe_load(f) or {}
    return Config(
        bind=raw.get("bind", "0.0.0.0"),
        port=int(raw.get("port", 8768)),
        hold_dir=raw.get("hold_dir", "/run/gpu-manager"),
        amd_poll_interval_s=int(raw.get("amd_poll_interval_s", 15)),
        amd_poll_timeout_s=float(raw.get("amd_poll_timeout_s", 3.0)),
        gpus=[GpuCfg(**g) for g in raw.get("gpus", [])],
        locks=[LockCfg(**l) for l in raw.get("locks", [])],
        sources=[SourceCfg(**s) for s in raw.get("sources", [])],
        presence=[PresenceCfg(**p) for p in raw.get("presence", [])],
        models=[ModelCfg(**m) for m in raw.get("models", [])],
        rogue_watch=RogueWatchCfg(**raw.get("rogue_watch", {})),
        stall_watch=StallWatchCfg(**raw.get("stall_watch", {})),
        mps=MpsCfg(**raw.get("mps", {})),
    )
