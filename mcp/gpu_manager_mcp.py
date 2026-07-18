"""gpu-manager MCP server — the standard surface for AI-agent sessions to use managed GPUs.

Wraps the gpu-manager HTTP API as MCP tools so any agent session discovers GPU
arbitration without prior knowledge. There is DELIBERATELY no
kill/preempt/cancel-other-jobs tool: admission is check-and-defer and
foreign processes are never touched.

Env: GPU_MANAGER_URL (default http://127.0.0.1:8768), GPU_MANAGER_TOKEN (required for
mutating tools). Run: `python gpu_manager_mcp.py`
(stdio) from a venv with fastmcp + requests.
"""
from __future__ import annotations

import os

import requests
from fastmcp import FastMCP

URL = os.environ.get("GPU_MANAGER_URL", "http://127.0.0.1:8768").rstrip("/")
TOKEN = os.environ.get("GPU_MANAGER_TOKEN", "")

mcp = FastMCP(
    "gpu-manager",
    instructions=(
        "GPU arbitration for shared hosts (batch cards + interactive/desktop cards). "
        "BEFORE running anything GPU-heavy (render, COLMAP, torch, ffmpeg-NVENC, "
        "model load), request a lease with request_lease and run your "
        "process with CUDA_VISIBLE_DEVICES set to the granted gpu_uuid; release when done. "
        "A deferral is normal — retry after retry_in_s. Never bypass a deferral: unmanaged "
        "GPU processes are flagged as rogues. Prefer the `gpu-lease` CLI wrapper "
        "for shell commands (it does lease+pin+heartbeat+release for you)."),
)


def _get(path: str) -> dict:
    r = requests.get(f"{URL}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def _post(path: str, body: dict | None = None) -> dict:
    r = requests.post(f"{URL}{path}", json=body,
                      headers={"Authorization": f"Bearer {TOKEN}"}, timeout=15)
    if r.status_code not in (200, 409):
        r.raise_for_status()
    return r.json()


@mcp.tool
def gpu_status() -> dict:
    """Live state of every managed GPU: VRAM, running compute/graphics processes, active
    leases, interactive-hold state, and any rogue (unmanaged) GPU processes."""
    return _get("/v1/gpu/status")


@mcp.tool
def gpu_queue() -> dict:
    """The merged GPU job queue (batch jobs from configured sources, model loads)
    with honest ETAs (null = unknown, never fabricated)."""
    return _get("/v1/gpu/queue")


@mcp.tool
def request_lease(gpu: str, initiator: str, label: str, vram_mib: int = 0,
                  exclusive: bool = True, ttl_s: int = 900, pid: int | None = None) -> dict:
    """Request GPU admission (check-and-defer; NEVER preempts anything). gpu = 'batch',
    'interactive', or a GPU UUID. On grant: run your process with CUDA_VISIBLE_DEVICES set
    to the returned gpu_uuid, heartbeat every few minutes, and release when done. On
    deferral you get machine-readable reasons + retry_in_s — wait and retry; do NOT run
    the job without a grant."""
    return _post("/v1/lease", {"gpu": gpu, "initiator": initiator, "label": label,
                               "vram_mib": vram_mib, "exclusive": exclusive,
                               "ttl_s": ttl_s, "pid": pid})


@mcp.tool
def heartbeat_lease(lease_id: str) -> dict:
    """Keep a granted lease alive (call every few minutes while your job runs)."""
    return _post(f"/v1/lease/{lease_id}/heartbeat")


@mcp.tool
def release_lease(lease_id: str) -> dict:
    """Release a granted lease when the job finishes (always do this)."""
    return _post(f"/v1/lease/{lease_id}/release")


@mcp.tool
def ensure_model(name: str) -> dict:
    """Ensure a declared model server (e.g. the OCR VLM) is resident on its GPU.
    Returns resident|loading|deferred{reasons}. Deferred = capacity is honestly busy;
    retry later — the running job is never disturbed."""
    return _post(f"/v1/models/{name}/ensure")


@mcp.tool
def set_hold(gpu_uuid: str, reason: str = "manual") -> dict:
    """Pause NEW batch admission on a GPU (running jobs finish). Use before interactive
    GPU work the presence poller can't see."""
    return _post(f"/v1/hold/{gpu_uuid}", {"reason": reason})


@mcp.tool
def clear_hold(gpu_uuid: str) -> dict:
    """Clear manual holds on a GPU (the NoMachine presence hold clears itself)."""
    r = requests.delete(f"{URL}/v1/hold/{gpu_uuid}",
                        headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    mcp.run()
