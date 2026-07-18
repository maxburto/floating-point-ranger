"""Device probes — the sensing layer.

NVML is the ground truth for NVIDIA cards: per-GPU VRAM plus the *compute and graphics*
process lists, so interactive desktop clients (compositor, Blender viewport, GIMP) are
visible without ever being "submitted" anywhere. Everything degrades to empty-but-honest
output if the driver/library is absent (dev boxes, CI).

Legacy flock files are probed non-destructively: a non-blocking exclusive flock attempt
tells us whether some consumer (e.g. the OCR worker's in-code lease) currently holds it.
"""
from __future__ import annotations

import contextlib
import fcntl
import glob
import os

try:
    import pynvml  # provided by nvidia-ml-py
    _NVML_IMPORTED = True
except Exception:  # noqa: BLE001
    _NVML_IMPORTED = False

_nvml_ready = False


def _ensure_nvml() -> bool:
    global _nvml_ready
    if not _NVML_IMPORTED:
        return False
    if not _nvml_ready:
        try:
            pynvml.nvmlInit()
            _nvml_ready = True
        except Exception:  # noqa: BLE001 — no driver on this host
            return False
    return True


def _proc_name(pid: int) -> str:
    with contextlib.suppress(OSError):
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv0 = f.read().split(b"\0", 1)[0].decode(errors="replace")
        # Electron apps rewrite argv0 into one giant space-joined string — keep the first token.
        argv0 = argv0.split(" ", 1)[0]
        if argv0:
            return os.path.basename(argv0)
    return "?"


def nvml_snapshot(wanted_uuids: list[str]) -> dict[str, dict]:
    """Return {uuid: {mem, processes[]}} for the requested GPUs (empty when NVML is unavailable)."""
    out: dict[str, dict] = {}
    if not _ensure_nvml():
        return out
    count = pynvml.nvmlDeviceGetCount()
    for i in range(count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        uuid = pynvml.nvmlDeviceGetUUID(h)
        if isinstance(uuid, bytes):
            uuid = uuid.decode()
        if wanted_uuids and uuid not in wanted_uuids:
            continue
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        procs = []
        for kind, getter in (("compute", pynvml.nvmlDeviceGetComputeRunningProcesses),
                             ("graphics", pynvml.nvmlDeviceGetGraphicsRunningProcesses)):
            try:
                rows = getter(h)
            except Exception:  # noqa: BLE001
                rows = []
            for p in rows:
                used = getattr(p, "usedGpuMemory", None)
                procs.append({
                    "pid": p.pid,
                    "name": _proc_name(p.pid),
                    "kind": kind,
                    "used_mib": (used // (1024 * 1024)) if used and used > 0 else None,
                })
        try:
            util_gpu = pynvml.nvmlDeviceGetUtilizationRates(h).gpu  # device-wide %, reliable on GeForce
        except Exception:  # noqa: BLE001
            util_gpu = None
        proc_util: dict[int, int] = {}  # {pid: smUtil%}; only NON-zero-util procs are returned,
        try:                            # and the whole call raises NOT_FOUND when the card is idle
            for su in pynvml.nvmlDeviceGetProcessUtilization(h, 0):
                proc_util[su.pid] = max(proc_util.get(su.pid, 0), su.smUtil)
        except Exception:  # noqa: BLE001 — NOT_FOUND (idle card) / unsupported → no samples
            pass
        out[uuid] = {
            "index": i,
            "mem_total_mib": mem.total // (1024 * 1024),
            "mem_used_mib": mem.used // (1024 * 1024),
            "mem_free_mib": mem.free // (1024 * 1024),
            "processes": procs,
            "util_gpu": util_gpu,
            "proc_util": proc_util,
        }
    return out


def flock_held(path: str) -> bool | None:
    """True if someone holds an exclusive flock on path; None if the file doesn't exist."""
    if not os.path.exists(path):
        return None
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def holds(hold_dir: str, gpu_uuid: str) -> list[str]:
    """Hold markers for a GPU: files named <hold_dir>/hold-<uuid>.<reason> (dot-delimited so a
    UUID that prefixes another can never cross-match)."""
    return [os.path.basename(p)
            for p in glob.glob(os.path.join(hold_dir, f"hold-{glob.escape(gpu_uuid)}.*"))]
