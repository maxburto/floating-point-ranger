#!/usr/bin/env python3
"""fpr-amd-agent — the Floating-Point Ranger's read-only AMD reporter (runs INSIDE the
guest that owns the card).

Exposes an AMD card's live state so the Ranger on another host can see and govern it
cross-node. Read-only BY CONSTRUCTION: it parses sysfs +
DRM fdinfo and serves JSON. It has no mutating surface, never signals a process, and never
touches the card — if this agent dies the Ranger fails open and AMD work simply runs unmanaged.

It runs inside the container (not on the Proxmox host) because the container can see MORE: the
host reads only aggregate `mem_info_*`, while per-process DRM fdinfo is visible here.

Measured facts on the reference card (Radeon Pro WX5100 / Polaris gfx803, in an
unprivileged LXC) that the parser is built around — verified live; do not "simplify"
these away:
  * sysfs lives at /sys/class/drm/renderD128/device/ — there is NO card0 node in the
    unprivileged LXC, so the usual /sys/class/drm/card*/ path finds nothing.
  * fdinfo memory values carry a UNIT suffix and are reported in KiB here (the kernel
    drm-usage-stats spec allows bytes / KiB / MiB), so always parse the unit.
  * drm-memory-<region> is a DEPRECATED amdgpu-only alias for drm-resident-<region>
    (identical values) — prefer drm-resident, fall back to drm-memory.
  * drm-engine-compute is the ONLY engine class exposed. There is no gfx/dec/enc, so VAAPI
    transcode (Jellyfin, VCE/UVD silicon) is INVISIBLE to engine accounting — visibility-only
    by necessity, not merely by policy. Never treat "no engine time" as "idle" for VAAPI.
  * ONE client holds the render node on SEVERAL fds sharing a drm-client-id, reporting
    IDENTICAL values on each — dedupe by client-id or you double-count its VRAM.
  * gpu_busy_percent reads 0 even with a model resident; it is useless as an activity signal.
    Activity comes from drm-engine-compute deltas over time (computed Ranger-side).
"""
from __future__ import annotations

import contextlib
import glob
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RENDER_NODE = os.environ.get("FPR_AMD_RENDER_NODE", "renderD128")
SYSFS_DEV = f"/sys/class/drm/{RENDER_NODE}/device"
BIND = os.environ.get("FPR_AMD_BIND", "0.0.0.0")
PORT = int(os.environ.get("FPR_AMD_PORT", "8769"))

_UNITS = {"KiB": 1024, "MiB": 1024 * 1024, "GiB": 1024 * 1024 * 1024}


def _to_mib(raw: str) -> int:
    """'653568 KiB' -> 638. A bare number is treated as bytes (the drm-usage-stats default);
    amdgpu always emits an explicit KiB suffix here, so that path should not fire on this card —
    if it ever does, the value is more likely KiB and would be under-reported 1024x."""
    parts = raw.split()
    try:
        val = int(parts[0])
    except (ValueError, IndexError):
        return 0
    mult = _UNITS.get(parts[1], 1) if len(parts) > 1 else 1
    return (val * mult) // (1024 * 1024)


def parse_fdinfo(text: str) -> dict:
    """Parse a /proc/<pid>/fdinfo/<fd> body into its drm-* keys (pure — unit-tested offline)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, val = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        if key.startswith("drm-"):
            out[key] = val.strip()
    return out


def _read_int(path: str) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def read_card() -> dict:
    """Aggregate card state from sysfs. gpu_busy_percent is reported but is NOT a reliable
    activity signal on this card (see module docstring) — the Ranger must not lean on it."""
    total = _read_int(f"{SYSFS_DEV}/mem_info_vram_total")
    used = _read_int(f"{SYSFS_DEV}/mem_info_vram_used")
    vis = _read_int(f"{SYSFS_DEV}/mem_info_vis_vram_used")
    busy = _read_int(f"{SYSFS_DEV}/gpu_busy_percent")
    mib = lambda b: (b // (1024 * 1024)) if b is not None else None  # noqa: E731
    # PCI address, so the Ranger can confirm it is talking to the card it thinks it is.
    # uevent states it explicitly (PCI_SLOT_NAME); realpath's basename is the fallback.
    pdev = None
    try:
        with open(f"{SYSFS_DEV}/uevent") as f:
            for line in f:
                if line.startswith("PCI_SLOT_NAME="):
                    pdev = line.split("=", 1)[1].strip()
                    break
    except OSError:
        pass
    if not pdev:
        with contextlib.suppress(OSError):
            pdev = os.path.basename(os.path.realpath(SYSFS_DEV)) or None
    return {
        "present": total is not None,
        "pdev": pdev,
        "vram_total_mib": mib(total),
        "vram_used_mib": mib(used),
        "vis_vram_used_mib": mib(vis),
        "vram_free_mib": (mib(total) - mib(used)) if (total is not None and used is not None) else None,
        "busy_percent": busy,
        "busy_percent_reliable": False,  # documented: reads 0 with a model resident
    }


def _comm(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip()
    except OSError:
        return "?"


def scan_clients() -> list[dict]:
    """Per-client GPU usage, deduped by drm-client-id (one client = many fds, same values)."""
    by_client: dict[str, dict] = {}
    # numeric pid order so a drm_file shared across a fork always reports the SAME (lowest) pid
    # instead of flapping between parent and child on successive polls
    pids = sorted((int(os.path.basename(p)) for p in glob.glob("/proc/[0-9]*")
                   if os.path.basename(p).isdigit()))
    for pid in pids:
        pid_dir = f"/proc/{pid}"
        try:
            fds = os.listdir(f"{pid_dir}/fd")
        except OSError:
            continue  # process exited mid-scan, or not ours to read
        for fd in fds:
            try:
                if not os.readlink(f"{pid_dir}/fd/{fd}").endswith(RENDER_NODE):
                    continue
                with open(f"{pid_dir}/fdinfo/{fd}") as f:
                    keys = parse_fdinfo(f.read())
            except OSError:
                continue
            if keys.get("drm-driver") != "amdgpu":
                continue
            cid = keys.get("drm-client-id")
            if not cid or cid in by_client:
                continue  # dedupe: further fds of this client repeat identical values
            # prefer drm-resident-*; drm-memory-* is the deprecated amdgpu alias
            vram = keys.get("drm-resident-vram") or keys.get("drm-memory-vram") or "0"
            gtt = keys.get("drm-resident-gtt") or keys.get("drm-memory-gtt") or "0"
            # NB: drm-engine-capacity-<class> is a distinct spec key that also carries the
            # drm-engine- prefix — excluded, or it shows up as a bogus "capacity-compute"
            # engine. Values are split defensively: an empty value must not kill the scan.
            engines = {k[len("drm-engine-"):]: (v.split() or ["0"])[0]
                       for k, v in keys.items()
                       if k.startswith("drm-engine-") and not k.startswith("drm-engine-capacity-")}
            by_client[cid] = {
                "client_id": cid,
                "pid": pid,
                "comm": _comm(pid),
                "pdev": keys.get("drm-pdev"),
                "vram_mib": _to_mib(vram),
                "gtt_mib": _to_mib(gtt),
                # cumulative ns per engine class; on this card only "compute" exists
                "engine_ns": {k: int(v) for k, v in engines.items() if v.isdigit()},
            }
    return sorted(by_client.values(), key=lambda c: (-c["vram_mib"], c["pid"]))


_CACHE_TTL_S = 2.0
_cache: dict = {"at": 0.0, "val": None}


def state() -> dict:
    """Memoized ~2s. The Ranger polls every 15s, so caching costs it nothing, but it stops any
    scanner from making the host walk all of /proc per request — the same guest typically
    also serves transcodes and we must not steal CPU from them."""
    now = time.time()
    cached = _cache["val"]
    if cached is not None and now - _cache["at"] < _CACHE_TTL_S:
        return cached
    val = {"ok": True, "render_node": RENDER_NODE, "card": read_card(),
           "clients": scan_clients(), "sampled_at": now}
    _cache["at"], _cache["val"] = now, val
    return val


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 10  # a stalled client must not hold a thread forever (slowloris + MemoryMax=128M)

    def do_GET(self):  # noqa: N802
        try:
            if self.path.startswith("/v1/amd/state"):
                body = json.dumps(state()).encode()
            elif self.path.startswith("/healthz"):
                body = json.dumps({"ok": True}).encode()
            else:
                self.send_error(404)
                return
        except Exception as e:  # noqa: BLE001 — a parse bug degrades THIS response only; the
            # endpoint stays up and the Ranger fails open to "offline" on ok:false.
            print(f"[fpr-amd-agent] state() failed: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            body = json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # journal stays quiet; the Ranger polls every few seconds
        pass


def main() -> int:
    srv = ThreadingHTTPServer((BIND, PORT), _Handler)
    print(f"[fpr-amd-agent] serving {RENDER_NODE} state on {BIND}:{PORT}", file=sys.stderr, flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
