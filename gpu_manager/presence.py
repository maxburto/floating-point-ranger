"""Presence poller — keeps interactive-hold markers honest.

Lives inside the service (not a standalone timer) so an OS rebuild can't silently drop it —
a standalone poller timer was once wiped by a host rebuild and interactive-hold protection
was silently down until noticed; that regression is what this placement closes.

nomachine provider: this NX build's `--list` has no Status column; a *connected* client
session carries its IPv4 in the Remote-IP column (the physical/shadow session shows "-").
Live client => touch `<hold_dir>/hold-<gpu_uuid>.nx`; none => remove the marker after
`cooldown_s`. Manual holds (`.manual` markers, set via the API) are independent.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import re
import subprocess
import time

from .config import Config

_NX_CONNECTED = re.compile(r"^[0-9]+ +\S+ +[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3} ", re.M)


def _nx_client_connected(command: str) -> bool:
    try:
        out = subprocess.run(command.split(), capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return bool(_NX_CONNECTED.search(out))


def _tick(cfg: Config) -> None:
    os.makedirs(cfg.hold_dir, exist_ok=True)
    for p in cfg.presence:
        if not p.enabled or p.kind != "nomachine":
            continue
        marker = os.path.join(cfg.hold_dir, f"hold-{p.gpu_uuid}.nx")
        if _nx_client_connected(p.command):
            with open(marker, "a"):
                os.utime(marker, None)
        elif os.path.exists(marker):
            if time.time() - os.stat(marker).st_mtime > p.cooldown_s:
                with contextlib.suppress(OSError):
                    os.remove(marker)


async def run(cfg: Config) -> None:
    interval = min((p.interval_s for p in cfg.presence if p.enabled), default=30)
    while True:
        await asyncio.to_thread(_tick, cfg)
        await asyncio.sleep(interval)
