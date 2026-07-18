"""Wake-on-use gates — always-on endpoints for drainable model servers.

A model with `gate_port` gets a TCP listener on that port (the address consumers already
know, e.g. :8771 for STT). The gate wakes the model on the FIRST BYTES of a connection —
not on connect, so monitoring TCP probes never wake anything — by calling ensure() until
resident, waiting for the backend to accept, then piping bytes both ways. Consumers keep
their original URLs; the model still idle-drains between uses. Cold-start latency = unit
start + model load (tens of seconds); typical OpenAI-style STT/TTS clients are
async/background and tolerate it.
"""
from __future__ import annotations

import asyncio
import contextlib
import sys

from . import models
from .admission import Leases
from .config import Config, ModelCfg

_WAKE_TIMEOUT_S = 240


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError, OSError):
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass


async def _ready(m: ModelCfg) -> bool:
    """App-level readiness: docker-proxy accepts TCP as soon as the container starts, long
    before the server inside listens — so when health_path is set, require an HTTP 200."""
    try:
        r, w = await asyncio.open_connection("127.0.0.1", m.port)
    except OSError:
        return False
    try:
        if not m.health_path:
            return True
        w.write(f"GET {m.health_path} HTTP/1.0\r\nHost: localhost\r\n\r\n".encode())
        await w.drain()
        line = await asyncio.wait_for(r.readline(), timeout=5)
        return b" 200" in line
    except (OSError, asyncio.TimeoutError):
        return False
    finally:
        with contextlib.suppress(ConnectionError, OSError):
            w.close()
            await w.wait_closed()


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                  m: ModelCfg, cfg: Config, leases: Leases) -> None:
    try:
        first = await reader.read(65536)
        if not first:
            return  # probe/portscan: connect-only, never wakes the model
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _WAKE_TIMEOUT_S
        backend = None
        while loop.time() < deadline:
            res = await asyncio.to_thread(models.ensure, cfg, leases, m.name)
            state = res.get("state")
            if state == "resident":
                if await _ready(m):
                    try:
                        backend = await asyncio.open_connection("127.0.0.1", m.port)
                        break
                    except OSError:
                        pass
                await asyncio.sleep(1.5)  # unit active, app still loading
            elif state == "loading":
                await asyncio.sleep(2.0)
            elif state == "deferred":
                await asyncio.sleep(min(res.get("retry_in_s") or 5, 10))
            else:
                print(f"[gate:{m.name}] ensure -> {res}", file=sys.stderr, flush=True)
                return
        if backend is None:
            print(f"[gate:{m.name}] wake timed out after {_WAKE_TIMEOUT_S}s",
                  file=sys.stderr, flush=True)
            return
        br, bw = backend
        bw.write(first)
        await bw.drain()
        await asyncio.gather(_pipe(reader, bw), _pipe(br, writer))
    finally:
        try:
            writer.close()
        except (ConnectionError, OSError):
            pass


async def run(cfg: Config, leases: Leases) -> None:
    servers = []
    for m in cfg.models:
        if not m.gate_port:
            continue
        if not m.port:
            print(f"[gate:{m.name}] gate_port set but no backend port — skipping",
                  file=sys.stderr, flush=True)
            continue

        def make(mm: ModelCfg):
            return lambda r, w: _handle(r, w, mm, cfg, leases)

        srv = await asyncio.start_server(make(m), host="0.0.0.0", port=m.gate_port)
        servers.append(srv)
        print(f"[gate:{m.name}] listening :{m.gate_port} -> 127.0.0.1:{m.port}",
              file=sys.stderr, flush=True)
    if not servers:
        return
    try:
        await asyncio.gather(*(s.serve_forever() for s in servers))
    finally:
        for s in servers:
            s.close()
