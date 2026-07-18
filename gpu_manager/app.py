"""gpu-manager — cluster GPU status, queue, holds, and lease admission.

Read surfaces (no auth, LAN):
  GET /healthz          liveness
  GET /v1/gpu/status    per configured GPU: VRAM, live compute+graphics processes,
                        lease holders, legacy-lock state, interactive-hold state
  GET /v1/gpu/queue     merged queue entries from all configured sources
  GET /                 dashboard

Mutating surfaces (Bearer GPU_MANAGER_TOKEN — fail closed if unset):
  POST   /v1/lease                      request admission {gpu, initiator, label, vram_mib,
                                        exclusive, ttl_s} -> granted | deferred with reasons
  POST   /v1/lease/{id}/heartbeat       keep a granted lease alive
  POST   /v1/lease/{id}/release         release
  POST   /v1/hold/{gpu_uuid}            set a manual interactive hold (reason in body)
  DELETE /v1/hold/{gpu_uuid}            clear manual holds

HARD INVARIANT: nothing in the admission path ever kills, evicts, or signals a GPU process;
admission is check-and-defer only. The refined never-preempt invariant
has ONE gated exception — the stall-watch reaper (`stall.py`) may evict a job
confirmed idle past the stall window whose owner does not vouch it alive, only when
`stall_watch.enforce` is on; never a working/vouched job, never the interactive desktop.
"""
from __future__ import annotations

import asyncio
import contextlib
import glob
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from . import (admission, amd, config as _config, gates, models, mps, presence, probes, rogue,
               sources, stall)
from .dashboard import PAGE

CFG = _config.load()
LEASES = admission.Leases(CFG)


async def _reaper() -> None:
    while True:
        await asyncio.to_thread(LEASES.reap)
        await asyncio.sleep(60)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    tasks = [asyncio.create_task(_reaper())]
    if CFG.presence:
        tasks.append(asyncio.create_task(presence.run(CFG)))
    if CFG.models:
        tasks.append(asyncio.create_task(models.drain_loop(CFG, LEASES)))
    if CFG.rogue_watch.enabled:
        tasks.append(asyncio.create_task(rogue.run(CFG, LEASES)))
    if getattr(CFG, "stall_watch", None) and CFG.stall_watch.enabled:
        tasks.append(asyncio.create_task(stall.run(CFG, LEASES)))
    if any(g.probe == "remote-amd" for g in CFG.gpus):
        tasks.append(asyncio.create_task(amd.run(CFG)))
    if any(m.gate_port for m in CFG.models):
        tasks.append(asyncio.create_task(gates.run(CFG, LEASES)))
    yield
    for t in tasks:
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t


APP = FastAPI(title="gpu-manager", version="0.2.0", lifespan=_lifespan)


def _auth(authorization: str | None) -> None:
    token = os.environ.get("GPU_MANAGER_TOKEN", "")
    if not token:
        raise HTTPException(503, "mutating API disabled: no GPU_MANAGER_TOKEN configured")
    if not (authorization or "").startswith("Bearer ") or \
            not secrets.compare_digest(authorization.removeprefix("Bearer "), token):
        raise HTTPException(401, "bad token")


@APP.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "version": APP.version}


@APP.get("/v1/gpu/status")
def gpu_status() -> JSONResponse:
    # read-only: lease expiry runs in the background reaper, never on an anonymous GET
    snap = probes.nvml_snapshot([g.uuid for g in CFG.gpus if g.probe != "remote-amd"])
    for g in CFG.gpus:  # cross-node cards come from their reporter agent (fail-open → offline)
        if g.probe == "remote-amd":
            rs = amd.probe_snapshot(g)
            if rs is not None:
                snap[g.uuid] = rs
    gpus = []
    for g in CFG.gpus:
        s = snap.get(g.uuid, {})
        locks = [{"path": l.path, "label": l.label, "held": probes.flock_held(l.path)}
                 for l in CFG.locks if l.gpu_uuid in (None, g.uuid)]
        hold_files = probes.holds(CFG.hold_dir, g.uuid)
        gpus.append({
            "uuid": g.uuid, "name": g.name, "role": g.role, "host": g.host,
            "capabilities": g.caps, "remote": g.probe == "remote-amd",
            "mps": mps.card_state(CFG, g.uuid),
            "online": bool(s),
            "mem_total_mib": s.get("mem_total_mib"), "mem_used_mib": s.get("mem_used_mib"),
            "mem_free_mib": s.get("mem_free_mib"),
            "processes": s.get("processes", []),
            "leases": [{"id": h["id"], "initiator": h["initiator"], "label": h["label"],
                        "vram_mib": h["vram_mib"], "exclusive": bool(h["exclusive"]),
                        "owner": h["owner"], "capability": h["capability"],
                        "needs_model": h["needs_model"], "job_type": h["job_type"],
                        "age_s": round(time.time() - h["created_at"])}
                       for h in LEASES.active(g.uuid)],
            "locks": locks,
            "hold": {"active": bool(hold_files), "markers": hold_files},
        })
    return JSONResponse({"ok": True, "gpus": gpus, "rogues": rogue.current(),
                         "flagged": stall.current(), "generated_at": time.time()})


@APP.get("/v1/gpu/queue")
def gpu_queue() -> JSONResponse:
    entries = sources.collect(CFG.sources)
    entries.sort(key=lambda e: (e["state"] != "running", e.get("created_at") or ""))
    return JSONResponse({"ok": True, "entries": entries, "generated_at": time.time()})


@APP.post("/v1/lease")
def lease_request(payload: dict = Body(...), authorization: str | None = Header(None)) -> JSONResponse:
    _auth(authorization)
    try:
        res = LEASES.request(
            gpu=str(payload["gpu"]),
            initiator=str(payload["initiator"]),
            label=str(payload.get("label", "job")),
            vram_mib=int(payload.get("vram_mib", 0)),
            exclusive=bool(payload.get("exclusive", True)),
            ttl_s=int(payload.get("ttl_s", 900)),
            pid=payload.get("pid"),
            owner=payload.get("owner"),
            capability=payload.get("capability"),
            needs_model=payload.get("needs_model"),
            job_type=str(payload.get("job_type", "oneshot")),
            health_url=payload.get("health_url"),
        )
    except KeyError as e:
        raise HTTPException(422, f"missing field: {e}") from e
    return JSONResponse(res, status_code=200 if res["granted"] else 409)


@APP.post("/v1/lease/{lease_id}/heartbeat")
def lease_heartbeat(lease_id: str, payload: dict = Body(default={}),
                    authorization: str | None = Header(None)) -> dict:
    _auth(authorization)
    pid = payload.get("pid") if isinstance(payload, dict) else None
    if not LEASES.heartbeat(lease_id, pid=int(pid) if pid else None):
        raise HTTPException(404, "no such granted lease")
    return {"ok": True}


@APP.post("/v1/lease/{lease_id}/release")
def lease_release(lease_id: str, authorization: str | None = Header(None)) -> dict:
    _auth(authorization)
    if not LEASES.release(lease_id):
        raise HTTPException(404, "no such granted lease")
    return {"ok": True}


@APP.get("/v1/models")
def models_status() -> JSONResponse:
    return JSONResponse({"ok": True, "models": models.status(CFG), "generated_at": time.time()})


@APP.post("/v1/models/{name}/ensure")
def models_ensure(name: str, authorization: str | None = Header(None)) -> JSONResponse:
    _auth(authorization)
    res = models.ensure(CFG, LEASES, name)
    code = {"unknown-model": 404, "deferred": 409, "error": 500}.get(res["state"], 200)
    return JSONResponse(res, status_code=code)


_REASON_RE = re.compile(r"[^A-Za-z0-9_-]")


@APP.post("/v1/hold/{gpu_uuid}")
def hold_set(gpu_uuid: str, payload: dict = Body(default={}),
             authorization: str | None = Header(None)) -> dict:
    _auth(authorization)
    if gpu_uuid not in [g.uuid for g in CFG.gpus]:
        raise HTTPException(404, "unknown gpu")
    reason = _REASON_RE.sub("", str(payload.get("reason", "manual")))[:40] or "manual"
    os.makedirs(CFG.hold_dir, exist_ok=True)
    marker = os.path.join(CFG.hold_dir, f"hold-{gpu_uuid}.{reason}")
    with open(marker, "a"):
        os.utime(marker, None)
    return {"ok": True, "marker": os.path.basename(marker)}


@APP.delete("/v1/hold/{gpu_uuid}")
def hold_clear(gpu_uuid: str, authorization: str | None = Header(None)) -> dict:
    _auth(authorization)
    if gpu_uuid not in [g.uuid for g in CFG.gpus]:
        raise HTTPException(404, "unknown gpu")
    removed = []
    for p in glob.glob(os.path.join(CFG.hold_dir, f"hold-{glob.escape(gpu_uuid)}.*")):
        if p.endswith(".nx"):
            continue  # NX presence marker is the poller's; it clears on disconnect+cooldown
        with contextlib.suppress(OSError):
            os.remove(p)
            removed.append(os.path.basename(p))
    return {"ok": True, "removed": removed}


_CONFIG_PATH = os.environ.get("GPU_MANAGER_CONFIG", "/opt/gpu-manager/config.yaml")


@APP.get("/v1/config", response_class=PlainTextResponse)
def config_get(authorization: str | None = Header(None)) -> str:
    _auth(authorization)
    with open(_CONFIG_PATH) as f:
        return f.read()


@APP.put("/v1/config")
def config_put(payload: dict = Body(...), authorization: str | None = Header(None)) -> dict:
    """Settings-panel save: validate the YAML by round-tripping it through the real loader,
    keep a timestamped backup, write atomically, then restart the service (delayed so this
    response reaches the caller). Running jobs are unaffected — leases live in SQLite and
    model units are independent systemd units; only the manager process restarts."""
    _auth(authorization)
    text = payload.get("yaml")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(422, "body must be {yaml: '<full config file text>'}")
    if len(text) > 256 * 1024:
        raise HTTPException(413, "config too large")
    tmp = _CONFIG_PATH + ".candidate"
    with open(tmp, "w") as f:
        f.write(text)
    try:
        cand = _config.load(tmp)
        if not cand.gpus:
            raise ValueError("config must declare at least one gpu")
    except Exception as e:  # noqa: BLE001 — surface the exact parse/shape error to the panel
        os.unlink(tmp)
        raise HTTPException(422, f"config rejected: {e}") from None
    backup = f"{_CONFIG_PATH}.bak-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    shutil.copy2(_CONFIG_PATH, backup)
    os.replace(tmp, _CONFIG_PATH)
    threading.Timer(0.8, lambda: subprocess.Popen(
        ["systemctl", "restart", "gpu-manager"])).start()
    return {"ok": True, "backup": os.path.basename(backup),
            "note": "saved; service restarting to apply (~3s)"}


@APP.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(PAGE)
