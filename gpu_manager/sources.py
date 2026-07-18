"""Queue sources — pluggable read-only taps that feed /v1/gpu/queue.

Every source yields normalized entries:
  {initiator, id, label, state (queued|running), created_at, eta_seconds|null}
ETAs are honest: derived from recent completed durations of the same job type when at
least 2 samples exist, otherwise null — never fabricated.

Kinds:
  sqlite-photogrammetry — read-only tap on the photogrammetry catalog job table
      (options: path). The DB is opened per-request in immutable/ro mode so the
      manager can never write or lock it against the live queue worker.
  exapp-jobs — a Nextcloud ExApp jobs endpoint (GET .../jobs behind the AppAPI proxy),
      with a short cache + serve-stale (such ExApps block during their synchronous
      submit).
      (options: url; env EXAPP_USER / EXAPP_PASSWORD for auth).
"""
from __future__ import annotations

import os
import sqlite3
import statistics
import time

import requests

from .config import SourceCfg

_EXAPP_ACTIVE = {"accepted", "created", "submitted"}


def _pg_eta(conn: sqlite3.Connection, job_type: str) -> float | None:
    rows = conn.execute(
        "SELECT created_at, updated_at FROM job WHERE status='done' AND type=? "
        "ORDER BY updated_at DESC LIMIT 5", (job_type,)).fetchall()
    durs = []
    for created, updated in rows:
        try:
            durs.append(max(0.0, (time.mktime(time.strptime(updated, "%Y-%m-%dT%H:%M:%S"))
                                  - time.mktime(time.strptime(created, "%Y-%m-%dT%H:%M:%S")))))
        except (ValueError, TypeError):
            continue
    if len(durs) < 2:
        return None
    return float(statistics.median(durs))


def _photogrammetry(cfg: SourceCfg) -> list[dict]:
    path = cfg.options.get("path", "/opt/photogrammetry/catalog.db")
    if not os.path.exists(path):
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    try:
        rows = conn.execute(
            "SELECT id, mode, type, status, created_at, resource FROM job "
            "WHERE status IN ('queued','waiting','running') ORDER BY id").fetchall()
        out = []
        for jid, mode, jtype, status, created, resource in rows:
            out.append({
                "initiator": cfg.initiator,
                "id": f"pg-{jid}",
                "label": f"{mode}/{jtype}",
                "state": "running" if status == "running" else "queued",
                "created_at": created,
                "resource": resource,
                "eta_seconds": _pg_eta(conn, jtype) if status != "running" else None,
            })
        return out
    finally:
        conn.close()


# serve-stale cache for the ExApp source (it blocks exactly when the queue is interesting)
_exapp_cache: dict = {"rows": [], "at": 0.0, "stale": True}


def _exapp(cfg: SourceCfg) -> list[dict]:
    url = cfg.options.get("url", "")
    user, pw = os.environ.get("EXAPP_USER", ""), os.environ.get("EXAPP_PASSWORD", "")
    if not url or not pw:
        return []
    now = time.time()
    if now - _exapp_cache["at"] >= 3.0:
        try:
            r = requests.get(url, auth=(user, pw), timeout=4)
            r.raise_for_status()
            _exapp_cache.update(rows=(r.json() or {}).get("jobs", []), at=now, stale=False)
        except Exception:  # noqa: BLE001 — ExApp busy mid-submit: serve last-good
            _exapp_cache["stale"] = True
    out = []
    for j in _exapp_cache["rows"]:
        status = j.get("status") or ""
        if status not in _EXAPP_ACTIVE:
            continue
        out.append({
            "initiator": cfg.initiator,
            "id": f"ocr-{j.get('job_id')}",
            "label": (j.get("filename") or j.get("source_path") or "scan").split("/")[-1],
            "state": "queued",
            "created_at": j.get("created_at"),
            "eta_seconds": None,
            "stale": _exapp_cache["stale"] or None,
        })
    return out


_KINDS = {"sqlite-photogrammetry": _photogrammetry, "exapp-jobs": _exapp}


def collect(sources: list[SourceCfg]) -> list[dict]:
    entries: list[dict] = []
    for cfg in sources:
        if not cfg.enabled:
            continue
        fn = _KINDS.get(cfg.kind)
        if fn is None:
            continue
        try:
            entries.extend(fn(cfg))
        except Exception:  # noqa: BLE001 — one broken source must not blank the queue
            entries.append({"initiator": cfg.initiator, "id": f"{cfg.initiator}-error",
                            "label": "source unavailable", "state": "error",
                            "created_at": None, "eta_seconds": None})
    return entries
