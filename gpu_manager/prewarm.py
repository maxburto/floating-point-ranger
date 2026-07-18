"""Nightly pre-warm: ensure every `prewarm: true` model during the batch window.

Run by gpu-manager-prewarm.timer (02:30). Deferrals are fine — admission stays honest even
at night; the idle-drain loop drains models naturally after the window.
"""
from __future__ import annotations

import json
import os
import urllib.request

from . import config as _config


def main() -> None:
    cfg = _config.load()
    tok = ""
    if os.path.exists("/opt/gpu-manager/env"):
        for line in open("/opt/gpu-manager/env"):
            if line.startswith("GPU_MANAGER_TOKEN="):
                tok = line.split("=", 1)[1].strip()
    for m in cfg.models:
        if not m.prewarm:
            continue
        req = urllib.request.Request(
            f"http://127.0.0.1:{cfg.port}/v1/models/{m.name}/ensure", method="POST",
            headers={"Authorization": f"Bearer {tok}"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                print(f"prewarm {m.name}: {json.loads(r.read())['state']}")
        except urllib.error.HTTPError as e:  # 409 deferred is a normal answer
            print(f"prewarm {m.name}: {e.read().decode()[:200]}")
        except (urllib.error.URLError, OSError) as e:  # manager down at 02:30: note, don't fail the unit
            print(f"prewarm {m.name}: manager unreachable ({e})")


if __name__ == "__main__":
    main()
