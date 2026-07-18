# Floating-Point Ranger

GPU arbitration for shared machines — a small FastAPI service that lets batch jobs, model
servers, and an interactive desktop share a handful of GPUs **without ever preempting
anything**.

Born in a homelab where one box runs a desktop session, renders, OCR pipelines, and several
on-demand model servers across mixed NVIDIA/AMD cards; extracted here as a generic,
config-only core.

## The one hard rule

**Never preempt.** Admission is check-and-defer: a job asks for a lease and is either
granted or told to wait (with machine-readable reasons and a retry hint). No code path
kills, evicts, or signals a GPU process it did not start. The single, gated exception is
the stall-watch reaper, which may — only when explicitly enabled — evict a job that has done
no GPU work for a configurable window *and* whose owner declines to vouch for it.

## What it does

- **Status + queue** — `GET /v1/gpu/status` (VRAM, live compute/graphics processes, leases,
  holds, rogues) and `GET /v1/gpu/queue` (merged view of pluggable job-queue sources), plus
  an auto-refreshing dashboard with a validated live-config editor.
- **Lease admission** — `POST /v1/lease` grants or defers on VRAM floors (NVML ground
  truth minus unrealized reservations), per-card exclusive lanes, holds, and capability
  routing (a CUDA job is never handed an AMD card and vice versa).
- **Interactive priority** — cards can be marked `interactive`; a presence poller (e.g.
  NoMachine sessions) or a manual hold pauses *new* batch admission there while running
  work always finishes.
- **Model residency** — declared systemd model units (llama.cpp, dockerized STT/TTS, …)
  are started only by the manager, hold non-exclusive VRAM reservations, idle-drain, and can
  sit behind **wake-on-use TCP gates** so consumers keep their URLs while models sleep.
- **Stall watch** — flags leases that stop doing GPU work (device-wide util, per-process
  samples, VRAM churn), asks the owner's `health_url` to vouch, and — only with
  `enforce: true` — evicts confirmed zombies. Flag-only by default.
- **Rogue watch** — unmanaged GPU processes are attributed to their owning unit/container/
  user and surfaced (journal + status + optional ntfy). Never touched.
- **Cross-node AMD reporting** — a read-only sysfs/DRM-fdinfo agent (`agent/`) runs in the
  guest that owns an AMD card; the manager polls it and fails open when it's unreachable.
- **MPS batch concurrency** — with NVIDIA MPS on a batch card, exclusive batch asks are
  downgraded server-side so jobs co-execute; a dead MPS daemon fails safe back to serial.

## Quickstart

```sh
uv pip install "floating-point-ranger @ git+https://github.com/maxwellburton/floating-point-ranger@v0.2.0"
cp config.example.yaml /etc/gpu-manager/config.yaml   # edit: your GPU UUIDs + roles
GPU_MANAGER_CONFIG=/etc/gpu-manager/config.yaml uvicorn gpu_manager.app:APP --port 8768
```

Systemd templates for the service, the MPS daemon, and the nightly model pre-warm live in
`systemd/`; the AMD reporter agent in `agent/`. Set `GPU_MANAGER_TOKEN` in the service
environment to enable the mutating API (it fails closed without one).

Run a command under a lease with the bundled wrapper (`bin/gpu-lease`): it requests
admission, pins `CUDA_VISIBLE_DEVICES` to the granted card, heartbeats, vouches liveness to
the stall watch, and releases on any exit path:

```sh
gpu-lease --vram 4000 --label render -- blender -b scene.blend -a
```

An MCP server (`mcp/gpu_manager_mcp.py`, `pip install "floating-point-ranger[mcp]"`) exposes
the same API to AI-agent sessions as discoverable tools.

## Tests

```sh
uv run pytest
```

The suite pins the safety invariants: never-preempt under holds, MPS fail-safe behaviour,
rollback-path correctness, capability routing (a legacy caller can never be handed a remote
card), degraded-agent fail-open, and the AMD fdinfo parser against real card output.

## License

MIT.
