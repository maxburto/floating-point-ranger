# Floating-Point Ranger

A small FastAPI service that lets batch jobs, model servers, and a live desktop share the
same GPUs without ever preempting anything.

I built this because my homelab treats two machines as one computer: two NVIDIA cards run
my desktop session, renders, an OCR pipeline, and a few on-demand model servers, and a
third AMD card lives in a separate guest. Ad-hoc lock files weren't cutting it - batch work
kept landing on the desktop card. The fix is one arbiter that knows every card: every job
asks it first, and nothing ever gets killed.

## The one hard rule

Never preempt. Admission is check-and-defer: a job asks for a lease and either gets the
card or gets told to wait, with machine-readable reasons and a retry hint. No code path
kills, evicts, or signals a GPU process it didn't start. The single gated exception is the
stall watch, which can evict a job that has done no GPU work for a configurable window and
whose owner won't vouch for it - and that enforcement is off by default.

## What it does

- Status and queue. `/v1/gpu/status` reports VRAM, live compute and graphics processes,
  leases, holds, and rogue processes; `/v1/gpu/queue` merges external job-queue sources.
  A dashboard includes a validated live-config editor.
- Lease admission. `POST /v1/lease` grants or defers on real NVML VRAM minus unrealized
  reservations, per-card exclusive lanes, holds, and capability routing - a CUDA job is
  never handed the AMD card.
- Interactive priority. Mark a card `interactive` and a presence poller (NoMachine in my
  case) or a manual hold pauses new batch admission there. Running work always finishes.
- Model residency. Declared systemd model units (llama.cpp, dockerized STT/TTS) start only
  via the manager, hold non-exclusive VRAM reservations, and drain when idle. Wake-on-use
  TCP gates keep consumer URLs stable while a model sleeps; the first real request wakes it.
- Stall watch. Flags leases with no GPU activity, asks the owner's `health_url` to vouch,
  and evicts confirmed zombies only when enforcement is explicitly enabled.
- Rogue watch. Unmanaged GPU processes are attributed to their owning unit, container, or
  user and surfaced. They're never touched.
- Cross-node AMD reporting. A read-only sysfs/fdinfo agent runs inside the guest that owns
  the AMD card; the manager polls it and fails open - agent down means the card shows
  offline and AMD work runs unmanaged.
- MPS batch concurrency. With NVIDIA MPS on a batch card, exclusive batch asks are
  downgraded server-side so jobs co-execute. A dead MPS daemon fails safe back to serial.

## Quickstart

```sh
uv pip install "floating-point-ranger @ git+https://github.com/maxwellburton/floating-point-ranger@v0.2.0"
cp config.example.yaml /etc/gpu-manager/config.yaml   # edit: your GPU UUIDs + roles
GPU_MANAGER_CONFIG=/etc/gpu-manager/config.yaml uvicorn gpu_manager.app:APP --port 8768
```

Systemd templates for the service, the MPS daemon, and the nightly model pre-warm are in
`systemd/`; the AMD reporter agent is in `agent/`. Set `GPU_MANAGER_TOKEN` in the service
environment to enable the mutating API - it fails closed without one.

`bin/gpu-lease` runs a command under a lease: it requests admission, pins
`CUDA_VISIBLE_DEVICES` to the granted card, heartbeats, vouches liveness to the stall
watch, and releases on any exit path:

```sh
gpu-lease --vram 4000 --label render -- blender -b scene.blend -a
```

An MCP server (`mcp/gpu_manager_mcp.py`, `pip install "floating-point-ranger[mcp]"`)
exposes the same API to AI-agent sessions as discoverable tools. Half the GPU consumers in
my homelab are built and operated by agents - an agent that can discover the arbiter stops
stealing the desktop card.

## Tests

```sh
uv run pytest
```

The suite pins the safety invariants: never-preempt under holds, MPS fail-safe behavior,
rollback-path correctness, capability routing (a legacy caller can never be handed a
remote card), degraded-agent fail-open, and the AMD fdinfo parser against real card output.

This is a hobby project extracted from my homelab's infra repo, so some defaults are
opinionated. Issues and PRs welcome.

## License

MIT.
