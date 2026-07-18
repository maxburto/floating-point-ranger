#!/usr/bin/env bash
# ExecStop for nvidia-mps.service.
#
# This lives in a script rather than an inline `ExecStop=` for a concrete reason, not style:
# an inline `pkill -f nvidia-cuda-mps-control` MATCHES THE STOP COMMAND'S OWN cmdline and kills
# its own shell — the unit then ends up `failed` and the pipe directory is left behind (observed
# live, 2026-07-18). A script's cmdline is just its path, so its patterns cannot match itself.
#
# Always exits 0: a best-effort teardown must never mark the unit failed.
set -u
PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/run/nvidia-mps}"
export CUDA_MPS_PIPE_DIRECTORY="$PIPE_DIR"

# 1. Graceful: drains connected clients, then the server exits.
#    --foreground -k so a control client blocked on a dead pipe is actually killed rather than
#    surviving as an orphan still holding it.
timeout --foreground -k 2 10 sh -c 'echo quit | /usr/bin/nvidia-cuda-mps-control' >/dev/null 2>&1

# 2. A client that died without synchronising its GPU work can hang that graceful quit (NVIDIA
#    documents it as leaving the server in an undefined state) — escalate rather than hang.
if pgrep -f '^/usr/bin/nvidia-cuda-mps-control -d' >/dev/null 2>&1; then
  timeout --foreground -k 2 10 sh -c 'echo quit_immediate | /usr/bin/nvidia-cuda-mps-control' \
    >/dev/null 2>&1
fi

# 3. Last resort. Patterns are anchored to the full binary path so they cannot hit an unrelated
#    cmdline that merely CONTAINS the name (e.g. an operator tailing the MPS log during exactly
#    the debugging session that prompted this stop). `pgrep -x` cannot match these >15-char
#    names, hence -f. Anything still alive is in our cgroup and systemd will reap it.
sleep 1
pkill -f '^/usr/bin/nvidia-cuda-mps-control -d' >/dev/null 2>&1
pkill -f '^/usr/bin/nvidia-cuda-mps-server' >/dev/null 2>&1
sleep 1

# 4. Remove the pipe directory LAST. This matters: a stale directory left behind by a failed
#    stop used to make the manager believe MPS was still alive, which kept batch leases
#    non-exclusive with no MPS actually running — the exclusive lane gone with nothing
#    replacing it. (server_up() now verifies the daemon pid too, so this is belt-and-braces.)
rm -rf "$PIPE_DIR"
exit 0
