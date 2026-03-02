#!/usr/bin/env python3
"""
Phase 3 unit test — UCCL intra-node CUDA IPC path via two separate OS processes.

Two processes are spawned on the same node.  Each creates a single nixl agent
backed by UCCL and registers a GPU buffer.  The client WRITES ones into the
server's GPU buffer.

Because the agents are in *different processes* on the *same node*:
  is_intra_node  = True   (same IP)
  is_same_process = False  (different PID → is_local_engine_addr() misses)
  has_ipc        = True   (VRAM buffer)
→ uccl_engine_write_vector dispatches to writev_ipc_async (CUDA IPC path).

Contrast with test_nixl_intranode.py (same-process), where the same-process
registry causes a direct gpuMemcpy() bypass instead.

Expected output from uccl_engine.cc in each subprocess:
  uccl_engine_connect: connection to <ip> is intra-node
  uccl_engine_accept:  connection from <ip> is intra-node

Metadata exchange protocol (via multiprocessing.Queue):
  server → client : ("meta",  agent_metadata_blob)
  server → client : ("descs", serialised_xfer_descs)
  client → server : ("meta",  agent_metadata_blob)
  client → server : ("done",  "")          ← after transfer completes
  server → client : ("pass",  "")          ← after buffer verified
             OR     ("fail",  reason)
"""

from __future__ import annotations

import multiprocessing as mp
import sys

BUF_ELEMS = 1024  # float32 elements (~4 KB)


# ---------------------------------------------------------------------------
# Subprocess entry points
# ---------------------------------------------------------------------------

def _server_proc(server_q: "mp.Queue[tuple]", client_q: "mp.Queue[tuple]") -> None:
    """Server: expose a zeros GPU buffer, wait for client WRITE, verify ones."""
    import torch

    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError:
        server_q.put(("error", "Failed to import NIXL"))
        return

    config = nixl_agent_config(backends=["UCCL"])
    agent = nixl_agent("server", config)

    if "UCCL" not in agent.backends:
        server_q.put(("error",
                      "UCCL backend not registered.  Set NIXL_PLUGIN_DIR."))
        return

    # Step 1: register GPU buffer BEFORE get_agent_metadata so that the
    # memory's public data (token hex) is included in the metadata blob.
    buf = torch.zeros(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
    reg = agent.register_memory(agent.get_reg_descs([buf]))

    # Step 2: publish metadata + serialised buffer descriptors to client.
    #   • get_agent_metadata() — includes conn info AND registered-mem tokens.
    #   • get_serialized_descs()  — the raw addr/len that the client needs for
    #     initialize_xfer; the client uses deserialize_descs() to reconstruct.
    server_meta = agent.get_agent_metadata()
    srv_descs   = reg.trim()
    server_q.put(("meta",  server_meta))
    server_q.put(("descs", agent.get_serialized_descs(srv_descs)))

    # Step 3: receive client metadata and establish bidirectional connection.
    tag, client_meta = client_q.get()
    assert tag == "meta", f"server: unexpected tag from client: {tag!r}"
    agent.add_remote_agent(client_meta)

    # Step 4: wait for client to signal that the WRITE transfer is complete.
    tag, _ = client_q.get()
    assert tag == "done", f"server: unexpected tag from client: {tag!r}"

    # Step 5: verify buffer contents.
    mean_val = buf.mean().item()
    if mean_val == 1.0:
        print("[server] PASS: GPU buffer filled with ones", flush=True)
        server_q.put(("pass", ""))
    else:
        msg = f"expected mean=1.0, got {mean_val}"
        print(f"[server] FAIL: {msg}", flush=True)
        server_q.put(("fail", msg))

    # Cleanup
    agent.deregister_memory(reg)
    agent.remove_remote_agent("client")


def _client_proc(server_q: "mp.Queue[tuple]", client_q: "mp.Queue[tuple]") -> None:
    """Client: receive server descs, WRITE a ones buffer into server GPU mem."""
    import torch

    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError:
        client_q.put(("error", "Failed to import NIXL"))
        return

    config = nixl_agent_config(backends=["UCCL"])
    agent = nixl_agent("client", config)

    if "UCCL" not in agent.backends:
        client_q.put(("error",
                      "UCCL backend not registered.  Set NIXL_PLUGIN_DIR."))
        return

    # Step 1: register local (source) GPU buffer.
    buf    = torch.ones(BUF_ELEMS, dtype=torch.float32, device="cuda:0")
    reg    = agent.register_memory(agent.get_reg_descs([buf]))
    cli_descs = reg.trim()

    # Step 2: send our metadata to server first (server is waiting for it after
    # it puts its own data onto server_q, so we must not block on server_q
    # before putting to client_q — avoid deadlock).
    client_meta = agent.get_agent_metadata()
    client_q.put(("meta", client_meta))

    # Step 3: receive server metadata and buffer descs.
    tag, server_meta   = server_q.get()
    assert tag == "meta",  f"client: unexpected tag from server: {tag!r}"
    tag, srv_serialized = server_q.get()
    assert tag == "descs", f"client: unexpected tag from server: {tag!r}"

    agent.add_remote_agent(server_meta)

    # Step 4: build and post the WRITE transfer.
    #   cli_descs   — local source (ones)
    #   remote_descs — server's destination (zeros, to be overwritten)
    remote_descs = agent.deserialize_descs(srv_serialized)
    handle = agent.initialize_xfer("WRITE", cli_descs, remote_descs, "server")
    state  = agent.transfer(handle)
    assert state != "ERR", "client: transfer() returned ERR"

    while True:
        state = agent.check_xfer_state(handle)
        assert state != "ERR", "client: check_xfer_state() returned ERR"
        if state == "DONE":
            break

    print("[client] PASS: GPU-to-GPU WRITE via CUDA IPC completed", flush=True)

    # Step 5: notify server; wait for its verification result.
    client_q.put(("done", ""))
    tag, msg = server_q.get()
    if tag == "fail":
        raise AssertionError(f"Server verification failed: {msg}")
    assert tag == "pass", f"client: unexpected server response: {tag!r}"

    # Cleanup
    agent.release_xfer_handle(handle)
    agent.deregister_memory(reg)
    agent.remove_remote_agent("server")


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------

def test_nixl_intranode_multiproc() -> None:
    """Spawn server + client as separate processes; verify CUDA IPC transfer."""
    print("=== UCCL intra-node IPC test (two OS processes) ===", flush=True)
    print("Exercises writev_ipc_async: different processes, same node, VRAM.",
          flush=True)
    print("Look for 'is intra-node' in the uccl_engine output below.",
          flush=True)
    print("", flush=True)

    # Use 'spawn' so each child starts a fresh Python interpreter — required
    # for CUDA safety (fork after CUDA init is unsafe) and mirrors production
    # use where separate ranks are separate OS processes.
    ctx = mp.get_context("spawn")

    server_q: mp.Queue = ctx.Queue()  # server → client
    client_q: mp.Queue = ctx.Queue()  # client → server

    server_p = ctx.Process(target=_server_proc, args=(server_q, client_q))
    client_p = ctx.Process(target=_client_proc, args=(server_q, client_q))

    server_p.start()
    client_p.start()

    timeout = 120  # seconds — allow time for process startup + CUDA init

    server_p.join(timeout=timeout)
    client_p.join(timeout=timeout)

    if server_p.is_alive():
        server_p.terminate()
        raise RuntimeError("Server process timed out")
    if client_p.is_alive():
        client_p.terminate()
        raise RuntimeError("Client process timed out")

    if server_p.exitcode != 0:
        raise RuntimeError(
            f"Server process exited with code {server_p.exitcode}")
    if client_p.exitcode != 0:
        raise RuntimeError(
            f"Client process exited with code {client_p.exitcode}")

    print("", flush=True)
    print("=== test_nixl_intranode_multiproc PASSED ===", flush=True)


if __name__ == "__main__":
    try:
        test_nixl_intranode_multiproc()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
