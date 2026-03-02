# UCCL Intra-Node Transfer: Implementation Plan

## End Goal

Enable the nixl uccl backend to automatically route transfers between **two separate nixl agents
on the same physical node** through CUDA IPC (`write_ipc` / `read_ipc` / `writev_ipc` /
`readv_ipc`) instead of RDMA. This applies to the standard two-agent transfer case
(`loadRemoteConnInfo` → `postXfer`) — not to be confused with `supportsLocal()`, which is a
nixl concept for a single agent transferring to itself.

When complete, a nixl agent that connects to a peer on the same machine will transparently use
the IPC data path, with no change required from the caller.

The IPC path (`write_ipc` / `read_ipc`) already exists and is tested at the Python / pybind11
level. This plan wires it into the `uccl_engine` C API and then into the nixl plugin.

---

## Nixl Plugin Code Path (reference)

The relevant flow through `src/plugins/uccl/uccl_backend.cpp`:

```
nixlUcclEngine::getConnInfo()            → uccl_engine_get_metadata()  → returns "ip:port?gpu"
nixlUcclEngine::loadRemoteConnInfo()     → uccl_engine_connect(ip, gpu, port)
nixlUcclEngine::startListener() [thread] → uccl_engine_accept()
nixlUcclEngine::postXfer()               → uccl_engine_read() / uccl_engine_write()
```

Key observations:
- `loadRemoteConnInfo` already has the remote IP (parsed from `remote_conn_info`)
- The local IP is available from `uccl_engine_get_metadata()`
- Comparing these two IPs is the correct and natural place to detect intra-node
- `startListener` stores accepted connections keyed by remote IP (`connected_agents_[ip_buf]`)

---

## Phase 0 — Full Code Flow (reference)

This section traces exactly what happens when a nixl user writes or reads memory between two
agents using the UCCL backend. Memory type (CPU/GPU) is noted where it matters.

Both `DRAM_SEG` (CPU/pinned) and `VRAM_SEG` (GPU) are supported — `getSupportedMems()` returns
both. `registerMem()` and `postXfer()` use the same code path for both; the RDMA engine handles
the rkey/lkey difference internally.

---

### Step 1 — Agent Creation (both sides, independent)

```python
config = nixl_agent_config(backends=["UCCL"])
agent  = nixl_agent("client", config)   # or "server"
```

**C++ (constructor):**
- `uccl_engine_create(num_cpus, in_python)` → allocates `Endpoint`, starts RDMA threads
- Spawns `startListener()` thread → immediately blocks in `uccl_engine_accept()`

---

### Step 2 — Memory Registration (both sides, BEFORE metadata exchange)

```python
descs          = agent.get_reg_descs(dataset)    # list of (addr, len, devId) descriptors
register_descs = agent.register_memory(descs)    # pins memory with RDMA engine
```

**C++ (`registerMem`):**
- `uccl_engine_reg(engine_, addr, len, &mr_id)` — pins the buffer with the RDMA NIC
  - Works for both CPU (pinned/DRAM) and GPU (VRAM) buffers
- `uccl_engine_prepare_fifo(engine_, mr_id, addr, len, fifo_item[64])` — pre-computes the
  64-byte `FifoItem` (contains rkey + lkey + base address) needed for one-sided RDMA
- Stores `nixlUcclBackendMD{ addr, len, mr_id, fifo_item }` in `mem_reg_info_`

The `fifo_item` is serialised as a hex string in `getPublicData()`.

> **Critical ordering**: `register_memory` must precede `get_agent_metadata()`.
> `getLocalMD()` serialises both connection info AND the registered memory's public data.
> If memory is not yet registered when the peer calls `add_remote_agent`, the remote
> memory section (`remoteSections[agent]`) stays empty, and `createXferReq` cannot find
> any backend capable of the transfer.

---

### Step 3 — Exchange metadata / connect (out-of-band)

```python
local_meta = agent.get_agent_metadata()   # returns bytes (conn info + memory section)
# user sends local_meta to the peer out-of-band (e.g. ZMQ)
agent.add_remote_agent(peer_meta)         # connect + load remote memory metadata
local_xfer_descs  = register_descs.trim()
remote_xfer_descs = agent.deserialize_descs(peer_xfer_desc_bytes)
```

**`getLocalMD()` (`get_agent_metadata`):**
- Serialises connection string (`"ip:port?gpu"`) from `getConnInfo()`
- Serialises registered memory section: for each descriptor, calls `getPublicData()`
  which emits the hex-encoded `fifo_item`

**C++ (`loadRemoteConnInfo`)** — called inside `add_remote_agent` → `loadRemoteMD`:
- Parses `"ip:port?gpu"` from peer metadata
- `uccl_engine_connect(engine_, ip, gpu, port)` — TCP + RDMA handshake to peer
- `uccl_engine_start_listener(conn)` — spawns completion-notification receiver thread
- Stores `conn` in `connected_agents_[remote_agent]` — keyed by **agent name** (string)

**C++ (peer `startListener` thread):**
- `uccl_engine_accept()` returns a new `uccl_conn_t*`
- `uccl_engine_start_listener(conn)` — same receiver thread on accepting side
- Stores `conn` in `connected_agents_[remote_ip]` — keyed by **remote IP** (string)

> **Key asymmetry**: the connecting side keys `connected_agents_` by agent name; the
> accepting side keys it by remote IP. Both sides look up by agent name in `prepXfer`,
> so the accepting side's entry must be re-keyed (or looked up differently) if agent
> name ≠ IP — this is the current behaviour and a known quirk.

**C++ (`loadRemoteSections`)** — also called inside `add_remote_agent` → `loadRemoteMD`:
- For each descriptor in the peer's serialised memory section, calls `loadRemoteMD`
  on the backend, which decodes the hex fifo_item into a new `nixlUcclBackendMD`
- Stores the result in `data->remoteSections[peer_name]`

**`deserialize_descs`** — Python `pickle.loads` of the peer's `nixlXferDList`:
- Carries addr/len/devId so `createXferReq` can look up the remote buffer in
  the already-loaded `remoteSections[peer_name]` via `populate()`

---

### Step 5 — Transfer Preparation (`initialize_xfer` / `prepXfer`)

```python
handle = agent.initialize_xfer("WRITE", local_xfer_descs, remote_xfer_descs, "server")
```

**C++ (`prepXfer`):**
- Looks up `uccl_conn_t*` for the remote agent name
- For each iovec pair (local[i], remote[i]):
  - Deserialises remote `fifo_item` → `FifoItem` struct
  - `uccl_engine_update_fifo(fifo_item, remote_addr, size)` — patches the exact sub-buffer
    address and size into the pre-computed FifoItem
- Creates `nixlUcclReqH{ conn, fifo_items[] }`

No data moves here.

---

### Step 6 — Transfer Execution (`transfer` / `postXfer`)

```python
state = agent.transfer(handle)          # returns "INPROG"
while agent.check_xfer_state(handle) != "DONE":
    pass
```

**C++ (`postXfer`):**
- Builds vectors of `(mr_id, local_addr, size)` for each iovec
- **NIXL_WRITE** → `uccl_engine_write_vector(conn, mr_ids, local_addrs, sizes, fifo_items, n)`
  - One-sided RDMA WRITE: local NIC pushes data directly into remote memory using rkey
  - Works for CPU→CPU, GPU→CPU, CPU→GPU, GPU→GPU (all via RDMA)
- **NIXL_READ** → `uccl_engine_read_vector(conn, mr_ids, local_addrs, sizes, fifo_items, n)`
  - One-sided RDMA READ: local NIC pulls data from remote memory using rkey
- Returns `NIXL_IN_PROG`; transfer is async

**C++ (`checkXfer`):**
- `uccl_engine_xfer_status(conn, transfer_id)` — polls NIC completion queue
- When done, sends a completion notification via TCP (`uccl_engine_send_notif`) so the remote
  side knows the transfer finished (important for the WRITE case where the writer is the only
  one that knows when data landed)

---

### Key architectural notes

| Aspect | Current (RDMA) path |
|--------|---------------------|
| Connection | TCP + RDMA QP pair via `uccl_engine_connect` / `uccl_engine_accept` |
| Memory token | 64-byte `FifoItem` with RDMA rkey + base address, pre-computed at `registerMem` |
| Transfer | One-sided RDMA WRITE or READ — remote side does **nothing** during data movement |
| Completion signal | TCP notification from initiator to remote side via `uccl_engine_send_notif` |
| CPU memory | Pinned DRAM, registered with RDMA NIC — same RDMA path as GPU |
| GPU memory | VRAM registered with RDMA NIC (GPUDirect) — same RDMA path as CPU |
| `connected_agents_` key | agent name on connect-side; remote IP on accept-side |

---

## Phases

### Phase 1 — Detection in `uccl_engine`  ✅ Done

`is_intra_node` bool added to `uccl_conn` struct in `uccl_engine.cc`.
Set by comparing remote IP with local IP (via `get_local_ip_from_engine()`) in both
`uccl_engine_connect()` and `uccl_engine_accept()`. A `std::cout` log prints the result.

**Files changed:** `uccl_engine.cc` only — no header or pybind11 changes needed.

#### Phase 1 unit test

`tests/test_nixl_intranode.py` — a self-contained test that creates two nixl agents
in the same process. Metadata is exchanged as plain variables; the C++ listener
threads handle the actual UCCL connections in the background. Replicates the exact
nixl code path:

```
registerMem → getConnInfo (get_agent_metadata) → add_remote_agent (loadRemoteConnInfo + loadRemoteSections)
→ initialize_xfer (prepXfer) → transfer (postXfer) → check_xfer_state (checkXfer)
```

A GPU-to-GPU WRITE (client GPU ones → server GPU zeros) is used to confirm the
connection is functional end-to-end, not just established.

**Runtime requirements:**
- nixl must be built from source with the UCCL plugin (`-Dplugins=UCCL` meson option).
  The plugin is output as `libplugin_UCCL.so` in the build tree.
- Set `NIXL_PLUGIN_DIR` to the directory containing `libplugin_UCCL.so` before running.
  nixl's plugin manager reads this env var at agent-creation time (before the C++ singleton
  initialises). Alternatively it looks for a `plugins/` directory next to the nixl `.so`.

```bash
export NIXL_PLUGIN_DIR=/path/to/nixl/build/src/plugins/uccl
cd /home/lirans/uccl/p2p
python tests/test_nixl_intranode.py
```

Expected output (from `std::cout` in `uccl_engine.cc`):
```
uccl_engine_connect: connection to <real-NIC-IP> is intra-node
uccl_engine_accept:  connection from <real-NIC-IP> is intra-node
[server] PASS: GPU buffer filled with ones by client WRITE
[client] PASS: GPU-to-GPU WRITE completed successfully
=== test_nixl_intranode PASSED ===
```

The test raises `RuntimeError` immediately if the UCCL backend was not loaded (e.g. if
`NIXL_PLUGIN_DIR` is wrong), before reaching any transfer code.

---

### Phase 2 — Unified token + IPC fast path (all branching in `uccl_engine`)

**Design principle:** the nixl plugin never sees RDMA vs IPC. All path selection lives in
`uccl_engine`. A new `uccl_mem_token_t` struct carries both tokens; C API functions branch
internally on `conn->is_intra_node && token.has_ipc`.

**Key implementation facts:**
- `advertise_ipc` does **not** use `conn_id` — it only calls `cudaIpcGetMemHandle`. Token can
  be computed at `registerMem` time, same as `FifoItem`. No ordering problem.
- `uccl_engine_xfer_status` (via `poll_async`) works for both RDMA and IPC `transfer_id`s —
  both use `TransferStatus*` cast to `uint64_t`. `checkXfer` needs **no changes**.

#### Step 1 — Add `uccl_mem_token_t` to `common.h`

```c
#define IPC_TOKEN_SIZE sizeof(IpcTransferInfo)   // ~88 bytes

typedef struct uccl_mem_token {
    char     fifo_buf[FIFO_SIZE];    // RDMA FifoItem — always computed
    char     ipc_buf[IPC_TOKEN_SIZE];// IpcTransferInfo — zeroed for CPU buffers
    uint64_t base_addr;              // registered region base address (for sub-buf patching)
    bool     has_ipc;                // true iff buffer is VRAM
} uccl_mem_token_t;
```

`IpcTransferInfo` must be moved from `engine.h` into `common.h` so that `uccl_engine.h`
exposes it without pulling in the full C++ `Endpoint` header.

#### Step 2 — Replace C API token functions in `uccl_engine.h` / `uccl_engine.cc`

| Old | New | Notes |
|-----|-----|-------|
| `uccl_engine_prepare_fifo(engine, mr, addr, len, fifo_buf)` | `uccl_engine_prepare_token(engine, mr, addr, len, token*)` | Calls `prepare_fifo` + `advertise_ipc` for VRAM |
| `uccl_engine_update_fifo(fifo_item, remote_addr, size)` | `uccl_engine_update_token(token*, sub_addr, size)` | Patches FifoItem addr **and** IPC offset from `base_addr` |
| `uccl_engine_write_vector(…, FifoItem vec, …)` | `uccl_engine_write_vector(…, uccl_mem_token_t vec, …)` | Dispatches to `writev_async` or `writev_ipc_async` |
| `uccl_engine_read_vector(…, FifoItem vec, …)` | `uccl_engine_read_vector(…, uccl_mem_token_t vec, …)` | Same |

**`uccl_engine_prepare_token` internals:**
```cpp
// RDMA token (always)
prepare_fifo(engine, mr, addr, len, token->fifo_buf);
token->base_addr = (uint64_t)addr;
// IPC token (GPU memory only)
if (get_dev_idx(addr) != -1) {
    advertise_ipc(engine, addr, len, token->ipc_buf);  // conn_id unused
    token->has_ipc = true;
}
```

**`uccl_engine_update_token` internals:**
```cpp
// RDMA: patch absolute sub-buffer address into FifoItem
FifoItem fi; deserialize_fifo_item(token->fifo_buf, &fi);
update_fifo_item(&fi, sub_addr, size);
serialize_fifo_item(fi, token->fifo_buf);
// IPC: recompute offset from CUDA-aligned base to sub_addr
if (token->has_ipc) {
    IpcTransferInfo* ipc = (IpcTransferInfo*)token->ipc_buf;
    uintptr_t aligned = token->base_addr & ~(kIpcAlignment - 1);
    ipc->offset = sub_addr - aligned;
    ipc->size   = size;
}
```

**`uccl_engine_write/read_vector` internals:**
```cpp
if (conn->is_intra_node && tokens[0].has_ipc) {
    // build IpcTransferInfo vec from token.ipc_buf
    endpoint->writev_ipc_async(conn->conn_id_, src_v, size_v, ipc_infos, n, transfer_id);
} else {
    // build FifoItem vec from token.fifo_buf
    endpoint->writev_async(conn->conn_id_, mr_ids, src_v, size_v, fifo_items, n, transfer_id);
}
```

#### Step 3 — Update nixl plugin (`uccl_backend.h` / `uccl_backend.cpp`)

**`nixlUcclBackendMD`:** replace `char fifo_item[FIFO_SIZE]` with `uccl_mem_token_t token`.

**`nixlUcclReqH`:** replace `std::vector<FifoItem> fifo_items` with
`std::vector<uccl_mem_token_t> tokens`.

Call-site changes (type rename only, no logic change):

| Function | Old | New |
|----------|-----|-----|
| `registerMem` | `uccl_engine_prepare_fifo(…, priv->fifo_item)` | `uccl_engine_prepare_token(…, &priv->token)` |
| `getPublicData` | hex-encode `priv->fifo_item` | hex-encode `priv->token` |
| `loadRemoteMD` | decode hex → `fifo_item` | decode hex → `token` |
| `prepXfer` | `deserialize_fifo_item` + `uccl_engine_update_fifo` | `uccl_engine_update_token(&rmd->token, …)` |
| `postXfer` | `uccl_engine_write_vector(…, fifo_items, …)` | `uccl_engine_write_vector(…, tokens, …)` |
| `checkXfer` | unchanged | unchanged |

The nixl plugin contains **zero** `is_intra_node` checks and **zero** IPC-specific code.

#### Step 4 — Unit test

Extend `tests/test_nixl_intranode.py`: after a successful intra-node transfer, verify that
the log contains `writev_ipc_async` (or add a counter/flag to the engine) rather than the
RDMA path. A second run with `is_intra_node` forced off (or a cross-node pair) should still
pass via RDMA, confirming the fallback works.

**Files changed:**
- `uccl/p2p/include/common.h` — add `uccl_mem_token_t`; move `IpcTransferInfo` here
- `uccl/p2p/engine.h` — remove `IpcTransferInfo` (now in `common.h`)
- `uccl/p2p/uccl_engine.h` — replace `prepare_fifo`/`update_fifo`, update `write/read_vector` signatures
- `uccl/p2p/uccl_engine.cc` — implement `prepare_token`, `update_token`, updated dispatch in `write/read_vector`
- `nixl/src/plugins/uccl/uccl_backend.h` — token type changes
- `nixl/src/plugins/uccl/uccl_backend.cpp` — call-site renames only

---

### Phase 3 — Multi-process intra-node unit test

**Goal:** verify that the CUDA IPC path (`writev_ipc_async`) works correctly for the
production case: **two separate OS processes** on the same node (different PIDs).

This is the complement to `test_nixl_intranode.py` (same process, direct `gpuMemcpy`
bypass). The two tests together confirm both sub-cases of `is_intra_node`:

| Test file | Processes | `is_same_process` | Path taken |
|-----------|-----------|-------------------|------------|
| `test_nixl_intranode.py` | 1 (both agents in-process) | `true` | direct `gpuMemcpy(fi.addr)` |
| `test_nixl_intranode_multiproc.py` | 2 (spawned subprocesses) | `false` | `writev_ipc_async` CUDA IPC |

#### Why two processes are needed

`cudaIpcOpenMemHandle` is **forbidden** within the same OS process that called
`cudaIpcGetMemHandle` on the same allocation. The same-process same-node case is
therefore handled by a direct `gpuMemcpy` path (Phase 2 `is_same_process` bypass).
The cross-process same-node case is the real CUDA IPC target; it requires separate
PIDs and thus cannot be tested from a single Python interpreter.

#### Same-process detection (`is_same_process`)

Added to `uccl_engine.cc` alongside Phase 2. Works without any protocol change:

1. `uccl_engine_get_metadata()` registers the engine's `(ip, port)` in a
   process-global map (`s_local_listen_ports`) when it is first called.
2. `uccl_engine_connect()` checks the registry: if `(remote_ip, remote_port)` is
   already there, the remote endpoint is another engine **in this process** →
   `conn->is_same_process = true`.
3. `uccl_engine_write/read_vector`: when `is_intra_node && is_same_process`, use
   `gpuMemcpy(fi.addr, ...)` directly — `fi.addr` is the peer's actual device pointer,
   accessible in the same address space. Returns sentinel `transfer_id = 0`.
4. `uccl_engine_xfer_status`: `transfer_id == 0` → return `true` immediately (already done).

The nixl plugin (`uccl_backend.cpp`) requires **no changes** for this detection.
The registry is populated before `uccl_engine_connect` is called because nixl always
calls `get_agent_metadata()` before `add_remote_agent()`.

#### Test design (`tests/test_nixl_intranode_multiproc.py`)

Uses `multiprocessing.get_context("spawn")` so each subprocess starts a fresh Python
interpreter (CUDA is not fork-safe). Metadata is exchanged via two
`multiprocessing.Queue` objects (one per direction).

**Deadlock-free exchange protocol:**
```
server_q (server → client):  ("meta", blob)  →  ("descs", bytes)
client_q (client → server):  ("meta", blob)
                              ("done", "")       ← after WRITE completes
server_q (server → client):  ("pass", "")       ← after buffer verified
                          or  ("fail", reason)
```

Server sends first (non-blocking queue puts, then blocks on client's meta).
Client sends its own meta first, then receives server's data — no circular wait.

**Why `get_serialized_descs` / `deserialize_descs` are needed:**
In the single-process test the two agents share memory so descriptors can be passed
as Python objects. Across processes the raw `(addr, len)` descriptor list must be
serialised; nixl's `get_serialized_descs` + `deserialize_descs` API does this. The
client passes the deserialized result as `remote_descs` to `initialize_xfer`.

**Files changed:**
- `uccl/p2p/uccl_engine.cc` — `is_same_process` field, `s_local_listen_ports` registry,
  direct `gpuMemcpy` path in `write/read_vector`, sentinel in `xfer_status`,
  `uccl_engine_set_same_process()` helper
- `uccl/p2p/uccl_engine.h` — declaration of `uccl_engine_set_same_process`
- `uccl/p2p/tests/test_nixl_intranode_multiproc.py` — new test (no C++ changes needed)

**Running:**
```bash
export NIXL_PLUGIN_DIR=/path/to/nixl/build/src/plugins/uccl
cd /home/lirans/uccl/p2p
python tests/test_nixl_intranode_multiproc.py
```

Expected output (interleaved from both subprocesses):
```
=== UCCL intra-node IPC test (two OS processes) ===
uccl_engine_connect: connection to <ip> is intra-node
uccl_engine_accept:  connection from <ip> is intra-node
[client] PASS: GPU-to-GPU WRITE via CUDA IPC completed
[server] PASS: GPU buffer filled with ones
=== test_nixl_intranode_multiproc PASSED ===
```

---

## Status

| Phase | Status |
|-------|--------|
| Phase 1 — `uccl_conn::is_intra_node` in `uccl_engine.cc` | ✅ Done |
| Phase 1 unit test — same-process intra-node (`test_nixl_intranode.py`) | ✅ Done |
| Phase 2 — `uccl_mem_token_t` + IPC dispatch + same-process bypass | ✅ Done |
| Phase 3 — multi-process unit test (`test_nixl_intranode_multiproc.py`) | ✅ Done |
