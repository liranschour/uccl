#!/usr/bin/env python3
"""
Test script for CPU↔GPU IPC transfers in send_ipc/recv_ipc
Tests all transfer scenarios:
  1. GPU → GPU (push model)
  2. CPU → GPU (push model)
  3. GPU → CPU (pull model)
"""
import torch
import torch.distributed as dist
import numpy as np
import sys
import os

# Add p2p to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uccl import p2p


def test_gpu_to_gpu(endpoint_sender, endpoint_receiver, conn_id_send, conn_id_recv):
    """Test GPU → GPU transfer (original functionality)"""
    print("\n[TEST 1] GPU → GPU Transfer")
    size = 1024 * 1024  # 1MB

    # Allocate GPU buffers
    send_buf = torch.ones(size // 4, dtype=torch.float32, device='cuda:0') * 42.0
    recv_buf = torch.zeros(size // 4, dtype=torch.float32, device='cuda:1')

    # Perform IPC transfer
    success_recv = endpoint_receiver.recv_ipc(conn_id_recv, recv_buf.data_ptr(), size)
    success_send = endpoint_sender.send_ipc(conn_id_send, send_buf.data_ptr(), size)

    # Verify
    assert success_send and success_recv, "Transfer failed"
    assert recv_buf.allclose(torch.tensor(42.0)), f"Data mismatch: got {recv_buf[0].item()}"
    print("✅ GPU → GPU transfer successful!")


def test_cpu_to_gpu(endpoint_sender, endpoint_receiver, conn_id_send, conn_id_recv):
    """Test CPU → GPU transfer (push model)"""
    print("\n[TEST 2] CPU → GPU Transfer")
    size = 1024 * 1024  # 1MB

    # Allocate CPU buffer (sender) and GPU buffer (receiver)
    send_buf = torch.ones(size // 4, dtype=torch.float32, pin_memory=True) * 123.0
    recv_buf = torch.zeros(size // 4, dtype=torch.float32, device='cuda:1')

    # Perform IPC transfer
    success_recv = endpoint_receiver.recv_ipc(conn_id_recv, recv_buf.data_ptr(), size)
    success_send = endpoint_sender.send_ipc(conn_id_send, send_buf.data_ptr(), size)

    # Verify
    assert success_send and success_recv, "Transfer failed"
    assert recv_buf.allclose(torch.tensor(123.0)), f"Data mismatch: got {recv_buf[0].item()}"
    print("✅ CPU → GPU transfer successful!")


def test_gpu_to_cpu(endpoint_sender, endpoint_receiver, conn_id_send, conn_id_recv):
    """Test GPU → CPU transfer (pull model)"""
    print("\n[TEST 3] GPU → CPU Transfer")
    size = 1024 * 1024  # 1MB

    # Allocate GPU buffer (sender) and CPU buffer (receiver)
    send_buf = torch.ones(size // 4, dtype=torch.float32, device='cuda:0') * 456.0
    recv_buf = torch.zeros(size // 4, dtype=torch.float32, pin_memory=True)

    # Perform IPC transfer
    success_recv = endpoint_receiver.recv_ipc(conn_id_recv, recv_buf.data_ptr(), size)
    success_send = endpoint_sender.send_ipc(conn_id_send, send_buf.data_ptr(), size)

    # Verify
    assert success_send and success_recv, "Transfer failed"
    assert recv_buf.allclose(torch.tensor(456.0)), f"Data mismatch: got {recv_buf[0].item()}"
    print("✅ GPU → CPU transfer successful!")


def main():
    print("=" * 60)
    print("Testing CPU↔GPU IPC Transfers")
    print("=" * 60)

    # Initialize distributed
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    assert world_size == 2, f"This test requires 2 processes, got {world_size}"

    # Set log level to see traces
    os.environ.setdefault("UCCL_P2P_LOG_LEVEL", "INFO")

    # Create endpoints
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    endpoint = p2p.Endpoint(local_rank, num_cpus=4)

    # Exchange metadata
    local_metadata = endpoint.get_metadata()
    if rank == 0:
        dist.send(torch.ByteTensor(list(local_metadata)), dst=1)
        remote_metadata_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_metadata_tensor, src=1)
        remote_metadata = bytes(remote_metadata_tensor.tolist())
    else:
        remote_metadata_tensor = torch.zeros(len(local_metadata), dtype=torch.uint8)
        dist.recv(remote_metadata_tensor, src=0)
        dist.send(torch.ByteTensor(list(local_metadata)), dst=1)
        remote_metadata = bytes(remote_metadata_tensor.tolist())

    remote_ip, remote_port, remote_gpu_idx = p2p.Endpoint.parse_metadata(remote_metadata)

    # Establish local IPC connections
    if rank == 0:
        print(f"[Rank {rank}] Connecting to rank 1 (GPU {remote_gpu_idx})...")
        conn_id = endpoint.connect_local(remote_gpu_idx)
        print(f"[Rank {rank}] Connected with conn_id={conn_id}")
    else:
        print(f"[Rank {rank}] Accepting connection from rank 0...")
        remote_gpu_idx_recv, conn_id = endpoint.accept_local()
        print(f"[Rank {rank}] Accepted connection with conn_id={conn_id}")

    # Synchronize before tests
    dist.barrier()

    # Run tests
    try:
        if rank == 0:
            # Rank 0 is sender
            test_gpu_to_gpu(endpoint, None, conn_id, None)
            test_cpu_to_gpu(endpoint, None, conn_id, None)
            test_gpu_to_cpu(endpoint, None, conn_id, None)
        else:
            # Rank 1 is receiver
            test_gpu_to_gpu(None, endpoint, None, conn_id)
            test_cpu_to_gpu(None, endpoint, None, conn_id)
            test_gpu_to_cpu(None, endpoint, None, conn_id)

        dist.barrier()

        if rank == 0:
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED!")
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
