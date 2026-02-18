#include "engine.h"
#include "endpoint_wrapper.h"
#include "util/util.h"
#include <arpa/inet.h>
#include <glog/logging.h>
#include <netinet/in.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <optional>
#include <sstream>
#include <thread>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

int const kMaxNumGPUs = 8;
std::once_flag glog_init_once;
constexpr uint32_t kGpuStreamId = 0;
thread_local bool inside_python = false;

inline void check_python_signals() {
  PyGILState_STATE gstate = PyGILState_Ensure();
  if (PyErr_CheckSignals() != 0) {
    std::cerr << "Python signal caught, exiting..." << std::endl;
    std::abort();
  }
  PyGILState_Release(gstate);
}

Endpoint::Endpoint(uint32_t const local_gpu_idx, uint32_t const num_cpus)
    : local_gpu_idx_(local_gpu_idx), num_cpus_(num_cpus) {
  std::cout << "Creating Engine with GPU index: " << local_gpu_idx
            << ", CPUs: " << num_cpus << std::endl;
  int n_streams = std::max(1, (int)kNumGpuRtStreams);

  int ngpus = 0;
  GPU_RT_CHECK(gpuGetDeviceCount(&ngpus));
  ipc_streams_.resize(ngpus);
  for (int i = 0; i < ngpus; ++i) {
    GPU_RT_CHECK(gpuSetDevice(i));
    ipc_streams_[i].resize(n_streams);
    for (int j = 0; j < n_streams; ++j) {
      GPU_RT_CHECK(
          gpuStreamCreateWithFlags(&ipc_streams_[i][j], gpuStreamNonBlocking));
    }
  }

  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
  streams_.resize(n_streams);
  for (int i = 0; i < n_streams; ++i) {
    GPU_RT_CHECK(gpuStreamCreateWithFlags(&streams_[i], gpuStreamNonBlocking));
  }

  std::call_once(glog_init_once,
                 []() { google::InitGoogleLogging("uccl_p2p"); });
  FLAGS_minloglevel = parseLogLevelFromEnv();
  FLAGS_logtostderr = true;
  google::InstallFailureSignalHandler();

  // Initialize the RDMA endpoint.
  ep_ = std::shared_ptr<NICEndpoint>(
      new NICEndpoint(local_gpu_idx_, INVALID_RANK_ID, 0, false));

  std::cout << "Engine initialized for GPU " << local_gpu_idx_ << std::endl;
  engine_initialized_ = true;

  send_unified_task_ring_ =
      uccl::create_ring(sizeof(UnifiedTask*), kTaskRingSize);
  recv_unified_task_ring_ =
      uccl::create_ring(sizeof(UnifiedTask*), kTaskRingSize);

  ipc_completion_ring_ =
      uccl::create_ring(sizeof(PendingIpcBatch*), kTaskRingSize);

  send_proxy_thread_ = std::thread(&Endpoint::send_proxy_thread_func, this);
  recv_proxy_thread_ = std::thread(&Endpoint::recv_proxy_thread_func, this);
  ipc_completion_thread_ =
      std::thread(&Endpoint::ipc_completion_thread_func, this);

  // Initialize UDS socket for local connections
  init_uds_socket();

  std::cout << "Endpoint initialized successfully" << std::endl;
}

Endpoint::Endpoint(uint32_t const num_cpus)
    : local_gpu_idx_(INVALID_GPU), num_cpus_(num_cpus) {
  std::cout << "Creating Engine with CPUs: " << num_cpus << std::endl;
  int n_streams = std::max(1, (int)kNumGpuRtStreams);

  int ngpus = 0;
  GPU_RT_CHECK(gpuGetDeviceCount(&ngpus));
  ipc_streams_.resize(ngpus);
  for (int i = 0; i < ngpus; ++i) {
    GPU_RT_CHECK(gpuSetDevice(i));
    ipc_streams_[i].resize(n_streams);
    for (int j = 0; j < n_streams; ++j) {
      GPU_RT_CHECK(
          gpuStreamCreateWithFlags(&ipc_streams_[i][j], gpuStreamNonBlocking));
    }
  }

  std::call_once(glog_init_once,
                 []() { google::InitGoogleLogging("uccl_p2p"); });

  google::InstallFailureSignalHandler();
  // Initialize the RDMA endpoint with lazy creation.
  ep_ = std::shared_ptr<NICEndpoint>(
      new NICEndpoint(INVALID_GPU, INVALID_RANK_ID, 0, false));
  std::cout << "Endpoint initialized successfully" << std::endl;
}

Endpoint::~Endpoint() {
  std::cout << "Destroying Engine..." << std::endl;

  stop_.store(true, std::memory_order_release);

  if (send_proxy_thread_.joinable()) {
    send_proxy_thread_.join();
  }
  if (recv_proxy_thread_.joinable()) {
    recv_proxy_thread_.join();
  }
  if (ipc_completion_thread_.joinable()) {
    ipc_completion_thread_.join();
  }

  if (send_unified_task_ring_ != nullptr) {
    free(send_unified_task_ring_);
  }
  if (recv_unified_task_ring_ != nullptr) {
    free(recv_unified_task_ring_);
  }
  if (ipc_completion_ring_ != nullptr) {
    free(ipc_completion_ring_);
  }

  {
    std::shared_lock<std::shared_mutex> lock(conn_mu_);
    for (auto& [conn_id, conn] : conn_id_to_conn_) {
      // Close UDS socket if it exists
      if (conn->uds_sockfd_ >= 0) {
        close(conn->uds_sockfd_);
      }
      delete conn;
    }
  }
  {
    std::shared_lock<std::shared_mutex> lock(mr_mu_);
    for (auto& [mr_id, mr] : mr_id_to_mr_) {
      delete mr;
    }
  }

  if (!streams_.empty()) {
    GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
    for (auto s : streams_)
      if (s) GPU_RT_CHECK(gpuStreamDestroy(s));
  }

  // Cleanup UDS socket
  cleanup_uds_socket();

  std::cout << "Engine destroyed" << std::endl;
}

void Endpoint::initialize_engine() {
  int n_streams = std::max(1, (int)kNumGpuRtStreams);
  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));
  streams_.resize(n_streams);
  for (int i = 0; i < n_streams; ++i) {
    GPU_RT_CHECK(gpuStreamCreateWithFlags(&streams_[i], gpuStreamNonBlocking));
  }

  numa_node_ = RdmaDeviceManager::instance().get_numa_node(local_gpu_idx_);

  // Initialize rdma contexts for devices used by the GPU
  initialize_rdma_ctx_for_gpu(ep_, local_gpu_idx_);
  std::cout << "Lazy creation of engine for GPU " << local_gpu_idx_
            << std::endl;

  // Initialize task rings
  send_unified_task_ring_ =
      uccl::create_ring(sizeof(UnifiedTask*), kTaskRingSize);
  recv_unified_task_ring_ =
      uccl::create_ring(sizeof(UnifiedTask*), kTaskRingSize);

  ipc_completion_ring_ =
      uccl::create_ring(sizeof(PendingIpcBatch*), kTaskRingSize);

  send_proxy_thread_ = std::thread(&Endpoint::send_proxy_thread_func, this);
  recv_proxy_thread_ = std::thread(&Endpoint::recv_proxy_thread_func, this);
  ipc_completion_thread_ =
      std::thread(&Endpoint::ipc_completion_thread_func, this);

  // Initialize UDS socket for local connections
  init_uds_socket();
}

bool Endpoint::connect(std::string ip_addr, int remote_gpu_idx, int remote_port,
                       uint64_t& conn_id) {
  std::cout << "Attempting to connect to " << ip_addr << ":" << remote_gpu_idx
            << " via port " << remote_port << std::endl;
  // Create a new connection ID
  conn_id = next_conn_id_.fetch_add(1);

  assert(local_gpu_idx_ != INVALID_GPU);

  std::future<ConnID> uccl_conn_id_future = std::async(
      std::launch::async, [this, remote_gpu_idx, &ip_addr, remote_port]() {
        return uccl_connect(ep_, remote_gpu_idx, ip_addr, remote_port);
      });

  // Check for Python signals (eg, ctrl+c) while waiting for connection
  while (uccl_conn_id_future.wait_for(std::chrono::seconds(0)) !=
         std::future_status::ready) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  ConnID uccl_conn_id = uccl_conn_id_future.get();

  // Store the connection ID.
  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_id_to_conn_[conn_id] =
        new Conn{conn_id, uccl_conn_id, ip_addr, remote_gpu_idx};
  }
  return true;
}

std::vector<uint8_t> Endpoint::get_metadata() {
  std::string ip_str = uccl::get_oob_ip();
  uint16_t port = get_p2p_listen_port(ep_);

  bool is_ipv6 = ip_str.find(':') != std::string::npos;
  size_t ip_len = is_ipv6 ? 16 : 4;

  // Additional 2 bytes for port and 4 bytes for local_gpu_idx_
  size_t total_len = ip_len + 2 + sizeof(int);
  std::vector<uint8_t> metadata(total_len);

  // Copy IP
  if (is_ipv6) {
    struct in6_addr ip6_bin;
    if (inet_pton(AF_INET6, ip_str.c_str(), &ip6_bin) != 1)
      throw std::runtime_error("Invalid IPv6 address: " + ip_str);
    std::memcpy(metadata.data(), &ip6_bin, 16);
  } else {
    struct in_addr ip4_bin;
    if (inet_pton(AF_INET, ip_str.c_str(), &ip4_bin) != 1)
      throw std::runtime_error("Invalid IPv4 address: " + ip_str);
    std::memcpy(metadata.data(), &ip4_bin, 4);
  }

  // Copy port in network byte order
  uint16_t net_port = htons(port);
  std::memcpy(metadata.data() + ip_len, &net_port, 2);

  // Copy local_gpu_idx_ in host byte order
  std::memcpy(metadata.data() + ip_len + 2, &local_gpu_idx_, sizeof(int));

  return metadata;
}

std::vector<uint8_t> Endpoint::get_unified_metadata() {
  int idx = 0;
  std::string ip_str = uccl::get_oob_ip();
  uint16_t port = get_p2p_listen_port(ep_);

  bool is_ipv6 = ip_str.find(':') != std::string::npos;
  size_t ip_len = is_ipv6 ? 16 : 4;

  // Additional 2 bytes for port and 4 bytes for local_gpu_idx_
  size_t total_len = ip_len + 2 + sizeof(int);
  std::vector<uint8_t> metadata(total_len);

  // Copy IP
  if (is_ipv6) {
    struct in6_addr ip6_bin;
    if (inet_pton(AF_INET6, ip_str.c_str(), &ip6_bin) != 1)
      throw std::runtime_error("Invalid IPv6 address: " + ip_str);
    std::memcpy(metadata.data(), &ip6_bin, 16);
  } else {
    struct in_addr ip4_bin;
    if (inet_pton(AF_INET, ip_str.c_str(), &ip4_bin) != 1)
      throw std::runtime_error("Invalid IPv4 address: " + ip_str);
    std::memcpy(metadata.data(), &ip4_bin, 4);
  }

  // Copy port in network byte order
  uint16_t net_port = htons(port);
  std::memcpy(metadata.data() + ip_len, &net_port, 2);

  // Copy local_gpu_idx_ in host byte order
  std::memcpy(metadata.data() + ip_len + 2, &idx, sizeof(int));

  return metadata;
}
std::tuple<std::string, uint16_t, int> Endpoint::parse_metadata(
    std::vector<uint8_t> const& metadata) {
  if (metadata.size() == 10) {
    // IPv4: 4 bytes IP, 2 bytes port, 4 bytes GPU idx
    char ip_str[INET_ADDRSTRLEN];
    if (inet_ntop(AF_INET, metadata.data(), ip_str, sizeof(ip_str)) ==
        nullptr) {
      throw std::runtime_error("Failed to parse IPv4 address from metadata");
    }

    uint16_t net_port;
    std::memcpy(&net_port, metadata.data() + 4, 2);
    uint16_t port = ntohs(net_port);

    int gpu_idx;
    std::memcpy(&gpu_idx, metadata.data() + 6, 4);

    return std::make_tuple(std::string(ip_str), port, gpu_idx);
  } else if (metadata.size() == 22) {
    // IPv6: 16 bytes IP, 2 bytes port, 4 bytes GPU idx
    char ip_str[INET6_ADDRSTRLEN];
    if (inet_ntop(AF_INET6, metadata.data(), ip_str, sizeof(ip_str)) ==
        nullptr) {
      throw std::runtime_error("Failed to parse IPv6 address from metadata");
    }

    uint16_t net_port;
    std::memcpy(&net_port, metadata.data() + 16, 2);
    uint16_t port = ntohs(net_port);

    int gpu_idx;
    std::memcpy(&gpu_idx, metadata.data() + 18, 4);

    return std::make_tuple(std::string(ip_str), port, gpu_idx);
  } else {
    throw std::runtime_error("Unexpected metadata length: " +
                             std::to_string(metadata.size()));
  }
}

bool Endpoint::accept(std::string& ip_addr, int& remote_gpu_idx,
                      uint64_t& conn_id) {
  std::cout << "Waiting to accept incoming connection..." << std::endl;

  // For demo purposes, simulate accepted connection
  conn_id = next_conn_id_.fetch_add(1);

  // Wait until engine is intialized to get the correct local_gpu_idx_
  while (!engine_initialized_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::future<ConnID> uccl_conn_id_future =
      std::async(std::launch::async, [this, &ip_addr, &remote_gpu_idx]() {
        return uccl_accept(ep_, ip_addr, &remote_gpu_idx);
      });

  // Check for Python signals (eg, ctrl+c) while waiting for connection
  while (uccl_conn_id_future.wait_for(std::chrono::seconds(0)) !=
         std::future_status::ready) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  ConnID uccl_conn_id = uccl_conn_id_future.get();

  // Store the connection ID.
  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    conn_id_to_conn_[conn_id] =
        new Conn{conn_id, uccl_conn_id, ip_addr, remote_gpu_idx};
  }

  return true;
}

bool Endpoint::reg(void const* data, size_t size, uint64_t& mr_id) {
  mr_id = next_mr_id_.fetch_add(1);

  if (!engine_initialized_) {
    int idx = uccl::get_dev_idx((void*)data);
    if (idx != -1) {
      // Pointer is on device idx
      local_gpu_idx_ = idx;
    } else {
      // Host memory/unknown memory type - fallback to dev 0
      local_gpu_idx_ = 0;
    }
    initialize_engine();
    engine_initialized_ = true;
  }

  P2PMhandle* mhandle = new P2PMhandle();
  if (!uccl_regmr(ep_, const_cast<void*>(data), size, mhandle)) {
    return false;
  }
  {
    std::unique_lock<std::shared_mutex> lock(mr_mu_);
    mr_id_to_mr_[mr_id] = new MR{mr_id, mhandle};
  }

  return true;
}

bool Endpoint::regv(std::vector<void const*> const& data_v,
                    std::vector<size_t> const& size_v,
                    std::vector<uint64_t>& mr_id_v) {
  if (data_v.size() != size_v.size())
    throw std::invalid_argument(
        "[Endpoint::regv] data_v/size_v length mismatch");

  size_t const n = data_v.size();
  mr_id_v.resize(n);

  {
    std::unique_lock<std::shared_mutex> lock(mr_mu_);
    mr_id_to_mr_.reserve(mr_id_to_mr_.size() + n);
  }

  for (size_t i = 0; i < n; ++i) {
    uint64_t id = next_mr_id_.fetch_add(1);
    P2PMhandle* mhandle = new P2PMhandle();

    if (!uccl_regmr(ep_, const_cast<void*>(data_v[i]), size_v[i], mhandle)) {
      std::cerr << "[Endpoint::regv] registration failed at i=" << i << '\n';
      return false;
    }

    {
      std::unique_lock<std::shared_mutex> lock(mr_mu_);
      mr_id_to_mr_[id] = new MR{id, mhandle};
    }
    mr_id_v[i] = id;
  }
  return true;
}

bool Endpoint::dereg(uint64_t mr_id) {
  {
    std::unique_lock<std::shared_mutex> lock(mr_mu_);
    auto it = mr_id_to_mr_.find(mr_id);
    if (it == mr_id_to_mr_.end()) {
      std::cerr << "[dereg] Error: Invalid mr_id " << mr_id << std::endl;
      return false;
    }
    auto mr = it->second;
    uccl_deregmr(ep_, mr->mhandle_);
    delete mr;
    mr_id_to_mr_.erase(mr_id);
  }
  return true;
}

bool Endpoint::send(uint64_t conn_id, uint64_t mr_id, void const* data,
                    size_t size) {
  DCHECK(size <= 0xffffffff) << "size must be less than 4GB";

  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[send] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  P2PMhandle* mhandle = get_mhandle(mr_id);
  if (unlikely(mhandle == nullptr)) {
    std::cerr << "[send] Error: Invalid mr_id " << mr_id << std::endl;
    return false;
  }

  ucclRequest ureq;

  auto* cur_data = const_cast<void*>(data);
  auto to_send = size;
  while (uccl_send_async(ep_, conn, mhandle, cur_data, to_send, &ureq) == -1)
    ;
  while (!uccl_poll_ureq_once(ep_, &ureq)) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
  }

  return true;
}

bool Endpoint::recv(uint64_t conn_id, uint64_t mr_id, void* data, size_t size) {
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[recv] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  P2PMhandle* mhandle = get_mhandle(mr_id);
  if (unlikely(mhandle == nullptr)) {
    std::cerr << "[recv] Error: Invalid mr_id " << mr_id << std::endl;
    return false;
  }
  int size_int = static_cast<int>(size);

  ucclRequest ureq;

  void* cur_data = data;
  while (uccl_recv_async(ep_, conn, mhandle, &cur_data, &size_int, 1, &ureq) ==
         -1)
    ;
  while (!uccl_poll_ureq_once(ep_, &ureq)) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
  }

  return true;
}

bool Endpoint::send_async(uint64_t conn_id, uint64_t mr_id, void const* data,
                          size_t size, uint64_t* transfer_id) {
  auto task_ptr = create_task(conn_id, mr_id, TaskType::SEND_NET,
                              const_cast<void*>(data), size);
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();
  while (jring_mp_enqueue_bulk(send_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::recv_async(uint64_t conn_id, uint64_t mr_id, void* data,
                          size_t size, uint64_t* transfer_id) {
  auto task_ptr = create_task(conn_id, mr_id, TaskType::RECV_NET, data, size);
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();
  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::sendv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                     std::vector<void const*> data_v,
                     std::vector<size_t> size_v, size_t num_iovs) {
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[sendv] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  if (num_iovs > kMaxVector) {
    std::cerr << "[sendv] Error: num_iovs > kMaxVector (" << kMaxVector << ")"
              << std::endl;
    return false;
  }

  ucclRequest ureq[kMaxVector] = {};
  bool sent[kMaxVector] = {false};
  bool done[kMaxVector] = {false};
  P2PMhandle* mhandles[kMaxVector] = {};

  // Check if mhandles are all valid
  for (int i = 0; i < num_iovs; i++) {
    mhandles[i] = get_mhandle(mr_id_v[i]);
    if (unlikely(mhandles[i] == nullptr)) {
      std::cerr << "[sendv] Error: Invalid mr_id " << mr_id_v[i] << std::endl;
      return false;
    }
  }

  while (1) {
    for (int i = 0; i < num_iovs; i++) {
      if (done[i]) continue;
      if (!sent[i]) {
        void* cur_data = (void*)data_v[i];
        size_t cur_size = size_v[i];

        auto mhandle = mhandles[i];

        auto rc =
            uccl_send_async(ep_, conn, mhandle, cur_data, cur_size, &ureq[i]);
        if (rc != -1) {
          sent[i] = true;
        }
      }

      if (sent[i] && !done[i]) {
        if (uccl_poll_ureq_once(ep_, &ureq[i])) {
          done[i] = true;
        }
      }
    }

    if (std::all_of(done, done + num_iovs, [](bool b) { return b; })) {
      break;
    }

    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
  }

  return true;
}

bool Endpoint::recvv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                     std::vector<void*> data_v, std::vector<size_t> size_v,
                     size_t num_iovs) {
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[recvv] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  if (num_iovs > kMaxVector) {
    std::cerr << "[recvv] Error: num_iovs > kMaxVector (" << kMaxVector << ")"
              << std::endl;
    return false;
  }

  ucclRequest ureq[kMaxVector] = {};
  bool done[kMaxVector] = {false};
  bool received[kMaxVector] = {false};
  P2PMhandle* mhandles[kMaxVector] = {};

  // Check if mhandles are all valid
  for (int i = 0; i < num_iovs; i++) {
    mhandles[i] = get_mhandle(mr_id_v[i]);
    if (unlikely(mhandles[i] == nullptr)) {
      std::cerr << "[recvv] Error: Invalid mr_id " << mr_id_v[i] << std::endl;
      return false;
    }
  }
  while (1) {
    for (int i = 0; i < num_iovs; i++) {
      if (done[i]) continue;

      if (!received[i]) {
        void* cur_data = data_v[i];
        size_t cur_size = size_v[i];

        auto mhandle = mhandles[i];

        int size_int = static_cast<int>(cur_size);

        auto rc = uccl_recv_async(ep_, conn, mhandle, &cur_data, &size_int, 1,
                                  &ureq[i]);
        if (rc != -1) {
          received[i] = true;
        }
      }

      if (received[i] && !done[i]) {
        if (uccl_poll_ureq_once(ep_, &ureq[i])) {
          done[i] = true;
        }
      }
    }

    if (std::all_of(done, done + num_iovs, [](bool b) { return b; })) {
      break;
    }

    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
  }

  return true;
}

bool Endpoint::read(uint64_t conn_id, uint64_t mr_id, void* dst, size_t size,
                    FifoItem const& slot_item) {
  DCHECK(size <= 0xffffffff) << "size must be < 4 GB";
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[read] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  P2PMhandle* mhandle = get_mhandle(mr_id);
  if (unlikely(mhandle == nullptr)) {
    std::cerr << "[read] Error: Invalid mr_id " << mr_id << std::endl;
    return false;
  }

  ucclRequest ureq = {};
  FifoItem curr_slot_item = slot_item;
  curr_slot_item.size = size;

  bool done = false;

  while (uccl_read_async(ep_, conn, mhandle, dst, size, curr_slot_item,
                         &ureq) == -1)
    ;

  while (!done) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
    if (uccl_poll_ureq_once(ep_, &ureq)) {
      done = true;
    }
  }

  return true;
}

bool Endpoint::read_async(uint64_t conn_id, uint64_t mr_id, void* dst,
                          size_t size, FifoItem const& slot_item,
                          uint64_t* transfer_id) {
  auto task_ptr =
      create_net_task(conn_id, mr_id, TaskType::READ_NET, dst, size, slot_item);
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::sendv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                           std::vector<void const*> data_v,
                           std::vector<size_t> size_v, size_t num_iovs,
                           uint64_t* transfer_id) {
  auto const_data_ptr =
      std::make_shared<std::vector<void const*>>(std::move(data_v));
  auto size_ptr = std::make_shared<std::vector<size_t>>(std::move(size_v));
  auto mr_id_ptr = std::make_shared<std::vector<uint64_t>>(std::move(mr_id_v));

  auto task_ptr = create_sendv_task(conn_id, std::move(const_data_ptr),
                                    std::move(size_ptr), std::move(mr_id_ptr));
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  while (jring_mp_enqueue_bulk(send_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::recvv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                           std::vector<void*> data_v,
                           std::vector<size_t> size_v, size_t num_iovs,
                           uint64_t* transfer_id) {
  // Use move semantics to reduce memory copies
  auto task_ptr = create_recvv_task(conn_id, std::move(data_v),
                                    std::move(size_v), std::move(mr_id_v));
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::readv(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                     std::vector<void*> dst_v, std::vector<size_t> size_v,
                     std::vector<FifoItem> slot_item_v, size_t num_iovs) {
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[readv] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  if (num_iovs > kMaxVector) {
    std::cerr << "[readv] Error: num_iovs > kMaxVector (" << kMaxVector << ")"
              << std::endl;
    return false;
  }

  ucclRequest ureq[kMaxVector] = {};
  FifoItem curr_slot_item[kMaxVector] = {};
  bool done[kMaxVector] = {false};
  bool read[kMaxVector] = {false};
  P2PMhandle* mhandles[kMaxVector] = {};

  // Check if mhandles are all valid
  for (int i = 0; i < num_iovs; i++) {
    mhandles[i] = get_mhandle(mr_id_v[i]);
    if (unlikely(mhandles[i] == nullptr)) {
      std::cerr << "[readv] Error: Invalid mr_id " << mr_id_v[i] << std::endl;
      return false;
    }
  }

  while (1) {
    for (int i = 0; i < num_iovs; i++) {
      if (done[i]) continue;

      if (!read[i]) {
        curr_slot_item[i] = slot_item_v[i];
        auto mhandle = mhandles[i];
        auto rc = uccl_read_async(ep_, conn, mhandle, dst_v[i], size_v[i],
                                  curr_slot_item[i], &ureq[i]);
        if (rc != -1) {
          read[i] = true;
        }
      }

      if (read[i] && !done[i]) {
        if (uccl_poll_ureq_once(ep_, &ureq[i])) {
          done[i] = true;
        }
      }
    }
    if (std::all_of(done, done + num_iovs, [](bool b) { return b; })) {
      break;
    }
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
  }

  return true;
}

bool Endpoint::readv_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                           std::vector<void*> dst_v, std::vector<size_t> size_v,
                           std::vector<FifoItem> slot_item_v, size_t num_iovs,
                           uint64_t* transfer_id) {
  // Use move semantics to reduce memory copies
  auto task_ptr =
      create_readv_task(conn_id, std::move(dst_v), std::move(size_v),
                        std::move(mr_id_v), std::move(slot_item_v));
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::write_async(uint64_t conn_id, uint64_t mr_id, void* src,
                           size_t size, FifoItem const& slot_item,
                           uint64_t* transfer_id) {
  auto task_ptr = create_net_task(conn_id, mr_id, TaskType::WRITE_NET, src,
                                  size, slot_item);
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  while (jring_mp_enqueue_bulk(send_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::writev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                      std::vector<void*> src_v, std::vector<size_t> size_v,
                      std::vector<FifoItem> slot_item_v, size_t num_iovs) {
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[writev] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  if (num_iovs > kMaxVector) {
    std::cerr << "[writev] Error: num_iovs > kMaxVector (" << kMaxVector << ")"
              << std::endl;
    return false;
  }

  ucclRequest ureq[kMaxVector] = {};
  FifoItem curr_slot_item[kMaxVector] = {};
  bool done[kMaxVector] = {false};
  bool written[kMaxVector] = {false};
  P2PMhandle* mhandles[kMaxVector] = {};

  // Check if mhandles are all valid
  for (int i = 0; i < num_iovs; i++) {
    mhandles[i] = get_mhandle(mr_id_v[i]);
    if (unlikely(mhandles[i] == nullptr)) {
      std::cerr << "[writev] Error: Invalid mr_id " << mr_id_v[i] << std::endl;
      return false;
    }
  }

  while (1) {
    for (int i = 0; i < num_iovs; i++) {
      if (done[i]) continue;

      if (!written[i]) {
        curr_slot_item[i] = slot_item_v[i];

        auto mhandle = mhandles[i];
        auto rc = uccl_write_async(ep_, conn, mhandle, src_v[i], size_v[i],
                                   curr_slot_item[i], &ureq[i]);
        if (rc != -1) {
          written[i] = true;
        }
      }

      if (written[i] && !done[i]) {
        if (uccl_poll_ureq_once(ep_, &ureq[i])) {
          done[i] = true;
        }
      }
    }

    if (std::all_of(done, done + num_iovs, [](bool b) { return b; })) {
      break;
    }

    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
  }

  return true;
}

bool Endpoint::writev_async(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                            std::vector<void*> src_v,
                            std::vector<size_t> size_v,
                            std::vector<FifoItem> slot_item_v, size_t num_iovs,
                            uint64_t* transfer_id) {
  // Use move semantics to reduce memory copies
  auto task_ptr =
      create_writev_task(conn_id, std::move(src_v), std::move(size_v),
                         std::move(mr_id_v), std::move(slot_item_v));
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  while (jring_mp_enqueue_bulk(send_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::write(uint64_t conn_id, uint64_t mr_id, void* src, size_t size,
                     FifoItem const& slot_item) {
  DCHECK(size <= 0xffffffff) << "size must be < 4 GB";
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[write] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }
  P2PMhandle* mhandle = get_mhandle(mr_id);
  if (unlikely(mhandle == nullptr)) {
    std::cerr << "[write] Error: Invalid mr_id " << mr_id << std::endl;
    return false;
  }

  ucclRequest ureq = {};
  FifoItem cur_slot_item = slot_item;
  cur_slot_item.size = size;
  while (uccl_write_async(ep_, conn, mhandle, src, size, cur_slot_item,
                          &ureq) == -1)
    ;

  while (!uccl_poll_ureq_once(ep_, &ureq)) {
    auto _ = inside_python ? (check_python_signals(), nullptr) : nullptr;
  }
  return true;
}

bool Endpoint::advertise(uint64_t conn_id, uint64_t mr_id, void* addr,
                         size_t len, char* out_buf) {
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[advertise] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }
  auto mhandle = get_mhandle(mr_id);
  if (unlikely(mhandle == nullptr)) {
    std::cerr << "[advertise] Error: Invalid mr_id " << mr_id << std::endl;
    return false;
  }
  if (prepare_fifo_metadata(ep_, conn, mhandle, addr, len, out_buf) == -1)
    return false;
  return true;
}

bool Endpoint::advertisev(uint64_t conn_id, std::vector<uint64_t> mr_id_v,
                          std::vector<void*> addr_v, std::vector<size_t> len_v,
                          std::vector<char*> out_buf_v, size_t num_iovs) {
  auto* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[advertisev] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }
  if (num_iovs > kMaxVector) {
    std::cerr << "[advertisev] Error: num_iovs > kMaxVector (" << kMaxVector
              << ")" << std::endl;
    return false;
  }

  P2PMhandle* mhandles[kMaxVector] = {};
  // Check if mhandles are all valid
  for (int i = 0; i < num_iovs; i++) {
    mhandles[i] = get_mhandle(mr_id_v[i]);
    if (unlikely(mhandles[i] == nullptr)) {
      std::cerr << "[advertisev] Error: Invalid mr_id " << mr_id_v[i]
                << std::endl;
      return false;
    }
  }

  for (size_t i = 0; i < num_iovs; ++i) {
    auto mhandle = mhandles[i];
    if (prepare_fifo_metadata(ep_, conn, mhandle, addr_v[i], len_v[i],
                              out_buf_v[i]) == -1) {
      return false;
    }
  }
  return true;
}

bool Endpoint::connect_local(int remote_gpu_idx, uint64_t& conn_id) {
  int retries = 5;
  int ret = -1;
  std::cout << "Connecting to remote GPU " << remote_gpu_idx << std::endl;

  std::string remote_socket_path = get_uds_socket_path(remote_gpu_idx);

  // Create socket for connection
  int sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
  CHECK_GE(sockfd, 0) << "Failed to create UDS socket for connection: "
                      << strerror(errno);
  fcntl(sockfd, F_SETFL, fcntl(sockfd, F_GETFL) | O_NONBLOCK);

  // Set up socket address
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, remote_socket_path.c_str(), sizeof(addr.sun_path) - 1);

  // Connect to remote socket
  for (int i = 0; i < retries; ++i) {
    ret = ::connect(sockfd, (struct sockaddr*)&addr, sizeof(addr));
    if (ret == 0) break;

    if (errno == ECONNREFUSED || errno == EAGAIN) {
      std::cerr << "Connect failed: " << strerror(errno) << ", retry "
                << (i + 1) << "/" << retries << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(200 * (i + 1)));
      continue;
    }
    break;
  }

  // Send our GPU index to the remote endpoint
  ret = uccl::send_message_nonblock(sockfd,
                                    static_cast<void const*>(&local_gpu_idx_),
                                    sizeof(local_gpu_idx_));
  CHECK_EQ(ret, sizeof(local_gpu_idx_)) << "Failed to send local GPU index";

  // Create a new connection ID for this local connection
  conn_id = next_conn_id_.fetch_add(1);

  // Create a special connection entry for local UDS connection with persistent
  // socket
  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    ConnID dummy_conn_id{nullptr, 0, 0, 0};
    conn_id_to_conn_[conn_id] =
        new Conn{conn_id, dummy_conn_id, "localhost", remote_gpu_idx, sockfd};
  }

  return true;
}

bool Endpoint::accept_local(int& remote_gpu_idx, uint64_t& conn_id) {
  std::cout << "Waiting to accept UDS connection" << std::endl;

  CHECK(uds_listen_fd_ >= 0) << "UDS socket not initialized";

  // Accept incoming connection
  struct sockaddr_un client_addr;
  socklen_t client_len = sizeof(client_addr);
  int client_fd =
      ::accept(uds_listen_fd_, (struct sockaddr*)&client_addr, &client_len);
  CHECK_GE(client_fd, 0) << "Failed to accept UDS connection: "
                         << strerror(errno);

  fcntl(client_fd, F_SETFL, fcntl(client_fd, F_GETFL) | O_NONBLOCK);

  // Receive remote GPU index
  auto ret = uccl::receive_message_nonblock(
      client_fd, static_cast<void*>(&remote_gpu_idx), sizeof(remote_gpu_idx));
  CHECK_EQ(ret, sizeof(remote_gpu_idx)) << "Failed to receive remote GPU index";

  // Create connection ID
  conn_id = next_conn_id_.fetch_add(1);

  // Store the connection with persistent socket
  {
    std::unique_lock<std::shared_mutex> lock(conn_mu_);
    ConnID dummy_conn_id{nullptr, 0, 0, 0};
    conn_id_to_conn_[conn_id] = new Conn{conn_id, dummy_conn_id, "localhost",
                                         remote_gpu_idx, client_fd};
  }

  return true;
}

bool Endpoint::send_ipc(uint64_t conn_id, void* data, size_t size) {
  CHECK(data != nullptr) << "send_ipc: data pointer is null!";

  // Get connection info
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[send_ipc] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  // Check if we have a valid persistent UDS socket (faster than string
  // comparison)
  CHECK_GE(conn->uds_sockfd_, 0)
      << "send_ipc only supports local connections with valid UDS socket";

  // Use the persistent UDS connection
  int sockfd = conn->uds_sockfd_;

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  // Detect if sender's data is CPU or GPU memory
  int send_dev_idx = uccl::get_dev_idx(data);
  bool send_is_gpu = (send_dev_idx >= 0);

  // Exchange memory type information
  // Receive receiver's memory type first
  uint32_t recv_mem_type = 0;
  auto ret = uccl::receive_message_nonblock(
      sockfd, static_cast<void*>(&recv_mem_type), sizeof(recv_mem_type));
  CHECK_EQ(ret, sizeof(recv_mem_type)) << "Failed to receive receiver memory type";

  // Send our memory type: 0=CPU, 1=GPU
  uint32_t send_mem_type = send_is_gpu ? 1 : 0;
  ret = uccl::send_message_nonblock(
      sockfd, static_cast<void const*>(&send_mem_type), sizeof(send_mem_type));
  CHECK_EQ(ret, sizeof(send_mem_type)) << "Failed to send memory type";

  bool recv_is_gpu = (recv_mem_type == 1);

  // Decide protocol based on memory types
  if (recv_is_gpu) {
    // Receiver has GPU: Use push model (sender writes to receiver's GPU)
    IpcTransferInfo transfer_info = {};
    ret = uccl::receive_message_nonblock(
        sockfd, static_cast<void*>(&transfer_info), sizeof(transfer_info));
    CHECK_EQ(ret, sizeof(transfer_info))
        << "Failed to receive IPC handle from receiver";
    CHECK_EQ(transfer_info.operation, 1) << "Invalid response from receiver";
    CHECK_EQ(transfer_info.size, size)
        << "Size mismatch: expected " << size << ", got " << transfer_info.size;

    // Open receiver's GPU memory
    void* base = nullptr;
    GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
    GPU_RT_CHECK(gpuIpcOpenMemHandle(&base, transfer_info.handle,
                                     gpuIpcMemLazyEnablePeerAccess));
    void* dst_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(base) + transfer_info.offset);

    std::vector<gpuStream_t>& dst_streams = ipc_streams_[conn->remote_gpu_idx_];
    int num_streams =
        std::min(dst_streams.size(),
                 size < kIpcSizePerEngine ? 1 : (size_t)size / kIpcSizePerEngine);

    // For CPU->GPU, use single stream (CPU doesn't benefit from multiple streams)
    if (!send_is_gpu) {
      num_streams = 1;
    }

    size_t chunk_size = size / num_streams;

    for (int i = 0; i < num_streams; ++i) {
      void* chunk_data = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(data) + i * chunk_size);
      void* chunk_dst_ptr = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
      auto copy_size = i == num_streams - 1 ? size - i * chunk_size : chunk_size;

      // CPU side calls cudaMemcpy if sender is CPU
      if (send_is_gpu) {
        GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst_ptr, chunk_data, copy_size,
                                    gpuMemcpyDeviceToDevice, dst_streams[i]));
      } else {
        GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst_ptr, chunk_data, copy_size,
                                    gpuMemcpyHostToDevice, dst_streams[i]));
      }
    }

    for (int i = 0; i < num_streams; ++i) {
      GPU_RT_CHECK(gpuStreamSynchronize(dst_streams[i]));
    }

    // Close the IPC memory handle
    GPU_RT_CHECK(gpuIpcCloseMemHandle(base));

    // Notify receiver of completion
    uint32_t completion = 1;
    ret = uccl::send_message_nonblock(
        sockfd, static_cast<void const*>(&completion), sizeof(completion));
    CHECK_EQ(ret, sizeof(completion)) << "Failed to send completion ack";

  } else {
    // Receiver has CPU: Use pull model (sender provides IPC handle)

    CHECK(send_is_gpu) << "send_ipc: CPU-to-CPU transfer not supported via IPC";

    GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

    // Create IPC handle for sender's GPU buffer
    IpcTransferInfo transfer_info = {};
    transfer_info.size = size;
    transfer_info.operation = 1;
    GPU_RT_CHECK(
        gpuIpcGetMemHandle(&transfer_info.handle, reinterpret_cast<void*>(data)));

    // Getting the base address
    void* base = nullptr;
    size_t base_size;
    GPU_RT_CHECK(gpuMemGetAddressRange(&base, &base_size, data));
    auto data_offset =
        reinterpret_cast<uintptr_t>(data) - reinterpret_cast<uintptr_t>(base);
    transfer_info.offset = data_offset;

    // Send IPC handle to receiver
    ret = uccl::send_message_nonblock(
        sockfd, static_cast<void const*>(&transfer_info), sizeof(transfer_info));
    CHECK_EQ(ret, sizeof(transfer_info)) << "Failed to send IPC handle to receiver";

    // Wait for receiver to complete the copy
    uint32_t completion = 0;
    ret = uccl::receive_message_nonblock(
        sockfd, static_cast<void*>(&completion), sizeof(completion));
    CHECK_EQ(ret, sizeof(completion))
        << "Failed to receive completion notification";
    CHECK_EQ(completion, 1) << "Receiver reported failure";

  }

  // We close all IPC memory handles when releasing this endpoint.

  return true;
}

bool Endpoint::recv_ipc(uint64_t conn_id, void* data, size_t size) {
  CHECK(data != nullptr) << "recv_ipc: data pointer is null!";

  // Get connection info
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[recv_ipc] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  // Check if we have a valid persistent UDS socket (faster than string
  // comparison)
  CHECK_GE(conn->uds_sockfd_, 0)
      << "recv_ipc only supports local connections with valid UDS socket";

  // Use the persistent UDS connection
  int client_fd = conn->uds_sockfd_;

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  // Detect if receiver's data is CPU or GPU memory
  int recv_dev_idx = uccl::get_dev_idx(data);
  bool recv_is_gpu = (recv_dev_idx >= 0);

  // Exchange memory type information
  // Send: 0=CPU, 1=GPU
  uint32_t recv_mem_type = recv_is_gpu ? 1 : 0;
  auto ret = uccl::send_message_nonblock(
      client_fd, static_cast<void const*>(&recv_mem_type), sizeof(recv_mem_type));
  CHECK_EQ(ret, sizeof(recv_mem_type)) << "Failed to send memory type";

  // Receive sender's memory type
  uint32_t send_mem_type = 0;
  ret = uccl::receive_message_nonblock(
      client_fd, static_cast<void*>(&send_mem_type), sizeof(send_mem_type));
  CHECK_EQ(ret, sizeof(send_mem_type)) << "Failed to receive sender memory type";

  bool send_is_gpu = (send_mem_type == 1);

  // Decide protocol based on memory types
  if (recv_is_gpu) {
    // Receiver has GPU: Create IPC handle (push model)
    IpcTransferInfo transfer_info = {};
    transfer_info.size = size;
    transfer_info.operation = 1;  // response
    GPU_RT_CHECK(
        gpuIpcGetMemHandle(&transfer_info.handle, reinterpret_cast<void*>(data)));

    // Getting the base address
    void* base = nullptr;
    size_t base_size;
    GPU_RT_CHECK(gpuMemGetAddressRange(&base, &base_size, data));
    auto data_offset =
        reinterpret_cast<uintptr_t>(data) - reinterpret_cast<uintptr_t>(base);
    transfer_info.offset = data_offset;

    ret = uccl::send_message_nonblock(
        client_fd, static_cast<void const*>(&transfer_info),
        sizeof(transfer_info));
    CHECK_EQ(ret, sizeof(transfer_info)) << "Failed to send IPC handle to sender";

    // Wait for sender to complete
    uint32_t completion = 0;
    ret = uccl::receive_message_nonblock(
        client_fd, static_cast<void*>(&completion), sizeof(completion));
    CHECK_EQ(ret, sizeof(completion))
        << "Failed to receive completion notification";
    CHECK_EQ(completion, 1) << "Sender reported failure";

  } else {
    // Receiver has CPU: Request sender's IPC handle (pull model)

    CHECK(send_is_gpu) << "recv_ipc: CPU-to-CPU transfer not supported via IPC";

    // Receive sender's IPC handle
    IpcTransferInfo transfer_info = {};
    ret = uccl::receive_message_nonblock(
        client_fd, static_cast<void*>(&transfer_info), sizeof(transfer_info));
    CHECK_EQ(ret, sizeof(transfer_info))
        << "Failed to receive IPC handle from sender";
    CHECK_EQ(transfer_info.size, size)
        << "Size mismatch: expected " << size << ", got " << transfer_info.size;

    // Open sender's GPU memory
    void* base = nullptr;
    GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
    GPU_RT_CHECK(gpuIpcOpenMemHandle(&base, transfer_info.handle,
                                     gpuIpcMemLazyEnablePeerAccess));
    void* src_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(base) + transfer_info.offset);

    // Copy from sender's GPU to receiver's CPU
    // CPU side calls cudaMemcpy (best practice)
    GPU_RT_CHECK(gpuMemcpy(data, src_ptr, size, gpuMemcpyDeviceToHost));

    // Close the IPC memory handle
    GPU_RT_CHECK(gpuIpcCloseMemHandle(base));

    // Notify sender of completion
    uint32_t completion = 1;
    ret = uccl::send_message_nonblock(
        client_fd, static_cast<void const*>(&completion), sizeof(completion));
    CHECK_EQ(ret, sizeof(completion)) << "Failed to send completion ack";

  }

  return true;
}

bool Endpoint::send_ipc_async(uint64_t conn_id, void const* data, size_t size,
                              uint64_t* transfer_id) {
  // Create a task for IPC send operation
  auto task_ptr = create_task(conn_id, 0, TaskType::SEND_IPC,
                              const_cast<void*>(data), size);
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  // For now, we'll use the existing async infrastructure but call our IPC
  // function In a real implementation, you might want a separate IPC task ring
  while (jring_mp_enqueue_bulk(send_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::recv_ipc_async(uint64_t conn_id, void* data, size_t size,
                              uint64_t* transfer_id) {
  // Create a task for IPC receive operation
  auto task_ptr = create_task(conn_id, 0, TaskType::RECV_IPC, data, size);
  if (unlikely(task_ptr == nullptr)) {
    return false;
  }

  auto* status = new TransferStatus();
  status->task_ptr = task_ptr;
  task_ptr->status_ptr = status;
  *transfer_id = reinterpret_cast<uint64_t>(status);

  UnifiedTask* task_raw = task_ptr.get();

  // For now, we'll use the existing async infrastructure but call our IPC
  // function In a real implementation, you might want a separate IPC task ring
  while (jring_mp_enqueue_bulk(recv_unified_task_ring_, &task_raw, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::write_ipc(uint64_t conn_id, void const* data, size_t size,
                         IpcTransferInfo const& info) {
  CHECK(data != nullptr) << "write_ipc: data pointer is null!";

  // Get connection info
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[write_ipc] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  // Open the remote IPC memory handle
  void* raw_dst_ptr = nullptr;
  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
  GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_dst_ptr, info.handle,
                                   gpuIpcMemLazyEnablePeerAccess));

  // Calculate destination pointer with offset
  void* dst_ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(raw_dst_ptr) + info.offset);

  // Detect if source data is CPU or GPU memory
  bool src_is_gpu = (uccl::get_dev_idx(const_cast<void*>(data)) >= 0);

  // Perform the memory copy using multiple streams for better performance
  std::vector<gpuStream_t>& dst_streams = ipc_streams_[conn->remote_gpu_idx_];
  int num_streams =
      std::min(dst_streams.size(),
               size < kIpcSizePerEngine ? 1 : (size_t)size / kIpcSizePerEngine);

  // For CPU->GPU, use single stream (CPU doesn't benefit from multiple streams)
  if (!src_is_gpu) {
    num_streams = 1;
  }

  size_t chunk_size = size / num_streams;

  for (int i = 0; i < num_streams; ++i) {
    // Split data and dst_ptr into n_streams chunks
    void* chunk_data = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(data) + i * chunk_size);
    void* chunk_dst_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
    auto copy_size = i == num_streams - 1 ? size - i * chunk_size : chunk_size;

    if (src_is_gpu) {
      GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst_ptr, chunk_data, copy_size,
                                  gpuMemcpyDeviceToDevice, dst_streams[i]));
    } else {
      GPU_RT_CHECK(gpuMemcpyAsync(chunk_dst_ptr, chunk_data, copy_size,
                                  gpuMemcpyHostToDevice, dst_streams[i]));
    }
  }

  // Wait for all streams to complete
  for (auto& stream : dst_streams) {
    GPU_RT_CHECK(gpuStreamSynchronize(stream));
  }

  // Close the IPC memory handle
  GPU_RT_CHECK(gpuIpcCloseMemHandle(raw_dst_ptr));

  return true;
}

bool Endpoint::read_ipc(uint64_t conn_id, void* data, size_t size,
                        IpcTransferInfo const& info) {
  CHECK(data != nullptr) << "read_ipc: data pointer is null!";

  // Get connection info
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[read_ipc] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  // Open the remote IPC memory handle
  void* raw_src_ptr = nullptr;
  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));
  GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_src_ptr, info.handle,
                                   gpuIpcMemLazyEnablePeerAccess));

  // Calculate source pointer with offset
  void* src_ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(raw_src_ptr) + info.offset);

  // Detect if destination data is CPU or GPU memory
  bool dst_is_gpu = (uccl::get_dev_idx(data) >= 0);

  // Perform the memory copy using multiple streams for better performance
  std::vector<gpuStream_t>& src_streams = ipc_streams_[conn->remote_gpu_idx_];
  int num_streams =
      std::min(src_streams.size(),
               size < kIpcSizePerEngine ? 1 : (size_t)size / kIpcSizePerEngine);

  // For GPU->CPU, use single stream
  if (!dst_is_gpu) {
    num_streams = 1;
  }

  size_t chunk_size = size / num_streams;

  for (int i = 0; i < num_streams; ++i) {
    // Split src_ptr and data into n_streams chunks
    void* chunk_src_ptr = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(src_ptr) + i * chunk_size);
    void* chunk_data = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(data) + i * chunk_size);
    auto copy_size = i == num_streams - 1 ? size - i * chunk_size : chunk_size;

    if (dst_is_gpu) {
      GPU_RT_CHECK(gpuMemcpyAsync(chunk_data, chunk_src_ptr, copy_size,
                                  gpuMemcpyDeviceToDevice, src_streams[i]));
    } else {
      GPU_RT_CHECK(gpuMemcpyAsync(chunk_data, chunk_src_ptr, copy_size,
                                  gpuMemcpyDeviceToHost, src_streams[i]));
    }
  }

  // Wait for all streams to complete
  for (auto& stream : src_streams) {
    GPU_RT_CHECK(gpuStreamSynchronize(stream));
  }

  // Close the IPC memory handle
  GPU_RT_CHECK(gpuIpcCloseMemHandle(raw_src_ptr));

  return true;
}

bool Endpoint::write_ipc_async(uint64_t conn_id, void const* data, size_t size,
                               IpcTransferInfo const& info,
                               uint64_t* transfer_id) {
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    return false;
  }

  auto* pending = new PendingIpcBatch();
  pending->status = new TransferStatus();
  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  pending->remote_device = conn->remote_gpu_idx_;
  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));

  // Open IPC handle
  void* raw_dst_ptr = nullptr;
  GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_dst_ptr, info.handle,
                                   gpuIpcMemLazyEnablePeerAccess));
  pending->raw_ptrs.push_back(raw_dst_ptr);

  void* dst_ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(raw_dst_ptr) + info.offset);

  bool src_is_gpu = (uccl::get_dev_idx(const_cast<void*>(data)) >= 0);
  std::vector<gpuStream_t>& streams = ipc_streams_[conn->remote_gpu_idx_];
  int num_streams =
      std::min(streams.size(),
               size < kIpcSizePerEngine ? 1 : (size_t)size / kIpcSizePerEngine);
  if (!src_is_gpu) num_streams = 1;

  size_t chunk_size = size / num_streams;
  for (int i = 0; i < num_streams; ++i) {
    void* chunk_data = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(data) + i * chunk_size);
    void* chunk_dst = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(dst_ptr) + i * chunk_size);
    auto copy_size = i == num_streams - 1 ? size - i * chunk_size : chunk_size;
    auto kind = src_is_gpu ? gpuMemcpyDeviceToDevice : gpuMemcpyHostToDevice;
    GPU_RT_CHECK(
        gpuMemcpyAsync(chunk_dst, chunk_data, copy_size, kind, streams[i]));
  }

  // Record events on used streams
  pending->events.resize(num_streams);
  for (int i = 0; i < num_streams; ++i) {
    GPU_RT_CHECK(gpuEventCreateWithFlags(&pending->events[i],
                                         gpuEventDisableTiming));
    GPU_RT_CHECK(gpuEventRecord(pending->events[i], streams[i]));
  }

  GPU_RT_CHECK(gpuSetDevice(orig_device));

  *transfer_id = reinterpret_cast<uint64_t>(pending->status);
  while (jring_mp_enqueue_bulk(ipc_completion_ring_, &pending, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::read_ipc_async(uint64_t conn_id, void* data, size_t size,
                              IpcTransferInfo const& info,
                              uint64_t* transfer_id) {
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    return false;
  }

  auto* pending = new PendingIpcBatch();
  pending->status = new TransferStatus();
  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  pending->remote_device = conn->remote_gpu_idx_;
  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));

  // Open IPC handle
  void* raw_src_ptr = nullptr;
  GPU_RT_CHECK(gpuIpcOpenMemHandle(&raw_src_ptr, info.handle,
                                   gpuIpcMemLazyEnablePeerAccess));
  pending->raw_ptrs.push_back(raw_src_ptr);

  void* src_ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(raw_src_ptr) + info.offset);

  bool dst_is_gpu = (uccl::get_dev_idx(data) >= 0);
  std::vector<gpuStream_t>& streams = ipc_streams_[conn->remote_gpu_idx_];
  int num_streams =
      std::min(streams.size(),
               size < kIpcSizePerEngine ? 1 : (size_t)size / kIpcSizePerEngine);
  if (!dst_is_gpu) num_streams = 1;

  size_t chunk_size = size / num_streams;
  for (int i = 0; i < num_streams; ++i) {
    void* chunk_src = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(src_ptr) + i * chunk_size);
    void* chunk_data = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(data) + i * chunk_size);
    auto copy_size = i == num_streams - 1 ? size - i * chunk_size : chunk_size;
    auto kind = dst_is_gpu ? gpuMemcpyDeviceToDevice : gpuMemcpyDeviceToHost;
    GPU_RT_CHECK(
        gpuMemcpyAsync(chunk_data, chunk_src, copy_size, kind, streams[i]));
  }

  // Record events on used streams
  pending->events.resize(num_streams);
  for (int i = 0; i < num_streams; ++i) {
    GPU_RT_CHECK(gpuEventCreateWithFlags(&pending->events[i],
                                         gpuEventDisableTiming));
    GPU_RT_CHECK(gpuEventRecord(pending->events[i], streams[i]));
  }

  GPU_RT_CHECK(gpuSetDevice(orig_device));

  *transfer_id = reinterpret_cast<uint64_t>(pending->status);
  while (jring_mp_enqueue_bulk(ipc_completion_ring_, &pending, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::writev_ipc(uint64_t conn_id, std::vector<void const*> data_v,
                          std::vector<size_t> size_v,
                          std::vector<IpcTransferInfo> info_v,
                          size_t num_iovs) {
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[writev_ipc] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });
  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));

  std::vector<gpuStream_t>& streams = ipc_streams_[conn->remote_gpu_idx_];

  // Process in batches to stay within CUDA IPC handle limit
  for (size_t start = 0; start < num_iovs; start += kMaxOpenIpcHandles) {
    size_t end = std::min(start + kMaxOpenIpcHandles, num_iovs);
    size_t batch_size = end - start;

    std::vector<void*> batch_ptrs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      GPU_RT_CHECK(gpuIpcOpenMemHandle(&batch_ptrs[i],
                                       info_v[start + i].handle,
                                       gpuIpcMemLazyEnablePeerAccess));
    }

    for (size_t i = 0; i < batch_size; ++i) {
      void* dst_ptr = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(batch_ptrs[i]) +
          info_v[start + i].offset);
      bool src_is_gpu =
          (uccl::get_dev_idx(const_cast<void*>(data_v[start + i])) >= 0);
      auto kind = src_is_gpu ? gpuMemcpyDeviceToDevice : gpuMemcpyHostToDevice;
      gpuStream_t stream = streams[(start + i) % streams.size()];
      GPU_RT_CHECK(gpuMemcpyAsync(dst_ptr, data_v[start + i],
                                   size_v[start + i], kind, stream));
    }

    size_t num_used = std::min(batch_size, streams.size());
    for (size_t i = 0; i < num_used; ++i) {
      GPU_RT_CHECK(gpuStreamSynchronize(streams[i]));
    }
    for (auto& ptr : batch_ptrs) {
      GPU_RT_CHECK(gpuIpcCloseMemHandle(ptr));
    }
  }

  return true;
}

bool Endpoint::readv_ipc(uint64_t conn_id, std::vector<void*> data_v,
                         std::vector<size_t> size_v,
                         std::vector<IpcTransferInfo> info_v,
                         size_t num_iovs) {
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    std::cerr << "[readv_ipc] Error: Invalid conn_id " << conn_id << std::endl;
    return false;
  }

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });
  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));

  std::vector<gpuStream_t>& streams = ipc_streams_[conn->remote_gpu_idx_];

  // Process in batches to stay within CUDA IPC handle limit
  for (size_t start = 0; start < num_iovs; start += kMaxOpenIpcHandles) {
    size_t end = std::min(start + kMaxOpenIpcHandles, num_iovs);
    size_t batch_size = end - start;

    std::vector<void*> batch_ptrs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      GPU_RT_CHECK(gpuIpcOpenMemHandle(&batch_ptrs[i],
                                       info_v[start + i].handle,
                                       gpuIpcMemLazyEnablePeerAccess));
    }

    for (size_t i = 0; i < batch_size; ++i) {
      void* src_ptr = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(batch_ptrs[i]) +
          info_v[start + i].offset);
      bool dst_is_gpu = (uccl::get_dev_idx(data_v[start + i]) >= 0);
      auto kind = dst_is_gpu ? gpuMemcpyDeviceToDevice : gpuMemcpyDeviceToHost;
      gpuStream_t stream = streams[(start + i) % streams.size()];
      GPU_RT_CHECK(gpuMemcpyAsync(data_v[start + i], src_ptr,
                                   size_v[start + i], kind, stream));
    }

    size_t num_used = std::min(batch_size, streams.size());
    for (size_t i = 0; i < num_used; ++i) {
      GPU_RT_CHECK(gpuStreamSynchronize(streams[i]));
    }
    for (auto& ptr : batch_ptrs) {
      GPU_RT_CHECK(gpuIpcCloseMemHandle(ptr));
    }
  }

  return true;
}

bool Endpoint::writev_ipc_async(uint64_t conn_id,
                                std::vector<void const*> data_v,
                                std::vector<size_t> size_v,
                                std::vector<IpcTransferInfo> info_v,
                                size_t num_iovs, uint64_t* transfer_id) {
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    return false;
  }

  auto* pending = new PendingIpcBatch();
  pending->status = new TransferStatus();
  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  pending->remote_device = conn->remote_gpu_idx_;
  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));

  std::vector<gpuStream_t>& streams = ipc_streams_[conn->remote_gpu_idx_];

  // Process in batches to stay within CUDA IPC handle limit.
  // Intermediate batches are sync'd and closed; the last batch stays open
  // for async completion via the completion thread.
  for (size_t start = 0; start < num_iovs; start += kMaxOpenIpcHandles) {
    size_t end = std::min(start + kMaxOpenIpcHandles, num_iovs);
    size_t batch_size = end - start;
    bool is_last_batch = (end == num_iovs);

    // Open handles for this batch
    std::vector<void*> batch_ptrs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      GPU_RT_CHECK(gpuIpcOpenMemHandle(&batch_ptrs[i],
                                       info_v[start + i].handle,
                                       gpuIpcMemLazyEnablePeerAccess));
    }

    // Issue async copies round-robin across streams
    for (size_t i = 0; i < batch_size; ++i) {
      void* dst_ptr = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(batch_ptrs[i]) +
          info_v[start + i].offset);
      bool src_is_gpu =
          (uccl::get_dev_idx(const_cast<void*>(data_v[start + i])) >= 0);
      auto kind = src_is_gpu ? gpuMemcpyDeviceToDevice : gpuMemcpyHostToDevice;
      gpuStream_t stream = streams[(start + i) % streams.size()];
      GPU_RT_CHECK(gpuMemcpyAsync(dst_ptr, data_v[start + i],
                                   size_v[start + i], kind, stream));
    }

    if (is_last_batch) {
      // Keep handles open — completion thread will close them
      pending->raw_ptrs = std::move(batch_ptrs);

      // Record events on used streams
      size_t num_used = std::min(num_iovs, streams.size());
      pending->events.resize(num_used);
      for (size_t i = 0; i < num_used; ++i) {
        GPU_RT_CHECK(gpuEventCreateWithFlags(&pending->events[i],
                                             gpuEventDisableTiming));
        GPU_RT_CHECK(gpuEventRecord(pending->events[i], streams[i]));
      }
    } else {
      // Sync and close intermediate batch
      size_t num_used = std::min(batch_size, streams.size());
      for (size_t i = 0; i < num_used; ++i) {
        GPU_RT_CHECK(gpuStreamSynchronize(streams[i]));
      }
      for (auto& ptr : batch_ptrs) {
        GPU_RT_CHECK(gpuIpcCloseMemHandle(ptr));
      }
    }
  }

  GPU_RT_CHECK(gpuSetDevice(orig_device));

  *transfer_id = reinterpret_cast<uint64_t>(pending->status);
  while (jring_mp_enqueue_bulk(ipc_completion_ring_, &pending, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::readv_ipc_async(uint64_t conn_id, std::vector<void*> data_v,
                               std::vector<size_t> size_v,
                               std::vector<IpcTransferInfo> info_v,
                               size_t num_iovs, uint64_t* transfer_id) {
  Conn* conn = get_conn(conn_id);
  if (unlikely(conn == nullptr)) {
    return false;
  }

  auto* pending = new PendingIpcBatch();
  pending->status = new TransferStatus();
  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  pending->remote_device = conn->remote_gpu_idx_;
  GPU_RT_CHECK(gpuSetDevice(conn->remote_gpu_idx_));

  std::vector<gpuStream_t>& streams = ipc_streams_[conn->remote_gpu_idx_];

  // Process in batches to stay within CUDA IPC handle limit.
  for (size_t start = 0; start < num_iovs; start += kMaxOpenIpcHandles) {
    size_t end = std::min(start + kMaxOpenIpcHandles, num_iovs);
    size_t batch_size = end - start;
    bool is_last_batch = (end == num_iovs);

    // Open handles for this batch
    std::vector<void*> batch_ptrs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      GPU_RT_CHECK(gpuIpcOpenMemHandle(&batch_ptrs[i],
                                       info_v[start + i].handle,
                                       gpuIpcMemLazyEnablePeerAccess));
    }

    // Issue async copies round-robin across streams
    for (size_t i = 0; i < batch_size; ++i) {
      void* src_ptr = reinterpret_cast<void*>(
          reinterpret_cast<uintptr_t>(batch_ptrs[i]) +
          info_v[start + i].offset);
      bool dst_is_gpu = (uccl::get_dev_idx(data_v[start + i]) >= 0);
      auto kind = dst_is_gpu ? gpuMemcpyDeviceToDevice : gpuMemcpyDeviceToHost;
      gpuStream_t stream = streams[(start + i) % streams.size()];
      GPU_RT_CHECK(gpuMemcpyAsync(data_v[start + i], src_ptr,
                                   size_v[start + i], kind, stream));
    }

    if (is_last_batch) {
      pending->raw_ptrs = std::move(batch_ptrs);

      size_t num_used = std::min(num_iovs, streams.size());
      pending->events.resize(num_used);
      for (size_t i = 0; i < num_used; ++i) {
        GPU_RT_CHECK(gpuEventCreateWithFlags(&pending->events[i],
                                             gpuEventDisableTiming));
        GPU_RT_CHECK(gpuEventRecord(pending->events[i], streams[i]));
      }
    } else {
      size_t num_used = std::min(batch_size, streams.size());
      for (size_t i = 0; i < num_used; ++i) {
        GPU_RT_CHECK(gpuStreamSynchronize(streams[i]));
      }
      for (auto& ptr : batch_ptrs) {
        GPU_RT_CHECK(gpuIpcCloseMemHandle(ptr));
      }
    }
  }

  GPU_RT_CHECK(gpuSetDevice(orig_device));

  *transfer_id = reinterpret_cast<uint64_t>(pending->status);
  while (jring_mp_enqueue_bulk(ipc_completion_ring_, &pending, 1,
                               nullptr) != 1) {
  }

  return true;
}

bool Endpoint::advertise_ipc(uint64_t conn_id, void* addr, size_t len,
                             char* out_buf) {
  CHECK(addr != nullptr) << "advertise_ipc: addr pointer is null!";
  CHECK(out_buf != nullptr) << "advertise_ipc: out_buf pointer is null!";

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  // Generate IPC memory handle for the address
  IpcTransferInfo transfer_info = {};  // Initialize to zero
  transfer_info.size = len;
  transfer_info.operation = 1;  // response

  // Calculate aligned address and offset
  auto addr_aligned = reinterpret_cast<uintptr_t>(addr) & ~(kIpcAlignment - 1);
  auto addr_offset = reinterpret_cast<uintptr_t>(addr) - addr_aligned;
  transfer_info.offset = addr_offset;

  GPU_RT_CHECK(gpuIpcGetMemHandle(&transfer_info.handle,
                                  reinterpret_cast<void*>(addr_aligned)));

  // Copy the transfer info to output buffer
  std::memcpy(out_buf, &transfer_info, sizeof(transfer_info));

  return true;
}

bool Endpoint::advertisev_ipc(uint64_t conn_id, std::vector<void*> addr_v,
                              std::vector<size_t> len_v,
                              std::vector<char*> out_buf_v, size_t num_iovs) {
  CHECK_EQ(addr_v.size(), num_iovs) << "addr_v size mismatch";
  CHECK_EQ(len_v.size(), num_iovs) << "len_v size mismatch";
  CHECK_EQ(out_buf_v.size(), num_iovs) << "out_buf_v size mismatch";

  int orig_device;
  GPU_RT_CHECK(gpuGetDevice(&orig_device));
  auto dev_reset =
      uccl::finally([&]() { GPU_RT_CHECK(gpuSetDevice(orig_device)); });

  GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));

  for (size_t i = 0; i < num_iovs; ++i) {
    CHECK(addr_v[i] != nullptr)
        << "advertisev_ipc: addr_v[" << i << "] is null!";
    CHECK(out_buf_v[i] != nullptr)
        << "advertisev_ipc: out_buf_v[" << i << "] is null!";

    // Generate IPC memory handle for each address
    IpcTransferInfo transfer_info = {};  // Initialize to zero
    transfer_info.size = len_v[i];
    transfer_info.operation = 1;  // response

    // Calculate aligned address and offset
    auto addr_aligned =
        reinterpret_cast<uintptr_t>(addr_v[i]) & ~(kIpcAlignment - 1);
    auto addr_offset = reinterpret_cast<uintptr_t>(addr_v[i]) - addr_aligned;
    transfer_info.offset = addr_offset;

    GPU_RT_CHECK(gpuIpcGetMemHandle(&transfer_info.handle,
                                    reinterpret_cast<void*>(addr_aligned)));

    // Copy the transfer info to output buffer
    std::memcpy(out_buf_v[i], &transfer_info, sizeof(transfer_info));
  }

  return true;
}

bool Endpoint::poll_async(uint64_t transfer_id, bool* is_done) {
  auto* status = reinterpret_cast<TransferStatus*>(transfer_id);
  *is_done = status->done.load(std::memory_order_acquire);
  if (*is_done) {
    delete status;
  }
  return true;
}

void Endpoint::init_uds_socket() {
  // Create UDS socket path based on local GPU index
  uds_socket_path_ = get_uds_socket_path(local_gpu_idx_);

  // Remove existing socket file if it exists
  unlink(uds_socket_path_.c_str());

  // Create socket
  uds_listen_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
  if (uds_listen_fd_ < 0) {
    std::cerr << "Failed to create UDS socket: " << strerror(errno)
              << std::endl;
    return;
  }

  // Set up socket address
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, uds_socket_path_.c_str(), sizeof(addr.sun_path) - 1);

  // Bind socket
  if (bind(uds_listen_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    std::cerr << "Failed to bind UDS socket to " << uds_socket_path_ << ": "
              << strerror(errno) << std::endl;
    close(uds_listen_fd_);
    uds_listen_fd_ = -1;
    return;
  }

  // Start listening
  if (listen(uds_listen_fd_, 5) < 0) {
    std::cerr << "Failed to listen on UDS socket: " << strerror(errno)
              << std::endl;
    close(uds_listen_fd_);
    uds_listen_fd_ = -1;
    unlink(uds_socket_path_.c_str());
    return;
  }

  std::cout << "UDS socket initialized at " << uds_socket_path_ << std::endl;
}

void Endpoint::cleanup_uds_socket() {
  if (uds_listen_fd_ >= 0) {
    close(uds_listen_fd_);
    uds_listen_fd_ = -1;
  }

  if (!uds_socket_path_.empty()) {
    unlink(uds_socket_path_.c_str());
    uds_socket_path_.clear();
  }
}

void Endpoint::ipc_completion_thread_func() {
  std::vector<PendingIpcBatch*> active;
  alignas(16) char buf[16];

  while (!stop_.load(std::memory_order_acquire)) {
    // Drain ring into active list
    while (jring_sc_dequeue_bulk(ipc_completion_ring_, buf, 1, nullptr) == 1) {
      active.push_back(*reinterpret_cast<PendingIpcBatch**>(buf));
    }

    // Poll each batch for completion
    for (auto it = active.begin(); it != active.end();) {
      bool all_done = true;
      for (auto& evt : (*it)->events) {
        if (gpuEventQuery(evt) != gpuSuccess) {
          all_done = false;
          break;
        }
      }
      if (all_done) {
        for (auto& evt : (*it)->events) {
          GPU_RT_CHECK(gpuEventDestroy(evt));
        }
        GPU_RT_CHECK(gpuSetDevice((*it)->remote_device));
        for (auto& ptr : (*it)->raw_ptrs) {
          GPU_RT_CHECK(gpuIpcCloseMemHandle(ptr));
        }
        TransferStatus* status = (*it)->status;
        delete *it;
        it = active.erase(it);
        // Mark done AFTER cleanup; done.store must be last since
        // poll_async deletes the TransferStatus when it sees done==true.
        status->done.store(true, std::memory_order_release);
      } else {
        ++it;
      }
    }

    if (active.empty()) {
      std::this_thread::yield();
    }
  }
}

void Endpoint::send_proxy_thread_func() {
  uccl::pin_thread_to_numa(numa_node_);
  // Use 16-byte buffer to avoid stringop-overflow warning from jring's 16-byte
  // bulk copy
  alignas(16) char task_buffer[16];
  UnifiedTask* task;

  while (!stop_.load(std::memory_order_acquire)) {
    if (jring_sc_dequeue_bulk(send_unified_task_ring_, task_buffer, 1,
                              nullptr) == 1) {
      task = *reinterpret_cast<UnifiedTask**>(task_buffer);
      switch (task->type) {
        case TaskType::SEND_IPC:
          send_ipc(task->conn_id, task->data, task->size);
          break;
        case TaskType::WRITE_NET:
          write(task->conn_id, task->mr_id, task->data, task->size,
                task->slot_item());
          break;
        case TaskType::SEND_NET:
          send(task->conn_id, task->mr_id, task->data, task->size);
          break;
        case TaskType::SENDV: {
          TaskBatch const& batch = task->task_batch();
          std::vector<void const*> const_data_v(
              batch.const_data_v(), batch.const_data_v() + batch.num_iovs);
          std::vector<size_t> size_v(batch.size_v(),
                                     batch.size_v() + batch.num_iovs);
          std::vector<uint64_t> mr_id_v(batch.mr_id_v(),
                                        batch.mr_id_v() + batch.num_iovs);

          sendv(task->conn_id, mr_id_v, const_data_v, size_v, batch.num_iovs);
          break;
        }
        case TaskType::WRITEV: {
          TaskBatch const& batch = task->task_batch();
          std::vector<void*> data_v(batch.data_v(),
                                    batch.data_v() + batch.num_iovs);
          std::vector<size_t> size_v(batch.size_v(),
                                     batch.size_v() + batch.num_iovs);
          std::vector<uint64_t> mr_id_v(batch.mr_id_v(),
                                        batch.mr_id_v() + batch.num_iovs);
          std::vector<FifoItem> slot_item_v(
              batch.slot_item_v(), batch.slot_item_v() + batch.num_iovs);

          writev(task->conn_id, mr_id_v, data_v, size_v, slot_item_v,
                 batch.num_iovs);
          break;
        }
        default:
          LOG(ERROR) << "Unexpected task type in send processing: "
                     << static_cast<int>(task->type);
          break;
      }
      task->status_ptr->task_ptr.reset();
      task->status_ptr->done.store(true, std::memory_order_release);
    }
  }
}

void Endpoint::recv_proxy_thread_func() {
  uccl::pin_thread_to_numa(numa_node_);
  // Use 16-byte buffer to avoid stringop-overflow warning from jring's 16-byte
  // bulk copy
  alignas(16) char task_buffer[16];
  UnifiedTask* task;

  while (!stop_.load(std::memory_order_acquire)) {
    if (jring_sc_dequeue_bulk(recv_unified_task_ring_, task_buffer, 1,
                              nullptr) == 1) {
      task = *reinterpret_cast<UnifiedTask**>(task_buffer);
      switch (task->type) {
        case TaskType::RECV_IPC:
          recv_ipc(task->conn_id, task->data, task->size);
          break;
        case TaskType::READ_NET:
          read(task->conn_id, task->mr_id, task->data, task->size,
               task->slot_item());
          break;
        case TaskType::RECV_NET:
          recv(task->conn_id, task->mr_id, task->data, task->size);
          break;
        case TaskType::RECVV: {
          TaskBatch const& batch = task->task_batch();
          std::vector<void*> data_v(batch.data_v(),
                                    batch.data_v() + batch.num_iovs);
          std::vector<size_t> size_v(batch.size_v(),
                                     batch.size_v() + batch.num_iovs);
          std::vector<uint64_t> mr_id_v(batch.mr_id_v(),
                                        batch.mr_id_v() + batch.num_iovs);

          recvv(task->conn_id, mr_id_v, data_v, size_v, batch.num_iovs);
          break;
        }
        case TaskType::READV: {
          TaskBatch const& batch = task->task_batch();
          std::vector<void*> data_v(batch.data_v(),
                                    batch.data_v() + batch.num_iovs);
          std::vector<size_t> size_v(batch.size_v(),
                                     batch.size_v() + batch.num_iovs);
          std::vector<uint64_t> mr_id_v(batch.mr_id_v(),
                                        batch.mr_id_v() + batch.num_iovs);
          std::vector<FifoItem> slot_item_v(
              batch.slot_item_v(), batch.slot_item_v() + batch.num_iovs);

          readv(task->conn_id, mr_id_v, data_v, size_v, slot_item_v,
                batch.num_iovs);
          break;
        }
        default:
          LOG(ERROR) << "Unexpected task type in receive processing: "
                     << static_cast<int>(task->type);
          break;
      }
      task->status_ptr->task_ptr.reset();
      task->status_ptr->done.store(true, std::memory_order_release);
    }
  }
}
