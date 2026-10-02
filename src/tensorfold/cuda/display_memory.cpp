// SPDX-License-Identifier: Apache-2.0
// Independent implementation of the Linux DRM dumb-buffer and CUDA registration APIs.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#if __has_include(<libdrm/drm.h>)
#include <libdrm/drm.h>
#include <libdrm/drm_mode.h>
#elif __has_include(<drm/drm.h>)
#include <drm/drm.h>
#include <drm/drm_mode.h>
#else
#error "Display storage needs the Linux DRM UAPI headers (libdrm-dev)."
#endif
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>

namespace {
void cuda_check(cudaError_t status, const char* operation) {
    TORCH_CHECK(status == cudaSuccess, "Display storage: ", operation, ": ", cudaGetErrorString(status));
}
void drm_check(int result, const char* operation, const std::string& card) {
    TORCH_CHECK(result >= 0, "Display storage: ", operation, " on ", card, ": ", std::strerror(errno));
}

struct Allocation {
    int fd = -1;
    uint32_t handle = 0;
    void* host = MAP_FAILED;
    void* gpu = nullptr;
    uint64_t size = 0;
    int device;
    bool registered = false;
    explicit Allocation(int index) : device(index) {}

    ~Allocation() noexcept {
        // The last tensor/view owns this object. Wait for outstanding work on ANY stream before unmapping.
        // If CUDA teardown or a device error prevents safe cleanup, leave it to process teardown instead.
        int previous = -1;
        if (registered) {
            cudaGetDevice(&previous);
            auto status = cudaSetDevice(device);
            if (status == cudaSuccess) status = cudaDeviceSynchronize();
            if (status == cudaSuccess) status = cudaHostUnregister(host);
            if (previous >= 0 && previous != device) cudaSetDevice(previous);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "Display storage cleanup deferred to process exit: %s\n",
                             cudaGetErrorString(status));
                return;
            }
        }
        if (host != MAP_FAILED) munmap(host, size);
        if (handle && fd >= 0) {
            drm_mode_destroy_dumb destroy{};
            destroy.handle = handle;
            if (ioctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy) < 0)
                std::fprintf(stderr, "Display storage: destroy dumb buffer: %s\n", std::strerror(errno));
        }
        if (fd >= 0) close(fd);
    }
};
} // namespace

pybind11::tuple allocate_display(int64_t requested, int64_t limit, const std::string& card, int device) {
    TORCH_CHECK(requested > 0 && limit >= requested, "Display storage: require 0 < requested bytes <= limit");
    TORCH_CHECK(device >= 0, "Display storage: a concrete CUDA device index is required");
    c10::cuda::CUDAGuard guard(device);
    cudaStreamCaptureStatus capture;
    cuda_check(cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(device), &capture), "capture status");
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "Display storage must be allocated before graph capture");
    int supported = 0;
    cuda_check(cudaDeviceGetAttribute(&supported, cudaDevAttrHostRegisterSupported, device), "host registration support");
    TORCH_CHECK(supported, "Display storage: GPU does not support CUDA host registration");
    auto owner = std::make_shared<Allocation>(device);
    owner->fd = open(card.c_str(), O_RDWR | O_CLOEXEC);
    drm_check(owner->fd, "open DRM card (pass --device and check permissions)", card);
    struct stat info{};
    drm_check(fstat(owner->fd, &info), "fstat", card);
    TORCH_CHECK(S_ISCHR(info.st_mode), "Display storage: DRM path is not a character device: ", card);
    char driver[128]{};
    drm_version version{};
    version.name = driver;
    version.name_len = sizeof(driver) - 1;
    drm_check(ioctl(owner->fd, DRM_IOCTL_VERSION, &version), "DRM_IOCTL_VERSION", card);
    TORCH_CHECK(std::string(driver) == "nvidia-drm", "Display storage needs nvidia-drm, found ", driver);
    char pci[32]{};
    cuda_check(cudaDeviceGetPCIBusId(pci, sizeof(pci), device), "CUDA PCI identity");
    const auto drm_pci = std::filesystem::canonical("/sys/dev/char/" + std::to_string(major(info.st_rdev)) +
                         ":" + std::to_string(minor(info.st_rdev)) + "/device").filename().string();
    unsigned int cd, cb, cs, cf, dd, db, ds, df;
    TORCH_CHECK(std::sscanf(pci, "%x:%x:%x.%x", &cd, &cb, &cs, &cf) == 4 &&
                std::sscanf(drm_pci.c_str(), "%x:%x:%x.%x", &dd, &db, &ds, &df) == 4 &&
                cd == dd && cb == db && cs == ds && cf == df,
                "Display storage: CUDA device ", pci, " does not match DRM device ", drm_pci);

    // A 64 KiB scanline keeps the allocation page aligned on GB10 and handles 1792 MiB below 32768 rows.
    constexpr uint64_t row_bytes = 16384 * 4;
    uint64_t height = (static_cast<uint64_t>(requested) + row_bytes - 1) / row_bytes;
    TORCH_CHECK(height <= std::numeric_limits<uint32_t>::max(), "Display storage: allocation too large");
    TORCH_CHECK(height * row_bytes <= static_cast<uint64_t>(limit),
                "Display storage: requested allocation rounded to 64 KiB exceeds limit");
    drm_mode_create_dumb create{};
    create.width = 16384;
    create.height = static_cast<uint32_t>(height);
    create.bpp = 32;
    drm_check(ioctl(owner->fd, DRM_IOCTL_MODE_CREATE_DUMB, &create), "DRM_IOCTL_MODE_CREATE_DUMB", card);
    owner->handle = create.handle;
    owner->size = create.size;
    TORCH_CHECK(create.size >= static_cast<uint64_t>(requested) && create.size <= static_cast<uint64_t>(limit),
                "Display storage: DRM allocated ", create.size, " bytes for ", requested, ", limit ", limit);
    TORCH_CHECK(create.size <= std::numeric_limits<size_t>::max(), "Display storage: mapping exceeds address space");
    drm_mode_map_dumb map{};
    map.handle = create.handle;
    drm_check(ioctl(owner->fd, DRM_IOCTL_MODE_MAP_DUMB, &map), "DRM_IOCTL_MODE_MAP_DUMB", card);
    owner->host = mmap(nullptr, create.size, PROT_READ | PROT_WRITE, MAP_SHARED, owner->fd, map.offset);
    TORCH_CHECK(owner->host != MAP_FAILED, "Display storage: mmap: ", std::strerror(errno));
    cuda_check(cudaHostRegister(owner->host, owner->size, cudaHostRegisterMapped | cudaHostRegisterIoMemory),
               "cudaHostRegister(Mapped | IoMemory)");
    owner->registered = true;
    cuda_check(cudaHostGetDevicePointer(&owner->gpu, owner->host, 0), "cudaHostGetDevicePointer");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(owner->gpu) % 256 == 0, "Display storage: GPU pointer is not 256-byte aligned");
    // Only this deleter releases the DRM/CUDA resources. PyTorch must never cudaFree this pointer.
    auto tensor = at::from_blob(owner->gpu, {requested}, [owner](void*) {},
                               at::TensorOptions().dtype(at::kByte).device(at::kCUDA, device));
    pybind11::dict receipt;
    receipt["requested_bytes"] = requested;
    receipt["allocated_bytes"] = create.size;
    receipt["card"] = card;
    receipt["driver"] = driver;
    receipt["device"] = device;
    receipt["device_pointer"] = reinterpret_cast<uintptr_t>(owner->gpu);
    receipt["pitch"] = create.pitch;
    receipt["pci_bus_id"] = pci;
    return pybind11::make_tuple(tensor, receipt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("allocate", &allocate_display);
}
