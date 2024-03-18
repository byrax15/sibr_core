#pragma once

#include "CUDA_SAFE_CALL_ALWAYS.h"
#include "GaussianProperties.hpp"
#include <vector>

namespace sibr {
struct GaussianScene {
    size_t start_index {}, count {};
    Pos position;
    float opacity = 1;

    GaussianScene() = default;

    GaussianScene(size_t start_index, size_t count)
        : start_index(start_index)
        , count(count)
    {
    }

public:
    template <typename CudaT, typename HostT, typename Callable>
    void for_each(CudaT* mapped_buffer, Callable&& c) const;

    template <typename CudaT, typename HostT>
    void EraseCompact(CudaT*& mapped_buffer, size_t old_count) const;

    auto begin() const { return start_index; }
    auto end() const { return start_index + count; }

private:
    template <typename CudaT, typename HostT>
    std::vector<HostT> MemcpyToHost(CudaT const* src_buffer) const;

    template <typename CudaT, typename HostT>
    void MemcpyToDevice(std::vector<HostT> const& upload, CudaT* dst_buffer) const;
};
}

template <typename CudaT, typename HostT>
std::vector<HostT> sibr::GaussianScene::MemcpyToHost(CudaT const* src_buffer) const
{
    static_assert(sizeof CudaT <= sizeof HostT);
    std::vector<HostT> copy(this->count);
    const auto start_ptr = reinterpret_cast<HostT const*>(src_buffer) + start_index;
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(copy.data(), start_ptr, count * sizeof(HostT), cudaMemcpyDeviceToHost));
    return copy;
}

template <typename CudaT, typename HostT>
void sibr::GaussianScene::MemcpyToDevice(std::vector<HostT> const& upload, CudaT* dst_buffer) const
{
    static_assert(sizeof CudaT <= sizeof HostT);
    constexpr auto size_ratio = sizeof(HostT) / sizeof(CudaT);
    const auto start_ptr = reinterpret_cast<HostT*>(dst_buffer) + start_index;
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(start_ptr, upload.data(), count * sizeof(HostT), cudaMemcpyHostToDevice));
}

template <typename CudaT, typename HostT, typename Callable>
void sibr::GaussianScene::for_each(CudaT* mapped_buffer, Callable&& c) const
{
    auto gauss = MemcpyToHost<CudaT, HostT>(mapped_buffer);
    std::for_each(gauss.begin(), gauss.end(), c);
    MemcpyToDevice<CudaT, HostT>(gauss, mapped_buffer);
}

template <typename CudaT, typename HostT>
void sibr::GaussianScene::EraseCompact(CudaT*& mapped_buffer, size_t old_count) const
{
    constexpr auto ByteSize = [](size_t count) { return (count * sizeof(HostT)); };

    HostT*& buf_old = reinterpret_cast<HostT*&>(mapped_buffer);
    HostT* buf_new {};

    CUDA_SAFE_CALL(cudaMalloc(&buf_new, ByteSize(old_count - count)));
    CUDA_SAFE_CALL(cudaMemcpy(buf_new, buf_old, ByteSize(start_index), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(buf_new + start_index, buf_old + end(), ByteSize(old_count - end()), cudaMemcpyDeviceToDevice));

    cudaFree(buf_old);
    mapped_buffer = reinterpret_cast<CudaT*>(buf_new);
}