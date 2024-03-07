#pragma once

#include <vector>
#include "CUDA_SAFE_CALL_ALWAYS.h"

namespace sibr {
struct GaussianScene {
    size_t start_index {}, count {};

    GaussianScene() = default;

    GaussianScene(size_t start_index, size_t count)
        : start_index(start_index)
        , count(count)
    {
    }

private:
    float opacity = 1;

public:
    /**
     * Set opacity member property.
     * Returns the opacity change ratio in [0,2)
     */
    [[nodiscard]] auto SetOpacity(float op)
    {
        const auto delta = 1 + (op - opacity);
        opacity = op;
        return delta;
    }

    template <typename CudaT, typename HostT, typename Callable>
    void for_each(CudaT* mapped_buffer, Callable&& c) const;

    template <typename CudaT, typename HostT>
    void EraseCompact(CudaT*& mapped_buffer, size_t old_size) const;

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
    HostT*& buf_old = reinterpret_cast<HostT*&>(mapped_buffer);
    HostT* buf_new {};
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc(&buf_new, (old_count - count) * sizeof(HostT)));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(buf_new, buf_old, start_index * sizeof(HostT), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(buf_new + start_index, buf_old + end(), (old_count - end()) * sizeof(HostT), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaFree(mapped_buffer));
    mapped_buffer = reinterpret_cast<CudaT*>(buf_new);
}