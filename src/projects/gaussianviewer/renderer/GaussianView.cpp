/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#include "GaussianView.hpp"
#include "CUDA_SAFE_CALL_ALWAYS.h"
#include "GaussianProperties.hpp"
#include "ImGuizmo.h"
#include <boost/asio.hpp>
#include <core/graphics/GUI.hpp>
#include <execution>
#include <rasterizer.h>
#include <thread>

float sigmoid(const float m1)
{
    return 1.0f / (1.0f + exp(-m1));
}

float inverse_sigmoid(const float m1)
{
    return log(m1 / (1.0f - m1));
}

// Load the Gaussians from the given file.
template <int D>
int loadPly(const char* filename,
    std::vector<Pos>& pos,
    std::vector<SHs<3>>& shs,
    std::vector<float>& opacities,
    std::vector<Scale>& scales,
    std::vector<Rot>& rot,
    sibr::Vector3f& minn,
    sibr::Vector3f& maxx)
{
    std::ifstream infile(filename, std::ios_base::binary);

    if (!infile.good())
        SIBR_ERR << "Unable to find model's PLY file, attempted:\n"
                 << filename << std::endl;

    // "Parse" header (it has to be a specific format anyway)
    std::string buff;
    std::getline(infile, buff);
    std::getline(infile, buff);

    std::string dummy;
    std::getline(infile, buff);
    std::stringstream ss(buff);
    int count;
    ss >> dummy >> dummy >> count;

    // Output number of Gaussians contained
    SIBR_LOG << "Loading " << count << " Gaussian splats" << std::endl;

    while (std::getline(infile, buff))
        if (buff.compare("end_header") == 0)
            break;

    // Read all Gaussians at once (AoS)
    std::vector<RichPoint<D>> points(count);
    infile.read((char*)points.data(), count * sizeof(RichPoint<D>));

    // Resize our SoA data
    pos.resize(count);
    shs.resize(count);
    scales.resize(count);
    rot.resize(count);
    opacities.resize(count);

    // Gaussians are done training, they won't move anymore. Arrange
    // them according to 3D Morton order. This means better cache
    // behavior for reading Gaussians that end up in the same tile
    // (close in 3D --> close in 2D).
    minn = sibr::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
    maxx = -minn;
    for (int i = 0; i < count; i++) {
        maxx = maxx.cwiseMax(points[i].pos);
        minn = minn.cwiseMin(points[i].pos);
    }
    std::vector<std::pair<uint64_t, int>> mapp(count);
    for (int i = 0; i < count; i++) {
        sibr::Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
        sibr::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
        sibr::Vector3i xyz = scaled.cast<int>();

        uint64_t code = 0;
        for (int i = 0; i < 21; i++) {
            code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
            code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
            code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
        }

        mapp[i].first = code;
        mapp[i].second = i;
    }
    auto sorter = [](const std::pair<uint64_t, int>& a, const std::pair<uint64_t, int>& b) {
        return a.first < b.first;
    };
    std::sort(mapp.begin(), mapp.end(), sorter);

    // Move data from AoS to SoA
    int SH_N = (D + 1) * (D + 1);
    for (int k = 0; k < count; k++) {
        int i = mapp[k].second;
        pos[k] = points[i].pos;

        // Normalize quaternion
        float length2 = 0;
        for (int j = 0; j < 4; j++)
            length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
        float length = sqrt(length2);
        for (int j = 0; j < 4; j++)
            rot[k].rot[j] = points[i].rot.rot[j] / length;

        // Exponentiate scale
        for (int j = 0; j < 3; j++)
            scales[k].scale[j] = exp(points[i].scale.scale[j]);

        // Activate alpha
        opacities[k] = sigmoid(points[i].opacity);

        shs[k].shs[0] = points[i].shs.shs[0];
        shs[k].shs[1] = points[i].shs.shs[1];
        shs[k].shs[2] = points[i].shs.shs[2];
        for (int j = 1; j < SH_N; j++) {
            shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
            shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
            shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
        }
    }
    return count;
}

void savePly(const char* filename,
    const std::vector<Pos>& pos,
    const std::vector<SHs<3>>& shs,
    const std::vector<float>& opacities,
    const std::vector<Scale>& scales,
    const std::vector<Rot>& rot,
    const std::vector<sibr::Vector3f>& minn,
    const std::vector<sibr::Vector3f>& maxx,
    FORWARD::Cull::Operator op)
{
    std::vector<RichPoint<3>> points;
    points.reserve(pos.size());
    for (int i = 0; i < pos.size(); ++i) {
        using namespace FORWARD::Cull;
        if (Boxes<sibr::Vector3f> { minn.data(), maxx.data(), static_cast<int>(minn.size()) }.TryCull(pos[i], op))
            continue;

        auto& p = points.emplace_back();

        p.pos = pos[i];
        p.rot = rot[i];
        // Exponentiate scale
        for (int j = 0; j < 3; j++)
            p.scale.scale[j] = log(scales[i].scale[j]);
        // Activate alpha
        p.opacity = inverse_sigmoid(opacities[i]);
        p.shs.shs[0] = shs[i].shs[0];
        p.shs.shs[1] = shs[i].shs[1];
        p.shs.shs[2] = shs[i].shs[2];
        for (int j = 1; j < 16; j++) {
            p.shs.shs[(j - 1) + 3] = shs[i].shs[j * 3 + 0];
            p.shs.shs[(j - 1) + 18] = shs[i].shs[j * 3 + 1];
            p.shs.shs[(j - 1) + 33] = shs[i].shs[j * 3 + 2];
        }
    }

    // Output number of Gaussians contained
    SIBR_LOG << "Saving " << points.size() << " Gaussian splats" << std::endl;

    std::ofstream outfile(filename, std::ios_base::binary);
    outfile << "ply\nformat binary_little_endian 1.0\nelement vertex " << points.size() << "\n";

    for (auto s : { "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2" })
        outfile << "property float " << s << std::endl;
    for (int i = 0; i < 45; i++)
        outfile << "property float f_rest_" << i << std::endl;
    for (auto s : { "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" })
        outfile << "property float " << s << std::endl;
    outfile << "end_header" << std::endl;

    outfile.write((char*)points.data(), sizeof(decltype(points)::value_type) * points.size());
}

template <typename CudaT, typename HostT>
void cudaReallocMemcpy(CudaT*& buf_realloc, size_t old_count, std::vector<HostT> const& append_data)
{
    static_assert(sizeof HostT >= sizeof CudaT);
    constexpr auto ByteSize = [](size_t count) { return (count * sizeof(HostT)); };

    const auto& buf_old = reinterpret_cast<HostT*&>(buf_realloc);
    HostT* buf_new {};

    cudaMalloc(&buf_new, ByteSize(old_count + append_data.size()));
    cudaMemcpy(buf_new, buf_old, ByteSize(old_count), cudaMemcpyDeviceToDevice);
    cudaMemcpy(buf_new + old_count, append_data.data(), ByteSize(append_data.size()), cudaMemcpyHostToDevice);
    CUDA_SAFE_CALL();
    cudaFree(buf_old);
    buf_realloc = reinterpret_cast<CudaT*>(buf_new);
}

template <typename CudaT, typename HostT>
static void cudaMallocMemcpy(CudaT*& buf, std::vector<HostT> const& data)
{
    cudaMalloc(&buf, data.size() * sizeof(HostT));
    cudaMemcpy(buf, data.data(), data.size() * sizeof(HostT), cudaMemcpyHostToDevice);
    CUDA_SAFE_CALL();
}

template <typename CudaT, typename HostT, bool copy_old = false>
static void cudaRealloc(CudaT*& buf, size_t count)
{
    CudaT* newbuf {};
    cudaMalloc(&newbuf, count * sizeof(HostT));
    if constexpr (copy_old) {
        cudaMemcpy(newbuf, buf, count * sizeof(HostT), cudaMemcpyDeviceToDevice);
    }
    cudaFree(buf);
    buf = newbuf;
}

static void glReallocGaussianData(sibr::GaussianData*& gData, size_t count,
    float* pos_cuda, float* rot_cuda, float* scale_cuda, float* opacity_cuda, float* shs_cuda)
{
    std::vector<Pos> all_pos(count);
    std::vector<Rot> all_rot(count);
    std::vector<float> all_opacity(count);
    std::vector<SHs<3>> all_shs(count);
    std::vector<Scale> all_scale(count);
    cudaMemcpy(all_pos.data(), pos_cuda, count * sizeof(Pos), cudaMemcpyDeviceToHost);
    cudaMemcpy(all_rot.data(), rot_cuda, count * sizeof(Rot), cudaMemcpyDeviceToHost);
    cudaMemcpy(all_opacity.data(), opacity_cuda, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(all_shs.data(), shs_cuda, count * sizeof(SHs<3>), cudaMemcpyDeviceToHost);
    cudaMemcpy(all_scale.data(), scale_cuda, count * sizeof(Scale), cudaMemcpyDeviceToHost);

    gData->set_buffers(count, (float*)all_pos.data(), (float*)all_rot.data(), (float*)all_scale.data(), (float*)all_opacity.data(), (float*)all_shs.data());
}

std::function<char*(size_t N)> resizeFunctional(void** ptr, size_t& S)
{
    auto lambda = [ptr, &S](size_t N) {
        if (N > S) {
            if (*ptr)
                CUDA_SAFE_CALL(cudaFree(*ptr));
            CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
            S = 2 * N;
        }
        return reinterpret_cast<char*>(*ptr);
    };
    return lambda;
}

namespace sibr {
// A simple copy renderer class. Much like the original, but this one
// reads from a buffer instead of a texture and blits the result to
// a render target.
class BufferCopyRenderer {

public:
    BufferCopyRenderer()
    {
        _shader.init("CopyShader",
            sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.vert"),
            sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.frag"));

        _flip.init(_shader, "flip");
        _width.init(_shader, "width");
        _height.init(_shader, "height");
    }

    void process(uint bufferID, IRenderTarget& dst, int width, int height, bool disableTest = true)
    {
        if (disableTest)
            glDisable(GL_DEPTH_TEST);
        else
            glEnable(GL_DEPTH_TEST);

        _shader.begin();
        _flip.send();
        _width.send();
        _height.send();

        dst.clear();
        dst.bind();

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

        sibr::RenderUtility::renderScreenQuad();

        dst.unbind();
        _shader.end();
    }

    /** \return option to flip the texture when copying. */
    bool& flip() { return _flip.get(); }
    int& width() { return _width.get(); }
    int& height() { return _height.get(); }

private:
    GLShader _shader;
    GLuniform<bool> _flip = false; ///< Flip the texture when copying.
    GLuniform<int> _width = 1000;
    GLuniform<int> _height = 800;
};
}

sibr::GaussianView::GaussianView(const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h, const char* file, bool* messageRead, int sh_degree, bool white_bg, bool useInterop, int device)
    : _scene(ibrScene)
    , _dontshow(messageRead)
    , _sh_degree(sh_degree)
    , sibr::ViewBase(render_w, render_h)
{
    int num_devices;
    CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
    _device = device;
    if (device >= num_devices) {
        if (num_devices == 0)
            SIBR_ERR << "No CUDA devices detected!";
        else
            SIBR_ERR << "Provided device index exceeds number of available CUDA devices!";
    }
    CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 7) {
        SIBR_ERR << "Sorry, need at least compute capability 7.0+!";
    }

    _copyRenderer = new BufferCopyRenderer();
    _copyRenderer->flip() = true;
    _copyRenderer->width() = render_w;
    _copyRenderer->height() = render_h;

    std::vector<uint> imgs_ulr;
    const auto& cams = ibrScene->cameras()->inputCameras();
    for (size_t cid = 0; cid < cams.size(); ++cid) {
        if (cams[cid]->isActive()) {
            imgs_ulr.push_back(uint(cid));
        }
    }
    _scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);

    // Load the PLY data (AoS) to the GPU (SoA)
    std::vector<Pos> pos;
    std::vector<Rot> rot;
    std::vector<Scale> scale;
    std::vector<float> opacity;
    std::vector<SHs<3>> shs;
    if (sh_degree == 1) {
        count = loadPly<1>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
    } else if (sh_degree == 2) {
        count = loadPly<2>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
    } else if (sh_degree == 3) {
        count = loadPly<3>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
    }

    const auto P = count;
    scenes.emplace_back(GaussianScene { 0, static_cast<size_t>(P) });

    // Allocate and fill the GPU data
    cudaMallocMemcpy(scene_space.pos_cuda, pos);
    cudaMallocMemcpy(scene_space.rot_cuda, rot);
    cudaMallocMemcpy(scene_space.shs_cuda, shs);
    cudaMallocMemcpy(scene_space.opacity_cuda, opacity);
    cudaMallocMemcpy(scene_space.scale_cuda, scale);

    cudaMalloc(&world_space.pos_cuda, count * sizeof(Pos));
    cudaMalloc(&world_space.rot_cuda, count * sizeof(Rot));
    cudaMalloc(&world_space.shs_cuda, count * sizeof(SHs<3>));
    cudaMalloc(&world_space.opacity_cuda, count * sizeof(float));
    cudaMalloc(&world_space.scale_cuda, count * sizeof(Scale));

    boxmin = { _scenemin };
    boxmax = { _scenemax };
    cudaMallocMemcpy(boxmin_cuda, boxmin);
    cudaMallocMemcpy(boxmax_cuda, boxmax);

    // Create space for view parameters
    CUDA_SAFE_CALL(cudaMalloc((void**)&view_cuda, sizeof(sibr::Matrix4f)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&proj_cuda, sizeof(sibr::Matrix4f)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cam_pos_cuda, 3 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&rect_cuda, 2 * P * sizeof(int)));

    const auto color = white_bg ? 1.f : 0.f;
    Vector3f bg(color, color, color);
    CUDA_SAFE_CALL(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(background_cuda, bg.data(), sizeof(Vector3f), cudaMemcpyHostToDevice));

    _gaussianRenderer = new GaussianSurfaceRenderer();

    // Create GL buffer ready for CUDA/GL interop
    glCreateBuffers(1, &imageBuffer);
    glNamedBufferStorage(imageBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

    if (useInterop) {
        if (cudaPeekAtLastError() != cudaSuccess) {
            SIBR_ERR << "A CUDA error occurred in setup:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
        }
        cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
        useInterop &= (cudaGetLastError() == cudaSuccess);
    }
    if (!useInterop) {
        fallback_bytes.resize(render_w * render_h * 3 * sizeof(float));
        cudaMalloc(&fallbackBufferCuda, fallback_bytes.size());
        _interop_failed = true;
    }

    geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
    binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
    imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);
}

void sibr::GaussianView::setScene(const sibr::BasicIBRScene::Ptr& newScene)
{
    _scene = newScene;

    // Tell the scene we are a priori using all active cameras.
    std::vector<uint> imgs_ulr;
    const auto& cams = newScene->cameras()->inputCameras();
    for (size_t cid = 0; cid < cams.size(); ++cid) {
        if (cams[cid]->isActive()) {
            imgs_ulr.push_back(uint(cid));
        }
    }
    _scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
}

void sibr::GaussianView::onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye)
{
    last_view = eye.view();
    last_proj = eye.proj();

    // Convert view and projection to target coordinate system
    auto view_mat = eye.view();
    view_mat.row(1) *= -1;
    view_mat.row(2) *= -1;

    auto proj_mat = eye.viewproj();
    proj_mat.row(1) *= -1;

    // Compute additional view parameters
    float tan_fovy = tan(eye.fovy() * 0.5f);
    float tan_fovx = tan_fovy * eye.aspect();

    // Copy frame-dependent data to GPU
    CUDA_SAFE_CALL(cudaMemcpy(view_cuda, view_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(proj_cuda, proj_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(cam_pos_cuda, &eye.position(), sizeof(float) * 3, cudaMemcpyHostToDevice));

    float* image_cuda = nullptr;
    if (!_interop_failed) {
        // Map OpenGL buffer resource for use with CUDA
        size_t bytes;
        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&image_cuda, &bytes, imageBufferCuda));
    } else {
        image_cuda = fallbackBufferCuda;
    }

    CudaRasterizer::Rasterizer::sceneToWorldAsync(scene_space, world_space, scenes.data(), scenes.size());
    auto& [pos_cuda, rot_cuda, scale_cuda, opacity_cuda, shs_cuda] = world_space;
    const auto num_rendered = CudaRasterizer::Rasterizer::forward(
        geomBufferFunc,
        binningBufferFunc,
        imgBufferFunc,
        count, _sh_degree, 16,
        background_cuda,
        _resolution.x(), _resolution.y(),
        pos_cuda,
        shs_cuda,
        nullptr,
        opacity_cuda,
        scale_cuda,
        _scalingModifier,
        rot_cuda,
        nullptr,
        view_cuda,
        proj_cuda,
        cam_pos_cuda,
        tan_fovx,
        tan_fovy,
        false,
        image_cuda,
        nullptr,
        rect_cuda,
        boxmin.size(),
        boxmin_cuda,
        boxmax_cuda,
        static_cast<FORWARD::Cull::Operator::Value>(selected_operation));

    if (!_interop_failed) {
        // Unmap OpenGL resource for use with OpenGL
        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
    } else {
        CUDA_SAFE_CALL(cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(), cudaMemcpyDeviceToHost));
        glNamedBufferSubData(imageBuffer, 0, fallback_bytes.size(), fallback_bytes.data());
    }

    /*
     * NECESSARY because the forward(...) returns without drawing if all gaussians were culled,
     * leaving a smear of gaussians in the framebuffer which lasted until at least one gaussian
     * remains after culling.
     */
    if (num_rendered <= 0) {
        dst.clear();
        return;
    } else {
        // Copy image contents to framebuffer
        _copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());
    }

    if (cudaPeekAtLastError() != cudaSuccess) {
        SIBR_ERR << "A CUDA error occurred during rendering:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
    }
}

void sibr::GaussianView::onUpdate(Input& input)
{
}

void sibr::GaussianView::onGUI()
{
    if (ImGui::Begin("Selection")) {
        const auto MemcpyBoxes = [&] {
            CUDA_SAFE_CALL(cudaMemcpy(boxmin_cuda, boxmin.data(), sizeof(Vector3f) * boxmin.size(), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(boxmax_cuda, boxmax.data(), sizeof(Vector3f) * boxmax.size(), cudaMemcpyHostToDevice));
        };

        const std::string selected_text = std::to_string(selected_box);
        if (ImGui::Button("Add Box")) {
            boxmin.emplace_back(_scenemin);
            boxmax.emplace_back(_scenemax);
            selected_box = boxmin.size() - 1;
            cudaRealloc<float, Vector3f>(boxmin_cuda, boxmin.size());
            cudaRealloc<float, Vector3f>(boxmax_cuda, boxmax.size());
            MemcpyBoxes();
        }
        ImGui::SameLine();
        if (ImGui::Button("Remove Box") && boxmin.size() > 1) {
            boxmin.erase(boxmin.begin() + selected_box);
            boxmax.erase(boxmax.begin() + selected_box);
            selected_box = std::min(static_cast<int>(boxmin.size() - 1), selected_box);
            cudaRealloc<float, Vector3f, true>(boxmin_cuda, boxmin.size());
            cudaRealloc<float, Vector3f, true>(boxmax_cuda, boxmax.size());
        }
        ImGui::SameLine();
        ImGui::Text("Active Boxes: %i/16", boxmin.size());

        assert(boxmin.size() == boxmax.size());
        ImGui::SliderInt("Crop Box", &selected_box, 0, boxmin.size() - 1);

        using FORWARD::Cull::Operator;
        if (ImGui::BeginCombo("Box Combination Op.", Operator::Names[selected_operation])) {
            for (int i = 0; i < Operator::Names.size(); ++i) {
                if (ImGui::Selectable(Operator::Names[i], selected_operation == i)) {
                    selected_operation = i;
                }
            }
            ImGui::EndCombo();
        }

        bool slid {};
        {
            ImGui::PushItemWidth(.45f * ImGui::GetWindowWidth());

            ImGui::TextColored({ 1, 0, 0, 1 }, "X");
            ImGui::SameLine();
            slid |= ImGui::SliderFloat("##min x", &boxmin[selected_box].x(), _scenemin.x(), _scenemax.x());
            ImGui::SameLine();
            slid |= ImGui::SliderFloat("##max x", &boxmax[selected_box].x(), _scenemin.x(), _scenemax.x());

            ImGui::TextColored({ 0, 1, 0, 1 }, "Y");
            ImGui::SameLine();
            slid |= ImGui::SliderFloat("##min y", &boxmin[selected_box].y(), _scenemin.y(), _scenemax.y());
            ImGui::SameLine();
            slid |= ImGui::SliderFloat("##max y", &boxmax[selected_box].y(), _scenemin.y(), _scenemax.y());

            ImGui::TextColored({ 0, 0, 1, 1 }, "Z");
            ImGui::SameLine();
            slid |= ImGui::SliderFloat("##min z", &boxmin[selected_box].z(), _scenemin.z(), _scenemax.z());
            ImGui::SameLine();
            slid |= ImGui::SliderFloat("##max z", &boxmax[selected_box].z(), _scenemin.z(), _scenemax.z());

            ImGui::PopItemWidth();
        }
        if (slid) {
            MemcpyBoxes();
        }
    }
    ImGui::End();

    if (ImGui::Begin("Scenes")) {
        if (ImGui::Button("Load subscene...")) {
            std::string fname;
            if (sibr::showFilePicker(fname, sibr::FilePickerMode::Default, "", "ply")) {
                std::vector<Pos> pos;
                std::vector<Rot> rot;
                std::vector<float> opacity;
                std::vector<SHs<3>> shs;
                std::vector<Scale> scale;
                sibr::Vector3f min, max;
                const auto append_count = std::invoke([&] {
                    switch (_sh_degree) {
                    case 1:
                        return loadPly<1>(fname.c_str(), pos, shs, opacity, scale, rot, min, max);
                    case 2:
                        return loadPly<2>(fname.c_str(), pos, shs, opacity, scale, rot, min, max);
                    case 3:
                        return loadPly<3>(fname.c_str(), pos, shs, opacity, scale, rot, min, max);
                    default:
                        SIBR_LOG << "Unsupported SH degree (" << _sh_degree << ")\n";
                        return 0;
                    }
                });
                if ((append_count + count) <= count) {
                    SIBR_LOG << "Skipping scene: Scene addition would cause certain integer overflow\n";
                }
                if (append_count > 0) {
                    scenes.emplace_back(GaussianScene {
                        static_cast<size_t>(count),
                        static_cast<size_t>(append_count) });
                    cudaReallocMemcpy(scene_space.pos_cuda, count, pos);
                    cudaReallocMemcpy(scene_space.rot_cuda, count, rot);
                    cudaReallocMemcpy(scene_space.opacity_cuda, count, opacity);
                    cudaReallocMemcpy(scene_space.shs_cuda, count, shs);
                    cudaReallocMemcpy(scene_space.scale_cuda, count, scale);
                    _scenemin = std::min(min, _scenemin);
                    _scenemax = std::max(max, _scenemax);
                    count += append_count;

                    cudaRealloc<float, Pos>(world_space.pos_cuda, (count));
                    cudaRealloc<float, Rot>(world_space.rot_cuda, (count));
                    cudaRealloc<float, float>(world_space.opacity_cuda, (count));
                    cudaRealloc<float, SHs<3>>(world_space.shs_cuda, (count));
                    cudaRealloc<float, Scale>(world_space.scale_cuda, (count));

                    // glReallocGaussianData(gData, count,
                    //     pos_cuda, rot_cuda, scale_cuda, opacity_cuda, shs_cuda);
                } else {
                    SIBR_LOG << "Skipping scene: no gaussian recognized in file\n";
                }
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Remove subscene") && scenes.size() > 1) {
            const auto& s = scenes[selected_scene];
            s.EraseCompact<float, Pos>(scene_space.pos_cuda, count);
            s.EraseCompact<float, Rot>(scene_space.rot_cuda, count);
            s.EraseCompact<float, float>(scene_space.opacity_cuda, count);
            s.EraseCompact<float, SHs<3>>(scene_space.shs_cuda, count);
            s.EraseCompact<float, Scale>(scene_space.scale_cuda, count);

            count -= s.count;
            cudaRealloc<float, Pos>(world_space.pos_cuda, (count));
            cudaRealloc<float, Rot>(world_space.rot_cuda, (count));
            cudaRealloc<float, float>(world_space.opacity_cuda, (count));
            cudaRealloc<float, SHs<3>>(world_space.shs_cuda, (count));
            cudaRealloc<float, Scale>(world_space.scale_cuda, (count));

            scenes.erase(std::next(scenes.begin(), selected_scene));
            scenes[0].start_index = 0;
            for (auto i = 1; i < scenes.size(); ++i) {
                scenes[i].start_index = scenes[i - 1].end();
            }
            selected_scene = selected_scene == scenes.size() ? selected_scene - 1 : selected_scene;

            // glReallocGaussianData(gData, count,
            //     pos_cuda, rot_cuda, scale_cuda, opacity_cuda, shs_cuda);
        }

        ImGui::SliderInt("Subscene", &selected_scene, 0, scenes.size() - 1);

        auto& s = scenes[selected_scene];
        ImGui::DragFloat3("Position", &s.position.x, .1f);
        ImGui::DragFloat3("Rotation", &s.rot.x);
        ImGui::DragFloat3("Scale", &s.scale.x, .1f, -5.f, 5.f);
        ImGui::SliderFloat("Opacity", &s.opacity, 0, 1);

        if (ImGui::Button("Save All...")) {
            std::string fname;
            if (sibr::showFilePicker(fname, sibr::FilePickerMode::Save, "", "ply")) {
                std::vector<Pos> pos(count);
                std::vector<Rot> rot(count);
                std::vector<float> opacity(count);
                std::vector<SHs<3>> shs(count);
                std::vector<Scale> scale(count);
                CUDA_SAFE_CALL(cudaMemcpy(pos.data(), world_space.pos_cuda, sizeof(Pos) * count, cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL(cudaMemcpy(rot.data(), world_space.rot_cuda, sizeof(Rot) * count, cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL(cudaMemcpy(opacity.data(), world_space.opacity_cuda, sizeof(float) * count, cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL(cudaMemcpy(shs.data(), world_space.shs_cuda, sizeof(SHs<3>) * count, cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL(cudaMemcpy(scale.data(), world_space.scale_cuda, sizeof(Scale) * count, cudaMemcpyDeviceToHost));
                savePly(fname.c_str(), pos, shs, opacity, scale, rot, boxmin, boxmax, static_cast<FORWARD::Cull::Operator::Value>(selected_operation));
            }
        }

        ImGui::Text("Gaussians count sum : %i", count);
        for (auto i = scenes.begin(); i != scenes.end(); ++i) {
            ImGui::Text("Scene #%i : start %i ; Sub-count %i", std::distance(scenes.begin(), i), i->start_index, i->count);
        }
    }
    ImGui::End();

    if (ImGui::Begin("Point view")) {
        using namespace ImGui;

        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(GetWindowPos().x, GetWindowPos().y, GetWindowWidth(), GetWindowHeight());

        using Mat = Matrix4f;
        std::vector<Mat> box_mats(boxmin.size());
        for (auto i = 0; i < boxmin.size(); ++i) {
            Vector3f transl = (boxmin[i] + boxmax[i]) / 2.f, rot = Vector3f::Zero(), scale = boxmax[i] - boxmin[i];
            ImGuizmo::RecomposeMatrixFromComponents(transl.data(), rot.data(), scale.data(), box_mats[i].data());
        }

        ImGuizmo::DrawCubes(last_view.data(), last_proj.data(), box_mats[0].data(), box_mats.size(), ImGuizmo::DrawMode::EDGES);
    }
    ImGui::End();

    if (!*_dontshow && !accepted && _interop_failed) {
        ImGui::OpenPopup("Error Using Interop");
    }

    if (!*_dontshow && !accepted && _interop_failed && ImGui::BeginPopupModal("Error Using Interop", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::SetItemDefaultFocus();
        ImGui::SetWindowFontScale(2.0f);
        ImGui::Text("This application tries to use CUDA/OpenGL interop.\n"
                    " It did NOT work for your current configuration.\n"
                    " For highest performance, OpenGL and CUDA must run on the same\n"
                    " GPU on an OS that supports interop.You can try to pass a\n"
                    " non-zero index via --device on a multi-GPU system, and/or try\n"
                    " attaching the monitors to the main CUDA card.\n"
                    " On a laptop with one integrated and one dedicated GPU, you can try\n"
                    " to set the preferred GPU via your operating system.\n\n"
                    " FALLING BACK TO SLOWER RENDERING WITH CPU ROUNDTRIP\n");

        ImGui::Separator();

        if (ImGui::Button("  OK  ")) {
            ImGui::CloseCurrentPopup();
            accepted = true;
        }
        ImGui::SameLine();
        ImGui::Checkbox("Don't show this message again", _dontshow);
        ImGui::EndPopup();
    }
}

sibr::GaussianView::~GaussianView()
{
    // Cleanup
    cudaFree(boxmin_cuda);
    cudaFree(boxmax_cuda);

    cudaFree(view_cuda);
    cudaFree(proj_cuda);
    cudaFree(cam_pos_cuda);
    cudaFree(background_cuda);
    cudaFree(rect_cuda);

    if (!_interop_failed) {
        cudaGraphicsUnregisterResource(imageBufferCuda);
    } else {
        cudaFree(fallbackBufferCuda);
    }
    glDeleteBuffers(1, &imageBuffer);

    cudaFree(geomPtr);
    cudaFree(binningPtr);
    cudaFree(imgPtr);

    delete _copyRenderer;
}

CudaRasterizer::UniqueGaussianProperties::~UniqueGaussianProperties()
{
    cudaFree(pos_cuda);
    cudaFree(rot_cuda);
    cudaFree(scale_cuda);
    cudaFree(opacity_cuda);
    cudaFree(shs_cuda);
}
