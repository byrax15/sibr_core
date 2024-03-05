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
#include "ImGuizmo.h"
#include <boost/asio.hpp>
#include <core/graphics/GUI.hpp>
#include <ranges>
#include <rasterizer.h>
#include <thread>

// Define the types and sizes that make up the contents of each Gaussian
// in the trained model.
typedef sibr::Vector3f Pos;
template <int D>
struct SHs {
    float shs[(D + 1) * (D + 1) * 3];
};
struct Scale {
    float scale[3];
};
struct Rot {
    float rot[4];
};
template <int D>
struct RichPoint {
    Pos pos;
    float n[3];
    SHs<D> shs;
    float opacity;
    Scale scale;
    Rot rot;
};

float sigmoid(const float m1)
{
    return 1.0f / (1.0f + exp(-m1));
}

float inverse_sigmoid(const float m1)
{
    return log(m1 / (1.0f - m1));
}

#define CUDA_SAFE_CALL_ALWAYS(A)              \
    A;                                        \
    cudaDeviceSynchronize();                  \
    if (cudaPeekAtLastError() != cudaSuccess) \
        SIBR_ERR << cudaGetErrorString(cudaGetLastError());

#if DEBUG || _DEBUG
#define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
#define CUDA_SAFE_CALL(A) A
#endif

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

    _pointbasedrenderer.reset(new PointBasedRenderer());
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
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&pos_cuda, sizeof(Pos) * P));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_cuda, sizeof(Rot) * P));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&shs_cuda, sizeof(SHs<3>) * P));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs.data(), sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_cuda, sizeof(float) * P));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity.data(), sizeof(float) * P, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_cuda, sizeof(Scale) * P));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, scale.data(), sizeof(Scale) * P, cudaMemcpyHostToDevice));

    boxmin = { _scenemin };
    boxmax = { _scenemax };
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&boxmin_cuda, sizeof(float3) * 16));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(boxmin_cuda, boxmin.data(), sizeof(float3) * 16, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&boxmax_cuda, sizeof(float3) * 16));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(boxmax_cuda, boxmax.data(), sizeof(float3) * 16, cudaMemcpyHostToDevice));

    // Create space for view parameters
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&view_cuda, sizeof(sibr::Matrix4f)));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_cuda, sizeof(sibr::Matrix4f)));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cam_pos_cuda, 3 * sizeof(float)));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * P * sizeof(int)));

    const auto color = white_bg ? 1.f : 0.f;
    Vector3f bg(color, color, color);
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg.data(), sizeof(Vector3f), cudaMemcpyHostToDevice));

    gData = new GaussianData(P,
        (float*)pos.data(),
        (float*)rot.data(),
        (float*)scale.data(),
        opacity.data(),
        (float*)shs.data());

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

sibr::Camera const* last_cam;
void sibr::GaussianView::onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye)
{
    last_cam = &eye;
    if (currMode == "Ellipsoids") {
        _gaussianRenderer->process(count, *gData, eye, dst, 0.2f);
    } else if (currMode == "Initial Points") {
        _pointbasedrenderer->process(_scene->proxies()->proxy(), eye, dst);
    } else {
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
            _fastCulling ? rect_cuda : nullptr,
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
    // Generate and update UI elements
    const std::string guiName = "3D Gaussians";
    if (ImGui::Begin(guiName.c_str())) {
        if (ImGui::BeginCombo("Render Mode", currMode.c_str())) {
            if (ImGui::Selectable("Splats"))
                currMode = "Splats";
            if (ImGui::Selectable("Initial Points"))
                currMode = "Initial Points";
            if (ImGui::Selectable("Ellipsoids"))
                currMode = "Ellipsoids";
            ImGui::EndCombo();
        }
        if (currMode == "Splats") {
            ImGui::SliderFloat("Scaling Modifier", &_scalingModifier, 0.001f, 1.0f);
        }
        ImGui::Checkbox("Fast culling", &_fastCulling);

        const auto MemcpyBoxes = [&] {
            CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(boxmin_cuda, boxmin.data()->data(), sizeof(float3) * boxmin.size(), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(boxmax_cuda, boxmax.data()->data(), sizeof(float3) * boxmax.size(), cudaMemcpyHostToDevice));
        };

        const std::string selected_text = std::to_string(selected_box);
        if (ImGui::Button("Add Box") && boxmin.size() < 16) {
            boxmin.emplace_back(_scenemin);
            boxmax.emplace_back(_scenemax);
            selected_box = boxmin.size() - 1;
            MemcpyBoxes();
        }
        ImGui::SameLine();
        if (ImGui::Button("Remove Box") && boxmin.size() > 1) {
            boxmin.pop_back();
            boxmax.pop_back();
            selected_box = std::max(selected_box - 1, 0);
            ImGui::BeginCombo("Select Crop Box", selected_text.c_str());
        }
        ImGui::SameLine();
        ImGui::Text("Active Boxes: %i/16", boxmin.size());

        if (ImGui::BeginCombo("Select Crop Box", selected_text.c_str())) {
            assert(boxmin.size() == boxmax.size());
            for (int i = 0; i < boxmin.size(); ++i) {
                const auto label = std::to_string(i);
                if (ImGui::Selectable(label.c_str(), selected_box == i)) {
                    selected_box = i;
                }
            }
            ImGui::EndCombo();
        }

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
        if (slid)
            MemcpyBoxes();

        ImGui::Begin("Point view");
        {
            using namespace ImGui;

            ImGuizmo::SetDrawlist();
            ImGuizmo::SetRect(GetWindowPos().x, GetWindowPos().y, GetWindowWidth(), GetWindowHeight());

            using Mat = Matrix4f;
            std::vector<Mat> box_mats(boxmin.size());
            for (auto i = 0; i < boxmin.size(); ++i) {
                Vector3f transl = (boxmin[i] + boxmax[i]) / 2.f, rot = Vector3f::Zero(), scale = boxmax[i] - boxmin[i];
                ImGuizmo::RecomposeMatrixFromComponents(transl.data(), rot.data(), scale.data(), box_mats[i].data());
            }

            ImGuizmo::DrawCubes(
                last_cam->view().data(), last_cam->proj().data(),
                box_mats[0].data(), box_mats.size(),
                ImGuizmo::DrawMode::EDGES);
        }
        ImGui::End();

        if (ImGui::Button("Save...")) {
            std::string fname;
            if (sibr::showFilePicker(fname, sibr::FilePickerMode::Save, "", "ply")) {
                std::vector<Pos> pos(count);
                std::vector<Rot> rot(count);
                std::vector<float> opacity(count);
                std::vector<SHs<3>> shs(count);
                std::vector<Scale> scale(count);
                CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos.data(), pos_cuda, sizeof(Pos) * count, cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot.data(), rot_cuda, sizeof(Rot) * count, cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity.data(), opacity_cuda, sizeof(float) * count, cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs.data(), shs_cuda, sizeof(SHs<3>) * count, cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale.data(), scale_cuda, sizeof(Scale) * count, cudaMemcpyDeviceToHost));
                savePly(fname.c_str(), pos, shs, opacity, scale, rot, boxmin, boxmax, static_cast<FORWARD::Cull::Operator::Value>(selected_operation));
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Load...")) {
            std::string fname;
            if (sibr::showFilePicker(fname, sibr::FilePickerMode::Default, "", "ply")) {
                /*std::vector<Pos> pos;
                std::vector<Rot> rot;
                std::vector<float> opacity;
                std::vector<SHs<3>> shs;
                std::vector<Scale> scale;
                sibr::Vector3f min, max;
                switch (_sh_degree) {
                case 1:
                    loadPly<1>(fname.c_str(), pos, shs, opacity, scale, rot, min, max);
                    break;
                case 2:
                    loadPly<2>(fname.c_str(), pos, shs, opacity, scale, rot, min, max);
                    break;
                case 3:
                    loadPly<3>(fname.c_str(), pos, shs, opacity, scale, rot, min, max);
                    break;
                default:
                    SIBR_LOG << "Unsupported SH degree " << _sh_degree << "\n";
                }*/
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Resize")) {
            static bool reduce = true;
            const float i = 1.f;
            for (const auto& s : scenes) {
                //++i;
                if (reduce)
                    s.for_each<float, sibr::Vector3f>(pos_cuda, [i](sibr::Vector3f& p) { p.y() += i; });
                else
                    s.for_each<float, sibr::Vector3f>(pos_cuda, [i](sibr::Vector3f& p) { p.y() -= i; });
            }
            reduce = !reduce;
        }
    }
    ImGui::End();

    if (!*_dontshow && !accepted && _interop_failed)
        ImGui::OpenPopup("Error Using Interop");

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
    cudaFree(pos_cuda);
    cudaFree(rot_cuda);
    cudaFree(scale_cuda);
    cudaFree(opacity_cuda);
    cudaFree(shs_cuda);

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

    if (geomPtr)
        cudaFree(geomPtr);
    if (binningPtr)
        cudaFree(binningPtr);
    if (imgPtr)
        cudaFree(imgPtr);

    delete _copyRenderer;
}

template <typename CudaT, typename HostT>
std::vector<HostT> sibr::GaussianScene::MemcpyToHost(CudaT const* src_buffer) const
{
    static_assert(sizeof CudaT <= sizeof HostT);
    std::vector<HostT> copy(this->size);
    const auto start_ptr = reinterpret_cast<HostT const*>(src_buffer) + start_index;
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(copy.data(), start_ptr, size * sizeof(HostT), cudaMemcpyDeviceToHost));
    return copy;
}

template <typename CudaT, typename HostT>
void sibr::GaussianScene::MemcpyToDevice(std::vector<HostT> const& upload, CudaT* dst_buffer) const
{
    static_assert(sizeof CudaT <= sizeof HostT);
    constexpr auto size_ratio = sizeof(HostT) / sizeof(CudaT);
    const auto start_ptr = reinterpret_cast<HostT*>(dst_buffer) + start_index;
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(start_ptr, upload.data(), size * sizeof(HostT), cudaMemcpyHostToDevice));
}

template <typename CudaT, typename HostT, typename Callable>
void sibr::GaussianScene::for_each(CudaT* mapped_buffer, Callable&& c) const
{
    auto gauss = MemcpyToHost<CudaT, HostT>(mapped_buffer);
    std::for_each(gauss.begin(), gauss.end(), c);
    MemcpyToDevice<CudaT, HostT>(gauss, mapped_buffer);
}