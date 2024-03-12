#pragma once


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