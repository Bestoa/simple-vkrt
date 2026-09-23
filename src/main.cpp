// ============================================================================
// vk-compute-pathtracer — engine mode.
// Cornell box composed from mesh assets (OBJ + MTL) placed by a scene file.
// Pure Vulkan compute, no ray tracing extensions, static rendering via
// progressive accumulation.
//
//   build:  cmake -B build && cmake --build build -j
//   run:    ./build/rt                          interactive preview (drag/WASD/scroll)
//           ./build/rt --offscreen 2048 --size 1920x1080   -> out.png
//           options: --scene path --size WxH --ires WxH --validate --novsync
//
// Engine data flow:
//   scene.txt -> entities -> OBJ/MTL load -> transform bake -> triangle soup
//             -> SAH BVH -> SSBOs -> compute path tracer (materials are data)
//   emissive triangles -> area-weighted light list (NEE direct sampling)
// ============================================================================

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ------------------------------------------------------------------- tiny math
struct v3 {
    float x, y, z;
};
static v3 operator+(v3 a, v3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static v3 operator-(v3 a, v3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static v3 operator*(v3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static v3 operator*(float s, v3 a) { return a * s; }
static v3 operator-(v3 a) { return {-a.x, -a.y, -a.z}; }
static float dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static v3 cross(v3 a, v3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static float len(v3 a) { return std::sqrt(dot(a, a)); }
static v3 norm(v3 a) { return a * (1.0f / std::max(len(a), 1e-12f)); }
static v3 vmin(v3 a, v3 b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
static v3 vmax(v3 a, v3 b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }

// ------------------------------------------------------------- GPU data layout
struct GPUTri {
    float v0[4], e1[4], e2[4], nm[4];  // nm.xyz = geometric normal, nm.w = material index
    float n0[4], n1[4], n2[4];         // per-vertex shading normals (smooth; fall back to nm)
};
struct GPUNode {
    float bmin[4], bmax[4];  // bmin.w = left child / first tri, bmax.w = tri count
};
struct GPUMaterial {
    float albType[4];   // rgb albedo, w = type
    float emitIor[4];   // rgb emission, w = IOR
};
struct GPULightTri {
    float data[4];  // x = tri index, y = cumulative pdf, z = area
};
struct UBOData {  // must match the Params block in pathtrace.comp (std140)
    float camPos[4], camRight[4], camUp[4], camFwd[4];
    float res[4];  // w, h, accumulated frames, tan(fov/2)
};

enum MatType { MAT_DIFFUSE = 0, MAT_MIRROR = 1, MAT_GLASS = 2, MAT_LIGHT = 3 };

struct Material {
    v3 albedo{0.73f, 0.73f, 0.73f};
    v3 emission{0, 0, 0};
    int type = MAT_DIFFUSE;
    float ior = 1.5f;
};
static float matLuma(const Material& m) {
    return 0.2126f * m.emission.x + 0.7152f * m.emission.y + 0.0722f * m.emission.z;
}

#define VKCHECK(x)                                                              \
    do {                                                                        \
        VkResult _r = (x);                                                      \
        if (_r != VK_SUCCESS) {                                                 \
            fprintf(stderr, "Vulkan error %d at %s:%d\n", _r, __FILE__, __LINE__); \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// --------------------------------------------------------------- OBJ + MTL load
struct ObjTri {
    int a, b, c;        // vertex indices
    int na, nb, nc;     // vertex-normal indices (-1 = use geometric face normal)
    int mtl;            // material index into model mats (-1 = none)
};
struct ObjModel {
    std::vector<v3> verts;
    std::vector<v3> vnorms;
    std::vector<ObjTri> tris;
    std::vector<Material> mats;
};

static std::string dirOf(const std::string& path) {
    size_t s = path.find_last_of("/\\");
    return (s == std::string::npos) ? "." : path.substr(0, s);
}

static std::vector<std::pair<std::string, Material>> parseMTL(const std::string& path) {
    std::vector<std::pair<std::string, Material>> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string line, cur = "";
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tok;
        if (!(ss >> tok)) continue;
        if (tok == "newmtl") {
            ss >> cur;
            out.push_back({cur, Material{}});
        } else if (!out.empty()) {
            Material& m = out.back().second;
            if (tok == "Kd") ss >> m.albedo.x >> m.albedo.y >> m.albedo.z;
            else if (tok == "Ke") ss >> m.emission.x >> m.emission.y >> m.emission.z;
            else if (tok == "Ni") ss >> m.ior;
            else if (tok == "d") {
                float d = 1; ss >> d;
                if (d < 0.999f) m.type = MAT_GLASS;
            } else if (tok == "Tr") {
                float t = 0; ss >> t;
                if (t > 0.001f) m.type = MAT_GLASS;
            }
        }
    }
    for (auto& [name, m] : out)
        if (matLuma(m) > 1e-3f) m.type = MAT_LIGHT;
    return out;
}

static bool loadOBJ(const std::string& path, ObjModel& model) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    std::map<std::string, int> matIdx;
    int curMat = -1;
    std::string line;
    auto vertIndex = [&](long i, size_t n) -> int {  // OBJ 1-based, negative = relative
        if (i > 0) return (int)(i - 1);
        return (int)((long)n + i);
    };
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tok;
        if (!(ss >> tok)) continue;
        if (tok == "v") {
            v3 p{};
            ss >> p.x >> p.y >> p.z;
            model.verts.push_back(p);
        } else if (tok == "vn") {
            v3 n{};
            ss >> n.x >> n.y >> n.z;
            model.vnorms.push_back(norm(n));
        } else if (tok == "mtllib") {
            std::string lib;
            ss >> lib;
            for (auto& [name, m] : parseMTL(dirOf(path) + "/" + lib)) {
                matIdx[name] = (int)model.mats.size();
                model.mats.push_back(m);
            }
        } else if (tok == "usemtl") {
            std::string name;
            ss >> name;
            auto it = matIdx.find(name);
            curMat = (it == matIdx.end()) ? -1 : it->second;
            if (curMat < 0) {  // unknown material: create it
                matIdx[name] = (int)model.mats.size();
                model.mats.push_back(Material{});
                curMat = (int)model.mats.size() - 1;
            }
        } else if (tok == "f") {
            std::vector<int> idx, nidx;
            std::string v;
            while (ss >> v) {
                // formats: "v", "v/vt", "v//vn", "v/vt/vn"
                long vi = strtol(v.c_str(), nullptr, 10);
                idx.push_back(vertIndex(vi, model.verts.size()));
                int ni = -1;
                size_t p1 = v.find('/');
                if (p1 != std::string::npos) {
                    size_t p2 = v.find('/', p1 + 1);
                    if (p2 != std::string::npos && p2 + 1 < v.size()) {
                        long vn = strtol(v.c_str() + p2 + 1, nullptr, 10);
                        if (vn != 0) ni = vertIndex(vn, model.vnorms.size());
                    }
                }
                nidx.push_back(ni);
            }
            for (size_t k = 1; k + 1 < idx.size(); ++k)  // fan triangulation
                model.tris.push_back({idx[0], idx[k], idx[k + 1],
                                      nidx[0], nidx[k], nidx[k + 1], curMat});
        }
        // vt / vn / s / o / g ... ignored (we use geometric face normals)
    }
    return !model.verts.empty() && !model.tris.empty();
}

// ------------------------------------------------------------------ scene file
struct Entity {
    std::string meshPath;          // OBJ path, or "__sphere__" for the native sphere primitive
    v3 pos{0, 0, 0};
    float rotYdeg = 0;
    v3 scale{1, 1, 1};
    bool hasOverride = false;
    Material overrideMat;
};

// Native sphere primitive (scene-level convenience): tessellated ONCE into an
// in-memory unit-sphere mesh with smooth vertex normals, then instanced through
// the same transform/bake/BVH pipeline as any other mesh. One intersection
// path in the renderer, no special-casing in the shader.
static ObjModel makeUnitSphere(int rings = 32, int sectors = 64) {
    ObjModel m;
    for (int r = 0; r <= rings; ++r) {
        float phi = 3.14159265f * r / rings;
        for (int s = 0; s <= sectors; ++s) {
            float th = 2 * 3.14159265f * s / sectors;
            v3 p{std::sin(phi) * std::cos(th), std::cos(phi), std::sin(phi) * std::sin(th)};
            m.verts.push_back(p);
            m.vnorms.push_back(p);  // unit sphere: normal == position
        }
    }
    int S = sectors + 1;
    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < sectors; ++s) {
            int a = r * S + s, b = a + S;
            if (r != 0) m.tris.push_back({a, b, a + 1, a, b, a + 1, -1});
            if (r != rings - 1) m.tris.push_back({a + 1, b, b + 1, a + 1, b, b + 1, -1});
        }
    }
    return m;
}

static bool parseMaterialTokens(std::istringstream& ss, Material& m, bool& has) {
    std::string kind;
    if (!(ss >> kind)) return false;
    has = true;
    if (kind == "auto") { has = false; return true; }
    if (kind == "diffuse") m.type = MAT_DIFFUSE;
    else if (kind == "mirror") m.type = MAT_MIRROR;
    else if (kind == "light") m.type = MAT_LIGHT;
    else if (kind == "glass") m.type = MAT_GLASS;
    else return false;
    ss >> m.albedo.x >> m.albedo.y >> m.albedo.z;
    if (m.type == MAT_LIGHT) m.emission = m.albedo;
    if (m.type == MAT_GLASS && ss >> m.ior) {}
    return true;
}

static bool loadScene(const std::string& path, std::vector<Entity>& entities) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open scene %s\n", path.c_str()); return false; }
    std::string line;
    while (std::getline(f, line)) {
        size_t h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        std::istringstream ss(line);
        std::string first;
        if (!(ss >> first)) continue;
        Entity e;
        bool ok = false;
        if (first == "sphere") {
            // sphere  cx cy cz  radius  material
            float r = 1;
            ok = (bool)(ss >> e.pos.x >> e.pos.y >> e.pos.z >> r);
            if (ok) {
                e.meshPath = "__sphere__";
                e.scale = {r, r, r};
                parseMaterialTokens(ss, e.overrideMat, e.hasOverride);
            }
        } else if (first.size() > 4 && first.compare(first.size() - 4, 4, ".obj") == 0) {
            e.meshPath = first;
            ok = (bool)(ss >> e.pos.x >> e.pos.y >> e.pos.z >> e.rotYdeg
                        >> e.scale.x >> e.scale.y >> e.scale.z);
            if (ok) parseMaterialTokens(ss, e.overrideMat, e.hasOverride);
        }
        if (!ok) {
            if (!first.empty() && first != "#")
                fprintf(stderr, "bad scene line: %s\n", line.c_str());
            continue;
        }
        entities.push_back(e);
    }
    return !entities.empty();
}

// ------------------------------------------------------- BVH (binned SAH) build
struct BVHBuilder {
    std::vector<GPUTri> tris;
    std::vector<GPUNode> nodes;
    std::vector<v3> cen, bmin, bmax;
    std::vector<int> indices;
    static const int BINS = 16;

    int allocNode() {
        GPUNode n{};
        nodes.push_back(n);
        return (int)nodes.size() - 1;
    }
    void build() {
        nodes.clear();
        nodes.reserve(tris.size() * 2);
        size_t n = tris.size();
        cen.resize(n); bmin.resize(n); bmax.resize(n);
        indices.resize(n);
        for (size_t i = 0; i < n; ++i) {
            v3 a{tris[i].v0[0], tris[i].v0[1], tris[i].v0[2]};
            v3 b = a + v3{tris[i].e1[0], tris[i].e1[1], tris[i].e1[2]};
            v3 c = a + v3{tris[i].e2[0], tris[i].e2[1], tris[i].e2[2]};
            bmin[i] = vmin(a, vmin(b, c));
            bmax[i] = vmax(a, vmax(b, c));
            cen[i] = (bmin[i] + bmax[i]) * 0.5f;
            indices[i] = (int)i;
        }
        recurse(allocNode(), 0, (int)n, 0);
        std::vector<GPUTri> out(n);
        for (size_t i = 0; i < n; ++i) out[i] = tris[indices[i]];
        tris.swap(out);
    }
    void recurse(int nodei, int first, int count, int depth) {
        v3 mn = bmin[indices[first]], mx = bmax[indices[first]];
        v3 cmn = cen[indices[first]], cmx = cen[indices[first]];
        for (int i = first + 1; i < first + count; ++i) {
            mn = vmin(mn, bmin[indices[i]]);
            mx = vmax(mx, bmax[indices[i]]);
            cmn = vmin(cmn, cen[indices[i]]);
            cmx = vmax(cmx, cen[indices[i]]);
        }
        for (int k = 0; k < 3; ++k) { nodes[nodei].bmin[k] = (&mn.x)[k]; nodes[nodei].bmax[k] = (&mx.x)[k]; }
        if (count <= 4 || depth > 40) {
            nodes[nodei].bmin[3] = (float)first;
            nodes[nodei].bmax[3] = (float)count;
            return;
        }
        float e[3] = {cmx.x - cmn.x, cmx.y - cmn.y, cmx.z - cmn.z};
        int axis = (e[0] > e[1] && e[0] > e[2]) ? 0 : (e[1] > e[2] ? 1 : 2);
        if (e[axis] < 1e-8f) {
            nodes[nodei].bmin[3] = (float)first;
            nodes[nodei].bmax[3] = (float)count;
            return;
        }
        float k1 = BINS / e[axis];
        float lo = axis == 0 ? cmn.x : (axis == 1 ? cmn.y : cmn.z);
        auto AX = [&](int i) { return axis == 0 ? cen[i].x : (axis == 1 ? cen[i].y : cen[i].z); };
        int binCount[BINS] = {0};
        v3 binMin[BINS], binMax[BINS];
        for (int b = 0; b < BINS; ++b) { binMin[b] = v3{1e30f, 1e30f, 1e30f}; binMax[b] = v3{-1e30f, -1e30f, -1e30f}; }
        for (int i = first; i < first + count; ++i) {
            int bi = (int)std::min((float)(BINS - 1), k1 * (AX(indices[i]) - lo));
            binCount[bi]++;
            binMin[bi] = vmin(binMin[bi], bmin[indices[i]]);
            binMax[bi] = vmax(binMax[bi], bmax[indices[i]]);
        }
        auto area = [](v3 a, v3 b) {
            v3 ex = b - a;
            return 2.0f * (ex.x * ex.y + ex.y * ex.z + ex.z * ex.x) + 1e-9f;
        };
        float leftArea[BINS], leftCount[BINS];
        v3 lmn = v3{1e30f, 1e30f, 1e30f}, lmx = v3{-1e30f, -1e30f, -1e30f};
        float lc = 0;
        for (int b = 0; b < BINS; ++b) {
            lmn = vmin(lmn, binMin[b]);
            lmx = vmax(lmx, binMax[b]);
            lc += binCount[b];
            leftArea[b] = area(lmn, lmx);
            leftCount[b] = lc;
        }
        float bestCost = 1e30f;
        int bestSplit = -1;
        float nodeArea = area(mn, mx);
        v3 rmn = v3{1e30f, 1e30f, 1e30f}, rmx = v3{-1e30f, -1e30f, -1e30f};
        float rc = 0;
        for (int b = BINS - 1; b > 0; --b) {
            rmn = vmin(rmn, binMin[b]);
            rmx = vmax(rmx, binMax[b]);
            rc += binCount[b];
            float cost = leftArea[b - 1] * leftCount[b - 1] + area(rmn, rmx) * rc;
            if (cost < bestCost) { bestCost = cost; bestSplit = b - 1; }
        }
        auto splitMedian = [&]() {
            int mid = count / 2;
            std::nth_element(indices.begin() + first, indices.begin() + first + mid,
                             indices.begin() + first + count,
                             [&](int a, int b) { return AX(a) < AX(b); });
            return mid;
        };
        int mid;
        if (bestSplit < 0 || bestCost >= count * nodeArea) {
            mid = splitMedian();
        } else {
            float thresh = lo + (bestSplit + 1) / k1;
            int* beg = &indices[first];
            int* end = &indices[first + count];
            int* midp = std::partition(beg, end, [&](int i) { return AX(i) < thresh; });
            mid = (int)(midp - beg);
            if (mid == 0 || mid == count) mid = splitMedian();
        }
        int L = allocNode(), R = allocNode();
        nodes[nodei].bmin[3] = (float)L;
        nodes[nodei].bmax[3] = 0;
        recurse(L, first, mid, depth + 1);
        recurse(R, first + mid, count - mid, depth + 1);
    }
};

// ------------------------------------------------------------------ PNG writer
static void be32(FILE* f, uint32_t v) {
    uint8_t b[4] = {(uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v};
    fwrite(b, 1, 4, f);
}
static void pngChunk(FILE* f, const char* type, const uint8_t* data, uint32_t ln2) {
    be32(f, ln2);
    fwrite(type, 1, 4, f);
    if (ln2) fwrite(data, 1, ln2, f);
    uint32_t crc = (uint32_t)crc32(0, (const Bytef*)type, 4);
    if (ln2) crc = (uint32_t)crc32(crc, data, ln2);
    be32(f, crc);
}
static bool writePNG(const char* path, int w, int h, const uint8_t* rgba) {
    std::vector<uint8_t> raw((size_t)h * (1 + (size_t)w * 4));
    for (int y = 0; y < h; ++y) {
        raw[(size_t)y * (1 + (size_t)w * 4)] = 0;
        memcpy(&raw[(size_t)y * (1 + (size_t)w * 4) + 1], rgba + (size_t)y * w * 4, (size_t)w * 4);
    }
    uLongf compSz = compressBound(raw.size());
    std::vector<uint8_t> comp(compSz);
    if (compress2(comp.data(), &compSz, raw.data(), (uLong)raw.size(), 6) != Z_OK) return false;
    comp.resize(compSz);
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    fwrite(sig, 1, 8, f);
    uint8_t ihdr[13];
    ihdr[0] = (uint8_t)(w >> 24); ihdr[1] = (uint8_t)(w >> 16); ihdr[2] = (uint8_t)(w >> 8); ihdr[3] = (uint8_t)w;
    ihdr[4] = (uint8_t)(h >> 24); ihdr[5] = (uint8_t)(h >> 16); ihdr[6] = (uint8_t)(h >> 8); ihdr[7] = (uint8_t)h;
    ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    pngChunk(f, "IHDR", ihdr, 13);
    pngChunk(f, "IDAT", comp.data(), (uint32_t)compSz);
    pngChunk(f, "IEND", nullptr, 0);
    fclose(f);
    return true;
}

// ------------------------------------------------------------------ Vulkan app
struct App {
    bool offscreen = false, validate = false, novsync = false;
    std::string scenePath = ASSET_DIR "scene.txt";
    int winW = 1280, winH = 720;
    int iw = 1280, ih = 720;

    GLFWwindow* window = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkQueue queue = VK_NULL_HANDLE;

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapImages;
    std::vector<VkImageView> swapViews;
    VkFormat swapFormat = VK_FORMAT_B8G8R8A8_UNORM;

    VkBuffer triBuf = VK_NULL_HANDLE, nodeBuf = VK_NULL_HANDLE, matBuf = VK_NULL_HANDLE, lightBuf = VK_NULL_HANDLE;
    VkDeviceMemory triMem = VK_NULL_HANDLE, nodeMem = VK_NULL_HANDLE, matMem = VK_NULL_HANDLE, lightMem = VK_NULL_HANDLE;
    VkBuffer accumBuf = VK_NULL_HANDLE, uboBuf = VK_NULL_HANDLE;
    VkDeviceMemory accumMem = VK_NULL_HANDLE, uboMem = VK_NULL_HANDLE;
    void* uboMapped = nullptr;
    // offline denoiser resources
    bool denoise = true;
    int snapSpp = 4096;   // P-key snapshot sample count (--snap N to change)
    VkBuffer gbufBuf = VK_NULL_HANDLE, momBuf = VK_NULL_HANDLE, scratchBuf = VK_NULL_HANDLE, resUboBuf = VK_NULL_HANDLE;
    VkDeviceMemory gbufMem = VK_NULL_HANDLE, momMem = VK_NULL_HANDLE, scratchMem = VK_NULL_HANDLE, resUboMem = VK_NULL_HANDLE;
    void* resUboMapped = nullptr;
    VkPipelineLayout pipeDenoiseLayout = VK_NULL_HANDLE, pipeFinalLayout = VK_NULL_HANDLE;
    VkPipeline pipeDenoise = VK_NULL_HANDLE, pipeFinal = VK_NULL_HANDLE;
    VkDescriptorSetLayout descDenoiseLayout = VK_NULL_HANDLE, descFinalLayout = VK_NULL_HANDLE;
    VkDescriptorSet sDenoiseAB = VK_NULL_HANDLE, sDenoiseBA = VK_NULL_HANDLE, sFinal = VK_NULL_HANDLE;

    VkImage displayImage = VK_NULL_HANDLE;
    VkDeviceMemory displayMem = VK_NULL_HANDLE;
    VkImageView displayView = VK_NULL_HANDLE;

    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore semAcquire = VK_NULL_HANDLE, semRender = VK_NULL_HANDLE;

    // world
    std::vector<GPUTri> tris;
    std::vector<GPUNode> bvhNodes;
    std::vector<GPUMaterial> materials;
    std::vector<GPULightTri> lightTris;
    std::vector<Material> matTable;  // host-side copy for light picking

    uint64_t frame = 0;
    bool scrollMoved = false;  // zoom must restart accumulation too (no ghosting)
    float yaw = 0.0f, pitch = 0.02f, dist = 8.0f;  // 8.0 + fov45: box ~65% frame width, full height
    v3 target{0.0f, 2.5f, 0.0f};

    // ------------------------------------------------------------- helpers
    uint32_t findMemory(uint32_t typeBits, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(physical, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
        return 0xFFFFFFFF;
    }
    void makeBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                    VkBuffer& buf, VkDeviceMemory& mem, void** mapped = nullptr, bool zero = false) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = usage;
        VKCHECK(vkCreateBuffer(device, &bi, nullptr, &buf));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, buf, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemory(req.memoryTypeBits, props);
        VKCHECK(vkAllocateMemory(device, &ai, nullptr, &mem));
        VKCHECK(vkBindBufferMemory(device, buf, mem, 0));
        if (mapped) VKCHECK(vkMapMemory(device, mem, 0, size, 0, mapped));
        if (zero) {
            void* p = mapped ? *mapped : nullptr;
            bool unmap = false;
            if (!p) { VKCHECK(vkMapMemory(device, mem, 0, size, 0, &p)); unmap = true; }
            memset(p, 0, (size_t)size);
            if (unmap) vkUnmapMemory(device, mem);
        }
    }
    void uploadBuffer(VkDeviceMemory mem, const void* data, size_t size) {
        void* p = nullptr;
        VKCHECK(vkMapMemory(device, mem, 0, size, 0, &p));
        memcpy(p, data, size);
        vkUnmapMemory(device, mem);
    }

    // ------------------------------------------------------------- world build
    void buildWorld() {
        std::vector<Entity> entities;
        if (!loadScene(scenePath, entities)) { fprintf(stderr, "empty scene\n"); exit(1); }

        std::map<std::string, ObjModel> meshCache;
        for (auto& e : entities) {
            auto it = meshCache.find(e.meshPath);
            if (it == meshCache.end()) {
                ObjModel m;
                if (e.meshPath == "__sphere__") m = makeUnitSphere();
                else {
                    std::string path = dirOf(scenePath) + "/" + e.meshPath;
                    if (!loadOBJ(path, m)) { fprintf(stderr, "failed to load %s\n", path.c_str()); exit(1); }
                }
                it = meshCache.emplace(e.meshPath, std::move(m)).first;
            }
            const ObjModel& mesh = it->second;

            // material indices: model materials appended globally, then override
            int base = (int)matTable.size();
            for (auto& m : mesh.mats) matTable.push_back(m);
            int matIdx = base;  // model default (first material) if tri has none
            if (e.hasOverride) {
                matTable.push_back(e.overrideMat);
                matIdx = (int)matTable.size() - 1;
            }

            // transform: scale -> rotate Y -> translate; normals rotate only
            float ca = std::cos(e.rotYdeg * 3.14159265f / 180.0f);
            float sa = std::sin(e.rotYdeg * 3.14159265f / 180.0f);
            auto xf = [&](v3 v) {
                v = v3{v.x * e.scale.x, v.y * e.scale.y, v.z * e.scale.z};
                v = v3{ca * v.x - sa * v.z, v.y, sa * v.x + ca * v.z};
                return v + e.pos;
            };
            auto xfn = [&](v3 n) {
                return norm(v3{ca * n.x - sa * n.z, n.y, sa * n.x + ca * n.z});
            };

            size_t start = tris.size();
            for (auto& t : mesh.tris) {
                v3 a = xf(mesh.verts[t.a]);
                v3 b = xf(mesh.verts[t.b]);
                v3 c = xf(mesh.verts[t.c]);
                v3 sn[3];
                sn[0] = (t.na >= 0 && (size_t)t.na < mesh.vnorms.size()) ? xfn(mesh.vnorms[t.na]) : v3{0, 0, 0};
                sn[1] = (t.nb >= 0 && (size_t)t.nb < mesh.vnorms.size()) ? xfn(mesh.vnorms[t.nb]) : v3{0, 0, 0};
                sn[2] = (t.nc >= 0 && (size_t)t.nc < mesh.vnorms.size()) ? xfn(mesh.vnorms[t.nc]) : v3{0, 0, 0};
                v3 n = norm(cross(b - a, c - a));
                if (sn[0].x == 0 && sn[0].y == 0 && sn[0].z == 0) {
                    sn[0] = sn[1] = sn[2] = n;  // no vn in model: flat shading
                } else if (dot(n, sn[0] + sn[1] + sn[2]) < 0) {
                    // winding disagrees with vertex normals: flip the triangle so
                    // the geometric normal shares a hemisphere with shading normals
                    std::swap(b, c);
                    std::swap(sn[1], sn[2]);
                    n = -n;
                }
                int mi = (t.mtl >= 0 && !e.hasOverride) ? base + t.mtl : matIdx;
                GPUTri g{};
                g.v0[0] = a.x; g.v0[1] = a.y; g.v0[2] = a.z;
                g.e1[0] = b.x - a.x; g.e1[1] = b.y - a.y; g.e1[2] = b.z - a.z;
                g.e2[0] = c.x - a.x; g.e2[1] = c.y - a.y; g.e2[2] = c.z - a.z;
                g.nm[0] = n.x; g.nm[1] = n.y; g.nm[2] = n.z; g.nm[3] = (float)mi;
                g.n0[0] = sn[0].x; g.n0[1] = sn[0].y; g.n0[2] = sn[0].z;
                g.n1[0] = sn[1].x; g.n1[1] = sn[1].y; g.n1[2] = sn[1].z;
                g.n2[0] = sn[2].x; g.n2[1] = sn[2].y; g.n2[2] = sn[2].z;
                tris.push_back(g);
            }
            printf("entity %-22s %6zu tris  (pos %.2f %.2f %.2f, rot %.0f, scale %.2f %.2f %.2f, %s)\n",
                   e.meshPath.c_str(), tris.size() - start, e.pos.x, e.pos.y, e.pos.z, e.rotYdeg,
                   e.scale.x, e.scale.y, e.scale.z, e.hasOverride ? "material override" : "mtl");
        }

        // BVH over the whole triangle soup
        BVHBuilder b;
        b.tris = tris;
        b.build();
        tris = b.tris;
        bvhNodes = b.nodes;

        // GPU materials
        materials.reserve(matTable.size());
        for (auto& m : matTable) {
            GPUMaterial g{};
            g.albType[0] = m.albedo.x; g.albType[1] = m.albedo.y; g.albType[2] = m.albedo.z;
            g.albType[3] = (float)m.type;
            g.emitIor[0] = m.emission.x; g.emitIor[1] = m.emission.y; g.emitIor[2] = m.emission.z;
            g.emitIor[3] = m.ior;
            materials.push_back(g);
        }

        // emissive triangle list with area-weighted CDF
        float cdf = 0;
        for (size_t i = 0; i < tris.size(); ++i) {
            int mi = (int)(tris[i].nm[3] + 0.5f);
            if (matLuma(matTable[mi]) < 1e-3f) continue;
            v3 e1{tris[i].e1[0], tris[i].e1[1], tris[i].e1[2]};
            v3 e2{tris[i].e2[0], tris[i].e2[1], tris[i].e2[2]};
            float area = 0.5f * len(cross(e1, e2));
            cdf += area * matLuma(matTable[mi]);
            GPULightTri lt{};
            lt.data[0] = (float)i;
            lt.data[1] = cdf;
            lt.data[2] = area;
            lightTris.push_back(lt);
        }
        printf("world: %zu tris, %zu BVH nodes, %zu materials, %zu light triangles (cdf %.2f)\n",
               tris.size(), bvhNodes.size(), materials.size(), lightTris.size(), cdf);
        if (lightTris.empty()) { fprintf(stderr, "scene has no emissive triangles — nothing to see\n"); exit(1); }
    }

    // ------------------------------------------------------------- vulkan init
    void initVulkan() {
        bool hasGui = glfwInit() == GLFW_TRUE;   // headless servers: render without a preview
        if (!hasGui) fprintf(stderr, "note: glfw unavailable, running headless\n");

        std::vector<const char*> exts;
        std::vector<const char*> layers;
        if (hasGui) {   // surface extensions serve the interactive window AND offscreen preview
            uint32_t gc = 0;
            const char** ge = glfwGetRequiredInstanceExtensions(&gc);
            if (ge)
                for (uint32_t i = 0; i < gc; ++i) exts.push_back(ge[i]);
        }
        VkInstanceCreateFlags flags = 0;
        {
            uint32_t c = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &c, nullptr);
            std::vector<VkExtensionProperties> ep(c);
            vkEnumerateInstanceExtensionProperties(nullptr, &c, ep.data());
            bool portability = false, dbg = false;
            for (auto& e : ep) {
                if (!strcmp(e.extensionName, "VK_KHR_portability_enumeration")) portability = true;
                if (!strcmp(e.extensionName, "VK_EXT_debug_utils")) dbg = true;
            }
            if (portability) { exts.push_back("VK_KHR_portability_enumeration"); flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR; }
            if (validate && dbg) {
                uint32_t lc = 0;
                vkEnumerateInstanceLayerProperties(&lc, nullptr);
                std::vector<VkLayerProperties> lps(lc);
                vkEnumerateInstanceLayerProperties(&lc, lps.data());
                for (auto& l : lps)
                    if (!strcmp(l.layerName, "VK_LAYER_KHRONOS_validation")) {
                        exts.push_back("VK_EXT_debug_utils");
                        layers.push_back("VK_LAYER_KHRONOS_validation");
                        break;
                    }
                if (layers.empty()) printf("note: validation layer not installed, skipping\n");
            }
        }
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "vk-compute-pathtracer";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.flags = flags;
        ci.pApplicationInfo = &app;
        ci.enabledExtensionCount = (uint32_t)exts.size();
        ci.ppEnabledExtensionNames = exts.data();
        ci.enabledLayerCount = (uint32_t)layers.size();
        ci.ppEnabledLayerNames = layers.data();
        VKCHECK(vkCreateInstance(&ci, nullptr, &instance));

        if (!layers.empty()) {
            auto create = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
                instance, "vkCreateDebugUtilsMessengerEXT");
            if (create) {
                VkDebugUtilsMessengerCreateInfoEXT mi{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
                mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
                mi.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
                                        const VkDebugUtilsMessengerCallbackDataEXT* d, void*) -> VkBool32 {
                    fprintf(stderr, "[validation] %s\n", d->pMessage);
                    return VK_FALSE;
                };
                create(instance, &mi, nullptr, &messenger);
            }
        }

        uint32_t dc = 0;
        VKCHECK(vkEnumeratePhysicalDevices(instance, &dc, nullptr));
        std::vector<VkPhysicalDevice> devs(dc);
        VKCHECK(vkEnumeratePhysicalDevices(instance, &dc, devs.data()));
        int best = -1, bestScore = -1;
        for (uint32_t i = 0; i < dc; ++i) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devs[i], &props);
            int score = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)     ? 300
                        : (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? 200
                        : (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)    ? 100
                                                                                        : 0;
            if (score > bestScore) { bestScore = score; best = (int)i; }
        }
        physical = devs[best];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physical, &props);
        uint32_t dec = 0;
        vkEnumerateDeviceExtensionProperties(physical, nullptr, &dec, nullptr);
        std::vector<VkExtensionProperties> dep(dec);
        vkEnumerateDeviceExtensionProperties(physical, nullptr, &dec, dep.data());
        bool hasPortabilitySubset = false, hasDriverProps = false;
        for (auto& e : dep) {
            if (!strcmp(e.extensionName, "VK_KHR_portability_subset")) hasPortabilitySubset = true;
            if (!strcmp(e.extensionName, "VK_KHR_driver_properties")) hasDriverProps = true;
        }
        if (hasDriverProps) {
            VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
            VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            p2.pNext = &drv;
            vkGetPhysicalDeviceProperties2(physical, &p2);
            printf("GPU: %s (%s)\n", props.deviceName, drv.driverName);
        } else {
            printf("GPU: %s\n", props.deviceName);
        }
        printf("Vulkan: %u.%u.%u | render res %dx%d | scene %s\n", VK_VERSION_MAJOR(props.apiVersion),
               VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion), iw, ih, scenePath.c_str());

        uint32_t qc = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &qc, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qc);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &qc, qf.data());
        bool foundQ = false;
        for (uint32_t i = 0; i < qc; ++i) {
            bool ok = (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT);
            if (ok) ok = glfwGetPhysicalDevicePresentationSupport(instance, physical, i) == GLFW_TRUE;
            if (ok) { queueFamily = i; foundQ = true; break; }
        }
        if (!foundQ) { fprintf(stderr, "no compute+graphics queue found\n"); exit(1); }

        std::vector<const char*> devExts;
        devExts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);   // preview window uses it too
        if (hasPortabilitySubset) devExts.push_back("VK_KHR_portability_subset");
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = queueFamily;
        qi.queueCount = 1;
        qi.pQueuePriorities = &prio;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        di.enabledExtensionCount = (uint32_t)devExts.size();
        di.ppEnabledExtensionNames = devExts.data();
        VKCHECK(vkCreateDevice(physical, &di, nullptr, &device));
        vkGetDeviceQueue(device, queueFamily, 0, &queue);

        // ---- world buffers
        makeBuffer(tris.size() * sizeof(GPUTri), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, triBuf, triMem);
        uploadBuffer(triMem, tris.data(), tris.size() * sizeof(GPUTri));
        makeBuffer(bvhNodes.size() * sizeof(GPUNode), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, nodeBuf, nodeMem);
        uploadBuffer(nodeMem, bvhNodes.data(), bvhNodes.size() * sizeof(GPUNode));
        makeBuffer(materials.size() * sizeof(GPUMaterial), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, matBuf, matMem);
        uploadBuffer(matMem, materials.data(), materials.size() * sizeof(GPUMaterial));
        makeBuffer(std::max<size_t>(lightTris.size(), 1) * sizeof(GPULightTri), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, lightBuf, lightMem);
        if (!lightTris.empty()) uploadBuffer(lightMem, lightTris.data(), lightTris.size() * sizeof(GPULightTri));
        makeBuffer(sizeof(UBOData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uboBuf, uboMem, &uboMapped);
        makeBuffer((VkDeviceSize)iw * ih * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   accumBuf, accumMem, nullptr, true);
        // denoiser resources: G-buffer, moments, scratch color, small res-UBO
        makeBuffer((VkDeviceSize)iw * ih * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   gbufBuf, gbufMem, nullptr, true);
        makeBuffer((VkDeviceSize)iw * ih * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   momBuf, momMem, nullptr, true);
        makeBuffer((VkDeviceSize)iw * ih * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   scratchBuf, scratchMem, nullptr, true);
        makeBuffer(16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   resUboBuf, resUboMem, &resUboMapped);
        ((int*)resUboMapped)[0] = iw;
        ((int*)resUboMapped)[1] = ih;

        // ---- display image
        VkImageCreateInfo ii2{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii2.imageType = VK_IMAGE_TYPE_2D;
        ii2.format = VK_FORMAT_R8G8B8A8_UNORM;
        ii2.extent = {(uint32_t)iw, (uint32_t)ih, 1};
        ii2.mipLevels = 1;
        ii2.arrayLayers = 1;
        ii2.samples = VK_SAMPLE_COUNT_1_BIT;
        ii2.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii2.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ii2.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VKCHECK(vkCreateImage(device, &ii2, nullptr, &displayImage));
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, displayImage, &req);
        uint32_t type = findMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == 0xFFFFFFFF) type = findMemory(req.memoryTypeBits, 0);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        VKCHECK(vkAllocateMemory(device, &ai, nullptr, &displayMem));
        VKCHECK(vkBindImageMemory(device, displayImage, displayMem, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = displayImage;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_R8G8B8A8_UNORM;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VKCHECK(vkCreateImageView(device, &vi, nullptr, &displayView));

        // ---- shader / pipeline / descriptors
        std::string spvPath = SHADER_DIR "pathtrace.comp.spv";
        FILE* f = fopen(spvPath.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", spvPath.c_str()); exit(1); }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<char> code(sz);
        if (fread(code.data(), 1, sz, f) != (size_t)sz) exit(1);
        fclose(f);
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize = code.size();
        sm.pCode = (const uint32_t*)code.data();
        VkShaderModule module;
        VKCHECK(vkCreateShaderModule(device, &sm, nullptr, &module));

        const VkShaderStageFlags cs = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutBinding b[9] = {
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, cs, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr},
            {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr},
            {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr},
            {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, cs, nullptr},
            {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr},
            {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr}};
        VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dl.bindingCount = 9;
        dl.pBindings = b;
        VKCHECK(vkCreateDescriptorSetLayout(device, &dl, nullptr, &descLayout));

        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &descLayout;
        VKCHECK(vkCreatePipelineLayout(device, &pl, nullptr, &pipelineLayout));

        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module = module;
        cp.stage.pName = "main";
        cp.layout = pipelineLayout;
        VKCHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp, nullptr, &pipeline));
        vkDestroyShaderModule(device, module, nullptr);

        // ---- denoise + finalize pipelines
        auto loadSpv = [&](const char* name) {
            std::string path = SHADER_DIR + std::string(name);
            FILE* g = fopen(path.c_str(), "rb");
            if (!g) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
            fseek(g, 0, SEEK_END);
            long sz2 = ftell(g);
            fseek(g, 0, SEEK_SET);
            std::vector<char> code2(sz2);
            if (fread(code2.data(), 1, sz2, g) != (size_t)sz2) exit(1);
            fclose(g);
            VkShaderModuleCreateInfo sm2{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            sm2.codeSize = code2.size();
            sm2.pCode = (const uint32_t*)code2.data();
            VkShaderModule mm;
            VKCHECK(vkCreateShaderModule(device, &sm2, nullptr, &mm));
            return mm;
        };
        {
            VkDescriptorSetLayoutBinding db[5] = {
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, cs, nullptr},
                {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr},
                {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr},
                {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr}};
            VkDescriptorSetLayoutCreateInfo ddl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            ddl.bindingCount = 5;
            ddl.pBindings = db;
            VKCHECK(vkCreateDescriptorSetLayout(device, &ddl, nullptr, &descDenoiseLayout));
            VkPushConstantRange pcr{};
            pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcr.size = 16;
            VkPipelineLayoutCreateInfo dpl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            dpl.setLayoutCount = 1;
            dpl.pSetLayouts = &descDenoiseLayout;
            dpl.pushConstantRangeCount = 1;
            dpl.pPushConstantRanges = &pcr;
            VKCHECK(vkCreatePipelineLayout(device, &dpl, nullptr, &pipeDenoiseLayout));
            VkComputePipelineCreateInfo dcp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            dcp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            dcp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            dcp.stage.module = loadSpv("denoise.comp.spv");
            dcp.stage.pName = "main";
            dcp.layout = pipeDenoiseLayout;
            VKCHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &dcp, nullptr, &pipeDenoise));
            vkDestroyShaderModule(device, dcp.stage.module, nullptr);

            VkDescriptorSetLayoutBinding fb[3] = {
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, cs, nullptr},
                {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, cs, nullptr}};
            VkDescriptorSetLayoutCreateInfo fdl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            fdl.bindingCount = 3;
            fdl.pBindings = fb;
            VKCHECK(vkCreateDescriptorSetLayout(device, &fdl, nullptr, &descFinalLayout));
            VkPipelineLayoutCreateInfo fpl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            fpl.setLayoutCount = 1;
            fpl.pSetLayouts = &descFinalLayout;
            VKCHECK(vkCreatePipelineLayout(device, &fpl, nullptr, &pipeFinalLayout));
            VkComputePipelineCreateInfo fcp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            fcp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            fcp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            fcp.stage.module = loadSpv("finalize.comp.spv");
            fcp.stage.pName = "main";
            fcp.layout = pipeFinalLayout;
            VKCHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &fcp, nullptr, &pipeFinal));
            vkDestroyShaderModule(device, fcp.stage.module, nullptr);
        }

        VkDescriptorPoolSize ps[3] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4}};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 8;
        dpi.poolSizeCount = 3;
        dpi.pPoolSizes = ps;
        VKCHECK(vkCreateDescriptorPool(device, &dpi, nullptr, &descPool));
        VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        da.descriptorPool = descPool;
        da.descriptorSetCount = 1;
        da.pSetLayouts = &descLayout;
        VKCHECK(vkAllocateDescriptorSets(device, &da, &descSet));

        VkDescriptorBufferInfo bi[6] = {
            {uboBuf, 0, sizeof(UBOData)},
            {accumBuf, 0, VK_WHOLE_SIZE},
            {triBuf, 0, VK_WHOLE_SIZE},
            {nodeBuf, 0, VK_WHOLE_SIZE},
            {matBuf, 0, VK_WHOLE_SIZE},
            {lightBuf, 0, VK_WHOLE_SIZE}};
        VkDescriptorImageInfo ii{VK_NULL_HANDLE, displayView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w[9] = {};
        for (int i = 0; i < 6; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = descSet;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
        w[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[6].dstSet = descSet;
        w[6].dstBinding = 6;
        w[6].descriptorCount = 1;
        w[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[6].pImageInfo = &ii;
        VkDescriptorBufferInfo bgb{gbufBuf, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo bmo{momBuf, 0, VK_WHOLE_SIZE};
        w[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[7].dstSet = descSet;
        w[7].dstBinding = 7;
        w[7].descriptorCount = 1;
        w[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[7].pBufferInfo = &bgb;
        w[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[8].dstSet = descSet;
        w[8].dstBinding = 8;
        w[8].descriptorCount = 1;
        w[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[8].pBufferInfo = &bmo;
        vkUpdateDescriptorSets(device, 9, w, 0, nullptr);

        // denoise / finalize descriptor sets
        {
            auto wbuf = [&](VkDescriptorSet s, uint32_t bnd, VkBuffer buf) {
                VkDescriptorBufferInfo bi2{buf, 0, VK_WHOLE_SIZE};
                VkWriteDescriptorSet w2{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w2.dstSet = s;
                w2.dstBinding = bnd;
                w2.descriptorCount = 1;
                w2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w2.pBufferInfo = &bi2;
                vkUpdateDescriptorSets(device, 1, &w2, 0, nullptr);
            };
            auto wubo = [&](VkDescriptorSet s, uint32_t bnd) {
                VkDescriptorBufferInfo bi2{resUboBuf, 0, VK_WHOLE_SIZE};
                VkWriteDescriptorSet w2{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w2.dstSet = s;
                w2.dstBinding = bnd;
                w2.descriptorCount = 1;
                w2.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                w2.pBufferInfo = &bi2;
                vkUpdateDescriptorSets(device, 1, &w2, 0, nullptr);
            };
            VkDescriptorSetAllocateInfo da2{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            da2.descriptorPool = descPool;
            da2.descriptorSetCount = 1;
            da2.pSetLayouts = &descDenoiseLayout;
            VKCHECK(vkAllocateDescriptorSets(device, &da2, &sDenoiseAB));
            VKCHECK(vkAllocateDescriptorSets(device, &da2, &sDenoiseBA));
            da2.pSetLayouts = &descFinalLayout;
            VKCHECK(vkAllocateDescriptorSets(device, &da2, &sFinal));
            // denoise sets: res, gbuf, moments, in, out
            wubo(sDenoiseAB, 0); wbuf(sDenoiseAB, 1, gbufBuf); wbuf(sDenoiseAB, 2, momBuf);
            wbuf(sDenoiseAB, 3, accumBuf); wbuf(sDenoiseAB, 4, scratchBuf);
            wubo(sDenoiseBA, 0); wbuf(sDenoiseBA, 1, gbufBuf); wbuf(sDenoiseBA, 2, momBuf);
            wbuf(sDenoiseBA, 3, scratchBuf); wbuf(sDenoiseBA, 4, accumBuf);
            // finalize: res, in (scratch holds the result after 5 odd iterations), image
            wubo(sFinal, 0); wbuf(sFinal, 1, scratchBuf);
            VkDescriptorImageInfo ii3{VK_NULL_HANDLE, displayView, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet w3{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w3.dstSet = sFinal;
            w3.dstBinding = 2;
            w3.descriptorCount = 1;
            w3.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w3.pImageInfo = &ii3;
            vkUpdateDescriptorSets(device, 1, &w3, 0, nullptr);
        }

        VkCommandPoolCreateInfo cmdp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cmdp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cmdp.queueFamilyIndex = queueFamily;
        VKCHECK(vkCreateCommandPool(device, &cmdp, nullptr, &cmdPool));
        VkCommandBufferAllocateInfo cba{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cba.commandPool = cmdPool;
        cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cba.commandBufferCount = 1;
        VKCHECK(vkAllocateCommandBuffers(device, &cba, &cmd));
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VKCHECK(vkCreateFence(device, &fi, nullptr, &fence));
        {
            VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VKCHECK(vkCreateSemaphore(device, &si, nullptr, &semAcquire));
            VKCHECK(vkCreateSemaphore(device, &si, nullptr, &semRender));
        }

        // transition display image UNDEFINED -> GENERAL (storage layout)
        VKCHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo tbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        tbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VKCHECK(vkBeginCommandBuffer(cmd, &tbi));
        VkImageMemoryBarrier tb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        tb.image = displayImage;
        tb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        tb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        tb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &tb);
        VKCHECK(vkEndCommandBuffer(cmd));
        submitWait(cmd);
    }

    void submitWait(VkCommandBuffer cb) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        VKCHECK(vkQueueSubmit(queue, 1, &si, fence));
        VKCHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        VKCHECK(vkResetFences(device, 1, &fence));
    }

    // ------------------------------------------------------------- frame record
    void recordFrame(uint32_t swapIdx, bool blit) {
        uint32_t gx = (uint32_t)((iw + 7) / 8), gy = (uint32_t)((ih + 7) / 8);
        VKCHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VKCHECK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
        vkCmdDispatch(cmd, gx, gy, 1);

        if (blit) {
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                 1, &mb, 0, nullptr, 0, nullptr);
            VkImageMemoryBarrier pre[2]{};
            pre[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            pre[0].image = displayImage;
            pre[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            pre[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            pre[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            pre[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            pre[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            pre[1] = pre[0];
            pre[1].image = swapImages[swapIdx];
            pre[1].srcAccessMask = 0;
            pre[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            pre[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            pre[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                 0, nullptr, 0, nullptr, 2, pre);

            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[1] = {iw, ih, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.dstOffsets[1] = {(int32_t)winW, (int32_t)winH, 1};
            vkCmdBlitImage(cmd, displayImage, VK_IMAGE_LAYOUT_GENERAL, swapImages[swapIdx],
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

            VkImageMemoryBarrier post{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            post.image = swapImages[swapIdx];
            post.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            post.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            post.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            post.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            post.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            post.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            post.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                                 0, nullptr, 0, nullptr, 1, &post);
        }
        VKCHECK(vkEndCommandBuffer(cmd));
    }

    // ------------------------------------------------------------- camera / UBO
    void updateUBO() {
        v3 pos = target + dist * v3{std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw)};
        v3 fwd = norm(target - pos);
        v3 right = norm(cross(fwd, v3{0, 1, 0}));
        v3 up = cross(right, fwd);
        UBOData u{};
        u.camPos[0] = pos.x; u.camPos[1] = pos.y; u.camPos[2] = pos.z;
        u.camRight[0] = right.x; u.camRight[1] = right.y; u.camRight[2] = right.z;
        u.camUp[0] = up.x; u.camUp[1] = up.y; u.camUp[2] = up.z;
        u.camFwd[0] = fwd.x; u.camFwd[1] = fwd.y; u.camFwd[2] = fwd.z;
        u.res[0] = (float)iw;
        u.res[1] = (float)ih;
        u.res[2] = (float)frame;
        u.res[3] = std::tan(45.0f * 3.14159265f / 180.0f * 0.5f);  // 45 deg fov
        memcpy(uboMapped, &u, sizeof(u));
    }

    // offline denoise (5 à-trous iterations) + finalize into the display image
    void runDenoiseFinalize() {
        if (!denoise) return;
        uint32_t gx = (uint32_t)((iw + 7) / 8), gy = (uint32_t)((ih + 7) / 8);
        VKCHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VKCHECK(vkBeginCommandBuffer(cmd, &bi));
        for (int it = 0; it < 5; ++it) {
            int32_t cfg[4] = {1 << it, it == 0 ? 1 : 0, 0, 0};
            VkDescriptorSet s = (it & 1) == 0 ? sDenoiseAB : sDenoiseBA;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeDenoise);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeDenoiseLayout, 0, 1, &s, 0, nullptr);
            vkCmdPushConstants(cmd, pipeDenoiseLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, cfg);
            vkCmdDispatch(cmd, gx, gy, 1);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &mb, 0, nullptr, 0, nullptr);
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeFinal);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeFinalLayout, 0, 1, &sFinal, 0, nullptr);
        vkCmdDispatch(cmd, gx, gy, 1);
        VKCHECK(vkEndCommandBuffer(cmd));
        submitWait(cmd);
        printf("denoise: 5x variance-guided a-trous + finalize\n");
    }

    // copy the display image back and write a PNG
    void readbackPNG(const char* path) {
        VkBuffer rb;
        VkDeviceMemory rm;
        makeBuffer((VkDeviceSize)iw * ih * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, rb, rm);
        VKCHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VKCHECK(vkBeginCommandBuffer(cmd, &bi));
        VkBufferImageCopy cpy{};
        cpy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        cpy.imageExtent = {(uint32_t)iw, (uint32_t)ih, 1};
        vkCmdCopyImageToBuffer(cmd, displayImage, VK_IMAGE_LAYOUT_GENERAL, rb, 1, &cpy);
        VKCHECK(vkEndCommandBuffer(cmd));
        submitWait(cmd);
        void* p = nullptr;
        VKCHECK(vkMapMemory(device, rm, 0, VK_WHOLE_SIZE, 0, &p));
        writePNG(path, iw, ih, (const uint8_t*)p);
        vkUnmapMemory(device, rm);
        vkDestroyBuffer(device, rb, nullptr);
        vkFreeMemory(device, rm, nullptr);
    }

    // high-quality still from the CURRENT camera (called by the P key):
    // re-accumulates from scratch for `frames` spp, writes a noise-free PNG
    void snapshotToPNG(const char* path, int frames) {
        printf("snapshot: rendering %d spp at %dx%d (window will pause)...\n", frames, iw, ih);
        auto t0 = std::chrono::steady_clock::now();
        for (int f = 0; f < frames; ++f) {
            frame = f;
            updateUBO();
            recordFrame(~0u, false);
            submitWait(cmd);
            if ((f & 127) == 127) {
                double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                printf("  %d/%d (%.0fs elapsed, ~%.0fs left)\n", f + 1, frames, el,
                       el / (f + 1) * (frames - f - 1));
            }
        }
        runDenoiseFinalize();   // overwrites the display image with the filtered result
        readbackPNG(path);
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("snapshot: wrote %s (%d spp%s, %.1fs)\n", path, frames, denoise ? ", denoised" : "", el);
    }

    // ------------------------------------------------------------- offscreen
    // Renders headless but (on desktop) opens a live preview window: every few
    // frames the converging image is blitted to screen and the title shows
    // progress. The written PNG is identical either way.
    int runOffscreen(int frames) {
        // live preview window (best effort — falls back to headless)
        bool preview = false;
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(winW, winH, "offscreen render", nullptr, nullptr);
        if (window) {
            glfwGetFramebufferSize(window, &winW, &winH);
            glfwPollEvents();
            VkResult rc = glfwCreateWindowSurface(instance, window, nullptr, &surface);
            if (rc != VK_SUCCESS) {
                glfwPollEvents();
                rc = glfwCreateWindowSurface(instance, window, nullptr, &surface);
            }
            if (rc == VK_SUCCESS) {
                createSwapchain();
                preview = true;
            } else {
                fprintf(stderr, "note: surface creation failed (%d), rendering headless\n", (int)rc);
                surface = VK_NULL_HANDLE;
            }
        } else {
            fprintf(stderr, "note: no window, rendering headless\n");
        }

        auto t0 = std::chrono::steady_clock::now();
        int f = 0;
        for (; f < frames; ++f) {
            frame = f;
            updateUBO();
            bool showFrame = preview && (f % 16 == 15);
            if (preview) glfwPollEvents();
            if (preview && glfwWindowShouldClose(window)) {
                printf("preview window closed — writing partial result (%d/%d spp)\n", f, frames);
                break;
            }
            if (showFrame) {
                VKCHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
                VKCHECK(vkResetFences(device, 1, &fence));
                uint32_t idx = 0;
                VkResult acq = vkAcquireNextImageKHR(device, swapchain, 1000000000ull, semAcquire, VK_NULL_HANDLE, &idx);
                if (acq == VK_SUCCESS || acq == VK_SUBOPTIMAL_KHR) {
                    recordFrame(idx, true);   // dispatch + blit to the preview window
                    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                    si.waitSemaphoreCount = 1;
                    si.pWaitSemaphores = &semAcquire;
                    si.pWaitDstStageMask = &waitStage;
                    si.commandBufferCount = 1;
                    si.pCommandBuffers = &cmd;
                    si.signalSemaphoreCount = 1;
                    si.pSignalSemaphores = &semRender;
                    VKCHECK(vkQueueSubmit(queue, 1, &si, fence));
                    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
                    pi.waitSemaphoreCount = 1;
                    pi.pWaitSemaphores = &semRender;
                    pi.swapchainCount = 1;
                    pi.pSwapchains = &swapchain;
                    pi.pImageIndices = &idx;
                    vkQueuePresentKHR(queue, &pi);
                    VKCHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
                } else if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
                    vkDeviceWaitIdle(device);
                    destroySwapchain();
                    createSwapchain();
                } else if (acq == VK_TIMEOUT || acq == VK_NOT_READY) {
                    // window minimized / image not ready: skip the preview this round
                    VKCHECK(vkResetFences(device, 1, &fence));
                    recordFrame(~0u, false);
                    submitWait(cmd);
                } else {
                    VKCHECK(acq);
                }
                char title[128];
                snprintf(title, sizeof(title), "offscreen render — %d / %d spp", f + 1, frames);
                glfwSetWindowTitle(window, title);
            } else {
                recordFrame(~0u, false);
                submitWait(cmd);
            }
            if ((f & 255) == 255) {
                double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                printf("  %d/%d spp (%.0fs elapsed, ~%.0fs left)\n", f + 1, frames, el,
                       el / (f + 1) * (frames - f - 1));
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        int done = f;
        runDenoiseFinalize();
        readbackPNG("out.png");
        double mpaths = (double)iw * ih * done / sec / 1e6;
        printf("offscreen: %d frames in %.2fs (%.2f ms/frame, %.1f Mpaths/s)%s\n",
               done, sec, sec * 1000.0 / done, mpaths, denoise ? " + denoise" : "");
        printf("wrote out.png (%dx%d, %d spp%s)\n", iw, ih, done, denoise ? ", denoised" : "");
        if (preview) {
            vkDeviceWaitIdle(device);
            destroySwapchain();
            vkDestroySurfaceKHR(instance, surface, nullptr);
            surface = VK_NULL_HANDLE;
            glfwDestroyWindow(window);
            window = nullptr;
        }
        return 0;
    }

    // ------------------------------------------------------------- swapchain
    void createSwapchain() {
        VkSurfaceCapabilitiesKHR caps;
        VKCHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps));
        uint32_t fc = 0;
        VKCHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &fc, nullptr));
        std::vector<VkSurfaceFormatKHR> formats(fc);
        VKCHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &fc, formats.data()));
        swapFormat = formats[0].format;
        for (auto& f : formats)
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { swapFormat = f.format; break; }
        uint32_t pc = 0;
        VKCHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &pc, nullptr));
        std::vector<VkPresentModeKHR> modes(pc);
        VKCHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &pc, modes.data()));
        VkPresentModeKHR pm = VK_PRESENT_MODE_FIFO_KHR;
        if (novsync) {
            for (auto m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) { pm = m; break; }
            if (pm == VK_PRESENT_MODE_FIFO_KHR)
                for (auto m : modes) if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) { pm = m; break; }
        }
        VkExtent2D ext = caps.currentExtent;
        if (ext.width == 0xFFFFFFFF) {
            ext.width = std::min(std::max((uint32_t)winW, caps.minImageExtent.width), caps.maxImageExtent.width);
            ext.height = std::min(std::max((uint32_t)winH, caps.minImageExtent.height), caps.maxImageExtent.height);
        }
        winW = (int)ext.width;
        winH = (int)ext.height;
        uint32_t count = std::max(3u, caps.minImageCount);
        if (caps.maxImageCount) count = std::min(count, caps.maxImageCount);
        VkSwapchainCreateInfoKHR si{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        si.surface = surface;
        si.minImageCount = count;
        si.imageFormat = swapFormat;
        si.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        si.imageExtent = ext;
        si.imageArrayLayers = 1;
        si.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        si.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        si.preTransform = caps.currentTransform;
        si.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        si.clipped = VK_TRUE;
        si.presentMode = pm;
        VKCHECK(vkCreateSwapchainKHR(device, &si, nullptr, &swapchain));
        uint32_t ic = 0;
        VKCHECK(vkGetSwapchainImagesKHR(device, swapchain, &ic, nullptr));
        swapImages.resize(ic);
        VKCHECK(vkGetSwapchainImagesKHR(device, swapchain, &ic, swapImages.data()));
        swapViews.resize(ic);
        for (uint32_t i = 0; i < ic; ++i) {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = swapImages[i];
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = swapFormat;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VKCHECK(vkCreateImageView(device, &vi, nullptr, &swapViews[i]));
        }
        printf("window %dx%d (render %dx%d), %u swap images, present mode %d\n",
               winW, winH, iw, ih, ic, (int)pm);
    }
    void destroySwapchain() {
        for (auto v : swapViews) vkDestroyImageView(device, v, nullptr);
        swapViews.clear();
        if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }
    void recreateSwapchain() {
        int fw, fh;
        glfwGetFramebufferSize(window, &fw, &fh);
        if (fw <= 0 || fh <= 0) return;
        if (fw == winW && fh == winH) return;
        vkDeviceWaitIdle(device);
        destroySwapchain();
        createSwapchain();
    }

    static void scrollCb(GLFWwindow* w, double, double dy) {
        App* app = (App*)glfwGetWindowUserPointer(w);
        app->dist = std::min(30.0f, std::max(1.5f, app->dist * std::exp((float)dy * 0.12f)));
        app->scrollMoved = true;
    }

    // ------------------------------------------------------------- windowed
    int runWindowed() {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(winW, winH, "vk-compute-pathtracer — engine mode (drag/WASD/scroll)", nullptr, nullptr);
        if (!window) { fprintf(stderr, "window failed\n"); return 1; }
        glfwSetWindowUserPointer(window, this);
        glfwSetScrollCallback(window, scrollCb);
        glfwGetFramebufferSize(window, &winW, &winH);
        glfwPollEvents();
        VkResult surfRc = glfwCreateWindowSurface(instance, window, nullptr, &surface);
        if (surfRc != VK_SUCCESS) {
            glfwPollEvents();
            surfRc = glfwCreateWindowSurface(instance, window, nullptr, &surface);
        }
        if (surfRc != VK_SUCCESS) {
            fprintf(stderr, "surface creation failed: %d\n", surfRc);
            return 1;
        }
        createSwapchain();

        printf("controls: drag = orbit, wheel = zoom, WASD = move, R = reset view\n");
        printf("          P = snapshot: render %d spp still from the current camera -> snapshot.png\n", snapSpp);
        double mx = 0, my = 0;
        glfwGetCursorPos(window, &mx, &my);
        auto lastT = std::chrono::steady_clock::now();
        int fpsFrames = 0;
        auto fpsT0 = lastT;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - lastT).count();
            lastT = now;

            bool moved = scrollMoved;
            scrollMoved = false;
            double cx, cy;
            glfwGetCursorPos(window, &cx, &cy);
            bool drag = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
            if (drag) {
                yaw -= (float)(cx - mx) * 0.005f;
                pitch = std::min(1.35f, std::max(-1.35f, pitch + (float)(cy - my) * 0.005f));
                if (fabs(cx - mx) + fabs(cy - my) > 1e-6) moved = true;
            }
            mx = cx; my = cy;

            v3 pos = target + dist * v3{std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw)};
            v3 fwd = norm(target - pos);
            v3 right = norm(cross(fwd, v3{0, 1, 0}));
            float spd = dt * dist * 0.7f;
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { target = target + fwd * spd; moved = true; }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { target = target - fwd * spd; moved = true; }
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { target = target - right * spd; moved = true; }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { target = target + right * spd; moved = true; }
            if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
                yaw = 0.0f; pitch = 0.02f; dist = 8.0f; target = {0.0f, 2.5f, 0.0f};
                moved = true;
            }
            static bool pWasUp = true;
            if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS && pWasUp) {
                snapshotToPNG("snapshot.png", snapSpp);
                frame = 0;  // restart preview accumulation
                pWasUp = false;
            } else if (glfwGetKey(window, GLFW_KEY_P) != GLFW_PRESS) {
                pWasUp = true;
            }
            if (moved) frame = 0;

            recreateSwapchain();
            VKCHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
            VKCHECK(vkResetFences(device, 1, &fence));
            uint32_t idx = 0;
            VkResult acq = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, semAcquire, VK_NULL_HANDLE, &idx);
            if (acq == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); continue; }

            updateUBO();
            recordFrame(idx, true);

            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.waitSemaphoreCount = 1;
            si.pWaitSemaphores = &semAcquire;
            si.pWaitDstStageMask = &waitStage;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &semRender;
            VKCHECK(vkQueueSubmit(queue, 1, &si, fence));

            VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &semRender;
            pi.swapchainCount = 1;
            pi.pSwapchains = &swapchain;
            pi.pImageIndices = &idx;
            VkResult pres = vkQueuePresentKHR(queue, &pi);
            if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
                recreateSwapchain();
                continue;
            }
            frame++;
            fpsFrames++;
            float el = std::chrono::duration<float>(now - fpsT0).count();
            if (el >= 1.0f) {
                printf("%.1f fps | %llu spp accumulated | %.1f Mpaths/s\n",
                       fpsFrames / el, (unsigned long long)frame,
                       fpsFrames * (double)iw * ih / el / 1e6);
                fpsFrames = 0;
                fpsT0 = now;
            }
        }
        vkDeviceWaitIdle(device);
        return 0;
    }
};

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    App app;
    int offFrames = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--offscreen")) {
            app.offscreen = true;
            offFrames = 256;
            if (i + 1 < argc && argv[i + 1][0] != '-') offFrames = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--size") && i + 1 < argc) {
            sscanf(argv[++i], "%dx%d", &app.winW, &app.winH);
            if (app.offscreen) { app.iw = app.winW; app.ih = app.winH; }
        } else if (!strcmp(argv[i], "--ires") && i + 1 < argc) {
            sscanf(argv[++i], "%dx%d", &app.iw, &app.ih);
        } else if (!strcmp(argv[i], "--scene") && i + 1 < argc) {
            app.scenePath = argv[++i];
        } else if (!strcmp(argv[i], "--snap") && i + 1 < argc) {
            app.snapSpp = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--nodenoise")) {
            app.denoise = false;
        } else if (!strcmp(argv[i], "--validate")) {
            app.validate = true;
        } else if (!strcmp(argv[i], "--novsync")) {
            app.novsync = true;
        }
    }
    if (app.offscreen && offFrames <= 0) offFrames = 256;

    app.buildWorld();
    app.initVulkan();
    return app.offscreen ? app.runOffscreen(offFrames) : app.runWindowed();
}
