# vk-compute-pathtracer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**纯 Vulkan compute shader 的路径追踪引擎 —— 不使用任何光追扩展、不需要 RT 硬件。**
BVH 遍历与全部求交在普通 compute shader 里手写完成;macOS 上经 MoltenVK 运行,
同一份代码可在 Windows/Linux 原生 Vulkan 及无头服务器上运行。

![Cornell box, 1024 spp + offline denoiser](docs/cornell_final.png)

*1920×1080 级静态渲染:康奈尔盒,含完整全局光照 —— 红绿墙色溢出、面光源软阴影、
玻璃球折射、镜面球反射。Apple M1 (MoltenVK) 上约 4 分钟出图。*

## 渐进收敛

每帧每像素 1 条随机路径(1 spp),静止后无限累积,噪声按 1/√N 消退:

| 16 spp | 256 spp(无降噪) | 256 spp(方差引导降噪) |
|:---:|:---:|:---:|
| ![16 spp](docs/spp_16.png) | ![256 spp raw](docs/spp_256_raw.png) | ![256 spp denoised](docs/spp_256_denoised.png) |

*右图与中图采样数完全相同 —— 降噪器从累积亮度矩取每像素方差,亮区不动、
暗区自适应平滑,边缘经 G-buffer 权重保护零损失(256 spp + 降噪 ≈ 裸跑 1000+ spp)。*

## 功能

- **静态渲染**:渐进累积,Welford 增量均值,数值有界永不停滞
- **NEE 直接光**:发光三角形自动识别,面积加权 CDF 采样,首弹射 4 样本
- **透射阴影**:阴影光线穿透玻璃((1−Fresnel)),软焦散、无黑本影
- **离线降噪**:à-trous 方差引导滤波 ×5,G-buffer 边缘停止
- **光滑法线**:OBJ `vn` 重心插值 + 绕序/法线自动和解(任意绕序 OBJ 安全)
- **原生球原语**:场景级 `sphere` 关键字,引擎内部网格化,渲染器单管线零特判
- **实时预览**:交互窗口(拖拽/滚轮/WASD),离屏渲染带实时预览窗 + 进度/ETA
- **P 键快照**:从当前视角出 4096 spp 无噪成品
- **相机动画**:关键帧文件(Catmull-Rom)→ 逐帧独立收敛 + 降噪 → PNG 序列

## 快速开始(macOS)

```bash
brew install cmake glfw glslang molten-vk vulkan-headers vulkan-loader vulkan-tools
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

./build/rt                                          # 交互预览,P = 4096 spp 快照
./build/rt --offscreen 4096 --size 1920x1080       # 静态渲染(带实时预览)-> out.png
./build/rt --anim assets/anim_flythrough.txt --size 1280x720 --spp 96 --out anim
# 常用: --scene path --ires WxH --snap N --fps 30 --nodenoise --novsync --validate
```

## 场景文件(`assets/scene.txt`)

```
#   mesh.obj  px py pz  rotYdeg  sx sy sz  material
#   sphere    cx cy cz  radius            material
#   material: auto | diffuse r g b | light r g b | mirror r g b | glass r g b ior
cornell_room.obj   0 0 0       0   1 1 1        auto
unit_box.obj      -1.08 0.99 -0.35  18   1.06 1.98 1.06  auto
unit_box.obj       1.25 0.53  0.35 -15   1.06 1.06 1.06  auto
sphere             0    0.68  1.70  0.68         glass  1 1 1 1.5
sphere             1.5  0.5   1.60  0.5           mirror 0.95 0.95 0.96
```

资产由 `assets/gen_assets.py` 生成;任何外部 OBJ(Blender 导出等)放入
`assets/` 加一行即可。`assets/scene_objtest.txt` 为全网格球对照示例。

## 相机动画(`assets/anim_flythrough.txt`)

![fly-through contact sheet](docs/anim_contact_sheet.png)

*8.6 秒飞越:远景接近 → 浅入盒中 → 0.9 秒快掉头锁定金属球 → 掠过玻璃球出盒 → 退回原始视角。
1280×720 × 30fps;`--fast` 预览 259 帧约 44 秒,正式渲染按 spp 定(96 spp ≈ 44 分钟,1024 spp 约 4.5 小时)。*

```
#   t  px py pz  tx ty tz     (秒,相机位置,注视点)
0.00   0.00 2.75 11.50    0.00 2.50  0.00
3.80   0.00 2.42 -0.20    0.00 2.30 -2.20    # 盒内中央,面向后墙
6.60   0.60 1.62  1.70    0.10 0.85  1.45    # 贴着玻璃球擦过
...
```

位置与注视点各自走 Catmull-Rom 样条(C1 连续,端点夹持自动缓入缓出)。
每个输出帧都是独立收敛的成品:设定相机 → 从零累积 N spp → 方差引导降噪
→ 写 `anim/frame_%04d.png`。渲染过程带实时预览窗,关窗即取消。

```bash
./build/rt --anim assets/anim_flythrough.txt --size 1280x720 --spp 96 --fps 30 --out anim
ffmpeg -framerate 30 -i anim/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 anim.mp4

# 快速预览(检查运镜,无视噪点):1 spp + 免降噪,283 帧约 1 分钟;加 --ires 640x360 约 20 秒
./build/rt --anim assets/anim_flythrough.txt --fast --size 1280x720 --out anim_fast
```

引擎侧相机与输入策略解耦:`setView(pos, look)` 是唯一的视图入口,
轨道相机、动画样条、未来的外部嵌入方都走同一条代码路径。

## 渲染管线

```
scene.txt → OBJ/MTL → 变换烘焙 → 三角形流 → SAH BVH → SSBO
                                                       ↓
pathtrace.comp(追踪 + NEE + 累积 + G-buffer/亮度矩 + tonemap)
                                                       ↓ (出图时)
denoise.comp ×5(à-trous 方差引导)→ finalize.comp → PNG
```

## 实测(Apple M1 / MoltenVK / Vulkan 1.1,7972 三角形)

| 项目 | 结果 |
|---|---|
| 交互预览 1280×720 | ~10 fps,静止持续收敛 |
| 降噪收益(256 spp vs 2048 参考) | 误差 −41%~−55%,边缘梯度零损失 |
| 1600×900 × 1024 spp + 降噪 | ~3 分钟 |
| 动画 1280×720 × 30fps × 96 spp/帧 | 283 帧 53 分钟(7.9 Mpaths/s 均值) |

## 已知边界

- 相机可移入几何体内部(双面着色不崩溃,但画面怪异)
- 镜面链经玻璃的光源计入与透射阴影存在轻微重复能量(视觉不可辨)
- MTL 纹理(`map_Kd`)未支持;材质为纯参数化
- 进程退出不逐个销毁 Vulkan 资源(由 OS 回收)

## License

[MIT](LICENSE)

## AI 生成说明 / AI Generation Notice

本项目的代码由 AI 辅助生成(GLM,经由 DeepSeek Harness),通过迭代式对话完成:
人类负责设计决策、需求定义、测试与审查。
This codebase was developed with AI assistance (GLM via DeepSeek Harness)
through iterative conversation; the human directed design decisions,
requirements, testing, and review.
