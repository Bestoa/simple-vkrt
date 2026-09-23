#!/usr/bin/env python3
"""Generate the Cornell box assets (OBJ + MTL) and the scene file.

Run once from the repo root:  python3 assets/gen_assets.py
The generated files are committed; regenerate after editing sizes/materials.
"""
import math
import os

A = os.path.dirname(os.path.abspath(__file__))


def fn(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def write_obj(path, name, verts, faces, mtllib=None, norms=None):
    """faces: list of (i, j, k, mtlname) with 0-based vertex indices.
    norms: optional per-vertex normals; faces emitted as 'f v//vn v//vn v//vn'."""
    with open(os.path.join(A, path), "w") as f:
        f.write(f"# {name}\n")
        if mtllib:
            f.write(f"mtllib {mtllib}\n")
        for x, y, z in verts:
            f.write(f"v {fn(x)} {fn(y)} {fn(z)}\n")
        if norms:
            for x, y, z in norms:
                f.write(f"vn {fn(x)} {fn(y)} {fn(z)}\n")
        cur = None
        for i, j, k, mtl in faces:
            if mtl != cur:
                if mtl:
                    f.write(f"usemtl {mtl}\n")
                cur = mtl
            if norms:
                f.write(f"f {i+1}//{i+1} {j+1}//{j+1} {k+1}//{k+1}\n")
            else:
                f.write(f"f {i+1} {j+1} {k+1}\n")


def write_mtl(path, mats):
    """mats: list of (name, kd_or_None, ke_or_None)."""
    with open(os.path.join(A, path), "w") as f:
        for name, kd, ke in mats:
            f.write(f"newmtl {name}\n")
            if kd:
                f.write(f"Kd {fn(kd[0])} {fn(kd[1])} {fn(kd[2])}\n")
            if ke:
                f.write(f"Ke {fn(ke[0])} {fn(ke[1])} {fn(ke[2])}\n")
            f.write("\n")


# ---------------------------------------------------------------- cornell room
W, H, D = 2.6, 5.2, 2.6
LH, LY = 0.65, 5.18  # light half-size / height

verts, faces = [], []


def V(x, y, z):
    verts.append((x, y, z))
    return len(verts) - 1


def quad(mtl, a, b, c, d):
    ia, ib, ic, idd = V(*a), V(*b), V(*c), V(*d)
    faces.append((ia, ib, ic, mtl))
    faces.append((ia, ic, idd, mtl))


quad("white", (-W, 0, -D), (W, 0, -D), (W, 0, D), (-W, 0, D))       # floor
quad("white", (-W, H, -D), (W, H, -D), (W, H, D), (-W, H, D))       # ceiling
quad("white", (-W, 0, -D), (W, 0, -D), (W, H, -D), (-W, H, -D))     # back wall
quad("red",   (-W, 0, -D), (-W, 0, D), (-W, H, D), (-W, H, -D))     # left wall
quad("green", (W, 0, -D), (W, 0, D), (W, H, D), (W, H, -D))         # right wall
quad("light", (-LH, LY, -LH), (LH, LY, -LH), (LH, LY, LH), (-LH, LY, LH))  # ceiling light

write_obj("cornell_room.obj", "Cornell box room", verts, faces, "cornell_room.mtl")
write_mtl("cornell_room.mtl", [
    ("white", (0.73, 0.73, 0.73), None),
    ("red",   (0.63, 0.065, 0.05), None),
    ("green", (0.14, 0.45, 0.091), None),
    ("light", (0, 0, 0), (17, 15, 12)),
])

# ---------------------------------------------------------------- unit box
s = 0.5
verts, faces = [], []
c = [(-s, -s, -s), (-s, -s, +s), (-s, +s, -s), (-s, +s, +s),
     (+s, -s, -s), (+s, -s, +s), (+s, +s, -s), (+s, +s, +s)]
quad("white", c[0], c[1], c[3], c[2])
quad("white", c[5], c[4], c[6], c[7])
quad("white", c[2], c[3], c[7], c[6])
quad("white", c[0], c[4], c[5], c[1])
quad("white", c[1], c[5], c[7], c[3])
quad("white", c[4], c[0], c[2], c[6])
write_obj("unit_box.obj", "Unit cube (centered, size 1)", verts, faces, "unit_box.mtl")
write_mtl("unit_box.mtl", [("white", (0.73, 0.73, 0.73), None)])

# ---------------------------------------------------------------- unit sphere (UV)
RINGS, SECTORS = 32, 64
verts, norms, faces = [], [], []
for r in range(RINGS + 1):
    phi = math.pi * r / RINGS
    for s_ in range(SECTORS + 1):
        th = 2 * math.pi * s_ / SECTORS
        p = (math.sin(phi) * math.cos(th), math.cos(phi), math.sin(phi) * math.sin(th))
        verts.append(p)
        norms.append(p)  # unit sphere: normal == position (smooth shading)
for r in range(RINGS):
    for s_ in range(SECTORS):
        a = r * (SECTORS + 1) + s_
        b = a + SECTORS + 1
        if r != 0:
            faces.append((a, b, a + 1, None))
        if r != RINGS - 1:
            faces.append((a + 1, b, b + 1, None))
write_obj("sphere.obj", f"Unit sphere (UV {RINGS}x{SECTORS}, smooth normals)", verts, faces, None, norms)

# ---------------------------------------------------------------- scene
with open(os.path.join(A, "scene.txt"), "w") as f:
    f.write("""# scene format (one entity per line):
#   mesh.obj  px py pz  rotYdeg  sx sy sz  material
#   sphere    cx cy cz  radius            material
#   material: auto | diffuse r g b | light r g b | mirror r g b | glass r g b ior
# 'auto' = use the OBJ's own MTL materials. Mesh paths are relative to this file.
# 'sphere' is a native engine primitive (tessellated internally with smooth normals).
cornell_room.obj   0 0 0       0   1 1 1        auto
unit_box.obj      -1.08 0.99 -0.35  18   1.06 1.98 1.06  auto
unit_box.obj       1.25 0.53  0.35 -15   1.06 1.06 1.06  auto
sphere             0    0.68  1.70  0.68         glass  1 1 1 1.5
sphere             1.5  0.5   1.60  0.5           mirror 0.95 0.95 0.96
""")

print("generated assets in", A)
for name in os.listdir(A):
    print("  ", name)
