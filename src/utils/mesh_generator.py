"""Prompt-conditioned procedural mesh generator.

This is a lightweight production path for the demo app: it turns a text prompt
into saved OBJ/MTL/SVG assets without requiring GPU diffusion dependencies.
The generated mesh is intentionally simple, but the API shape mirrors a future
text-to-3D backend.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class MeshAsset:
    job_id: str
    prompt: str
    name: str
    obj_path: Path
    mtl_path: Path
    stl_path: Path
    preview_path: Path
    metadata_path: Path
    vertices: int
    faces: int
    kind: str
    material: str
    quality: dict


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "dreamer-mesh"


def _prompt_seed(prompt: str) -> int:
    return int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)


def _choose_kind(prompt: str) -> str:
    p = prompt.lower()
    if any(word in p for word in ("chair", "seat", "stool", "armchair", "sofa")):
        return "chair"
    if any(word in p for word in ("table", "desk", "bench")):
        return "block"
    if any(word in p for word in ("crystal", "gem", "diamond", "rock")):
        return "crystal"
    if any(word in p for word in ("bust", "statue", "angel", "head", "face")):
        return "bust"
    if any(word in p for word in ("cup", "vase", "bottle", "tower", "pillar")):
        return "vessel"
    if any(word in p for word in ("cube", "box", "chest", "robot")):
        return "block"
    return "organic"


def _choose_material(prompt: str) -> tuple[str, tuple[float, float, float]]:
    p = prompt.lower()
    if re.search(r"\b(marble|stone)\b", p):
        return "polished_marble", (0.86, 0.82, 0.72)
    if re.search(r"\bgold(?:en)?\b", p):
        return "warm_gold", (1.0, 0.67, 0.23)
    if re.search(r"\b(wood|wooden|carved_wood)\b", p):
        return "carved_wood", (0.55, 0.32, 0.16)
    if re.search(r"\b(green|painted_wood)\b", p):
        return "painted_wood", (0.42, 0.58, 0.30)
    if re.search(r"\bred\b", p):
        return "red_ceramic", (0.82, 0.12, 0.10)
    if re.search(r"\bblue\b", p):
        return "blue_glass", (0.18, 0.44, 0.86)
    return "dreamer_clay", (0.62, 0.72, 0.78)


def _radius_for(kind: str, theta: float, y: float, seed: int) -> float:
    wobble = 0.035 * math.sin(theta * 5 + seed * 0.013) + 0.025 * math.sin(theta * 9 + y * 7)
    if kind == "bust":
        shoulders = 0.72 * math.exp(-((y + 0.58) / 0.28) ** 2)
        neck = 0.28 * math.exp(-((y + 0.12) / 0.22) ** 2)
        head = 0.48 * math.exp(-((y - 0.38) / 0.36) ** 2)
        halo = 0.07 * math.exp(-((y - 0.72) / 0.08) ** 2) * (1 + math.sin(theta * 2) * 0.35)
        return max(0.08, shoulders + neck + head + halo + wobble)
    if kind == "vessel":
        body = 0.42 + 0.22 * math.sin((y + 0.95) * math.pi)
        neck = -0.18 * math.exp(-((y - 0.48) / 0.24) ** 2)
        lip = 0.12 * math.exp(-((y - 0.82) / 0.08) ** 2)
        return max(0.10, body + neck + lip + wobble)
    if kind == "crystal":
        facets = 0.48 + 0.12 * math.cos(theta * 6)
        taper = 1.0 - 0.52 * abs(y)
        return max(0.05, facets * taper)
    if kind == "block":
        squareish = 0.55 / max(abs(math.cos(theta)), abs(math.sin(theta)), 0.72)
        bevel = 1.0 - 0.18 * abs(y)
        return max(0.12, squareish * bevel)
    return max(0.08, 0.52 * math.sqrt(max(0.0, 1 - y * y)) + wobble)


def _add_box(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> None:
    cx, cy, cz = center
    sx, sy, sz = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    base = len(vertices) + 1
    vertices.extend(
        [
            (cx - sx, cy - sy, cz - sz),
            (cx + sx, cy - sy, cz - sz),
            (cx + sx, cy + sy, cz - sz),
            (cx - sx, cy + sy, cz - sz),
            (cx - sx, cy - sy, cz + sz),
            (cx + sx, cy - sy, cz + sz),
            (cx + sx, cy + sy, cz + sz),
            (cx - sx, cy + sy, cz + sz),
        ]
    )
    quads = [
        (1, 2, 3, 4),
        (5, 8, 7, 6),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 8, 4),
        (4, 8, 5, 1),
    ]
    for a, b, c, d in quads:
        faces.append((base + a - 1, base + b - 1, base + c - 1))
        faces.append((base + a - 1, base + c - 1, base + d - 1))


def _build_chair_mesh(seed: int) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    del seed
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    _add_box(vertices, faces, center=(0.0, -0.15, 0.0), size=(1.25, 0.18, 1.1))
    _add_box(vertices, faces, center=(0.0, 0.55, 0.47), size=(1.25, 1.2, 0.16))
    for x in (-0.48, 0.48):
        for z in (-0.38, 0.38):
            _add_box(vertices, faces, center=(x, -0.75, z), size=(0.16, 1.15, 0.16))
    _add_box(vertices, faces, center=(-0.73, 0.05, 0.0), size=(0.16, 0.18, 1.08))
    _add_box(vertices, faces, center=(0.73, 0.05, 0.0), size=(0.16, 0.18, 1.08))
    _add_box(vertices, faces, center=(0.0, 0.95, 0.55), size=(1.12, 0.16, 0.16))
    return vertices, faces


def _build_mesh(kind: str, seed: int, rings: int = 30, segments: int = 48) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    if kind == "chair":
        return _build_chair_mesh(seed)

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for i in range(rings + 1):
        y = -1.0 + 2.0 * i / rings
        for j in range(segments):
            theta = 2.0 * math.pi * j / segments
            radius = _radius_for(kind, theta, y, seed)
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            if kind == "bust" and y > 0.18:
                # Subtle nose/face plane so bust prompts look directional.
                face = max(0.0, math.cos(theta))
                x += 0.08 * face * math.exp(-((y - 0.38) / 0.20) ** 2)
                z *= 1.0 - 0.16 * face
            vertices.append((x, y, z))
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j + 1
            b = i * segments + ((j + 1) % segments) + 1
            c = (i + 1) * segments + j + 1
            d = (i + 1) * segments + ((j + 1) % segments) + 1
            faces.append((a, c, b))
            faces.append((b, c, d))
    bottom_center = len(vertices) + 1
    vertices.append((0.0, -1.0, 0.0))
    top_center = len(vertices) + 1
    vertices.append((0.0, 1.0, 0.0))
    for j in range(segments):
        a = j + 1
        b = ((j + 1) % segments) + 1
        faces.append((bottom_center, b, a))
        c = rings * segments + j + 1
        d = rings * segments + ((j + 1) % segments) + 1
        faces.append((top_center, c, d))
    return vertices, faces


def _write_obj(path: Path, mtl_name: str, material_name: str, vertices: Iterable[tuple[float, float, float]], faces: Iterable[tuple[int, int, int]]) -> None:
    lines = [f"mtllib {mtl_name}", f"usemtl {material_name}"]
    lines.extend(f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in vertices)
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mtl(path: Path, material_name: str, color: tuple[float, float, float]) -> None:
    r, g, b = color
    path.write_text(
        "\n".join(
            [
                f"newmtl {material_name}",
                f"Kd {r:.4f} {g:.4f} {b:.4f}",
                "Ka 0.1200 0.1200 0.1200",
                "Ks 0.4500 0.4500 0.4500",
                "Ns 64.0000",
                "illum 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normal(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> tuple[float, float, float]:
    n = _cross(_sub(b, a), _sub(c, a))
    mag = _length(n)
    if mag < 1e-12:
        return (0.0, 0.0, 0.0)
    return (n[0] / mag, n[1] / mag, n[2] / mag)


def _mesh_quality(vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> dict:
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    area = 0.0
    signed_volume = 0.0
    edges: dict[tuple[int, int], int] = {}
    for a_i, b_i, c_i in faces:
        a, b, c = vertices[a_i - 1], vertices[b_i - 1], vertices[c_i - 1]
        area += 0.5 * _length(_cross(_sub(b, a), _sub(c, a)))
        signed_volume += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        ) / 6.0
        for u, v in ((a_i, b_i), (b_i, c_i), (c_i, a_i)):
            edge = tuple(sorted((u, v)))
            edges[edge] = edges.get(edge, 0) + 1
    boundary_edges = sum(1 for count in edges.values() if count != 2)
    return {
        "watertight": boundary_edges == 0,
        "boundary_edges": boundary_edges,
        "surface_area": round(area, 4),
        "volume": round(abs(signed_volume), 4),
        "bbox": {
            "x": [round(min(xs), 4), round(max(xs), 4)],
            "y": [round(min(ys), 4), round(max(ys), 4)],
            "z": [round(min(zs), 4), round(max(zs), 4)],
        },
    }


def _write_stl(path: Path, name: str, vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> None:
    lines = [f"solid {name}"]
    for a_i, b_i, c_i in faces:
        a, b, c = vertices[a_i - 1], vertices[b_i - 1], vertices[c_i - 1]
        nx, ny, nz = _normal(a, b, c)
        lines.append(f"  facet normal {nx:.6f} {ny:.6f} {nz:.6f}")
        lines.append("    outer loop")
        lines.append(f"      vertex {a[0]:.6f} {a[1]:.6f} {a[2]:.6f}")
        lines.append(f"      vertex {b[0]:.6f} {b[1]:.6f} {b[2]:.6f}")
        lines.append(f"      vertex {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_preview(path: Path, prompt: str, kind: str, color: tuple[float, float, float], seed: int) -> None:
    r, g, b = [int(c * 255) for c in color]
    accent = f"rgb({r},{g},{b})"
    label = prompt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")[:70]
    if kind == "chair":
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" viewBox="0 0 640 420">
  <rect width="640" height="420" fill="#0d1117"/>
  <ellipse cx="202" cy="328" rx="150" ry="25" fill="#000" opacity=".32"/>
  <g transform="translate(80 38)">
    <rect x="88" y="176" width="178" height="42" rx="8" fill="{accent}" stroke="#e6edf3" stroke-opacity=".34" stroke-width="3"/>
    <rect x="106" y="58" width="146" height="126" rx="10" fill="{accent}" opacity=".86" stroke="#e6edf3" stroke-opacity=".34" stroke-width="3"/>
    <rect x="78" y="210" width="28" height="118" rx="6" fill="{accent}" opacity=".9"/>
    <rect x="246" y="210" width="28" height="118" rx="6" fill="{accent}" opacity=".9"/>
    <rect x="104" y="220" width="24" height="108" rx="6" fill="{accent}" opacity=".75"/>
    <rect x="224" y="220" width="24" height="108" rx="6" fill="{accent}" opacity=".75"/>
    <rect x="58" y="160" width="28" height="80" rx="8" fill="{accent}" opacity=".72"/>
    <rect x="270" y="160" width="28" height="80" rx="8" fill="{accent}" opacity=".72"/>
    <path d="M118 82 C152 64 206 64 242 82" fill="none" stroke="#fff" stroke-opacity=".28" stroke-width="5"/>
  </g>
  <text x="338" y="118" fill="#e6edf3" font-family="Arial" font-size="28" font-weight="700">Generated Chair Mesh</text>
  <text x="338" y="154" fill="#7d8590" font-family="Arial" font-size="16">Kind: chair</text>
  <text x="338" y="184" fill="#7d8590" font-family="Arial" font-size="16">{label}</text>
  <text x="338" y="246" fill="{accent}" font-family="Arial" font-size="15">OBJ + STL saved locally</text>
  <text x="338" y="276" fill="#7d8590" font-family="Arial" font-size="14">Seat, back, legs, and arm rests</text>
</svg>"""
        path.write_text(svg, encoding="utf-8")
        return

    points = []
    for i in range(32):
        theta = 2 * math.pi * i / 32
        rad = 84 + 9 * math.sin(theta * 5 + seed)
        points.append(f"{160 + math.cos(theta) * rad:.1f},{150 + math.sin(theta) * rad:.1f}")
    silhouette = " ".join(points)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" viewBox="0 0 640 420">
  <defs>
    <radialGradient id="g" cx="35%" cy="25%" r="70%">
      <stop offset="0" stop-color="white" stop-opacity=".92"/>
      <stop offset=".45" stop-color="{accent}" stop-opacity=".95"/>
      <stop offset="1" stop-color="#111827"/>
    </radialGradient>
  </defs>
  <rect width="640" height="420" fill="#0d1117"/>
  <g transform="translate(160 45)">
    <ellipse cx="0" cy="238" rx="118" ry="22" fill="#000" opacity=".32"/>
    <polygon points="{silhouette}" transform="translate(-160 -20)" fill="url(#g)" stroke="{accent}" stroke-width="3"/>
    <path d="M-70 250 C-25 274 42 274 82 246" fill="none" stroke="#e6edf3" opacity=".35" stroke-width="3"/>
  </g>
  <text x="338" y="118" fill="#e6edf3" font-family="Arial" font-size="28" font-weight="700">Generated Mesh</text>
  <text x="338" y="154" fill="#7d8590" font-family="Arial" font-size="16">Kind: {kind}</text>
  <text x="338" y="184" fill="#7d8590" font-family="Arial" font-size="16">{label}</text>
  <text x="338" y="246" fill="{accent}" font-family="Arial" font-size="15">OBJ + MTL saved locally</text>
  <text x="338" y="276" fill="#7d8590" font-family="Arial" font-size="14">Also exports STL and metadata</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def generate_mesh(prompt: str, output_root: str | Path = "outputs/generated") -> MeshAsset:
    prompt = (prompt or "").strip() or "A dreamer world object"
    seed = _prompt_seed(prompt)
    kind = _choose_kind(prompt)
    material_name, color = _choose_material(prompt)
    slug = _slugify(prompt)
    job_id = f"{slug}-{seed:08x}"
    out_dir = Path(output_root) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    vertices, faces = _build_mesh(kind, seed)
    obj_path = out_dir / f"{slug}.obj"
    mtl_path = out_dir / f"{slug}.mtl"
    stl_path = out_dir / f"{slug}.stl"
    preview_path = out_dir / "preview.svg"
    metadata_path = out_dir / "metadata.json"
    quality = _mesh_quality(vertices, faces)
    _write_obj(obj_path, mtl_path.name, material_name, vertices, faces)
    _write_mtl(mtl_path, material_name, color)
    _write_stl(stl_path, slug, vertices, faces)
    _write_preview(preview_path, prompt, kind, color, seed)
    metadata_path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "prompt": prompt,
                "kind": kind,
                "material": material_name,
                "vertices": len(vertices),
                "faces": len(faces),
                "quality": quality,
                "files": {
                    "obj": obj_path.name,
                    "mtl": mtl_path.name,
                    "stl": stl_path.name,
                    "preview": preview_path.name,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return MeshAsset(
        job_id=job_id,
        prompt=prompt,
        name=slug,
        obj_path=obj_path,
        mtl_path=mtl_path,
        stl_path=stl_path,
        preview_path=preview_path,
        metadata_path=metadata_path,
        vertices=len(vertices),
        faces=len(faces),
        kind=kind,
        material=material_name,
        quality=quality,
    )
