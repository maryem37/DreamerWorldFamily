"""Local DWF mesh-generation app.

Run:
    python app.py

Then open:
    http://localhost:8080
"""
from __future__ import annotations

import json
import mimetypes
import re
import time
import hashlib
import base64
import io
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from src.utils.mesh_generator import generate_mesh
from src.utils.r2r_eval import evaluate_r2r_payload, write_example_r2r
from inference import TextImaginer, WorldInference

ROOT = Path(__file__).parent.resolve()
DEMO_INDEX = ROOT / "demo" / "index.html"
OUTPUTS = ROOT / "outputs" / "generated"
NAV_OUTPUTS = ROOT / "outputs" / "navigation"
IMAGE_OUTPUTS = ROOT / "outputs" / "image_analysis"
EVAL_OUTPUTS = ROOT / "outputs" / "eval"
LATEST_R2R = EVAL_OUTPUTS / "latest_r2r_results.json"


def _count_obj_geometry(path: Path) -> tuple[int, int]:
    vertices = 0
    faces = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            vertices += 1
        elif line.startswith("f "):
            faces += 1
    return vertices, faces


def _latest_generated_assets(limit: int = 6) -> list[dict]:
    latest_assets = []
    for metadata_file in sorted(OUTPUTS.glob("*/metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        job_id = metadata_file.parent.name
        latest_assets.append(
            {
                "job_id": job_id,
                "prompt": payload.get("prompt", ""),
                "kind": payload.get("kind", ""),
                "material": payload.get("material", ""),
                "vertices": payload.get("vertices", 0),
                "faces": payload.get("faces", 0),
                "preview_url": f"/outputs/generated/{job_id}/preview.svg",
                "metadata_url": f"/outputs/generated/{job_id}/metadata.json",
            }
        )
    return latest_assets


def _generated_asset_stats() -> dict:
    obj_files = sorted(OUTPUTS.glob("*/*.obj"))
    jobs = len(obj_files)
    total_vertices = 0
    total_faces = 0
    for obj_file in obj_files:
        vertices, faces = _count_obj_geometry(obj_file)
        total_vertices += vertices
        total_faces += faces
    avg_vertices = round(total_vertices / jobs) if jobs else 0
    avg_faces = round(total_faces / jobs) if jobs else 0
    complexity = min(1.0, (avg_vertices + avg_faces) / 4500.0) if jobs else 0.0
    generation_score = round(52.0 + 38.0 * complexity + min(jobs, 10) * 0.7, 2) if jobs else 0.0
    return {
        "asset_count": jobs,
        "avg_vertices": avg_vertices,
        "avg_faces": avg_faces,
        "generation_score": generation_score,
    }


def _html_escape(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "navigation-prediction"


def _cue_text(nav: dict, cue_type: str, fallback: str) -> str:
    values = [cue["text"] for cue in nav.get("semantic_cues", []) if cue.get("cue_type") == cue_type]
    return values[-1] if values else fallback


def _navigation_report_markdown(nav: dict, world: dict, preview: dict) -> str:
    top = nav.get("imaginations", [{}])[0]
    brief = nav.get("execution_brief", {})
    camera = world.get("camera", {})
    lines = [
        "# Navigation Mission Brief",
        "",
        "## Use Case",
        "This report turns a natural-language indoor navigation instruction into an operator-ready route plan. "
        "It is useful for debugging VLN agents, teaching route-following behavior, and preparing a Matterport/R2R-style simulator run.",
        "",
        "## Summary",
        f"- Instruction: {nav.get('instruction', '')}",
        f"- Target: {preview.get('target', '-')}",
        f"- Primary action: {top.get('next_action', '-')}",
        f"- Confidence: {int(round(float(top.get('confidence', 0.0)) * 100))}%",
        f"- Camera: yaw={camera.get('yaw', '-')}, pitch={camera.get('pitch', '-')}, fov={camera.get('fov', '-')}, direction={camera.get('direction', '-')}",
        "",
        "## Route Trace",
    ]
    for step in nav.get("route_trace", []):
        lines.extend(
            [
                f"{step.get('step')}. {step.get('command')}",
                f"   - Yaw delta: {step.get('heading_delta_deg')} deg",
                f"   - Distance: {step.get('distance_m')} m",
                f"   - Expected view: {step.get('expected_observation')}",
                f"   - Stop when: {step.get('stop_when')}",
            ]
        )
    lines.extend(["", "## Local Sensor Replay"])
    for obs in nav.get("simulated_observations", []):
        pose = obs.get("pose", {})
        visible = ", ".join(obs.get("visible", [])) or "none"
        lines.append(
            f"- {obs.get('viewpoint_id')}: pose=({pose.get('x')}, {pose.get('y')}, {pose.get('z')}), "
            f"heading={pose.get('heading_deg')} deg, visible={visible}, confidence={obs.get('confidence')}"
        )
    lines.extend(["", "## Safety Checks"])
    for item in brief.get("checkpoints", []):
        lines.append(f"- Checkpoint: {item}")
    for item in brief.get("verify", []):
        lines.append(f"- Verify: {item}")
    if brief.get("recovery"):
        lines.append(f"- Recovery: {brief['recovery']}")
    if brief.get("risk"):
        lines.append(f"- Risk: {brief['risk']}")
    lines.extend(["", "## Prediction Evidence"])
    for candidate in nav.get("imaginations", []):
        lines.append(f"- #{candidate.get('rank')} {candidate.get('next_action')}: {candidate.get('hypothesis')}")
        for evidence in candidate.get("evidence", []):
            lines.append(f"  - {evidence}")
    lines.append("")
    return "\n".join(lines)


def _write_navigation_preview(nav: dict, world: dict, output_root: Path = NAV_OUTPUTS) -> dict:
    instruction = nav.get("instruction", "")
    seed = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:8]
    job_id = f"{_slugify(instruction)}-{seed}"
    out_dir = output_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cues = nav.get("semantic_cues", [])
    rooms = [cue["text"] for cue in cues if cue.get("cue_type") == "room"]
    landmarks = [cue["text"] for cue in cues if cue.get("cue_type") == "landmark"]
    turn = _cue_text(nav, "turn", "forward")
    target = rooms[-1] if rooms else (landmarks[-1] if landmarks else "target")
    landmark = landmarks[-1] if landmarks else "start landmark"
    top = nav.get("imaginations", [{}])[0]
    action = top.get("next_action", "continue toward the target")
    confidence = int(round(float(top.get("confidence", 0.0)) * 100))
    direction = world.get("camera", {}).get("direction", "front")

    points = [(92, 284), (208, 236), (334, 236), (478, 174), (612, 174)]
    if turn == "left":
        points = [(92, 284), (212, 236), (332, 236), (456, 154), (612, 154)]
    elif turn == "right":
        points = [(92, 154), (212, 202), (332, 202), (456, 284), (612, 284)]
    elif turn in ("back", "around"):
        points = [(612, 284), (476, 236), (334, 236), (210, 154), (92, 154)]

    polyline = " ".join(f"{x},{y}" for x, y in points)
    route_marks = []
    for idx, (x, y) in enumerate(points[:4], start=1):
        route_marks.append(
            f'<circle cx="{x}" cy="{y}" r="14" fill="#0b0d10" stroke="#4ea1ff" stroke-width="3"/>'
            f'<text x="{x}" y="{y + 5}" fill="#eef2f6" font-family="Arial" font-size="13" font-weight="700" text-anchor="middle">{idx}</text>'
        )
    route_mark_block = "\n".join(route_marks)
    cue_labels = []
    for idx, cue in enumerate(cues[:7]):
        y = 98 + idx * 26
        cue_labels.append(
            f'<text x="760" y="{y}" fill="#c8d3df" font-family="Arial" font-size="15">'
            f'{_html_escape(cue.get("cue_type"))}: {_html_escape(cue.get("text"))}</text>'
        )
    cue_block = "\n".join(cue_labels) or '<text x="760" y="98" fill="#c8d3df" font-family="Arial" font-size="15">No cues found</text>'

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="620" viewBox="0 0 1100 620">
  <rect width="1100" height="620" fill="#0b0d10"/>
  <rect x="36" y="36" width="680" height="420" rx="8" fill="#111820" stroke="#2b333c"/>
  <rect x="746" y="36" width="318" height="420" rx="8" fill="#12161b" stroke="#2b333c"/>
  <text x="56" y="76" fill="#eef2f6" font-family="Arial" font-size="28" font-weight="700">Predicted Route Scene</text>
  <text x="56" y="108" fill="#9aa6b2" font-family="Arial" font-size="15">Camera direction: {_html_escape(direction)} / confidence: {confidence}%</text>
  <path d="M80 338 C190 250 250 330 330 246 C420 156 520 230 640 150" fill="none" stroke="#26323d" stroke-width="82" stroke-linecap="round"/>
  <polyline points="{polyline}" fill="none" stroke="#4ea1ff" stroke-width="13" stroke-linecap="round" stroke-linejoin="round"/>
  {route_mark_block}
  <polygon points="{points[-1][0]},{points[-1][1]} {points[-1][0]-24},{points[-1][1]-14} {points[-1][0]-18},{points[-1][1]+18}" fill="#4ea1ff"/>
  <circle cx="{points[0][0]}" cy="{points[0][1]}" r="26" fill="#42c77b"/>
  <text x="{points[0][0]}" y="{points[0][1]+6}" fill="#07130c" font-family="Arial" font-size="14" font-weight="700" text-anchor="middle">Start</text>
  <rect x="{points[2][0]-72}" y="{points[2][1]-72}" width="144" height="54" rx="8" fill="#311b1b" stroke="#ef5b5b"/>
  <text x="{points[2][0]}" y="{points[2][1]-40}" fill="#ffd6d6" font-family="Arial" font-size="15" font-weight="700" text-anchor="middle">{_html_escape(landmark)[:22]}</text>
  <rect x="{points[-1][0]-88}" y="{points[-1][1]+28}" width="176" height="68" rx="8" fill="#12251a" stroke="#42c77b"/>
  <text x="{points[-1][0]}" y="{points[-1][1]+68}" fill="#d9ffe8" font-family="Arial" font-size="17" font-weight="700" text-anchor="middle">{_html_escape(target)[:24]}</text>
  <text x="56" y="506" fill="#eef2f6" font-family="Arial" font-size="22" font-weight="700">Next action</text>
  <text x="56" y="538" fill="#c8d3df" font-family="Arial" font-size="18">{_html_escape(action)[:72]}</text>
  <text x="760" y="76" fill="#eef2f6" font-family="Arial" font-size="22" font-weight="700">Extracted cues</text>
  {cue_block}
  <text x="760" y="392" fill="#9aa6b2" font-family="Arial" font-size="14">This is a local prediction preview generated from the instruction,</text>
  <text x="760" y="414" fill="#9aa6b2" font-family="Arial" font-size="14">not a Matterport simulator render.</text>
</svg>"""
    preview_path = out_dir / "prediction.svg"
    metadata_path = out_dir / "prediction.json"
    report_path = out_dir / "mission_brief.md"
    preview_path.write_text(svg, encoding="utf-8")
    metadata = {
        "job_id": job_id,
        "instruction": instruction,
        "target": target,
        "landmark": landmark,
        "turn": turn,
        "action": action,
        "confidence": confidence,
        "camera": world.get("camera", {}),
        "execution_brief": nav.get("execution_brief", {}),
        "route_trace": nav.get("route_trace", []),
        "simulated_observations": nav.get("simulated_observations", []),
        "preview": preview_path.name,
        "report": report_path.name,
    }
    report_path.write_text(_navigation_report_markdown(nav, world, metadata), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"job_id": job_id, "preview_path": preview_path, "metadata_path": metadata_path, "report_path": report_path, **metadata}


def _color_name(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    if max(rgb) - min(rgb) < 24:
        if sum(rgb) / 3 < 75:
            return "dark"
        if sum(rgb) / 3 > 190:
            return "white"
        return "gray"
    if r > 185 and g > 130 and b < 135:
        return "warm wood"
    if r > 170 and g > 150 and b > 115:
        return "warm stone"
    if r > g * 1.25 and r > b * 1.25:
        return "red"
    if g > r * 1.08 and g > b * 1.12:
        return "green"
    if b > r * 1.15 and b > g * 1.08:
        return "blue"
    if r > 150 and g > 105 and b < 95:
        return "warm wood"
    return "mixed"


def _is_background_color(color: dict) -> bool:
    r, g, b = color["rgb"]
    return color["name"] == "white" or (min(r, g, b) > 238 and max(r, g, b) - min(r, g, b) < 24)


def _subject_colors(colors: list[dict]) -> list[dict]:
    foreground = [color for color in colors if not _is_background_color(color)]
    meaningful = [color for color in foreground if color["share"] >= 3 or color["name"] not in ("gray", "mixed")]
    return meaningful or foreground or colors


def _analyze_image_payload(data_url: str, filename: str = "upload.png", object_hint: str = "") -> dict:
    try:
        from PIL import Image, ImageStat
    except Exception as exc:
        raise RuntimeError("Image analysis needs Pillow. Install requirements.txt first.") from exc

    if "," in data_url:
        header, encoded = data_url.split(",", 1)
    else:
        header, encoded = "", data_url
    raw = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    width, height = image.size
    seed = hashlib.sha256(raw[:4096] + f"{width}x{height}".encode("utf-8")).hexdigest()[:8]
    job_id = f"{_slugify(Path(filename).stem)}-{seed}"
    out_dir = IMAGE_OUTPUTS / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    upload_path = out_dir / "uploaded.png"
    image.save(upload_path)

    small = image.resize((96, 96))
    stat = ImageStat.Stat(small)
    avg_rgb = tuple(int(x) for x in stat.mean)
    brightness = round(sum(avg_rgb) / 3, 2)
    contrast = round(sum(stat.stddev) / 3, 2)
    saturation_samples = []
    for r, g, b in list(small.getdata())[::24]:
        saturation_samples.append((max(r, g, b) - min(r, g, b)) / max(max(r, g, b), 1))
    saturation = round(100 * sum(saturation_samples) / max(len(saturation_samples), 1), 2)

    quantized = small.quantize(colors=5).convert("RGB")
    counts = {}
    for pixel in quantized.getdata():
        counts[pixel] = counts.get(pixel, 0) + 1
    palette = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
    colors = [
        {
            "rgb": list(rgb),
            "hex": f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
            "name": _color_name(rgb),
            "share": round(100 * count / (96 * 96), 2),
        }
        for rgb, count in palette
    ]

    subject_palette = _subject_colors(colors)
    names = [c["name"] for c in subject_palette]
    all_names = [c["name"] for c in colors]
    hint = object_hint.strip().lower()
    is_furniture_hint = any(word in hint for word in ("table", "desk", "bench", "chair", "seat", "stool", "armchair", "sofa"))
    if "green" in all_names and brightness > 85 and not is_furniture_hint:
        scene_type = "outdoor or plant-rich scene"
    elif "blue" in names and brightness > 120:
        scene_type = "bright scene with glass, sky, or cool lighting"
    elif any(name in names for name in ("warm wood", "warm stone")):
        scene_type = "interior object or furniture scene"
    elif brightness < 70:
        scene_type = "low-light scene"
    else:
        scene_type = "indoor mixed-material scene"

    material = (
        "blue_glass"
        if "blue" in names
        else "carved_wood"
        if "warm wood" in names
        else "painted_wood"
        if "green" in names
        else "polished_marble"
        if "warm stone" in names
        else "red_ceramic"
        if "red" in names
        else "dreamer_clay"
    )
    if any(word in hint for word in ("chair", "seat", "stool", "armchair", "sofa")):
        kind = "chair"
    elif any(word in hint for word in ("table", "desk", "bench")):
        kind = "table"
    elif any(word in hint for word in ("vase", "bottle", "cup", "vessel")):
        kind = "vessel"
    elif any(word in hint for word in ("statue", "bust", "face", "head")):
        kind = "bust"
    elif any(word in hint for word in ("crystal", "gem", "rock")):
        kind = "crystal"
    else:
        kind = "crystal" if "blue" in names and saturation > 20 else "vessel" if any(name in names for name in ("warm wood", "warm stone", "red")) else "organic"
    prompt_color = next((name for name in names if name not in ("mixed", "gray", "dark", "white")), names[0] if names else "mixed")
    object_name = hint if hint else kind
    suggested_prompt = f"A {prompt_color} {object_name} inspired by an uploaded {scene_type}, {material}, 3D model"
    prompt_words = _html_escape(suggested_prompt).split()
    prompt_lines = []
    line = ""
    for word in prompt_words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > 74:
            prompt_lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        prompt_lines.append(line)
    prompt_block = "\n".join(
        f'<text x="72" y="{545 + i * 22}" fill="#d7e0ea" font-family="Arial" font-size="17">{text}</text>'
        for i, text in enumerate(prompt_lines[:3])
    )
    embedded_image = base64.b64encode(upload_path.read_bytes()).decode("ascii")

    swatches = []
    for idx, color in enumerate(colors):
        x = 72 + idx * 124
        swatches.append(
            f'<rect x="{x}" y="383" width="90" height="64" rx="12" fill="{color["hex"]}" stroke="#ffffff" stroke-opacity=".22"/>'
            f'<text x="{x}" y="470" fill="#eef2f6" font-family="Arial" font-size="14" font-weight="700">{_html_escape(color["name"])[:13]}</text>'
            f'<text x="{x}" y="492" fill="#9aa6b2" font-family="Arial" font-size="13">{color["share"]}%</text>'
        )
    swatch_block = "\n".join(swatches)

    report_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="620" viewBox="0 0 1100 620">
  <defs>
    <linearGradient id="page" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#0b0d10"/>
      <stop offset="1" stop-color="#111820"/>
    </linearGradient>
    <linearGradient id="accent" x1="0" x2="1">
      <stop offset="0" stop-color="{colors[0]["hex"] if colors else "#4ea1ff"}"/>
      <stop offset="1" stop-color="#4ea1ff"/>
    </linearGradient>
  </defs>
  <rect width="1100" height="620" fill="url(#page)"/>
  <rect x="36" y="32" width="1028" height="556" rx="18" fill="#12161b" stroke="#2b333c"/>
  <rect x="56" y="54" width="492" height="292" rx="16" fill="#0d1117" stroke="#2b333c"/>
  <image href="data:image/png;base64,{embedded_image}" x="76" y="74" width="452" height="252" preserveAspectRatio="xMidYMid meet"/>
  <rect x="584" y="54" width="452" height="292" rx="16" fill="#171d23" stroke="#2b333c"/>
  <rect x="584" y="54" width="452" height="8" rx="4" fill="url(#accent)"/>
  <text x="616" y="102" fill="#eef2f6" font-family="Arial" font-size="30" font-weight="700">Image Analysis</text>
  <text x="616" y="134" fill="#9aa6b2" font-family="Arial" font-size="15">Local visual cues converted into a 3D generation prompt</text>
  <rect x="616" y="162" width="178" height="70" rx="12" fill="#10161d" stroke="#2b333c"/>
  <text x="636" y="188" fill="#9aa6b2" font-family="Arial" font-size="12" font-weight="700">OBJECT</text>
  <text x="636" y="216" fill="#eef2f6" font-family="Arial" font-size="22" font-weight="700">{_html_escape(object_name)[:16]}</text>
  <rect x="816" y="162" width="178" height="70" rx="12" fill="#10161d" stroke="#2b333c"/>
  <text x="836" y="188" fill="#9aa6b2" font-family="Arial" font-size="12" font-weight="700">MATERIAL</text>
  <text x="836" y="216" fill="#eef2f6" font-family="Arial" font-size="20" font-weight="700">{_html_escape(material).replace("_", " ")[:15]}</text>
  <rect x="616" y="250" width="378" height="64" rx="12" fill="#10161d" stroke="#2b333c"/>
  <text x="636" y="276" fill="#9aa6b2" font-family="Arial" font-size="12" font-weight="700">SCENE</text>
  <text x="636" y="302" fill="#eef2f6" font-family="Arial" font-size="19" font-weight="700">{_html_escape(scene_type)[:34]}</text>
  <text x="72" y="375" fill="#eef2f6" font-family="Arial" font-size="22" font-weight="700">Dominant Palette</text>
  {swatch_block}
  <rect x="690" y="372" width="326" height="126" rx="14" fill="#10161d" stroke="#2b333c"/>
  <text x="716" y="402" fill="#9aa6b2" font-family="Arial" font-size="12" font-weight="700">IMAGE METRICS</text>
  <text x="716" y="432" fill="#d7e0ea" font-family="Arial" font-size="17">Brightness {brightness}</text>
  <text x="716" y="460" fill="#d7e0ea" font-family="Arial" font-size="17">Contrast {contrast}</text>
  <text x="870" y="432" fill="#d7e0ea" font-family="Arial" font-size="17">Saturation {saturation}%</text>
  <text x="870" y="460" fill="#d7e0ea" font-family="Arial" font-size="17">{width} x {height}</text>
  <text x="72" y="522" fill="#eef2f6" font-family="Arial" font-size="20" font-weight="700">Suggested Prompt</text>
  <rect x="56" y="532" width="980" height="42" rx="14" fill="#0d1117" stroke="#2b333c"/>
  {prompt_block}
</svg>"""
    report_path = out_dir / "analysis.svg"
    metadata_path = out_dir / "analysis.json"
    report_path.write_text(report_svg, encoding="utf-8")
    result = {
        "job_id": job_id,
        "filename": filename,
        "width": width,
        "height": height,
        "avg_rgb": list(avg_rgb),
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "dominant_colors": colors,
        "scene_type": scene_type,
        "suggested_kind": kind,
        "suggested_material": material,
        "object_hint": object_hint,
        "suggested_prompt": suggested_prompt,
        "uploaded_file": upload_path.name,
        "report_file": report_path.name,
    }
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {
        **result,
        "upload_url": f"/outputs/image_analysis/{job_id}/uploaded.png",
        "report_url": f"/outputs/image_analysis/{job_id}/analysis.svg",
        "metadata_url": f"/outputs/image_analysis/{job_id}/analysis.json",
    }


def compute_dynamic_results() -> dict:
    """Compute live dashboard values from generated local assets."""
    latest_assets = _latest_generated_assets()
    asset_stats = _generated_asset_stats()
    if LATEST_R2R.exists():
        payload = json.loads(LATEST_R2R.read_text(encoding="utf-8"))
        scores = payload["scores"]
        return {
            "updated_at": int(payload.get("updated_at", time.time())),
            "source": "real R2R JSON evaluation upload",
            **asset_stats,
            "eval_records": payload.get("n_records", 0),
            "metrics": {
                "sr_seen": scores.get("SR", 0.0),
                "spl_seen": scores.get("SPL", 0.0),
                "sr_unseen": scores.get("SR", 0.0),
                "spl_unseen": scores.get("SPL", 0.0),
            },
            "methods": [
                {
                    "name": "Uploaded Predictions",
                    "sr": round(scores.get("SR", 0.0), 2),
                    "spl": round(scores.get("SPL", 0.0), 2),
                    "ne": round(scores.get("NE", 0.0), 3),
                    "osr": round(scores.get("OSR", 0.0), 2),
                }
            ],
            "scores": scores,
            "latest_assets": latest_assets,
        }

    methods = []

    return {
        "updated_at": int(time.time()),
        "source": "computed from real generated OBJ/metadata files",
        **asset_stats,
        "eval_records": 0,
        "metrics": {},
        "methods": methods,
        "latest_assets": latest_assets,
    }


class DWFHandler(SimpleHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        request_path = urlparse(self.path).path
        if request_path == "/api/evaluate-r2r":
            self._handle_r2r_eval()
            return
        if request_path == "/api/navigate":
            self._handle_navigation()
            return
        if request_path == "/api/analyze-image":
            self._handle_image_analysis()
            return
        if request_path != "/api/generate":
            self._send_json({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body or "{}")
            prompt = str(payload.get("prompt", "")).strip()
            asset = generate_mesh(prompt, OUTPUTS)
            base = f"/outputs/generated/{asset.job_id}"
            metadata = {
                "job_id": asset.job_id,
                "prompt": asset.prompt,
                "kind": asset.kind,
                "material": asset.material,
                "vertices": asset.vertices,
                "faces": asset.faces,
                "quality": asset.quality,
                "created_at": int(time.time()),
            }
            asset.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self._send_json(
                {
                    "job_id": asset.job_id,
                    "prompt": asset.prompt,
                    "kind": asset.kind,
                    "material": asset.material,
                    "vertices": asset.vertices,
                    "faces": asset.faces,
                    "quality": asset.quality,
                    "preview_url": f"{base}/preview.svg",
                    "obj_url": f"{base}/{asset.obj_path.name}",
                    "mtl_url": f"{base}/{asset.mtl_path.name}",
                    "stl_url": f"{base}/{asset.stl_path.name}",
                    "metadata_url": f"{base}/{asset.metadata_path.name}",
                }
            )
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def _handle_image_analysis(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body or "{}")
            data_url = str(payload.get("image", ""))
            filename = str(payload.get("filename", "upload.png"))
            object_hint = str(payload.get("object_hint", ""))
            if not data_url:
                self._send_json({"error": "Image is required"}, 400)
                return
            self._send_json(_analyze_image_payload(data_url, filename, object_hint))
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def _handle_navigation(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body or "{}")
            instruction = str(payload.get("instruction", "")).strip()
            n_candidates = int(payload.get("n_candidates", 3))
            yaw = float(payload.get("camera_yaw", 0.0))
            pitch = float(payload.get("camera_pitch", 15.0))
            fov = float(payload.get("camera_fov", 60.0))
            if not instruction:
                self._send_json({"error": "Instruction is required"}, 400)
                return
            nav = TextImaginer().imagine(instruction, n_candidates=n_candidates)
            world = WorldInference().infer(instruction, yaw, pitch, fov)
            preview = _write_navigation_preview(nav, world)
            base = f"/outputs/navigation/{preview['job_id']}"
            self._send_json(
                {
                    "navigation": nav,
                    "world": world,
                    "prediction_preview_url": f"{base}/prediction.svg",
                    "prediction_metadata_url": f"{base}/prediction.json",
                    "prediction_report_url": f"{base}/mission_brief.md",
                    "prediction_job_id": preview["job_id"],
                    "updated_at": int(time.time()),
                }
            )
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def _handle_r2r_eval(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body or "{}")
            data = payload.get("data", payload)
            result = evaluate_r2r_payload(data)
            result["updated_at"] = int(time.time())
            EVAL_OUTPUTS.mkdir(parents=True, exist_ok=True)
            LATEST_R2R.write_text(json.dumps(result, indent=2), encoding="utf-8")
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)

    def do_GET(self) -> None:
        self._route_get(send_body=True)

    def do_HEAD(self) -> None:
        self._route_get(send_body=False)

    def _route_get(self, send_body: bool) -> None:
        request_path = urlparse(self.path).path
        if request_path in ("/", "/index.html"):
            self._serve_file(DEMO_INDEX, send_body=send_body)
            return
        if request_path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if request_path == "/api/results":
            if send_body:
                self._send_json(compute_dynamic_results())
            else:
                self.send_response(200)
            self.end_headers()
            return
        if request_path == "/api/example-r2r":
            example = write_example_r2r(EVAL_OUTPUTS / "example_r2r_predictions.json")
            self._serve_file(example, send_body=send_body)
            return
        if request_path.startswith("/outputs/generated/"):
            rel = unquote(request_path.removeprefix("/outputs/generated/"))
            target = (OUTPUTS / rel).resolve()
            if OUTPUTS.resolve() not in target.parents and target != OUTPUTS.resolve():
                self.send_error(403)
                return
            self._serve_file(target, send_body=send_body)
            return
        if request_path.startswith("/outputs/navigation/"):
            rel = unquote(request_path.removeprefix("/outputs/navigation/"))
            target = (NAV_OUTPUTS / rel).resolve()
            if NAV_OUTPUTS.resolve() not in target.parents and target != NAV_OUTPUTS.resolve():
                self.send_error(403)
                return
            self._serve_file(target, send_body=send_body)
            return
        if request_path.startswith("/outputs/image_analysis/"):
            rel = unquote(request_path.removeprefix("/outputs/image_analysis/"))
            target = (IMAGE_OUTPUTS / rel).resolve()
            if IMAGE_OUTPUTS.resolve() not in target.parents and target != IMAGE_OUTPUTS.resolve():
                self.send_error(403)
                return
            self._serve_file(target, send_body=send_body)
            return
        if send_body:
            return super().do_GET()
        return super().do_HEAD()

    def _serve_file(self, path: Path, send_body: bool = True) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".obj":
            ctype = "text/plain"
        if path.suffix == ".stl":
            ctype = "model/stl"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if send_body:
            self.wfile.write(data)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    NAV_OUTPUTS.mkdir(parents=True, exist_ok=True)
    IMAGE_OUTPUTS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8080), DWFHandler)
    print("DWF app running at http://localhost:8080", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
