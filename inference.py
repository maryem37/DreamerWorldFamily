"""DWF inference pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.mesh_generator import generate_mesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class NavigationCue:
    cue_type: str
    text: str
    step: int


@dataclass
class ImaginationCandidate:
    rank: int
    hypothesis: str
    next_action: str
    confidence: float
    evidence: List[str]


@dataclass
class ExecutionBrief:
    goal: str
    immediate_action: str
    checkpoints: List[str]
    verify: List[str]
    recovery: str
    risk: str


@dataclass
class RouteStep:
    step: int
    command: str
    heading_delta_deg: int
    distance_m: float
    expected_observation: str
    stop_when: str


@dataclass
class SimulatedObservation:
    step: int
    viewpoint_id: str
    pose: Dict[str, float]
    visible: List[str]
    progress: float
    confidence: float


class TextImaginer:
    """Deterministic semantic imagination for navigation instructions.

    This is not a trained VLN model; it is a real parser/ranker that turns the
    user's instruction into structured cues and grounded next-state hypotheses.
    The output changes with the instruction and includes evidence so bad inputs
    are easier to diagnose.
    """

    ROOM_WORDS = {
        "bathroom",
        "bedroom",
        "kitchen",
        "hallway",
        "corridor",
        "living room",
        "dining room",
        "office",
        "garage",
        "balcony",
        "lobby",
        "foyer",
        "stairs",
        "staircase",
    }
    OBJECT_PATTERNS = [
        r"\b(?:red|blue|green|white|black|brown|yellow|gray|grey|wooden|metal|glass|large|small|round|square)\s+[a-z]+(?:\s+[a-z]+)?",
        r"\b(?:sofa|couch|table|chair|bed|door|window|sink|toilet|shower|tv|television|cabinet|counter|fridge|painting|lamp)\b",
    ]
    TURN_WORDS = {"left", "right", "around", "back"}
    MOVE_WORDS = {"walk", "go", "move", "continue", "enter", "exit", "pass", "approach", "towards", "toward", "stop"}

    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = checkpoint

    def _split_steps(self, instruction: str) -> List[str]:
        parts = re.split(r"[.;]|\bthen\b|\band then\b", instruction, flags=re.IGNORECASE)
        return [p.strip(" ,") for p in parts if p.strip(" ,")]

    def _extract_cues(self, instruction: str) -> List[NavigationCue]:
        cues: List[NavigationCue] = []
        for step_idx, step in enumerate(self._split_steps(instruction), start=1):
            lower = step.lower()
            for room in sorted(self.ROOM_WORDS, key=len, reverse=True):
                if re.search(rf"\b{re.escape(room)}\b", lower):
                    cues.append(NavigationCue("room", room, step_idx))
            for turn in self.TURN_WORDS:
                if re.search(rf"\b{turn}\b", lower):
                    cues.append(NavigationCue("turn", turn, step_idx))
            for move in self.MOVE_WORDS:
                if re.search(rf"\b{move}\b", lower):
                    cues.append(NavigationCue("motion", move, step_idx))
            for pattern in self.OBJECT_PATTERNS:
                for match in re.finditer(pattern, lower):
                    text = match.group(0).strip()
                    if text not in self.ROOM_WORDS:
                        cues.append(NavigationCue("landmark", text, step_idx))
        deduped: List[NavigationCue] = []
        seen = set()
        for cue in cues:
            if cue.cue_type == "landmark" and any(
                prior.cue_type == "landmark" and cue.text in prior.text and cue.step == prior.step for prior in deduped
            ):
                continue
            key = (cue.cue_type, cue.text, cue.step)
            if key not in seen:
                seen.add(key)
                deduped.append(cue)
        return deduped

    def _latest(self, cues: List[NavigationCue], cue_type: str) -> Optional[NavigationCue]:
        selected = [cue for cue in cues if cue.cue_type == cue_type]
        return selected[-1] if selected else None

    def _candidate_list(self, cues: List[NavigationCue], n_candidates: int) -> List[ImaginationCandidate]:
        landmarks = [cue for cue in cues if cue.cue_type == "landmark"]
        rooms = [cue for cue in cues if cue.cue_type == "room"]
        turns = [cue for cue in cues if cue.cue_type == "turn"]
        motions = [cue for cue in cues if cue.cue_type == "motion"]

        target_room = rooms[-1].text if rooms else "the next connected area"
        last_landmark = landmarks[-1].text if landmarks else "the strongest visible landmark"
        turn = turns[-1].text if turns else "forward"
        action = "enter" if any(c.text == "enter" for c in motions) else "continue"

        candidates = [
            ImaginationCandidate(
                rank=1,
                hypothesis=f"The agent is near {last_landmark} and should {action} toward {target_room}.",
                next_action=self._next_action(turn, target_room),
                confidence=self._confidence(cues, required=("landmark", "room")),
                evidence=self._evidence(cues, ("landmark", "room", "turn")),
            ),
            ImaginationCandidate(
                rank=2,
                hypothesis=f"The route likely transitions through a hallway or doorway before reaching {target_room}.",
                next_action="slow down at the next opening and verify the room label or fixtures",
                confidence=self._confidence(cues, required=("room",)),
                evidence=self._evidence(cues, ("room", "motion")),
            ),
            ImaginationCandidate(
                rank=3,
                hypothesis=f"If {last_landmark} is no longer visible, recover by rotating toward the last specified turn cue.",
                next_action=f"scan {turn} and reacquire the referenced landmark",
                confidence=self._confidence(cues, required=("turn", "landmark")),
                evidence=self._evidence(cues, ("turn", "landmark")),
            ),
        ]
        return candidates[: max(1, n_candidates)]

    def _next_action(self, turn: str, target_room: str) -> str:
        if turn in {"left", "right"}:
            return f"turn {turn}, then move into {target_room}"
        if turn in {"around", "back"}:
            return f"turn {turn} and check for the path into {target_room}"
        return f"move forward toward {target_room}"

    def _confidence(self, cues: List[NavigationCue], required: tuple[str, ...]) -> float:
        if not cues:
            return 0.15
        present = {cue.cue_type for cue in cues}
        score = 0.35 + 0.15 * min(len(cues), 4)
        score += 0.15 * sum(1 for cue_type in required if cue_type in present)
        return round(min(score, 0.92), 2)

    def _evidence(self, cues: List[NavigationCue], cue_types: tuple[str, ...]) -> List[str]:
        evidence = [f"step {cue.step}: {cue.cue_type}={cue.text}" for cue in cues if cue.cue_type in cue_types]
        return evidence[:6] or ["no strong semantic cue found"]

    def _execution_brief(self, cues: List[NavigationCue], candidates: List[ImaginationCandidate]) -> ExecutionBrief:
        rooms = [cue.text for cue in cues if cue.cue_type == "room"]
        landmarks = [cue.text for cue in cues if cue.cue_type == "landmark"]
        turns = [cue.text for cue in cues if cue.cue_type == "turn"]
        motions = [cue.text for cue in cues if cue.cue_type == "motion"]

        goal = rooms[-1] if rooms else (landmarks[-1] if landmarks else "the next target")
        immediate_action = candidates[0].next_action if candidates else f"move toward {goal}"
        checkpoints: List[str] = []
        if "stairs" in rooms or "staircase" in rooms:
            checkpoints.append("Confirm you have cleared the stairs before searching for the next landmark.")
        if landmarks:
            checkpoints.append(f"Keep {landmarks[-1]} as the last known anchor before the turn.")
        if turns:
            checkpoints.append(f"After the anchor, rotate {turns[-1]} and look for an open doorway.")
        if any(move in motions for move in ("enter", "exit", "pass")):
            checkpoints.append(f"Cross the threshold only when the destination matches {goal}.")
        if not checkpoints:
            checkpoints.append("Advance slowly and keep the next visible landmark centered.")

        verify = [f"Destination should read visually as {goal}."]
        if goal == "bathroom":
            verify.append("Look for bathroom fixtures such as a sink, toilet, shower, mirror, or tiled surfaces.")
        if landmarks:
            verify.append(f"Do not leave the area until {landmarks[-1]} has been positively matched.")

        recovery_turn = turns[-1] if turns else "left and right"
        recovery = f"If the target is not visible, stop, scan {recovery_turn}, and return to the last confirmed anchor."
        risk = "High risk: stairs plus a room transition need slower movement." if any(room in rooms for room in ("stairs", "staircase")) else "Normal risk: verify the doorway before entering."
        return ExecutionBrief(goal, immediate_action, checkpoints[:4], verify[:3], recovery, risk)

    def _route_trace(self, cues: List[NavigationCue]) -> List[RouteStep]:
        rooms = [cue.text for cue in cues if cue.cue_type == "room"]
        landmarks = [cue.text for cue in cues if cue.cue_type == "landmark"]
        turns = [cue.text for cue in cues if cue.cue_type == "turn"]
        target = rooms[-1] if rooms else "target area"
        anchor = landmarks[-1] if landmarks else "next landmark"
        turn = turns[-1] if turns else "forward"
        turn_delta = {"left": -90, "right": 90, "around": 180, "back": 180}.get(turn, 0)

        trace = [
            RouteStep(
                1,
                "advance from current viewpoint",
                0,
                1.2,
                "stairs or landing should remain in peripheral view",
                "path ahead is level and unobstructed",
            ),
            RouteStep(
                2,
                f"move toward {anchor}",
                0,
                2.0,
                f"{anchor} grows larger and stays near center frame",
                f"{anchor} is within one room segment",
            ),
        ]
        if turn_delta:
            trace.append(
                RouteStep(
                    3,
                    f"rotate {turn}",
                    turn_delta,
                    0.0,
                    "doorway or room opening appears after rotation",
                    "opening is centered enough to enter",
                )
            )
        trace.append(
            RouteStep(
                len(trace) + 1,
                f"enter {target}",
                0,
                1.0,
                f"inside view should match {target}",
                f"{target} visual evidence is visible",
            )
        )
        return trace

    def _simulated_observations(self, trace: List[RouteStep], cues: List[NavigationCue]) -> List[SimulatedObservation]:
        landmarks = [cue.text for cue in cues if cue.cue_type == "landmark"]
        rooms = [cue.text for cue in cues if cue.cue_type == "room"]
        anchor = landmarks[-1] if landmarks else "landmark"
        target = rooms[-1] if rooms else "target area"
        x, y, heading = 0.0, 0.0, 0.0
        observations: List[SimulatedObservation] = []
        total = max(len(trace), 1)
        for idx, step in enumerate(trace, start=1):
            heading = (heading + step.heading_delta_deg) % 360
            if step.distance_m:
                x += round(step.distance_m * 0.7, 2)
                y += round(step.distance_m * 0.18 if idx < total else step.distance_m * 0.45, 2)
            visible = ["stairs"] if idx == 1 and any(room in rooms for room in ("stairs", "staircase")) else []
            if idx >= 2:
                visible.append(anchor)
            if idx >= total - 1:
                visible.append("doorway")
            if idx == total:
                visible.append(target)
            observations.append(
                SimulatedObservation(
                    step=idx,
                    viewpoint_id=f"vp_{idx:03d}",
                    pose={"x": round(x, 2), "y": round(y, 2), "z": 1.6, "heading_deg": round(heading, 1)},
                    visible=list(dict.fromkeys(visible)),
                    progress=round(idx / total, 2),
                    confidence=round(max(0.48, 0.92 - (idx - 1) * 0.06), 2),
                )
            )
        return observations

    def imagine(self, instruction: str, n_candidates: int = 3) -> Dict:
        instruction = instruction.strip()
        cues = self._extract_cues(instruction)
        candidates = self._candidate_list(cues, n_candidates)
        route_trace = self._route_trace(cues)
        target = self._latest(cues, "room") or self._latest(cues, "landmark")
        state = (
            f"Parsed {len(cues)} navigation cues. "
            f"Current target: {target.text if target else 'unknown, need more context'}."
        )
        return {
            "instruction": instruction,
            "state_estimation": state,
            "semantic_cues": [asdict(cue) for cue in cues],
            "imaginations": [asdict(candidate) for candidate in candidates],
            "execution_brief": asdict(self._execution_brief(cues, candidates)),
            "route_trace": [asdict(step) for step in route_trace],
            "simulated_observations": [asdict(obs) for obs in self._simulated_observations(route_trace, cues)],
            "n_candidates": len(candidates),
            "mode": "semantic_parser",
        }


class Text3DGenerator:
    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = checkpoint

    def generate(self, prompt: str, output_dir: str, geo_iters: int = 2000, app_iters: int = 1000) -> Dict:
        started = time.perf_counter()
        asset = generate_mesh(prompt, output_dir)
        elapsed = time.perf_counter() - started
        result = {
            "prompt": prompt,
            "geo_iters": geo_iters,
            "app_iters": app_iters,
            "elapsed_seconds": round(elapsed, 4),
            "output_dir": str(Path(output_dir)),
            "job_id": asset.job_id,
            "kind": asset.kind,
            "material": asset.material,
            "vertices": asset.vertices,
            "faces": asset.faces,
            "quality": asset.quality,
            "files": {
                "obj": str(asset.obj_path),
                "mtl": str(asset.mtl_path),
                "stl": str(asset.stl_path),
                "preview": str(asset.preview_path),
                "metadata": str(asset.metadata_path),
            },
            "modules": {"backend": "procedural_mesh", "checkpoint": self.checkpoint},
        }
        with open(Path(output_dir) / "generation_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result


def _yaw_to_direction(yaw: float) -> str:
    yaw = yaw % 360
    if yaw < 45 or yaw >= 315:
        return "front"
    if yaw < 135:
        return "right"
    if yaw < 225:
        return "back"
    return "left"


class WorldInference:
    def infer(self, instruction: str, camera_yaw: float = 0.0, camera_pitch: float = 15.0, camera_fov: float = 60.0) -> Dict:
        imagination = TextImaginer().imagine(instruction)
        top = imagination["imaginations"][0] if imagination["imaginations"] else {}
        return {
            "instruction": instruction,
            "camera": {"yaw": camera_yaw, "pitch": camera_pitch, "fov": camera_fov, "direction": _yaw_to_direction(camera_yaw)},
            "navigation": imagination,
            "recommended_action": top.get("next_action"),
            "confidence": top.get("confidence", 0.0),
            "world_features_shape": [1, 32, 1024],
            "generation_ready": True,
            "mode": "semantic_world_inference",
        }


def parse_args():
    p = argparse.ArgumentParser(description="DWF Inference")
    p.add_argument("--task", choices=["imagine", "generate_3d", "world", "figures"], default="imagine")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--instruction", default="Walk down the stairs and walk towards the red sofa. Turn left and enter the bathroom.")
    p.add_argument("--prompt", default="A marble bust of an angel, 3D model, high resolution")
    p.add_argument("--n_candidates", type=int, default=3)
    p.add_argument("--geo_iters", type=int, default=2000)
    p.add_argument("--app_iters", type=int, default=1000)
    p.add_argument("--camera_yaw", type=float, default=0.0)
    p.add_argument("--camera_pitch", type=float, default=15.0)
    p.add_argument("--camera_fov", type=float, default=60.0)
    p.add_argument("--output_dir", default="outputs/inference")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.task == "imagine":
        result = TextImaginer(args.checkpoint).imagine(args.instruction, args.n_candidates)
        out = Path(args.output_dir) / "imagination_result.json"
    elif args.task == "generate_3d":
        result = Text3DGenerator(args.checkpoint).generate(args.prompt, str(Path(args.output_dir) / "3d_generation"), args.geo_iters, args.app_iters)
        out = Path(args.output_dir) / "generation_result.json"
    elif args.task == "world":
        result = WorldInference().infer(args.instruction, args.camera_yaw, args.camera_pitch, args.camera_fov)
        out = Path(args.output_dir) / "world_result.json"
    else:
        from src.utils.visualization import generate_demo_figures

        generate_demo_figures(str(Path(args.output_dir) / "figures"))
        result = {"figures": str(Path(args.output_dir) / "figures")}
        out = Path(args.output_dir) / "figures_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    log.info("Saved to %s", out)


if __name__ == "__main__":
    main()
