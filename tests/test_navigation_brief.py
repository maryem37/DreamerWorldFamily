from inference import TextImaginer
from app import _navigation_report_markdown


def test_navigation_prediction_includes_execution_brief():
    result = TextImaginer().imagine(
        "Walk down the stairs and walk towards the red sofa. Turn left and enter the bathroom."
    )

    brief = result["execution_brief"]
    assert brief["goal"] == "bathroom"
    assert brief["immediate_action"] == "turn left, then move into bathroom"
    assert any("red sofa" in item for item in brief["checkpoints"])
    assert any("sink" in item for item in brief["verify"])
    assert "scan left" in brief["recovery"]
    assert "stairs" in brief["risk"]


def test_navigation_prediction_includes_route_trace():
    result = TextImaginer().imagine(
        "Walk down the stairs and walk towards the red sofa. Turn left and enter the bathroom."
    )

    trace = result["route_trace"]
    assert [step["step"] for step in trace] == [1, 2, 3, 4]
    assert trace[1]["command"] == "move toward red sofa"
    assert trace[2]["command"] == "rotate left"
    assert trace[2]["heading_delta_deg"] == -90
    assert trace[-1]["command"] == "enter bathroom"


def test_navigation_prediction_includes_simulated_observations():
    result = TextImaginer().imagine(
        "Walk down the stairs and walk towards the red sofa. Turn left and enter the bathroom."
    )

    observations = result["simulated_observations"]
    assert observations[0]["viewpoint_id"] == "vp_001"
    assert observations[-1]["viewpoint_id"] == "vp_004"
    assert observations[2]["pose"]["heading_deg"] == 270.0
    assert "red sofa" in observations[1]["visible"]
    assert "bathroom" in observations[-1]["visible"]


def test_navigation_report_markdown_contains_use_case_and_route():
    nav = TextImaginer().imagine(
        "Walk down the stairs and walk towards the red sofa. Turn left and enter the bathroom."
    )
    world = {"camera": {"yaw": 0, "pitch": 15, "fov": 60, "direction": "front"}}
    preview = {"target": "bathroom"}

    report = _navigation_report_markdown(nav, world, preview)

    assert "# Navigation Mission Brief" in report
    assert "## Use Case" in report
    assert "turn left" in report
    assert "vp_004" in report
    assert "bathroom" in report
