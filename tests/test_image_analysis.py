import base64
import io

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image

from app import _analyze_image_payload, _color_name


def _data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_muted_green_is_named_green():
    assert _color_name((155, 175, 126)) == "green"


def test_image_prompt_uses_green_subject_over_white_background():
    image = Image.new("RGB", (700, 700), "white")
    for x in range(210, 490):
        for y in range(260, 560):
            image.putpixel((x, y), (155, 175, 126))

    result = _analyze_image_payload(_data_url(image), "green_table.png", "table")

    assert result["dominant_colors"][0]["name"] == "white"
    assert any(color["name"] == "green" for color in result["dominant_colors"])
    assert result["suggested_kind"] == "table"
    assert result["suggested_material"] == "painted_wood"
    assert result["suggested_prompt"].startswith("A green table ")
