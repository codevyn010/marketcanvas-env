"""MCP server exposing the MarketCanvas environment as LLM-callable tools.

Run with: python -m marketcanvas.server
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any

from mcp.server import MCPServer
from mcp.types import ImageContent, TextContent

from marketcanvas.canvas import Canvas
from marketcanvas.actions import execute_semantic_action
from marketcanvas.observer import get_semantic_state, render_to_array
from marketcanvas.rewards import TargetSpec, compute_reward

logger = logging.getLogger(__name__)

mcp = MCPServer("marketcanvas")

_canvas = Canvas()
_target_spec: TargetSpec | None = None
_step_count: int = 0
_render_count: int = 0
_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


@mcp.tool()
def reset_canvas(target_prompt: str) -> str:
    """Reset the canvas and begin a new design episode.

    Args:
        target_prompt: Natural language description of the desired marketing asset.
    """
    global _canvas, _target_spec, _step_count
    _canvas = Canvas()
    _target_spec = TargetSpec.from_prompt(target_prompt)
    _step_count = 0
    state = get_semantic_state(_canvas, _step_count)
    return json.dumps(state, indent=2)


@mcp.tool()
def get_canvas_state() -> str:
    """Return the current semantic state of the canvas as JSON."""
    state = get_semantic_state(_canvas, _step_count)
    return json.dumps(state, indent=2)


@mcp.tool()
def execute_action(action_name: str, params: str = "{}") -> str:
    """Execute a design action on the canvas.

    Args:
        action_name: One of add_text, add_shape, add_image, move_element,
                     resize_element, change_color, set_text, remove_element, done.
        params: JSON string of action parameters. Examples:
                add_text: {"content": "Hello", "x": 100, "y": 50, "width": 200, "height": 40}
                add_shape: {"x": 100, "y": 200, "width": 150, "height": 50, "variant": "button", "fill_color": [255, 255, 0]}
                move_element: {"element_id": 1, "x": 200, "y": 100}
                change_color: {"element_id": 1, "color": [255, 0, 0]}
    """
    global _step_count
    _step_count += 1

    kwargs = json.loads(params)

    for key in ("fill_color", "text_color", "background_color", "color", "border_color"):
        if key in kwargs and isinstance(kwargs[key], list):
            kwargs[key] = tuple(kwargs[key])

    result = execute_semantic_action(_canvas, action_name, **kwargs)
    state = get_semantic_state(_canvas, _step_count)
    return json.dumps({"action_result": result, "state": state}, indent=2)


@mcp.tool()
def get_current_reward() -> str:
    """Compute and return the current reward breakdown.

    Returns a JSON object with per-component scores and the final scalar reward.
    """
    if _target_spec is None:
        return json.dumps({"error": "No target prompt set. Call reset_canvas first."})
    scores = compute_reward(_canvas, _target_spec)
    return json.dumps(scores, indent=2)


@mcp.tool()
def render_canvas() -> list[TextContent | ImageContent]:
    """Render the current canvas state to a PNG image and save to output/."""
    global _render_count
    _render_count += 1

    arr = render_to_array(_canvas)
    from PIL import Image
    img = Image.fromarray(arr)

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    filename = f"canvas_{_render_count:03d}.png"
    filepath = os.path.join(_OUTPUT_DIR, filename)
    img.save(filepath, format="PNG")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return [
        ImageContent(data=b64, mimeType="image/png"),
        TextContent(type="text", text=f"Canvas rendered ({_canvas.width}x{_canvas.height}). Saved to {filepath}"),
    ]


if __name__ == "__main__":
    import sys
    print("MarketCanvas MCP server running on stdio. Waiting for client connection...", file=sys.stderr)
    mcp.run(transport="stdio")
