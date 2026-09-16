from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np
from gymnasium import spaces

from marketcanvas.canvas import Canvas
from marketcanvas.elements import ShapeVariant
from marketcanvas.observer import MAX_ELEMENTS


class ActionType(IntEnum):
    NO_OP = 0
    ADD_TEXT = 1
    ADD_SHAPE = 2
    ADD_IMAGE = 3
    MOVE = 4
    RESIZE = 5
    CHANGE_COLOR = 6
    REMOVE = 7
    SET_TEXT = 8
    DONE = 9


NUM_ACTION_TYPES = len(ActionType)

# Parameter vector layout (fixed-size, interpreted per action type):
#   [0] target_id (normalized 0-1, scaled to element ID range)
#   [1] x (normalized)
#   [2] y (normalized)
#   [3] width (normalized)
#   [4] height (normalized)
#   [5] r (normalized 0-1)
#   [6] g (normalized 0-1)
#   [7] b (normalized 0-1)
#   [8] shape_variant (0 = rectangle, 1 = button)
#   [9] font_size (normalized, maps to 8-72)
PARAM_DIM = 10


def build_action_space() -> spaces.Dict:
    return spaces.Dict({
        "action_type": spaces.Discrete(NUM_ACTION_TYPES),
        "params": spaces.Box(low=0.0, high=1.0, shape=(PARAM_DIM,), dtype=np.float32),
    })


def decode_params(params: np.ndarray, canvas: Canvas) -> dict[str, Any]:
    """Denormalize the parameter vector into concrete values."""
    element_ids = [e.id for e in canvas.elements]

    return {
        "target_id": element_ids[int(params[0] * (len(element_ids) - 1))] if element_ids else 0,
        "x": int(params[1] * canvas.width),
        "y": int(params[2] * canvas.height),
        "width": max(20, int(params[3] * canvas.width)),
        "height": max(20, int(params[4] * canvas.height)),
        "color": (int(params[5] * 255), int(params[6] * 255), int(params[7] * 255)),
        "shape_variant": ShapeVariant.BUTTON if params[8] > 0.5 else ShapeVariant.RECTANGLE,
        "font_size": int(8 + params[9] * 64),
    }


def execute_action(canvas: Canvas, action: dict[str, Any]) -> dict[str, Any]:
    """Execute an action dict on the canvas. Returns an info dict."""
    action_type = ActionType(action["action_type"])
    params = np.array(action["params"], dtype=np.float32) if not isinstance(action["params"], np.ndarray) else action["params"]
    p = decode_params(params, canvas)

    result: dict[str, Any] = {"action": action_type.name, "success": True}

    if action_type == ActionType.NO_OP:
        pass

    elif action_type == ActionType.DONE:
        result["done"] = True

    elif action_type == ActionType.ADD_TEXT:
        if len(canvas.elements) >= MAX_ELEMENTS:
            result["success"] = False
            result["reason"] = "max elements reached"
        else:
            elem = canvas.add_text(
                content="Text",
                x=p["x"], y=p["y"],
                width=p["width"], height=p["height"],
                font_size=p["font_size"],
                text_color=p["color"],
            )
            result["element_id"] = elem.id

    elif action_type == ActionType.ADD_SHAPE:
        if len(canvas.elements) >= MAX_ELEMENTS:
            result["success"] = False
            result["reason"] = "max elements reached"
        else:
            elem = canvas.add_shape(
                x=p["x"], y=p["y"],
                width=p["width"], height=p["height"],
                variant=p["shape_variant"],
                fill_color=p["color"],
            )
            result["element_id"] = elem.id

    elif action_type == ActionType.ADD_IMAGE:
        if len(canvas.elements) >= MAX_ELEMENTS:
            result["success"] = False
            result["reason"] = "max elements reached"
        else:
            elem = canvas.add_image(
                x=p["x"], y=p["y"],
                width=p["width"], height=p["height"],
                fill_color=p["color"],
            )
            result["element_id"] = elem.id

    elif action_type == ActionType.MOVE:
        result["success"] = canvas.move_element(p["target_id"], p["x"], p["y"])

    elif action_type == ActionType.RESIZE:
        result["success"] = canvas.resize_element(p["target_id"], p["width"], p["height"])

    elif action_type == ActionType.CHANGE_COLOR:
        elem = canvas.get_element(p["target_id"])
        if elem is None:
            result["success"] = False
        else:
            props = {}
            if hasattr(elem, "fill_color"):
                props["fill_color"] = p["color"]
            if hasattr(elem, "text_color"):
                props["text_color"] = p["color"]
            canvas.update_element(p["target_id"], **props)

    elif action_type == ActionType.REMOVE:
        result["success"] = canvas.remove_element(p["target_id"])

    elif action_type == ActionType.SET_TEXT:
        result["success"] = canvas.update_element(p["target_id"], content="Text")

    return result


def execute_semantic_action(canvas: Canvas, action_name: str, **kwargs: Any) -> dict[str, Any]:
    """Execute a named action with explicit keyword arguments.

    Provides the clean interface used by the MCP server and demo script,
    bypassing the normalized parameter vector encoding.
    """
    result: dict[str, Any] = {"action": action_name, "success": True}

    if action_name == "add_text":
        if len(canvas.elements) >= MAX_ELEMENTS:
            result["success"] = False
            return result
        elem = canvas.add_text(
            content=kwargs.get("content", "Text"),
            x=kwargs.get("x", 0),
            y=kwargs.get("y", 0),
            width=kwargs.get("width", 200),
            height=kwargs.get("height", 50),
            font_size=kwargs.get("font_size", 16),
            text_color=kwargs.get("text_color", (0, 0, 0)),
            background_color=kwargs.get("background_color", (255, 255, 255)),
        )
        result["element_id"] = elem.id

    elif action_name == "add_shape":
        if len(canvas.elements) >= MAX_ELEMENTS:
            result["success"] = False
            return result
        variant_str = kwargs.get("variant", "rectangle")
        variant = ShapeVariant(variant_str) if isinstance(variant_str, str) else variant_str
        elem = canvas.add_shape(
            x=kwargs.get("x", 0),
            y=kwargs.get("y", 0),
            width=kwargs.get("width", 100),
            height=kwargs.get("height", 50),
            variant=variant,
            fill_color=kwargs.get("fill_color", (200, 200, 200)),
            border_color=kwargs.get("border_color", (0, 0, 0)),
        )
        result["element_id"] = elem.id

    elif action_name == "add_image":
        if len(canvas.elements) >= MAX_ELEMENTS:
            result["success"] = False
            return result
        elem = canvas.add_image(
            x=kwargs.get("x", 0),
            y=kwargs.get("y", 0),
            width=kwargs.get("width", 150),
            height=kwargs.get("height", 150),
            fill_color=kwargs.get("fill_color", (128, 128, 128)),
            label=kwargs.get("label", ""),
        )
        result["element_id"] = elem.id

    elif action_name == "move_element":
        result["success"] = canvas.move_element(
            kwargs["element_id"], kwargs["x"], kwargs["y"],
        )

    elif action_name == "resize_element":
        result["success"] = canvas.resize_element(
            kwargs["element_id"], kwargs["width"], kwargs["height"],
        )

    elif action_name == "change_color":
        elem = canvas.get_element(kwargs["element_id"])
        if elem is None:
            result["success"] = False
        else:
            props = {}
            color = kwargs.get("color", (200, 200, 200))
            if hasattr(elem, "fill_color"):
                props["fill_color"] = color
            if hasattr(elem, "text_color"):
                props["text_color"] = color
            canvas.update_element(kwargs["element_id"], **props)

    elif action_name == "set_text":
        result["success"] = canvas.update_element(
            kwargs["element_id"], content=kwargs.get("content", ""),
        )

    elif action_name == "remove_element":
        result["success"] = canvas.remove_element(kwargs["element_id"])

    elif action_name == "done":
        result["done"] = True

    else:
        result["success"] = False
        result["reason"] = f"unknown action: {action_name}"

    return result


class CursorState:
    """Tracks cursor position and selected element for low-level interaction."""

    def __init__(self):
        self.x: int = 0
        self.y: int = 0
        self.selected_id: int | None = None

    def move(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def click(self, canvas: Canvas) -> dict[str, Any]:
        hits = canvas.elements_at(self.x, self.y)
        if hits:
            self.selected_id = hits[0].id
            return {"action": "click", "selected": self.selected_id}
        self.selected_id = None
        return {"action": "click", "selected": None}

    def drag(self, canvas: Canvas, to_x: int, to_y: int) -> dict[str, Any]:
        if self.selected_id is None:
            return {"action": "drag", "success": False}
        success = canvas.move_element(self.selected_id, to_x, to_y)
        self.x, self.y = to_x, to_y
        return {"action": "drag", "success": success, "element_id": self.selected_id}

    def type_text(self, canvas: Canvas, text: str) -> dict[str, Any]:
        if self.selected_id is None:
            return {"action": "type", "success": False}
        success = canvas.update_element(self.selected_id, content=text)
        return {"action": "type", "success": success, "element_id": self.selected_id}
