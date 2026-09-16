from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gymnasium import spaces

from marketcanvas.canvas import Canvas
from marketcanvas.elements import CanvasElement, ElementType, ShapeVariant, TextElement, ShapeElement, ImageElement

MAX_ELEMENTS = 20
MAX_STEPS = 50
ELEMENT_FEATURE_DIM = 8


def build_observation_space(canvas_width: int, canvas_height: int) -> spaces.Dict:
    return spaces.Dict({
        "elements": spaces.Box(
            low=0.0,
            high=1.0,
            shape=(MAX_ELEMENTS, ELEMENT_FEATURE_DIM),
            dtype=np.float32,
        ),
        "element_count": spaces.Discrete(MAX_ELEMENTS + 1),
        "canvas_meta": spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        ),
    })


def _encode_element(elem: CanvasElement, canvas_w: int, canvas_h: int) -> np.ndarray:
    """Encode a single element as a normalized feature vector."""
    type_map = {ElementType.TEXT: 0.0, ElementType.SHAPE: 0.5, ElementType.IMAGE: 1.0}
    return np.array([
        elem.x / canvas_w,
        elem.y / canvas_h,
        elem.width / canvas_w,
        elem.height / canvas_h,
        elem.z_index / max(MAX_ELEMENTS, 1),
        type_map.get(elem.element_type, 0.0),
        1.0,  # active flag
        0.0,  # reserved
    ], dtype=np.float32)


def get_gymnasium_observation(canvas: Canvas, step_count: int = 0) -> dict[str, Any]:
    elements = canvas.elements[:MAX_ELEMENTS]
    obs = np.zeros((MAX_ELEMENTS, ELEMENT_FEATURE_DIM), dtype=np.float32)
    for i, elem in enumerate(elements):
        obs[i] = _encode_element(elem, canvas.width, canvas.height)
    return {
        "elements": obs,
        "element_count": len(elements),
        "canvas_meta": np.array([
            step_count / MAX_STEPS,
            len(elements) / MAX_ELEMENTS,
        ], dtype=np.float32),
    }


def get_semantic_state(canvas: Canvas, step_count: int = 0) -> dict[str, Any]:
    """Full JSON-serializable state for MCP / LLM consumption."""
    elements = canvas.elements
    state = canvas.to_dict()
    state["canvas"]["step_count"] = step_count

    relationships = []
    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            if a.overlaps(b):
                relationships.append({
                    "type": "overlaps",
                    "elements": [a.id, b.id],
                    "intersection_area": a.intersection_area(b),
                })
            elif _adjacent(a, b, threshold=10):
                relationships.append({
                    "type": "adjacent",
                    "elements": [a.id, b.id],
                })

    state["relationships"] = relationships
    return state


def render_to_array(canvas: Canvas) -> np.ndarray:
    """Render the canvas to an RGB numpy array (H, W, 3)."""
    img = _render_pil(canvas)
    return np.array(img)


def render_to_png(canvas: Canvas, path: str) -> None:
    img = _render_pil(canvas)
    img.save(path)


def _render_pil(canvas: Canvas) -> Image.Image:
    img = Image.new("RGB", (canvas.width, canvas.height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for elem in canvas.elements:
        if isinstance(elem, ShapeElement):
            if elem.shape_variant is ShapeVariant.BUTTON:
                radius = min(elem.width, elem.height) // 6
                draw.rounded_rectangle(
                    [elem.x, elem.y, elem.right - 1, elem.bottom - 1],
                    radius=radius,
                    fill=elem.fill_color,
                    outline=elem.border_color,
                    width=2,
                )
            else:
                draw.rectangle(
                    [elem.x, elem.y, elem.right - 1, elem.bottom - 1],
                    fill=elem.fill_color,
                    outline=elem.border_color,
                    width=2,
                )

        elif isinstance(elem, ImageElement):
            draw.rectangle(
                [elem.x, elem.y, elem.right - 1, elem.bottom - 1],
                fill=elem.fill_color,
                outline=(80, 80, 80),
            )
            label = elem.label or "IMG"
            _draw_centered_text(draw, elem, label, (80, 80, 80))

        elif isinstance(elem, TextElement):
            draw.rectangle(
                [elem.x, elem.y, elem.right - 1, elem.bottom - 1],
                fill=elem.background_color,
            )
            _draw_centered_text(draw, elem, elem.content, elem.text_color, elem.font_size)

    return img


_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    elem: CanvasElement,
    text: str,
    color: tuple[int, int, int],
    font_size: int = 14,
) -> None:
    font = _load_font(font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = elem.x + (elem.width - tw) / 2
    ty = elem.y + (elem.height - th) / 2
    draw.text((tx, ty), text, fill=color, font=font)


def _adjacent(a: CanvasElement, b: CanvasElement, threshold: int = 10) -> bool:
    if a.overlaps(b):
        return False
    gap_x = max(0, max(a.x, b.x) - min(a.right, b.right))
    gap_y = max(0, max(a.y, b.y) - min(a.bottom, b.bottom))
    return (gap_x + gap_y) <= threshold


class PygameDisplay:
    """Live rendering window using Pygame. Created once, reused across steps."""

    def __init__(self, width: int, height: int, fps: int = 10):
        import pygame
        pygame.init()
        self._pg = pygame
        self._width = width
        self._height = height
        self._fps = fps
        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MarketCanvas-Env")
        self._clock = pygame.time.Clock()

    def render(self, canvas: Canvas) -> None:
        self._pump_events()
        rgb_array = render_to_array(canvas)
        surface = self._pg.surfarray.make_surface(rgb_array.transpose(1, 0, 2))
        self._screen.blit(surface, (0, 0))
        self._pg.display.flip()
        self._clock.tick(self._fps)

    def close(self) -> None:
        self._pg.quit()

    def _pump_events(self) -> None:
        for event in self._pg.event.get():
            if event.type == self._pg.QUIT:
                self.close()
                raise SystemExit
