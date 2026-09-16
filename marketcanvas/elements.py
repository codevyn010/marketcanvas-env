from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ElementType(str, Enum):
    TEXT = "text"
    SHAPE = "shape"
    IMAGE = "image"


class ShapeVariant(str, Enum):
    RECTANGLE = "rectangle"
    BUTTON = "button"


@dataclass
class CanvasElement:
    id: int
    x: int
    y: int
    width: int
    height: int
    z_index: int = 0
    element_type: ElementType = ElementType.TEXT

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def area(self) -> int:
        return self.width * self.height

    def overlaps(self, other: CanvasElement) -> bool:
        return (
            self.x < other.right
            and self.right > other.x
            and self.y < other.bottom
            and self.bottom > other.y
        )

    def intersection_area(self, other: CanvasElement) -> int:
        dx = max(0, min(self.right, other.right) - max(self.x, other.x))
        dy = max(0, min(self.bottom, other.bottom) - max(self.y, other.y))
        return dx * dy

    def contains_point(self, px: int, py: int) -> bool:
        return self.x <= px < self.right and self.y <= py < self.bottom

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.element_type.value,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "z_index": self.z_index,
        }


@dataclass
class TextElement(CanvasElement):
    content: str = ""
    font_size: int = 16
    text_color: tuple[int, int, int] = (0, 0, 0)
    background_color: tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self):
        self.element_type = ElementType.TEXT

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "content": self.content,
            "font_size": self.font_size,
            "text_color": list(self.text_color),
            "background_color": list(self.background_color),
        })
        return d


@dataclass
class ShapeElement(CanvasElement):
    shape_variant: ShapeVariant = ShapeVariant.RECTANGLE
    fill_color: tuple[int, int, int] = (200, 200, 200)
    border_color: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        self.element_type = ElementType.SHAPE

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "shape_variant": self.shape_variant.value,
            "fill_color": list(self.fill_color),
            "border_color": list(self.border_color),
        })
        return d


@dataclass
class ImageElement(CanvasElement):
    fill_color: tuple[int, int, int] = (128, 128, 128)
    label: str = ""

    def __post_init__(self):
        self.element_type = ElementType.IMAGE

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "fill_color": list(self.fill_color),
            "label": self.label,
        })
        return d
