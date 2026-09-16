from __future__ import annotations

from typing import Any

from marketcanvas.elements import (
    CanvasElement,
    ImageElement,
    ShapeElement,
    ShapeVariant,
    TextElement,
)


class Canvas:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self._elements: dict[int, CanvasElement] = {}
        self._next_id = 1
        self._next_z = 0

    def clear(self) -> None:
        self._elements.clear()
        self._next_id = 1
        self._next_z = 0

    @property
    def elements(self) -> list[CanvasElement]:
        return sorted(self._elements.values(), key=lambda e: e.z_index)

    def get_element(self, element_id: int) -> CanvasElement | None:
        return self._elements.get(element_id)

    def _register(self, elem: CanvasElement) -> CanvasElement:
        self._clamp(elem)
        self._elements[elem.id] = elem
        self._next_id += 1
        self._next_z += 1
        return elem

    def add_text(
        self,
        content: str,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        font_size: int = 16,
        text_color: tuple[int, int, int] = (0, 0, 0),
        background_color: tuple[int, int, int] = (255, 255, 255),
    ) -> TextElement:
        elem = TextElement(
            id=self._next_id,
            x=x, y=y, width=width, height=height,
            z_index=self._next_z,
            content=content,
            font_size=font_size,
            text_color=text_color,
            background_color=background_color,
        )
        return self._register(elem)

    def add_shape(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        variant: ShapeVariant = ShapeVariant.RECTANGLE,
        fill_color: tuple[int, int, int] = (200, 200, 200),
        border_color: tuple[int, int, int] = (0, 0, 0),
    ) -> ShapeElement:
        elem = ShapeElement(
            id=self._next_id,
            x=x, y=y, width=width, height=height,
            z_index=self._next_z,
            shape_variant=variant,
            fill_color=fill_color,
            border_color=border_color,
        )
        return self._register(elem)

    def add_image(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        fill_color: tuple[int, int, int] = (128, 128, 128),
        label: str = "",
    ) -> ImageElement:
        elem = ImageElement(
            id=self._next_id,
            x=x, y=y, width=width, height=height,
            z_index=self._next_z,
            fill_color=fill_color,
            label=label,
        )
        return self._register(elem)

    def remove_element(self, element_id: int) -> bool:
        return self._elements.pop(element_id, None) is not None

    def move_element(self, element_id: int, x: int, y: int) -> bool:
        elem = self._elements.get(element_id)
        if elem is None:
            return False
        elem.x = x
        elem.y = y
        self._clamp(elem)
        return True

    def resize_element(self, element_id: int, width: int, height: int) -> bool:
        elem = self._elements.get(element_id)
        if elem is None:
            return False
        elem.width = max(1, width)
        elem.height = max(1, height)
        self._clamp(elem)
        return True

    def update_element(self, element_id: int, **props: Any) -> bool:
        elem = self._elements.get(element_id)
        if elem is None:
            return False
        for key, value in props.items():
            if hasattr(elem, key) and key not in ("id", "element_type"):
                setattr(elem, key, value)
        self._clamp(elem)
        return True

    def elements_at(self, px: int, py: int) -> list[CanvasElement]:
        hits = [e for e in self._elements.values() if e.contains_point(px, py)]
        return sorted(hits, key=lambda e: e.z_index, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canvas": {
                "width": self.width,
                "height": self.height,
                "element_count": len(self._elements),
            },
            "elements": [e.to_dict() for e in self.elements],
        }

    def _clamp(self, elem: CanvasElement) -> None:
        elem.x = max(0, min(elem.x, self.width - 1))
        elem.y = max(0, min(elem.y, self.height - 1))
        elem.width = max(1, min(elem.width, self.width - elem.x))
        elem.height = max(1, min(elem.height, self.height - elem.y))
