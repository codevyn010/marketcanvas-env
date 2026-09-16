from __future__ import annotations

from dataclasses import dataclass, field

from marketcanvas.canvas import Canvas
from marketcanvas.elements import (
    CanvasElement,
    ElementType,
    ShapeElement,
    ShapeVariant,
    TextElement,
)


@dataclass
class TargetConstraint:
    element_type: ElementType
    role: str
    color_hint: tuple[int, int, int] | None = None
    present: bool = False


@dataclass
class TargetSpec:
    constraints: list[TargetConstraint] = field(default_factory=list)
    require_contrast: str = "AA"

    @staticmethod
    def from_prompt(prompt: str) -> TargetSpec:
        """Parse a natural language prompt into a structured target specification.

        This runs once at episode reset — the reward function evaluates
        against the resulting spec, not the raw prompt string.
        """
        prompt_lower = prompt.lower()
        constraints = []

        headline_keywords = ["headline", "heading", "title", "header"]
        if any(kw in prompt_lower for kw in headline_keywords):
            constraints.append(TargetConstraint(
                element_type=ElementType.TEXT,
                role="headline",
            ))

        subtext_keywords = ["subtext", "subtitle", "description", "body text", "tagline"]
        if any(kw in prompt_lower for kw in subtext_keywords):
            constraints.append(TargetConstraint(
                element_type=ElementType.TEXT,
                role="subtext",
            ))

        cta_keywords = ["cta", "button", "call to action", "call-to-action"]
        if any(kw in prompt_lower for kw in cta_keywords):
            color = _extract_color_hint(prompt_lower)
            constraints.append(TargetConstraint(
                element_type=ElementType.SHAPE,
                role="cta_button",
                color_hint=color,
            ))

        image_keywords = ["image", "photo", "picture", "illustration", "logo", "icon"]
        if any(kw in prompt_lower for kw in image_keywords):
            constraints.append(TargetConstraint(
                element_type=ElementType.IMAGE,
                role="image",
            ))

        contrast_level = "AA"
        if "aaa" in prompt_lower or "high contrast" in prompt_lower:
            contrast_level = "AAA"
        elif "contrast" in prompt_lower:
            contrast_level = "AA"

        return TargetSpec(constraints=constraints, require_contrast=contrast_level)


def _extract_color_hint(prompt: str) -> tuple[int, int, int] | None:
    color_map = {
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "purple": (128, 0, 128),
        "pink": (255, 192, 203),
    }
    for name, rgb in color_map.items():
        if name in prompt:
            return rgb
    return None


def _linearize(c: float) -> float:
    if c <= 0.03928:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(r: int, g: int, b: int) -> float:
    rs, gs, bs = r / 255.0, g / 255.0, b / 255.0
    return 0.2126 * _linearize(rs) + 0.7152 * _linearize(gs) + 0.0722 * _linearize(bs)


def contrast_ratio(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    l1 = relative_luminance(*c1)
    l2 = relative_luminance(*c2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def score_constraints(canvas: Canvas, spec: TargetSpec) -> float:
    if not spec.constraints:
        return 1.0

    elements = canvas.elements
    satisfied = 0

    for constraint in spec.constraints:
        matched = False
        for elem in elements:
            if elem.element_type != constraint.element_type:
                continue
            if elem.area < 100:
                continue

            if constraint.role == "cta_button":
                if not (isinstance(elem, ShapeElement) and elem.shape_variant == ShapeVariant.BUTTON):
                    continue
                if constraint.color_hint and elem.fill_color != constraint.color_hint:
                    continue
                matched = True
                break

            if constraint.role == "headline" and isinstance(elem, TextElement):
                if elem.content.strip():
                    matched = True
                    break

            if constraint.role == "subtext" and isinstance(elem, TextElement):
                if elem.content.strip():
                    matched = True
                    break

            if constraint.role == "image":
                matched = True
                break

        if matched:
            constraint.present = True
            satisfied += 1

    return satisfied / len(spec.constraints)


def score_overlap(canvas: Canvas) -> float:
    elements = canvas.elements
    if len(elements) < 2:
        return 1.0

    canvas_area = canvas.width * canvas.height

    # Exclude elements covering >80% of the canvas (background panels)
    foreground = [e for e in elements if e.area < canvas_area * 0.8]
    if len(foreground) < 2:
        return 1.0

    total_intersection = 0
    for i, a in enumerate(foreground):
        for b in foreground[i + 1:]:
            # Skip intentional containment (e.g., text label inside a button)
            if _is_contained(a, b) or _is_contained(b, a):
                continue
            total_intersection += a.intersection_area(b)

    normalized = total_intersection / canvas_area
    return max(0.0, 1.0 - normalized * 10)


def _is_contained(inner: CanvasElement, outer: CanvasElement) -> bool:
    """Check if inner is fully contained within outer (intentional layering)."""
    return (
        inner.x >= outer.x
        and inner.y >= outer.y
        and inner.right <= outer.right
        and inner.bottom <= outer.bottom
        and inner.area < outer.area
    )


def score_alignment(canvas: Canvas) -> float:
    elements = canvas.elements
    if not elements:
        return 0.0

    canvas_cx = canvas.width / 2
    canvas_area = canvas.width * canvas.height
    tolerance = 15

    # Exclude full-canvas background panels from pairwise edge checks
    foreground = [e for e in elements if e.area < canvas_area * 0.8]

    # Center-of-canvas alignment (primary signal)
    center_score = 0.0
    for elem in elements:
        dist = abs(elem.center_x - canvas_cx) / canvas.width
        center_score += max(0.0, 1.0 - dist * 3)
    center_score /= max(len(elements), 1)

    # Pairwise alignment between foreground elements only
    pair_score = 0.0
    pair_checks = 0
    for i, a in enumerate(foreground):
        for b in foreground[i + 1:]:
            if abs(a.center_x - b.center_x) <= tolerance:
                pair_score += 1.0
            pair_checks += 1

    pair_score = pair_score / max(pair_checks, 1)

    return 0.6 * center_score + 0.4 * pair_score


def score_contrast(canvas: Canvas, spec: TargetSpec) -> float:
    text_elements = [e for e in canvas.elements if isinstance(e, TextElement)]
    if not text_elements:
        return 1.0

    threshold = 7.0 if spec.require_contrast == "AAA" else 4.5
    passing = 0

    for elem in text_elements:
        ratio = contrast_ratio(elem.text_color, elem.background_color)
        if ratio >= threshold:
            passing += 1

    return passing / len(text_elements)


def score_balance(canvas: Canvas) -> float:
    elements = canvas.elements
    if not elements:
        return 0.0

    total_area = sum(e.area for e in elements)
    if total_area == 0:
        return 0.0

    cx = sum(e.center_x * e.area for e in elements) / total_area
    cy = sum(e.center_y * e.area for e in elements) / total_area

    max_dist = ((canvas.width / 2) ** 2 + (canvas.height / 2) ** 2) ** 0.5
    dist = ((cx - canvas.width / 2) ** 2 + (cy - canvas.height / 2) ** 2) ** 0.5

    return 1.0 - dist / max_dist


WEIGHTS = {
    "constraint": 0.35,
    "overlap": 0.20,
    "alignment": 0.15,
    "contrast": 0.15,
    "balance": 0.15,
}


def compute_reward(canvas: Canvas, spec: TargetSpec) -> dict[str, float]:
    """Compute the composite reward with per-component breakdown.

    Returns a dict with individual scores and the final scalar reward in [-1, 1].
    """
    scores = {
        "constraint": score_constraints(canvas, spec),
        "overlap": score_overlap(canvas),
        "alignment": score_alignment(canvas),
        "contrast": score_contrast(canvas, spec),
        "balance": score_balance(canvas),
    }

    weighted = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    scores["raw_weighted"] = weighted
    scores["reward"] = 2.0 * weighted - 1.0

    return scores
