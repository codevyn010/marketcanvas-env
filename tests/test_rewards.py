import pytest
from marketcanvas.canvas import Canvas
from marketcanvas.elements import ShapeVariant
from marketcanvas.rewards import (
    TargetSpec,
    compute_reward,
    contrast_ratio,
    relative_luminance,
    score_alignment,
    score_balance,
    score_constraints,
    score_contrast,
    score_overlap,
)


class TestWCAG:
    def test_black_white_contrast(self):
        ratio = contrast_ratio((0, 0, 0), (255, 255, 255))
        assert abs(ratio - 21.0) < 0.1

    def test_same_color_contrast(self):
        ratio = contrast_ratio((128, 128, 128), (128, 128, 128))
        assert abs(ratio - 1.0) < 0.01

    def test_luminance_black(self):
        assert relative_luminance(0, 0, 0) == 0.0

    def test_luminance_white(self):
        assert abs(relative_luminance(255, 255, 255) - 1.0) < 0.01


class TestTargetSpec:
    def test_parse_headline_and_cta(self):
        spec = TargetSpec.from_prompt(
            "Create a banner with a headline and a yellow CTA button"
        )
        roles = [c.role for c in spec.constraints]
        assert "headline" in roles
        assert "cta_button" in roles

    def test_parse_color_hint(self):
        spec = TargetSpec.from_prompt("A yellow CTA button")
        cta = [c for c in spec.constraints if c.role == "cta_button"][0]
        assert cta.color_hint == (255, 255, 0)

    def test_parse_image(self):
        spec = TargetSpec.from_prompt("A banner with an image and title")
        roles = [c.role for c in spec.constraints]
        assert "image" in roles
        assert "headline" in roles

    def test_contrast_level_default(self):
        spec = TargetSpec.from_prompt("A banner with good contrast")
        assert spec.require_contrast == "AA"

    def test_contrast_level_aaa(self):
        spec = TargetSpec.from_prompt("A banner with AAA contrast")
        assert spec.require_contrast == "AAA"


class TestConstraintScoring:
    def test_all_constraints_met(self):
        canvas = Canvas()
        canvas.add_text("Headline", x=100, y=50, width=200, height=40)
        canvas.add_shape(
            x=100, y=200, width=150, height=50,
            variant=ShapeVariant.BUTTON, fill_color=(255, 255, 0),
        )
        spec = TargetSpec.from_prompt("headline and yellow CTA button")
        assert score_constraints(canvas, spec) == 1.0

    def test_missing_constraint(self):
        canvas = Canvas()
        canvas.add_text("Headline", x=100, y=50, width=200, height=40)
        spec = TargetSpec.from_prompt("headline and yellow CTA button")
        score = score_constraints(canvas, spec)
        assert 0.0 < score < 1.0

    def test_empty_canvas(self):
        canvas = Canvas()
        spec = TargetSpec.from_prompt("headline and CTA button")
        assert score_constraints(canvas, spec) == 0.0


class TestOverlapScoring:
    def test_no_overlap(self):
        canvas = Canvas()
        canvas.add_shape(x=0, y=0, width=100, height=100)
        canvas.add_shape(x=200, y=200, width=100, height=100)
        assert score_overlap(canvas) == 1.0

    def test_full_overlap(self):
        canvas = Canvas()
        canvas.add_shape(x=50, y=50, width=100, height=100)
        canvas.add_shape(x=50, y=50, width=100, height=100)
        score = score_overlap(canvas)
        assert score < 1.0


class TestContrastScoring:
    def test_good_contrast(self):
        canvas = Canvas()
        canvas.add_text(
            "Hello", x=10, y=10, width=100, height=30,
            text_color=(0, 0, 0), background_color=(255, 255, 255),
        )
        spec = TargetSpec(require_contrast="AA")
        assert score_contrast(canvas, spec) == 1.0

    def test_bad_contrast(self):
        canvas = Canvas()
        canvas.add_text(
            "Hello", x=10, y=10, width=100, height=30,
            text_color=(200, 200, 200), background_color=(220, 220, 220),
        )
        spec = TargetSpec(require_contrast="AA")
        assert score_contrast(canvas, spec) == 0.0


class TestCompositeReward:
    def test_empty_canvas_low_reward(self):
        canvas = Canvas()
        spec = TargetSpec.from_prompt("headline and CTA button")
        scores = compute_reward(canvas, spec)
        assert scores["reward"] < 0.0

    def test_reward_range(self):
        canvas = Canvas()
        canvas.add_text("Title", x=300, y=100, width=200, height=50)
        spec = TargetSpec.from_prompt("headline")
        scores = compute_reward(canvas, spec)
        assert -1.0 <= scores["reward"] <= 1.0
