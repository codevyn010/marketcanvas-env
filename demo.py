"""Demonstration of the MarketCanvas environment.

Initializes the environment with a sample marketing prompt, executes a
sequence of deliberate design actions, and prints the resulting state
and reward breakdown. The final canvas is saved as a PNG.

Usage:
    python demo.py            # headless mode (PNG output only)
    python demo.py --live     # live Pygame window showing step-by-step build
"""
import argparse
import os
import json
import time

from marketcanvas.canvas import Canvas
from marketcanvas.actions import execute_semantic_action
from marketcanvas.observer import get_semantic_state, render_to_png, PygameDisplay
from marketcanvas.rewards import TargetSpec, compute_reward


TARGET_PROMPT = (
    "Create a Summer Sale email banner with a headline, "
    "a yellow CTA button, and good contrast"
)


def main():
    parser = argparse.ArgumentParser(description="MarketCanvas-Env demo")
    parser.add_argument("--live", action="store_true", help="Show live Pygame window")
    args = parser.parse_args()

    canvas = Canvas(width=800, height=600)
    spec = TargetSpec.from_prompt(TARGET_PROMPT)

    display = None
    if args.live:
        display = PygameDisplay(canvas.width, canvas.height, fps=2)

    print(f"Target prompt: {TARGET_PROMPT}")
    print(f"Parsed constraints: {[c.role for c in spec.constraints]}")
    print(f"Required contrast: {spec.require_contrast}")
    print("-" * 60)

    actions = [
        ("add_shape", dict(
            x=0, y=0, width=800, height=600,
            variant="rectangle",
            fill_color=(25, 25, 112),
        )),
        ("add_text", dict(
            content="SUMMER SALE",
            x=200, y=80, width=400, height=60,
            font_size=48,
            text_color=(255, 255, 255),
            background_color=(25, 25, 112),
        )),
        ("add_text", dict(
            content="Up to 50% off everything",
            x=225, y=180, width=350, height=40,
            font_size=20,
            text_color=(200, 200, 220),
            background_color=(25, 25, 112),
        )),
        ("add_shape", dict(
            x=275, y=280, width=250, height=60,
            variant="button",
            fill_color=(255, 255, 0),
            border_color=(200, 200, 0),
        )),
        ("add_text", dict(
            content="SHOP NOW",
            x=310, y=290, width=180, height=40,
            font_size=24,
            text_color=(25, 25, 25),
            background_color=(255, 255, 0),
        )),
        ("add_image", dict(
            x=300, y=400, width=200, height=120,
            fill_color=(70, 130, 180),
            label="Product",
        )),
    ]

    for i, (action_name, kwargs) in enumerate(actions, 1):
        result = execute_semantic_action(canvas, action_name, **kwargs)
        elem_count = len(canvas.elements)
        status = "OK" if result["success"] else "FAIL"
        print(f"Step {i}: {action_name:15s} -> [{status}]  (elements on canvas: {elem_count})")

        if display is not None:
            display.render(canvas)
            time.sleep(0.8)

    print("-" * 60)
    print("\nFinal semantic state:")
    state = get_semantic_state(canvas, step_count=len(actions))
    print(json.dumps(state, indent=2))

    print("\nReward breakdown:")
    scores = compute_reward(canvas, spec)
    for key, value in scores.items():
        label = key.replace("_", " ").title()
        print(f"  {label:20s}: {value:+.4f}")

    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", "demo_result.png")
    render_to_png(canvas, output_path)
    print(f"\nCanvas saved to {output_path}")

    if display is not None:
        print("\nLive window open — close it or press Ctrl+C to exit.")
        try:
            while True:
                display.render(canvas)
        except (SystemExit, KeyboardInterrupt):
            pass
        finally:
            display.close()


if __name__ == "__main__":
    main()
