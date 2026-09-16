"""MarketCanvas — interactive RL environment showcase.

Launches a browser-based UI that demonstrates the environment's
state-action-reward loop. Enter a target prompt, click Generate,
and watch the agent build a design step-by-step while reward
components update in real time.

Usage:
    python interactive.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import webbrowser
from threading import Timer

from flask import Flask, render_template, request, jsonify

from marketcanvas.canvas import Canvas
from marketcanvas.elements import ShapeVariant
from marketcanvas.actions import execute_semantic_action
from marketcanvas.observer import get_semantic_state, render_to_png
from marketcanvas.rewards import TargetSpec, compute_reward

app = Flask(__name__)


def _build_action_sequence(prompt: str, target_spec: TargetSpec, cw: int = 800, ch: int = 600) -> list[dict]:
    """Build a sequence of semantic actions from the parsed prompt.

    Returns a list of action dicts that can be executed one-by-one via
    execute_semantic_action(), producing a step-by-step agent replay.
    """
    roles = [c.role for c in target_spec.constraints]
    if not roles:
        roles = ["headline", "cta_button"]

    bg_color = _extract_bg_color(prompt)
    text_color = _readable_text_color(bg_color)
    sub_color = _soften(text_color)
    headline_text = _extract_theme(prompt)
    subtitle_text = _extract_subtitle(prompt)
    steps = []

    steps.append({
        "action": "add_shape",
        "step_label": "Setting up background",
        "x": 0, "y": 0, "width": cw, "height": ch,
        "variant": "rectangle",
        "fill_color": list(bg_color),
    })

    y_cursor = int(ch * 0.13)

    if "headline" in roles:
        steps.append({
            "action": "add_text",
            "step_label": f"Adding headline: \"{headline_text}\"",
            "content": headline_text, "x": int(cw * 0.10), "y": y_cursor,
            "width": int(cw * 0.80), "height": 60,
            "font_size": 42,
            "text_color": list(text_color),
            "background_color": list(bg_color),
        })
        y_cursor += 90

    # Always add a subtitle for richer layouts
    steps.append({
        "action": "add_text",
        "step_label": f"Adding subtitle: \"{subtitle_text}\"",
        "content": subtitle_text,
        "x": int(cw * 0.18), "y": y_cursor,
        "width": int(cw * 0.64), "height": 40,
        "font_size": 20,
        "text_color": list(sub_color),
        "background_color": list(bg_color),
    })
    y_cursor += 70

    if "image" in roles:
        img_w, img_h = 200, 120
        steps.append({
            "action": "add_image",
            "step_label": "Placing product image",
            "x": (cw - img_w) // 2, "y": y_cursor,
            "width": img_w, "height": img_h,
            "fill_color": [70, 130, 180],
            "label": "Product",
        })
        y_cursor += img_h + 30

    if "cta_button" in roles:
        constraint = next((c for c in target_spec.constraints if c.role == "cta_button"), None)
        btn_color = constraint.color_hint if constraint and constraint.color_hint else (255, 255, 0)
        btn_text_color = _readable_text_color(btn_color)
        cta_label = _extract_cta_label(prompt)
        btn_w, btn_h = 250, 55
        steps.append({
            "action": "add_shape",
            "step_label": f"Creating CTA button ({cta_label})",
            "x": (cw - btn_w) // 2, "y": y_cursor,
            "width": btn_w, "height": btn_h,
            "variant": "button",
            "fill_color": list(btn_color),
            "border_color": list(_darken(btn_color)),
        })
        steps.append({
            "action": "add_text",
            "step_label": "Adding button label",
            "content": cta_label,
            "x": (cw - btn_w) // 2 + 20, "y": y_cursor + 8,
            "width": btn_w - 40, "height": btn_h - 16,
            "font_size": 22,
            "text_color": list(btn_text_color),
            "background_color": list(btn_color),
        })

    return steps


def _extract_theme(prompt: str) -> str:
    """Extract a short marketing headline from the prompt.

    Strips instruction verbs and structural descriptions, keeps only
    the core subject/theme as a 1-4 word headline.
    """
    text = prompt.strip()

    # Remove leading instruction verbs
    text = re.sub(
        r'^(create|make|design|build|generate|launch|start|set\s+up)\s+'
        r'(a\s+new|an?\s+new|me\s+a|me\s+an|a|an|the)?\s*',
        '', text, flags=re.IGNORECASE,
    ).strip()

    # Cut at structural description boundaries
    text = re.split(
        r'\s*[,.]?\s*\b(with|which|that|having|containing|including|where)\b',
        text, maxsplit=1, flags=re.IGNORECASE,
    )[0].strip()

    # Remove format/container words
    text = re.sub(
        r'\b(email\s+)?(banner|email|page|ad|poster|flyer|card|layout|'
        r'design|template|application|app|landing|website|site|promo)\b',
        '', text, flags=re.IGNORECASE,
    ).strip()

    # Remove filler like "for", "of", "about" at edges
    text = re.sub(r'\s+(for|of|about)\s*$', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'^(for|of|about)\s+', '', text, flags=re.IGNORECASE).strip()

    # Clean up punctuation and extra spaces
    text = re.sub(r'\s+', ' ', text).strip(" ,.-")

    # Cap at ~4 words to keep it headline-sized
    words = text.split()
    if len(words) > 4:
        text = " ".join(words[:4])

    if len(text) < 2:
        text = "Sale"

    return text.upper()


def _extract_subtitle(prompt: str) -> str:
    """Generate a contextual subtitle based on the prompt theme."""
    p = prompt.lower()
    if "black friday" in p:
        return "Biggest deals of the year"
    if "summer" in p:
        return "Up to 50% off everything"
    if "winter" in p or "holiday" in p or "christmas" in p:
        return "Seasonal savings start now"
    if "flash" in p:
        return "Hurry — offer ends soon"
    if "launch" in p or "new" in p:
        return "Introducing something fresh"
    if "food" in p or "order" in p or "restaurant" in p:
        return "Fresh flavors delivered to you"
    if "sale" in p or "off" in p or "discount" in p:
        return "Limited time offer"
    if "free" in p or "trial" in p:
        return "No credit card required"
    return "Don't miss out"


def _extract_cta_label(prompt: str) -> str:
    """Extract a CTA button label from the prompt context."""
    p = prompt.lower()
    cta_labels = {
        "shop": "SHOP NOW",
        "buy": "BUY NOW",
        "sign up": "SIGN UP",
        "subscribe": "SUBSCRIBE",
        "learn more": "LEARN MORE",
        "get started": "GET STARTED",
        "download": "DOWNLOAD",
        "order": "ORDER NOW",
        "book": "BOOK NOW",
        "register": "REGISTER",
        "join": "JOIN NOW",
        "try": "TRY FREE",
    }
    for keyword, label in cta_labels.items():
        if keyword in p:
            return label
    if "sale" in p or "off" in p or "discount" in p or "black friday" in p:
        return "SHOP NOW"
    if "food" in p or "restaurant" in p or "order" in p:
        return "ORDER NOW"
    return "GET STARTED"


_BG_COLOR_MAP = {
    "black": (20, 20, 20),
    "white": (255, 255, 255),
    "dark": (30, 30, 40),
    "navy": (25, 25, 112),
    "red": (140, 20, 20),
    "blue": (20, 40, 120),
    "green": (20, 80, 40),
    "gray": (60, 60, 65),
    "grey": (60, 60, 65),
    "purple": (60, 20, 100),
}


def _extract_bg_color(prompt: str) -> tuple:
    """Parse a background color from the prompt. Defaults to dark navy."""
    p = prompt.lower()
    for name, rgb in _BG_COLOR_MAP.items():
        if name + " background" in p or name + " bg" in p:
            return rgb
        if "background" in p and name in p:
            return rgb
    if "dark" in p:
        return (30, 30, 40)
    return (25, 25, 112)


def _readable_text_color(bg: tuple) -> tuple:
    """Return white or black text depending on background luminance."""
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (255, 255, 255) if lum < 140 else (20, 20, 20)


def _soften(color: tuple) -> tuple:
    """Slightly mute a text color for subtitles."""
    return tuple(max(0, min(255, int(c * 0.8 + 40))) for c in color)


def _darken(color: tuple) -> tuple:
    return tuple(max(0, int(c * 0.8)) for c in color)


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response

canvas = Canvas(800, 600)
target_prompt = "Create a Summer Sale banner with a headline, a yellow CTA button, and good contrast"
spec = TargetSpec.from_prompt(target_prompt)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state", methods=["GET"])
def get_state():
    state = get_semantic_state(canvas)
    scores = compute_reward(canvas, spec)
    return jsonify({"state": state, "scores": scores, "prompt": target_prompt})


@app.route("/api/generate", methods=["POST"])
def generate():
    """Parse prompt, clear canvas, and return the action sequence for replay."""
    global target_prompt, spec
    data = request.json
    target_prompt = data.get("prompt", target_prompt)
    spec = TargetSpec.from_prompt(target_prompt)
    constraints = [c.role for c in spec.constraints]

    canvas.clear()
    steps = _build_action_sequence(target_prompt, spec, canvas.width, canvas.height)

    state = get_semantic_state(canvas)
    scores = compute_reward(canvas, spec)
    return jsonify({
        "state": state,
        "scores": scores,
        "constraints": constraints,
        "steps": steps,
        "total_steps": len(steps),
    })


@app.route("/api/step", methods=["POST"])
def execute_step():
    """Execute a single action from the replay sequence."""
    data = request.json
    action_name = data.pop("action")
    data.pop("step_label", None)

    # Convert color lists to tuples
    for key in ("fill_color", "text_color", "background_color", "border_color", "color"):
        if key in data and isinstance(data[key], list):
            data[key] = tuple(data[key])

    # Convert variant string to enum
    if "variant" in data:
        data["variant"] = ShapeVariant(data["variant"])

    result = execute_semantic_action(canvas, action_name, **data)
    state = get_semantic_state(canvas)
    scores = compute_reward(canvas, spec)
    return jsonify({"result": result, "state": state, "scores": scores})


@app.route("/api/action", methods=["POST"])
def do_action():
    data = request.json
    action_name = data.pop("action")

    for key in ("fill_color", "text_color", "background_color", "border_color", "color"):
        if key in data and isinstance(data[key], list):
            data[key] = tuple(data[key])

    result = execute_semantic_action(canvas, action_name, **data)
    state = get_semantic_state(canvas)
    scores = compute_reward(canvas, spec)
    return jsonify({"result": result, "state": state, "scores": scores})


@app.route("/api/move", methods=["POST"])
def move():
    data = request.json
    canvas.move_element(data["element_id"], int(data["x"]), int(data["y"]))
    state = get_semantic_state(canvas)
    scores = compute_reward(canvas, spec)
    return jsonify({"state": state, "scores": scores})


@app.route("/api/resize", methods=["POST"])
def resize():
    data = request.json
    canvas.resize_element(data["element_id"], int(data["width"]), int(data["height"]))
    state = get_semantic_state(canvas)
    scores = compute_reward(canvas, spec)
    return jsonify({"state": state, "scores": scores})


@app.route("/api/clear", methods=["POST"])
def clear():
    canvas.clear()
    state = get_semantic_state(canvas)
    scores = compute_reward(canvas, spec)
    return jsonify({"state": state, "scores": scores})


@app.route("/api/render", methods=["GET"])
def render_png():
    from PIL import Image
    from marketcanvas.observer import render_to_array
    arr = render_to_array(canvas)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"image": f"data:image/png;base64,{b64}"})


@app.route("/api/save", methods=["POST"])
def save():
    os.makedirs("output", exist_ok=True)
    render_to_png(canvas, "output/interactive_result.png")
    return jsonify({"saved": "output/interactive_result.png"})


def open_browser():
    webbrowser.open("http://127.0.0.1:8080")


if __name__ == "__main__":
    Timer(1.0, open_browser).start()
    print("MarketCanvas — RL Environment Showcase")
    print("Open http://127.0.0.1:8080 in your browser")
    app.run(host="0.0.0.0", port=8080, debug=False)
