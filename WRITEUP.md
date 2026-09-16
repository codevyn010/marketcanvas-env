# MarketCanvas-Env: Design Document

## 1. State and Action Space

### Observation

The environment exposes two observation modes, generated from the same canonical canvas state:

**Semantic (JSON)** — A DOM-like tree listing every element with its properties and computed spatial relationships. This is the primary interface for LLM agents via MCP — it's structured, human-readable, and cheap to serialize.

Example (2 elements on canvas):
```json
{
  "canvas": {"width": 800, "height": 600, "element_count": 2, "step_count": 3},
  "elements": [
    {"id": 1, "type": "text", "x": 200, "y": 80, "width": 400, "height": 60,
     "content": "SUMMER SALE", "font_size": 48,
     "text_color": [255, 255, 255], "background_color": [25, 25, 112]},
    {"id": 2, "type": "shape", "x": 275, "y": 280, "width": 250, "height": 60,
     "shape_variant": "button", "fill_color": [255, 255, 0]}
  ],
  "relationships": [
    {"type": "adjacent", "elements": [1, 2]}
  ]
}
```

**Gymnasium tensor** — A fixed-size `(20, 8)` float32 array of normalized element features, zero-padded for inactive slots. Each row encodes `[x, y, width, height, z_index, type, active_flag, reserved]`, all normalized to `[0, 1]`.

Example (same 2 elements, 18 slots zero-padded):
```
elements[0] = [0.250, 0.133, 0.500, 0.100, 0.00, 0.0, 1.0, 0.0]  # text at (200,80)
elements[1] = [0.344, 0.467, 0.313, 0.100, 0.05, 0.5, 1.0, 0.0]  # button at (275,280)
elements[2..19] = [0, 0, 0, 0, 0, 0, 0, 0]                        # inactive padding

element_count = 2
canvas_meta = [0.06, 0.10]  # step_progress, fill_ratio
```

The fixed shape is a deliberate bounded-observation design for compatibility with standard RL libraries (SB3, RLlib, CleanRL) that require stable tensor shapes for batched training. Variable-length or graph-based observations are possible with custom policies but would add significant complexity for a 20-element canvas.

### Actions

**High-level semantic** — `add_text`, `move_element`, `change_color`, `done`, etc. Each maps to exactly one canvas mutation. An agent can build a complete layout in 6-10 steps. This is also what the MCP tools expose, so the RL action vocabulary and the LLM tool surface are identical.

**Low-level computer-use** — `mouse_move`, `click`, `drag`, `type`. A thin adapter that translates cursor events into canvas ops via hit-testing. Included to demonstrate dual-interface support, not as a production computer-use harness.

**Trade-off:** High-level actions give tighter credit assignment and shorter horizons, but they bake in assumptions about what operations exist. Low-level actions are more general (an agent could learn to use *any* UI) but require hundreds of steps for the same layout, making exploration exponentially harder. The semantic interface is the primary training interface; the low-level adapter provides a path toward future computer-use research.

## 2. Reward Function

### Why sparse terminal

Reward is computed only at episode end — intermediate steps return 0.0. This keeps the optimization objective clean: the agent is scored on the final layout, not on how it got there. Shaping rewards (giving partial credit at each step) would improve credit assignment for longer episodes, but every shaping term is a new surface for reward hacking and changes the optimization landscape in hard-to-predict ways.

### Components

| Component | Weight | What it measures |
|---|---|---|
| Constraint satisfaction | 0.35 | Are the requested elements present, correctly typed, visible, color-matched? |
| Overlap penalty | 0.20 | Pairwise AABB intersection among foreground elements. Intentional containment (text inside a button) is excluded. |
| Alignment | 0.15 | 60% center-of-canvas alignment + 40% pairwise center alignment between foreground elements (15px tolerance). |
| WCAG contrast | 0.15 | Fraction of text elements meeting AA (4.5:1) or AAA (7:1) contrast ratio. |
| Visual balance | 0.15 | Area-weighted centroid displacement from canvas center. |

Sub-scores are each `[0, 1]`. Final reward: `2 * weighted_sum - 1` mapping to `[-1, 1]`.

The target prompt is parsed once at `reset()` into a `TargetSpec` — the reward function never touches the raw string. This keeps reward computation deterministic and fast.

### Known exploits

These are real failure modes, not hypothetical edge cases:

1. **Element stacking** — Place all elements at the same coordinates. Constraints are satisfied, overlap penalty is moderate. A minimum-separation term would fix this.
2. **Contrast gaming** — Black-on-white everywhere scores 21:1 ratio regardless of whether it matches the prompt's aesthetic intent. Needs a color-diversity or prompt-matching sub-reward.
3. **Alignment collapse** — Cluster everything at canvas center. Maximizes both alignment and balance simultaneously. A spatial spread penalty would counteract it.
4. **Invisible elements** — Create elements just above the 100px area threshold. Technically present, practically invisible. Needs a size-proportionality check.
5. **Edge stacking** — Push elements to canvas edges to avoid overlap. Bounds clamping prevents going off-canvas, but edge-hugging still games the overlap score. A canvas utilization reward would help.

These mitigations should be added based on observed training behavior rather than pre-emptively — each additional reward term increases tuning burden and interaction complexity. The current 5-component formulation intentionally covers the major axes (content, layout, accessibility) without over-constraining optimization.

## 3. Scaling to 10K Parallel Rollouts

10K active environments doesn't mean 10K simultaneous VLM calls. The environments just hold canvas state — they're cheap. Policy inference is where the GPU budget goes, and that gets batched separately.

Three things to worry about, in order:

1. **VLM inference** is the dominant cost. The exact GPU footprint depends on model size, precision, sequence length, and batch size — not a single number. What matters is batching aggressively and keeping a bounded number of requests in flight, not one call per environment.

2. **Rendering** sounds fine at ~1-3ms per frame, until it's 10K envs x 50 steps = 500K renders per iteration. But if the policy reads semantic JSON instead of pixels, rendering is skipped entirely. That's probably the right starting point.

3. **Python GIL** is less of an issue than it seems — `AsyncVectorEnv` already forks into separate processes. The real question is how many environments to pack per worker, which needs profiling.

The current implementation is a single-process Python environment — works for development and MCP, but needs a redesign at 10K scale. Key changes:

- Canvas state moves to flat numpy arrays (shared memory, no serialization)
- PIL rendering removed from the step loop (on-demand only, or headless GPU if pixel observations are needed)
- Environments and VLM inference scale independently — 10K active envs might only need a few hundred batched inference requests in flight

```
┌─────────────────────────────────────────────────┐
│                  PPO Learner                     │
│            (collects trajectories,               │
│             updates policy weights)              │
└────────────────────┬────────────────────────────-┘
                     │ batched observations
                     ▼
┌──────────────────────────────────────────────────┐
│             VLM Inference Service                 │
│     (vLLM / continuous batching / paged KV)       │
│      bounded concurrency, not 1:1 per env         │
└────────────────────┬─────────────────────────────┘
                     │ actions
                     ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ CPU      │  │ CPU      │  │ CPU      │   Ray actors or
│ Worker   │  │ Worker   │  │ Worker   │   AsyncVectorEnv
│ (N envs) │  │ (N envs) │  │ (N envs) │   + shared memory
└──────────┘  └──────────┘  └──────────┘
```

Pragmatic first step: train on semantic JSON with a text LLM — no rendering, no VLM. Measure where time actually goes before building GPU infrastructure.

## 4. What's Built

The implementation covers the full environment stack plus two integration surfaces:

- **Canvas engine** (Python) — Deterministic 800x600 canvas with typed elements (text, shape, image), sequential IDs, z-ordering, bounds clamping, and hit-testing. 18 unit tests.
- **Gymnasium environment** — Registered as `marketcanvas/MarketCanvas-v0`. Fixed-size observation space, composite action space, sparse terminal reward. Passes `check_env` and 100 random rollout episodes.
- **Reward pipeline** — 5-component composite reward with WCAG contrast checking, parsed once from prompt into `TargetSpec`. 16 unit tests.
- **MCP server** — 5 tools (`reset_canvas`, `get_canvas_state`, `execute_action`, `get_current_reward`, `render_canvas`) over stdio transport. Directly callable from Claude Desktop or any MCP client.
- **Interactive UI** (Flask) — Browser-based showcase at `localhost:8080`. Enter a prompt, watch the agent build a layout step-by-step with live reward breakdown. Elements are draggable and resizable after generation.
- **Docker support** — Single-command build and run for both demo and interactive modes.

### Demo output (`python demo.py`)

Prompt: *"Create a Summer Sale email banner with a headline, a yellow CTA button, and good contrast"*

```
Step 1: add_shape  -> [OK]  (elements: 1)   # background
Step 2: add_text   -> [OK]  (elements: 2)   # headline
Step 3: add_text   -> [OK]  (elements: 3)   # subtitle
Step 4: add_shape  -> [OK]  (elements: 4)   # CTA button
Step 5: add_text   -> [OK]  (elements: 5)   # button label
Step 6: add_image  -> [OK]  (elements: 6)   # product image

Reward breakdown:
  Constraint   : 1.000
  Overlap      : 1.000
  Alignment    : 1.000
  Contrast     : 1.000
  Balance      : 0.993
  Final reward : +0.998
```

![Demo output](output/demo_result.png)

### MCP session (LLM driving the environment)

The MCP server was connected to an LLM client via stdio. Given a free-form prompt, the model autonomously called `reset_canvas`, issued a sequence of `execute_action` calls to build the layout, checked `get_current_reward` for scoring, and called `render_canvas` to produce the final image — all through tool-calling with no manual intervention.

![MCP output — generated entirely through LLM tool calls](output/mcp_output.png)
