# MarketCanvas-Env

A deterministic 2D design canvas environment for training RL agents to create marketing assets. Exposes both a standard Gymnasium interface and an MCP server for LLM tool-calling.

See [WRITEUP.md](WRITEUP.md) for design rationale and technical discussion.

## Setup

### Option A: Local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Option B: Docker

```bash
docker build -t marketcanvas .
docker run marketcanvas                              # run demo
docker run -p 8080:8080 marketcanvas python interactive.py  # interactive UI
```

## Demo

```bash
python demo.py
```

Produces a sample banner layout and saves the result to `output/demo_result.png`.

## Tests

```bash
pytest tests/ -v
```

## Interactive Showcase

```bash
pip install flask
python interactive.py
```

Opens a browser UI that demonstrates the environment's state→action→reward loop. Enter a target prompt, click **Generate**, and watch the agent build a design step-by-step with live reward updates.

Sample prompts that work well:

- `Create a Summer Sale banner with a headline, a yellow CTA button, and good contrast`
- `Design a Black Friday flash sale ad with a bold headline, a red buy now button, and a product image on dark background`
- `Launch a fitness app landing page with a headline, a green sign up button, and an image`
- `Create a holiday discount banner with a headline, a subtitle, a blue shop now button, and an image`

After generation, elements can be manually dragged, resized, or edited for further refinement.

## MCP Server

```bash
python -m marketcanvas.server
```

Exposes tools: `reset_canvas`, `get_canvas_state`, `execute_action`, `get_current_reward`, `render_canvas`.

Rendered images are saved incrementally to `output/canvas_001.png`, `canvas_002.png`, etc.

### Connecting an MCP client

Add the following to the client's MCP server configuration (e.g., `claude_desktop_config.json`, Cursor settings, or any MCP-compatible client):

```json
{
  "mcpServers": {
    "marketcanvas": {
      "command": "python",
      "args": ["-m", "marketcanvas.server"],
      "cwd": "/absolute/path/to/marketcanvas-env"
    }
  }
}
```

Replace `/absolute/path/to/marketcanvas-env` with the actual project path. The server communicates over stdio — the client launches it automatically.

## Project Structure

```
marketcanvas/
├── elements.py    # Element dataclasses (Text, Shape, Image)
├── canvas.py      # Core 2D canvas engine
├── observer.py    # Observation space (semantic JSON + visual RGB)
├── actions.py     # Action space (high-level semantic + low-level cursor)
├── rewards.py     # Composite reward (constraints, overlap, alignment, WCAG, balance)
├── env.py         # Gymnasium environment wrapper
└── server.py      # MCP server integration
```
