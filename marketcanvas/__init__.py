from gymnasium.envs.registration import register

register(
    id="marketcanvas/MarketCanvas-v0",
    entry_point="marketcanvas.env:MarketCanvasEnv",
    max_episode_steps=50,
    nondeterministic=False,
)
