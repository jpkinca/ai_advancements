"""Project root placeholder.

Source code lives exclusively under the `src/` directory to avoid name collisions
with the upstream `tensortrade` library. Run modules using one of:

	python -m src.mvp_pipeline  --start YYYY-MM-DD --end YYYY-MM-DD
	python -m src.train_mvp     [training args]

Avoid importing from the root package directly; treat `src` as the code root.
"""
__all__: list[str] = []
