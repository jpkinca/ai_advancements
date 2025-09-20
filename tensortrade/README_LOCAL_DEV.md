# TensorTrade MVP Local Development

Single source layout: all Python modules are in `src/`.

## Running Pipelines

Historical ingestion + feature scaffold:

```bash
python -m src.mvp_pipeline --start 2025-05-01 --end 2025-08-01 --interval 1d --limit 5
```

## Running Training

```bash
python -m src.train_mvp --months 3 --limit 5 --steps 5000 --with-features --action-scheme vol_target
```

## Key Flags

- `--advanced-fetch` : enable richer IBKR fallback logic.
- `--with-features`  : include engineered return/vol/volume features.
- `--action-scheme {vol_target,discrete_simple}`
- `--reward-scheme {risk_aware,returns}`
- `--eval-episodes N` : run N evaluation episodes after training.

## Environment Variables

Set `DATABASE_URL` for PostgreSQL before running.

## Rationale

Previously code was split between root and `src/`. Wrapper removed; use module invocation via `python -m src.<module>` for clarity and to avoid overshadowing upstream `tensortrade` library.
