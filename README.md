# PEFT + Conformal 360° FoV (Undergrad Project)

**Goal:** Predict future VR viewport tiles with a personalized Transformer, conformal prediction sets, and an online α-controller.

## Repo layout
- `src/` – model, conformal calibration, controller, simulator
- `tests/` – unit tests (geometry, set→tiles, controller)
- `scripts/` – data prep, training, evaluation
- `configs/` – horizons, grid, controller gains
- `data/` – (local only; not committed)

## Getting started
1. Clone the repo and create a Python 3.11 env.
2. `pip install -r requirements.txt`
3. `pytest -q`
