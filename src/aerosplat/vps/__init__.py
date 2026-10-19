"""Visual positioning against the twin (ADR-0015 layer A).

* `atlas`, `features`, `geometry`, `localize`, `perturb` — numpy + OpenCV only: these run on the
  companion (Raspberry Pi 5) and never import torch / gsplat.
* `build` — atlas assembly from any renderer; `twin` — the site run's splat as that renderer
  (ground station, GPU); `bench` — the feasibility bench (`python -m aerosplat.vps.bench`).
"""

# The operating point (owner delegated the choice, 2026-10-19; ADR-0015 §6–7): AKAZE at 960×540.
# Bench on dataset_1910: 94 % fixes, p95 0.15 m / 0.13°, 0 false fixes; ORB 960×540 is 2x faster
# (90 %, 0.23 m) but less repeatable, and AKAZE 1920×1080 (0.09 m) costs 3x more time than this
# for a gain the 0.5 m planning σ does not use. On the Pi 5 this is expected around 2–4 Hz
# (estimate: PC single-thread 145 ms); measure there before flying.
DEFAULT_EXTRACTOR = "akaze"
DEFAULT_RESOLUTION = (960, 540)

