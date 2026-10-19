"""Visual positioning against the twin (ADR-0015 layer A).

* `atlas`, `features`, `geometry`, `localize`, `perturb` — numpy + OpenCV only: these run on the
  companion (Raspberry Pi 5) and never import torch / gsplat.
* `build` — atlas assembly from any renderer; `twin` — the site run's splat as that renderer
  (ground station, GPU); `bench` — the feasibility bench (`python -m aerosplat.vps.bench`).
"""
