"""3DGS scene as a standard INRIA-layout .ply (SuperSplat, gsplat, Postshot, nerfstudio).

Layout (binary little-endian, one `vertex` element, float32 properties in this order):
x y z · nx ny nz (zeros) · f_dc_0..2 · f_rest_0..3K-1 · opacity · scale_0..2 · rot_0..3

* `f_dc` / `f_rest` are SH coefficients; `f_rest` is **channel-major**, i.e. index
  c * K + k for colour channel c and SH basis k >= 1 (INRIA's
  `features_rest.transpose(1, 2).flatten(1)`); K = (d+1)^2 - 1.
* `opacity` is the **logit**, `scale_i` the **log** of the standard deviation, `rot` a
  quaternion w, x, y, z -- the raw, pre-activation values every reader applies
  sigmoid / exp / normalise to.

Our `.pt` stores activated values and says so (ADR-0002); the contract is read, never
guessed, and a `.pt` without it is refused. A scene trained without SH (`rgb` only) is
written as degree 0: f_dc = (rgb - 0.5) / C0.

Coordinates stay in **Site ENU** (metres, z up). A PLY has no CRS, and moving Gaussians to
UTM would put 4e6-sized numbers into float32 (0.5 m steps). The origin and frame are in
the header comments and the export manifest. Gaussians with non-finite values or a zero
quaternion (diverged in training) are dropped and counted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from aerosplat.export.georef import ExportRefused
from aerosplat.gs.sh import C0, num_sh_bases, sh_degree_of

OPACITY_EPS = 1e-6
SCALE_FLOOR = 1e-12


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, OPACITY_EPS, 1.0 - OPACITY_EPS)
    return np.log(p / (1.0 - p))


def inria_arrays(scene: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Activated `.pt` dict (numpy or torch values) -> raw INRIA arrays + statistics."""
    def arr(key: str) -> np.ndarray:
        v = scene[key]
        v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        return v.astype(np.float64)

    contract = scene.get("activations")
    if not isinstance(contract, dict) or not {"scale_is_log", "opacity_is_logit"} <= set(contract):
        raise ExportRefused("the splat .pt has no activation contract (ADR-0002): cannot "
                            "tell whether its scales are log-space and opacities logits")
    missing = [k for k in ("xyz", "scale", "rotation", "opacity") if k not in scene]
    if missing or ("sh" not in scene and "rgb" not in scene):
        raise ExportRefused(f"the splat .pt lacks {missing or ['sh/rgb']}: not a 3DGS scene")
    xyz, scale, rot = arr("xyz"), arr("scale"), arr("rotation")
    opacity = arr("opacity").reshape(-1)
    if "sh" in scene:
        sh = arr("sh")                                   # (N, B, 3)
        if sh.ndim != 3 or sh.shape[2] != 3:
            raise ExportRefused(f"sh has shape {sh.shape}, expected (N, (d+1)^2, 3)")
        degree = sh_degree_of(sh.shape[1])
    else:
        sh = ((arr("rgb") - 0.5) / C0)[:, None, :]
        degree = 0
    log_scale = scale if contract["scale_is_log"] else np.log(np.maximum(scale, SCALE_FLOOR))
    logit = opacity if contract["opacity_is_logit"] else _logit(opacity)

    n = len(xyz)
    keep = (np.isfinite(xyz).all(1) & np.isfinite(log_scale).all(1) & np.isfinite(rot).all(1)
            & np.isfinite(logit) & np.isfinite(sh.reshape(n, -1)).all(1)
            & (np.linalg.norm(rot, axis=1) > 1e-8))
    rot = rot[keep] / np.linalg.norm(rot[keep], axis=1, keepdims=True)
    out = {"xyz": xyz[keep], "f_dc": sh[keep, 0, :],
           # (N, B-1, 3) -> (N, 3, B-1) -> channel-major flat
           "f_rest": np.transpose(sh[keep, 1:, :], (0, 2, 1)).reshape(int(keep.sum()), -1),
           "opacity": logit[keep], "scale": log_scale[keep], "rot": rot}
    stats = {"gaussians": int(keep.sum()), "dropped_invalid": int(n - keep.sum()),
             "sh_degree": degree,
             "colour": "sh" if "sh" in scene else "rgb (written as SH degree 0)",
             "source_activations": dict(contract)}
    return out, stats


def property_names(sh_rest: int) -> list[str]:
    return (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
            + [f"f_rest_{i}" for i in range(sh_rest)]
            + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])


def write_inria_ply(path: Path, arrays: dict[str, np.ndarray],
                    comments: Optional[list[str]] = None) -> int:
    n = len(arrays["xyz"])
    rest = arrays["f_rest"].shape[1]
    names = property_names(rest)
    cols = np.concatenate([
        arrays["xyz"], np.zeros((n, 3)), arrays["f_dc"], arrays["f_rest"],
        arrays["opacity"].reshape(n, 1), arrays["scale"], arrays["rot"]], axis=1)
    data = np.ascontiguousarray(cols, dtype="<f4")
    lines = ["ply", "format binary_little_endian 1.0"]
    for c in comments or []:
        lines.append("comment " + " ".join(str(c).split()))     # one line, no newlines
    lines.append(f"element vertex {n}")
    lines += [f"property float {p}" for p in names]
    lines.append("end_header")
    tmp = path.with_name(f".{path.stem}.partial{path.suffix}")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii", errors="replace"))
        f.write(data.tobytes())
    tmp.replace(path)
    return n


def read_inria_ply(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """Reader for files in this layout (binary LE float32 vertex properties).
    Returns (arrays as in `inria_arrays`, header comments)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        props: list[str] = []
        comments: list[str] = []
        n = None
        fmt = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: header has no end_header")
            tok = line.decode("ascii", errors="replace").strip().split()
            if not tok:
                continue
            if tok[0] == "end_header":
                break
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "comment":
                comments.append(" ".join(tok[1:]))
            elif tok[0] == "element":
                if tok[1] != "vertex":
                    raise ValueError(f"{path}: unexpected element {tok[1]}")
                n = int(tok[2])
            elif tok[0] == "property":
                if tok[1] != "float":
                    raise ValueError(f"{path}: property {tok[-1]} is {tok[1]}, expected float")
                props.append(tok[2])
        if fmt != "binary_little_endian" or n is None:
            raise ValueError(f"{path}: expected a binary_little_endian vertex PLY")
        data = np.frombuffer(f.read(n * len(props) * 4), dtype="<f4").reshape(n, len(props))
    col = {p: i for i, p in enumerate(props)}
    rest = sorted((p for p in props if p.startswith("f_rest_")), key=lambda s: int(s[7:]))

    def take(names: list[str]) -> np.ndarray:
        return data[:, [col[p] for p in names]].astype(np.float64)

    return {"xyz": take(["x", "y", "z"]), "f_dc": take(["f_dc_0", "f_dc_1", "f_dc_2"]),
            "f_rest": take(rest) if rest else np.zeros((n, 0)),
            "opacity": take(["opacity"])[:, 0],
            "scale": take(["scale_0", "scale_1", "scale_2"]),
            "rot": take(["rot_0", "rot_1", "rot_2", "rot_3"])}, comments


def inria_to_activated(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The inverse of `inria_arrays` (activated values and (N, B, 3) SH), for round trips."""
    n = len(arrays["xyz"])
    rest = arrays["f_rest"].shape[1] // 3
    sh = np.empty((n, 1 + rest, 3))
    sh[:, 0, :] = arrays["f_dc"]
    sh[:, 1:, :] = np.transpose(arrays["f_rest"].reshape(n, 3, rest), (0, 2, 1))
    if num_sh_bases(sh_degree_of(1 + rest)) != 1 + rest:
        raise ValueError(f"{1 + rest} SH bases is not a full degree")
    return {"xyz": arrays["xyz"], "sh": sh, "opacity": 1.0 / (1.0 + np.exp(-arrays["opacity"])),
            "scale": np.exp(arrays["scale"]),
            "rotation": arrays["rot"] / np.linalg.norm(arrays["rot"], axis=1, keepdims=True)}
