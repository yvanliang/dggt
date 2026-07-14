from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


GAUSSIAN_PLY_PROPERTIES = [
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
]
GAUSSIAN_SH_C0 = 0.28209479177387814


def _to_float_tensor(value: Any, *, shape_last: int | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().to(torch.float32).contiguous()
    elif isinstance(value, np.ndarray):
        tensor = torch.tensor(value.tolist(), dtype=torch.float32)
    else:
        tensor = torch.tensor(value, dtype=torch.float32)
    if shape_last is not None and tensor.dim() == 1:
        tensor = tensor.view(-1, shape_last)
    return tensor.contiguous()


def _write_binary_float_tensor(path: Path, header: str, matrix: torch.Tensor) -> None:
    matrix = matrix.detach().cpu().to(torch.float32).contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        try:
            matrix.numpy().tofile(f)
        except Exception:
            np.asarray(matrix.view(-1).tolist(), dtype=np.float32).tofile(f)


def _sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp(0.0, 1.0)
    return (rgb - 0.5) / GAUSSIAN_SH_C0


def _logit(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(x / (1.0 - x))


def read_gaussian_ply(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "rb") as f:
        header_lines: list[str] = []
        vertex_count = None
        property_names: list[str] = []
        header_bytes = 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")
            header_bytes += len(line)
            text = line.decode("ascii").strip()
            header_lines.append(text)
            if text.startswith("element vertex"):
                vertex_count = int(text.split()[-1])
            elif text.startswith("property"):
                property_names.append(text.split()[-1])
            elif text == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"Missing vertex count in PLY header: {path}")

        if property_names != GAUSSIAN_PLY_PROPERTIES:
            raise ValueError(
                f"Unsupported gaussian PLY schema for {path}. "
                f"Expected {GAUSSIAN_PLY_PROPERTIES}, got {property_names}"
            )

        payload = np.fromfile(f, dtype=np.float32, count=vertex_count * len(property_names))
        if payload.size != vertex_count * len(property_names):
            raise ValueError(
                f"PLY payload truncated for {path}: expected {vertex_count * len(property_names)} floats, "
                f"got {payload.size}"
            )
        payload = payload.reshape(vertex_count, len(property_names))

    means = payload[:, 0:3].astype(np.float32, copy=False)
    normals = payload[:, 3:6].astype(np.float32, copy=False)
    features_dc = payload[:, 6:9].astype(np.float32, copy=False)
    opacity_raw = payload[:, 9:10].astype(np.float32, copy=False)
    scale_raw = payload[:, 10:13].astype(np.float32, copy=False)
    quats = payload[:, 13:17].astype(np.float32, copy=False)

    return {
        "path": str(path),
        "vertex_count": int(vertex_count),
        "property_names": list(property_names),
        "means": means,
        "normals": normals,
        "features_dc": features_dc,
        "rgb": np.clip(GAUSSIAN_SH_C0 * features_dc + 0.5, 0.0, 1.0).astype(np.float32, copy=False),
        "opacity_raw": opacity_raw,
        "opacities": _sigmoid_numpy(opacity_raw).astype(np.float32, copy=False),
        "scale_raw": scale_raw,
        "scales": np.exp(scale_raw).astype(np.float32, copy=False),
        "quats": quats,
        "header_bytes": int(header_bytes),
        "header_lines": header_lines,
    }


def write_gaussian_ply(gaussians: dict[str, Any], path: str | Path) -> None:
    means = _to_float_tensor(gaussians["means"], shape_last=3)
    count = means.shape[0]

    normals = gaussians.get("normals")
    if normals is None:
        normals_t = torch.zeros((count, 3), dtype=torch.float32)
    else:
        normals_t = _to_float_tensor(normals, shape_last=3)

    if "features_dc" in gaussians:
        features_dc = _to_float_tensor(gaussians["features_dc"], shape_last=3)
    else:
        colors = _to_float_tensor(gaussians["features_dc_rgb"], shape_last=3)
        features_dc = _rgb_to_sh_dc(colors)

    if "opacities_raw" in gaussians:
        opacity_raw = _to_float_tensor(gaussians["opacities_raw"], shape_last=1)
    else:
        opacities = _to_float_tensor(gaussians["opacities"], shape_last=1)
        opacity_raw = _logit(opacities)

    if "scales_raw" in gaussians:
        scale_raw = _to_float_tensor(gaussians["scales_raw"], shape_last=3)
    else:
        scales = _to_float_tensor(gaussians["scales"], shape_last=3).clamp_min(1e-8)
        scale_raw = torch.log(scales)

    quats = _to_float_tensor(gaussians["quats"], shape_last=4)

    matrix = torch.cat([means, normals_t, features_dc, opacity_raw, scale_raw, quats], dim=1)
    if matrix.shape != (count, len(GAUSSIAN_PLY_PROPERTIES)):
        raise ValueError(f"Unexpected gaussian matrix shape {tuple(matrix.shape)}")

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {count}",
    ]
    header.extend(f"property float {name}" for name in GAUSSIAN_PLY_PROPERTIES)
    header.append("end_header")
    header_text = "\n".join(header) + "\n"
    _write_binary_float_tensor(Path(path), header_text, matrix)


def write_point_ply(points: Any, colors: Any, path: str | Path, opacities: Any | None = None) -> None:
    points_t = _to_float_tensor(points, shape_last=3)
    colors_t = _to_float_tensor(colors, shape_last=3).clamp(0.0, 1.0)
    if points_t.shape[0] != colors_t.shape[0]:
        raise ValueError(
            f"Point/color count mismatch: {points_t.shape[0]} vs {colors_t.shape[0]}"
        )

    # MeshLab compatibility is better with uchar vertex colors.
    colors_u8 = torch.round(colors_t * 255.0).to(torch.uint8).contiguous()
    count = points_t.shape[0]
    opacity_t = None
    if opacities is not None:
        opacity_t = _to_float_tensor(opacities, shape_last=1).reshape(-1).clamp(0.0, 1.0)
        if int(opacity_t.shape[0]) != int(count):
            raise ValueError(f"Point/opacity count mismatch: {count} vs {opacity_t.shape[0]}")
    opacity_header = "property float opacity\n" if opacity_t is not None else ""
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"{opacity_header}"
        "end_header\n"
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        dtype = [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        if opacity_t is not None:
            dtype.append(("opacity", "<f4"))
        payload = np.empty(count, dtype=dtype)
        xyz = points_t.detach().cpu().numpy()
        rgb = colors_u8.detach().cpu().numpy()
        payload["x"] = xyz[:, 0]
        payload["y"] = xyz[:, 1]
        payload["z"] = xyz[:, 2]
        payload["red"] = rgb[:, 0]
        payload["green"] = rgb[:, 1]
        payload["blue"] = rgb[:, 2]
        if opacity_t is not None:
            payload["opacity"] = opacity_t.detach().cpu().numpy()
        payload.tofile(f)
