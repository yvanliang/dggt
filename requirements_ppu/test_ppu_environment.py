#!/usr/bin/env python3
"""Smoke test for DGGT on Alibaba PPU torch 2.9 environments.

This script intentionally avoids checkpoint downloads and large model forwards.
It verifies the pieces most likely to break after environment migration:

  - torch / torchvision / torchaudio imports and a small PPU tensor op
  - gsplat import and a minimal rasterization call
  - pointops2 native extension import and a minimal grouping op
  - OpenCV/Open3D/NumPy basic ops
  - key DGGT module forwards on cuda/PPU
  - pip dependency conflicts, reported as diagnostics

Usage:
  python requirements_ppu/test_ppu_environment.py
  python requirements_ppu/test_ppu_environment.py --strict-pip-check
  python requirements_ppu/test_ppu_environment.py --skip-pointops2
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _run_check(name: str, fn: Callable[[], str | None]) -> CheckResult:
    try:
        detail = fn() or ""
        print(f"[PASS] {name}{': ' + detail if detail else ''}")
        return CheckResult(name=name, ok=True, detail=detail)
    except Exception as exc:  # noqa: BLE001 - smoke test should report all failures.
        print(f"[FAIL] {name}: {exc}")
        traceback.print_exc()
        return CheckResult(name=name, ok=False, detail=str(exc))


def _require_cuda_tensor(name: str, tensor: object) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"{name} is not a torch.Tensor: {type(tensor)}")
    if tensor.device.type != "cuda":
        raise RuntimeError(f"{name} is not on PPU/cuda: device={tensor.device}")
    if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains non-finite values")


def _open3d_import_error_hint(exc: OSError) -> RuntimeError:
    message = str(exc)
    if "libGL.so.1" in message:
        return RuntimeError(
            "Open3D requires system OpenGL runtime libGL.so.1, but it is missing. "
            "On Ubuntu/Debian run: apt-get update && apt-get install -y libgl1 libglib2.0-0"
        )
    return RuntimeError(f"Open3D import failed: {message}")


def check_versions() -> str:
    import numpy as np

    packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "triton",
        "gsplat",
        "numpy",
        "opencv-python",
        "opencv-python-headless",
        "open3d",
        "protobuf",
        "packaging",
        "diffusers",
        "transformers",
    ]
    lines = [
        f"python={sys.version.split()[0]}",
        f"executable={sys.executable}",
        f"platform={platform.platform()}",
        f"cwd={os.getcwd()}",
    ]
    lines.extend(f"{pkg}={_version(pkg)}" for pkg in packages)
    if np.__version__.split(".", 1)[0] != "1":
        raise RuntimeError(f"DGGT PPU requirements expect NumPy 1.x, got {np.__version__}")
    return "; ".join(lines)


def check_torch_ppu() -> str:
    import torch

    if not torch.__version__.startswith("2.9."):
        raise RuntimeError(f"expected torch 2.9.x, got {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False; PPU runtime is not visible")

    device = torch.device("cuda:0")
    x = torch.randn(64, 64, device=device, dtype=torch.float32, requires_grad=True)
    y = (x @ x.T).mean()
    y.backward()
    torch.cuda.synchronize(device)
    if not torch.isfinite(x.grad).all():
        raise RuntimeError("torch backward produced non-finite gradient")

    return (
        f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
        f"device={torch.cuda.get_device_name(0)}"
    )


def check_torchvision_ops() -> str:
    import torch
    import torchvision
    from torchvision.ops import nms

    device = torch.device("cuda:0")
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]],
        device=device,
    )
    scores = torch.tensor([0.9, 0.8, 0.7], device=device)
    keep = nms(boxes, scores, 0.5)
    _require_cuda_tensor("torchvision nms output", keep)
    if keep.tolist() != [0, 2]:
        raise RuntimeError(f"unexpected torchvision.ops.nms output: {keep.tolist()}")
    return f"torchvision={torchvision.__version__}, nms_keep={keep.tolist()}, device={keep.device}"


def check_cv_open3d() -> str:
    import cv2
    import numpy as np

    try:
        import open3d as o3d
    except OSError as exc:
        raise _open3d_import_error_hint(exc) from exc

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.zeros((3, 3), dtype=np.float64))
    if gray.shape != (8, 8) or len(cloud.points) != 3:
        raise RuntimeError("OpenCV/Open3D basic operation failed")
    return f"cv2={cv2.__version__}, open3d={o3d.__version__}"


def check_gsplat() -> str:
    import torch
    import gsplat
    from gsplat.rendering import rasterization

    device = torch.device("cuda:0")
    means = torch.tensor([[0.0, 0.0, 2.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[0.05, 0.05, 0.05]], device=device)
    opacities = torch.tensor([0.9], device=device)
    colors = torch.tensor([[1.0, 0.2, 0.1]], device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor(
        [[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    render, alpha, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=32,
        height=32,
        render_mode="RGB",
    )
    torch.cuda.synchronize(device)
    if render.shape != (1, 32, 32, 3) or alpha.shape != (1, 32, 32, 1):
        raise RuntimeError(f"unexpected gsplat shapes: render={tuple(render.shape)}, alpha={tuple(alpha.shape)}")
    if not torch.isfinite(render).all() or not torch.isfinite(alpha).all():
        raise RuntimeError("gsplat output contains non-finite values")
    _require_cuda_tensor("gsplat render", render)
    _require_cuda_tensor("gsplat alpha", alpha)
    return f"gsplat={getattr(gsplat, '__version__', 'unknown')}, alpha_sum={alpha.sum().item():.6f}, meta_keys={len(meta)}"


def check_pointops2() -> str:
    import torch

    # The local setup.py installs package name pointops2, while some in-repo
    # code imports through third_party.pointops2. Prefer installed package,
    # then fall back to source-tree import for diagnostics.
    try:
        pointops = importlib.import_module("pointops2.pointops")
    except ModuleNotFoundError:
        pointops = importlib.import_module("third_party.pointops2.functions.pointops")

    device = torch.device("cuda:0")
    features = torch.arange(8, device=device, dtype=torch.float32).view(4, 2).contiguous()
    idx = torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.int32).contiguous()
    grouped = pointops.grouping(features, idx)
    torch.cuda.synchronize(device)
    expected = torch.tensor(
        [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
        device=device,
    )
    if not torch.allclose(grouped, expected):
        raise RuntimeError(f"pointops2 grouping mismatch: {grouped}")
    _require_cuda_tensor("pointops2 grouped", grouped)
    return f"module={pointops.__name__}, grouping_shape={tuple(grouped.shape)}"


def check_dggt_aggregator_forward() -> str:
    import torch
    import torch.nn as nn

    from dggt.layers import PatchEmbed
    from dggt.models.aggregator import Aggregator

    class MiniPatchEmbed(nn.Module):
        def __init__(self, img_size: int, patch_size: int, embed_dim: int) -> None:
            super().__init__()
            self.patch = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.patch(x)

        def get_intermediate_layers(self, x: torch.Tensor, n: int = 24) -> list[torch.Tensor]:
            tokens = self.patch(x)
            return [tokens for _ in range(int(n))]

    device = torch.device("cuda:0")
    model = Aggregator(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        depth=1,
        num_heads=4,
        num_register_tokens=2,
        patch_embed="conv",
        rope_freq=100,
    ).to(device).eval()
    model.patch_embed = MiniPatchEmbed(img_size=28, patch_size=14, embed_dim=32).to(device).eval()

    images = torch.rand(1, 2, 3, 28, 28, device=device)
    with torch.no_grad():
        output_list, output_with_tokens, dino_tokens, image_feature, patch_start_idx = model(images)

    if patch_start_idx != 3:
        raise RuntimeError(f"unexpected patch_start_idx={patch_start_idx}")
    if len(output_list) != 1 or len(output_with_tokens) != 1:
        raise RuntimeError(f"unexpected aggregator output lengths: {len(output_list)}, {len(output_with_tokens)}")
    _require_cuda_tensor("aggregator output_list[0]", output_list[0])
    _require_cuda_tensor("aggregator output_with_tokens[0]", output_with_tokens[0])
    _require_cuda_tensor("aggregator dino_tokens[0]", dino_tokens[0])
    _require_cuda_tensor("aggregator image_feature", image_feature)
    return (
        f"output={tuple(output_list[0].shape)}, output_with_tokens={tuple(output_with_tokens[0].shape)}, "
        f"image_feature={tuple(image_feature.shape)}"
    )


def check_dggt_camera_head_forward() -> str:
    import torch

    from dggt.heads.camera_head import CameraHead

    device = torch.device("cuda:0")
    model = CameraHead(dim_in=64, trunk_depth=1, num_heads=4).to(device).eval()
    tokens = torch.randn(1, 2, 5, 64, device=device)
    with torch.no_grad():
        outputs = model([tokens], num_iterations=2)
    if len(outputs) != 2:
        raise RuntimeError(f"expected 2 camera iterations, got {len(outputs)}")
    for idx, output in enumerate(outputs):
        _require_cuda_tensor(f"camera output[{idx}]", output)
        if output.shape != (1, 2, 9):
            raise RuntimeError(f"unexpected camera output shape: {tuple(output.shape)}")
    return f"iterations={len(outputs)}, shape={tuple(outputs[-1].shape)}"


def check_dggt_joint_scene_tokenizer_forward() -> str:
    import torch

    from dggt.models.joint_scene_tokenizer import JointSceneTokenizer

    device = torch.device("cuda:0")
    model = JointSceneTokenizer(
        latent_dim=48,
        hidden_dim=64,
        num_layers=2,
        num_block_pairs=1,
        num_heads=4,
        layer_attn_depth=1,
        layer_attn_heads=4,
        stream_dim=16,
        detail_dim=16,
    ).to(device).eval()
    image_tokens = [torch.randn(1, 2, 4, 48, device=device) for _ in range(2)]
    frame_positions = torch.tensor([[0, 1]], device=device)
    with torch.no_grad():
        z = model.encode(image_tokens, patch_grid=(2, 2), frame_positions_1d=frame_positions)
        decoded = model.decode(z, patch_grid=(2, 2), frame_positions_1d=frame_positions)
    _require_cuda_tensor("joint_scene_tokenizer z", z)
    if z.shape != (1, 2, 4, 48):
        raise RuntimeError(f"unexpected tokenizer latent shape: {tuple(z.shape)}")
    if len(decoded) != 2:
        raise RuntimeError(f"expected 2 decoded levels, got {len(decoded)}")
    for idx, tensor in enumerate(decoded):
        _require_cuda_tensor(f"joint_scene_tokenizer decoded[{idx}]", tensor)
        if tensor.shape != (1, 2, 4, 48):
            raise RuntimeError(f"unexpected decoded[{idx}] shape: {tuple(tensor.shape)}")
    return f"z={tuple(z.shape)}, decoded={[tuple(t.shape) for t in decoded]}"


def check_dggt_feature_splatter_forward() -> str:
    import torch

    from dggt.models.feature_splatter import FeatureSplatter
    from dggt.models.gaussian_pointers import GaussianPointers, SCENE_OBJECT_ID, SRC_KIND_SCENE

    device = torch.device("cuda:0")
    model = FeatureSplatter(channels=4, chunk_channels=2, num_levels=2, patch_grid=(4, 4)).to(device).eval()

    gaussians = [
        {
            "means": torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.2]], device=device),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device),
            "scales": torch.tensor([[0.05, 0.05, 0.05], [0.04, 0.04, 0.04]], device=device),
            "opacities": torch.tensor([0.9, 0.8], device=device),
        }
    ]
    pointers = [
        GaussianPointers(
            src_kind=torch.full((2,), SRC_KIND_SCENE, device=device, dtype=torch.long),
            object_id=torch.full((2,), SCENE_OBJECT_ID, device=device, dtype=torch.long),
            view_n=torch.tensor([0, 0], device=device, dtype=torch.long),
            patch_idx=torch.tensor([0, 1], device=device, dtype=torch.long),
            visible_mask=torch.ones(2, device=device, dtype=torch.bool),
        )
    ]
    lut_scene = [
        torch.rand(1, 1, 16, 4, device=device),
        torch.rand(1, 1, 16, 4, device=device),
    ]
    cameras = {
        "viewmats": torch.eye(4, device=device).view(1, 1, 4, 4),
        "Ks": torch.tensor([[[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]]], device=device),
    }
    with torch.no_grad():
        outputs = model(
            gaussians_dggt=gaussians,
            pointers=pointers,
            lut_scene=lut_scene,
            lut_asset_dict=None,
            cameras_dggt=cameras,
            H=32,
            W=32,
            pool_to=4,
        )
    if len(outputs) != 2:
        raise RuntimeError(f"expected 2 feature splatter levels, got {len(outputs)}")
    for idx, tensor in enumerate(outputs):
        _require_cuda_tensor(f"feature_splatter output[{idx}]", tensor)
        if tensor.shape != (1, 1, 16, 4):
            raise RuntimeError(f"unexpected feature_splatter output[{idx}] shape: {tuple(tensor.shape)}")
    return f"outputs={[tuple(t.shape) for t in outputs]}"


def check_dggt_forward_imports() -> str:
    # Keep these imports after forward tests so failures point to the specific
    # compute path first. This also covers modules used by training scripts.
    modules = [
        "datasets.dataset",
        "dggt.models.vggt",
        "dggt.models.scene_flow",
        "dggt.losses.rgb_render_loss",
        "dggt.utils.gs",
    ]
    for module in modules:
        try:
            importlib.import_module(module)
        except OSError as exc:
            if "open3d" in traceback.format_exc() or "libGL.so.1" in str(exc):
                raise _open3d_import_error_hint(exc) from exc
            raise
    return ", ".join(modules)


def check_pip_conflicts(strict: bool) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = proc.stdout.strip()
    if proc.returncode != 0 and strict:
        raise RuntimeError(output)
    if proc.returncode != 0:
        return "reported conflicts, but not failing smoke test:\n" + output
    return output or "pip check passed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gsplat", action="store_true")
    parser.add_argument("--skip-pointops2", action="store_true")
    parser.add_argument("--strict-pip-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks: list[tuple[str, Callable[[], str | None]]] = [
        ("versions", check_versions),
        ("torch_ppu", check_torch_ppu),
        ("torchvision_ops", check_torchvision_ops),
        ("cv_open3d", check_cv_open3d),
    ]
    if not args.skip_gsplat:
        checks.append(("gsplat_minimal_rasterization", check_gsplat))
    if not args.skip_pointops2:
        checks.append(("pointops2_minimal_op", check_pointops2))
    checks.extend(
        [
            ("dggt_aggregator_forward", check_dggt_aggregator_forward),
            ("dggt_camera_head_forward", check_dggt_camera_head_forward),
            ("dggt_joint_scene_tokenizer_forward", check_dggt_joint_scene_tokenizer_forward),
            ("dggt_feature_splatter_forward", check_dggt_feature_splatter_forward),
            ("dggt_forward_imports", check_dggt_forward_imports),
            ("pip_check", lambda: check_pip_conflicts(args.strict_pip_check)),
        ]
    )

    results = [_run_check(name, fn) for name, fn in checks]
    failed = [result.name for result in results if not result.ok]
    if failed:
        print("\nFAILED checks:", ", ".join(failed))
        return 1
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
