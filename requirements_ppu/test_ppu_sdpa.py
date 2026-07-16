#!/usr/bin/env python3
"""Isolated SDPA compatibility and performance tests for Alibaba PPU.

The production-shaped cases mirror Scene Flow pretraining with batch size 1,
sequence length 10, a 25x37 patch grid, and the 29-frame DGGT context.

Every case/backend pair runs in a fresh subprocess. This is required because a
device-side illegal memory access leaves the current accelerator context unusable.
Only the explicitly requested fused backend is enabled; this script never enables
or falls back to the unfused SDPA implementation.

Examples:
  python requirements_ppu/test_ppu_sdpa.py
  python requirements_ppu/test_ppu_sdpa.py --backend efficient
  python requirements_ppu/test_ppu_sdpa.py --backend flash --case tokenizer_layer
  python requirements_ppu/test_ppu_sdpa.py --backend efficient --include-backward
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    heads: int
    query_tokens: int
    key_tokens: int
    head_dim: int
    operation: str = "self"
    chunk_size: int = 0
    backward: bool = False
    large: bool = False
    description: str = ""


FORWARD_CASES = (
    Case(
        name="tokenizer_layer",
        batch=9250,
        heads=8,
        query_tokens=4,
        key_tokens=4,
        head_dim=144,
        description="Tokenizer LayerAttnStack: B*S*P batches over four feature levels",
    ),
    Case(
        name="tokenizer_pool",
        batch=9250,
        heads=8,
        query_tokens=1,
        key_tokens=4,
        head_dim=144,
        operation="mha",
        description="Tokenizer LearnedQueryPool using torch.nn.MultiheadAttention",
    ),
    Case(
        name="tokenizer_frame",
        batch=10,
        heads=16,
        query_tokens=925,
        key_tokens=925,
        head_dim=72,
        description="Tokenizer per-frame attention",
    ),
    Case(
        name="tokenizer_global",
        batch=925,
        heads=16,
        query_tokens=10,
        key_tokens=10,
        head_dim=72,
        description="Tokenizer cross-frame attention at each patch location",
    ),
    Case(
        name="aggregator_frame",
        batch=29,
        heads=16,
        query_tokens=930,
        key_tokens=930,
        head_dim=64,
        description="DGGT per-frame attention including five special tokens",
    ),
    Case(
        name="aggregator_global",
        batch=1,
        heads=16,
        query_tokens=26970,
        key_tokens=26970,
        head_dim=64,
        large=True,
        description="DGGT 29-frame global attention over 29*930 tokens",
    ),
)


BACKWARD_CASES = (
    Case(
        name="scene_flow_trunk_backward",
        batch=1,
        heads=20,
        query_tokens=12000,
        key_tokens=12000,
        head_dim=72,
        backward=True,
        large=True,
        description="Representative SceneFlow trunk training attention",
    ),
    Case(
        name="scene_flow_ddt_backward",
        batch=1,
        heads=16,
        query_tokens=12000,
        key_tokens=12000,
        head_dim=128,
        backward=True,
        large=True,
        description="Representative SceneFlow DDT-head training attention",
    ),
)


WORKAROUND_CASES = (
    Case(
        name="tokenizer_pool_b1",
        batch=1,
        heads=8,
        query_tokens=1,
        key_tokens=4,
        head_dim=144,
        operation="mha",
        description="LearnedQueryPool baseline with one flattened patch batch",
    ),
    Case(
        name="tokenizer_pool_b1024",
        batch=1024,
        heads=8,
        query_tokens=1,
        key_tokens=4,
        head_dim=144,
        operation="mha",
        description="LearnedQueryPool candidate using fused-attention batch chunks of 1024",
    ),
    Case(
        name="tokenizer_pool_b4096",
        batch=4096,
        heads=8,
        query_tokens=1,
        key_tokens=4,
        head_dim=144,
        operation="mha",
        description="LearnedQueryPool candidate using fused-attention batch chunks of 4096",
    ),
    Case(
        name="tokenizer_pool_b8192",
        batch=8192,
        heads=8,
        query_tokens=1,
        key_tokens=4,
        head_dim=144,
        operation="mha",
        description="LearnedQueryPool candidate using fused-attention batch chunks of 8192",
    ),
    Case(
        name="tokenizer_pool_q2",
        batch=9250,
        heads=8,
        query_tokens=2,
        key_tokens=4,
        head_dim=144,
        operation="mha",
        description="LearnedQueryPool candidate padding the fused query length to two",
    ),
    Case(
        name="tokenizer_pool_q4",
        batch=9250,
        heads=8,
        query_tokens=4,
        key_tokens=4,
        head_dim=144,
        operation="mha",
        description="LearnedQueryPool candidate padding the fused query length to four",
    ),
    Case(
        name="tokenizer_pool_chunk4096",
        batch=9250,
        heads=8,
        query_tokens=1,
        key_tokens=4,
        head_dim=144,
        operation="mha",
        chunk_size=4096,
        description="Production encoder pool with internal fused-attention batch chunking",
    ),
    Case(
        name="tokenizer_unpool_chunk4096",
        batch=9250,
        heads=8,
        query_tokens=4,
        key_tokens=1,
        head_dim=144,
        operation="mha",
        chunk_size=4096,
        description="Production decoder unpool with internal fused-attention batch chunking",
    ),
    Case(
        name="tokenizer_pool_long_chunk4096",
        batch=26825,
        heads=8,
        query_tokens=1,
        key_tokens=4,
        head_dim=144,
        operation="mha",
        chunk_size=4096,
        large=True,
        description="29-frame encoder pool with internal fused-attention batch chunking",
    ),
    Case(
        name="tokenizer_unpool_long_chunk4096",
        batch=26825,
        heads=8,
        query_tokens=4,
        key_tokens=1,
        head_dim=144,
        operation="mha",
        chunk_size=4096,
        large=True,
        description="29-frame decoder unpool with internal fused-attention batch chunking",
    ),
)


ALL_CASES = FORWARD_CASES + WORKAROUND_CASES + BACKWARD_CASES
CASE_BY_NAME = {case.name: case for case in ALL_CASES}
BACKEND_NAMES = ("flash", "efficient")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Test production-shaped PPU SDPA kernels in isolated subprocesses.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=BACKEND_NAMES,
        help="Backend to force. Repeat to test both; omitted means flash and efficient.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(CASE_BY_NAME),
        help="Case to run. Repeat for multiple cases; omitted means all forward cases.",
    )
    parser.add_argument(
        "--include-backward",
        action="store_true",
        help="Also run the two large representative SceneFlow backward cases.",
    )
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="Skip cases marked large, including aggregator_global and backward cases.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600, help="Per-case timeout in seconds.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--list-cases", action="store_true")

    # Internal options used only by isolated subprocess workers.
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-backend", choices=BACKEND_NAMES, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-case", choices=tuple(CASE_BY_NAME), help=argparse.SUPPRESS)
    return parser.parse_args()


def list_cases() -> None:
    print("Forward cases:")
    for case in FORWARD_CASES:
        print(f"  {case.name:28s} {format_shape(case):34s} {case.description}")
    print("LearnedQueryPool fused-attention workaround candidates:")
    for case in WORKAROUND_CASES:
        print(f"  {case.name:28s} {format_shape(case):34s} {case.description}")
    print("Backward cases (enabled with --include-backward):")
    for case in BACKWARD_CASES:
        print(f"  {case.name:28s} {format_shape(case):34s} {case.description}")


def format_shape(case: Case) -> str:
    if case.operation == "mha":
        shape = (
            f"MHA B={case.batch}, H={case.heads}, "
            f"L={case.query_tokens}, S={case.key_tokens}, D={case.head_dim}"
        )
        if case.chunk_size > 0:
            shape += f", chunk={case.chunk_size}"
        return shape
    return (
        f"q=[{case.batch},{case.heads},{case.query_tokens},{case.head_dim}] "
        f"k/v=[{case.batch},{case.heads},{case.key_tokens},{case.head_dim}]"
    )


def selected_backend(torch_module, name: str):
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as exc:
        raise RuntimeError("torch.nn.attention.sdpa_kernel is required (expected in PPU torch 2.9)") from exc

    backend = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
    }[name]
    # Passing exactly one backend prevents any implicit fallback.
    return sdpa_kernel(backend)


def make_packed_self_qkv(torch_module, case: Case, device):
    packed = torch_module.empty(
        case.batch,
        case.query_tokens,
        3,
        case.heads,
        case.head_dim,
        device=device,
        dtype=torch_module.bfloat16,
    )
    packed.normal_(mean=0.0, std=0.02)
    packed.requires_grad_(case.backward)
    q, k, v = packed.unbind(dim=2)
    return packed, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


def run_sdpa_case(torch_module, functional, case: Case, device):
    packed, q, k, v = make_packed_self_qkv(torch_module, case, device)

    def invoke():
        return functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    return packed, invoke


def run_mha_case(torch_module, case: Case, device):
    embed_dim = case.heads * case.head_dim
    module = torch_module.nn.MultiheadAttention(
        embed_dim,
        case.heads,
        batch_first=True,
        device=device,
        dtype=torch_module.bfloat16,
    ).eval()
    module.requires_grad_(False)
    query_batch = 1 if case.chunk_size > 0 else case.batch
    query = torch_module.empty(
        query_batch,
        case.query_tokens,
        embed_dim,
        device=device,
        dtype=torch_module.bfloat16,
    ).normal_(mean=0.0, std=0.02)
    key_value = torch_module.empty(
        case.batch,
        case.key_tokens,
        embed_dim,
        device=device,
        dtype=torch_module.bfloat16,
    ).normal_(mean=0.0, std=0.02)

    def invoke():
        if case.chunk_size > 0:
            outputs = []
            for start in range(0, case.batch, case.chunk_size):
                end = min(start + case.chunk_size, case.batch)
                query_chunk = query.expand(end - start, -1, -1)
                output_chunk, _ = module(
                    query_chunk,
                    key_value[start:end],
                    key_value[start:end],
                    need_weights=False,
                )
                outputs.append(output_chunk)
            return torch_module.cat(outputs, dim=0)
        output, _ = module(query, key_value, key_value, need_weights=False)
        return output

    return None, invoke


def reset_peak_memory(torch_module, device) -> None:
    try:
        torch_module.cuda.reset_peak_memory_stats(device)
    except (AttributeError, RuntimeError):
        pass


def peak_memory_mib(torch_module, device) -> float | None:
    try:
        return float(torch_module.cuda.max_memory_allocated(device)) / (1024.0**2)
    except (AttributeError, RuntimeError):
        return None


def worker(args: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("PPU is not visible through torch.cuda")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be >= 0 and --iterations must be > 0")

    backend_name = str(args._worker_backend)
    case = CASE_BY_NAME[str(args._worker_case)]
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    if case.operation == "mha":
        grad_source, invoke = run_mha_case(torch, case, device)
    else:
        grad_source, invoke = run_sdpa_case(torch, F, case, device)

    print(
        f"[WORKER] torch={torch.__version__} device={torch.cuda.get_device_name(device)} "
        f"backend={backend_name} case={case.name}",
        flush=True,
    )
    print(f"[WORKER] {format_shape(case)} backward={case.backward}", flush=True)

    reset_peak_memory(torch, device)
    with selected_backend(torch, backend_name):
        for _ in range(args.warmup):
            output = invoke()
            if case.backward:
                output.float().square().mean().backward()
                assert grad_source is not None
                grad_source.grad = None
        torch.cuda.synchronize(device)

        started = time.perf_counter()
        output = None
        for _ in range(args.iterations):
            output = invoke()
            if case.backward:
                output.float().square().mean().backward()
                assert grad_source is not None
                grad_source.grad = None
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

    assert output is not None
    finite = bool(torch.isfinite(output).all().item())
    if not finite:
        raise RuntimeError("attention output contains non-finite values")

    seconds_per_iteration = elapsed / args.iterations
    forward_flops = (
        4.0
        * case.batch
        * case.heads
        * case.query_tokens
        * case.key_tokens
        * case.head_dim
    )
    result = {
        "status": "PASS",
        "backend": backend_name,
        "case": case.name,
        "backward": case.backward,
        "seconds_per_iteration": seconds_per_iteration,
        "estimated_forward_tflops": forward_flops / seconds_per_iteration / 1.0e12,
        "peak_memory_mib": peak_memory_mib(torch, device),
        "output_shape": list(output.shape),
        "output_stride": list(output.stride()),
    }
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


def parent(args: argparse.Namespace) -> int:
    backends = args.backend or list(BACKEND_NAMES)
    if args.case:
        cases = [CASE_BY_NAME[name] for name in args.case]
    else:
        cases = list(FORWARD_CASES)
        if args.include_backward:
            cases.extend(BACKWARD_CASES)
    if args.skip_large:
        cases = [case for case in cases if not case.large]
    if not cases:
        raise ValueError("no cases selected")

    script = Path(__file__).resolve()
    results: list[tuple[str, str, int]] = []
    print(f"Python: {sys.executable}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"Backends: {', '.join(backends)}")
    print(f"Cases: {', '.join(case.name for case in cases)}")
    print("No fallback backend is enabled by this test.\n", flush=True)

    for backend_name in backends:
        for case in cases:
            print("=" * 88)
            print(f"RUN backend={backend_name} case={case.name}: {case.description}", flush=True)
            command = [
                sys.executable,
                str(script),
                "--_worker",
                "--_worker-backend",
                backend_name,
                "--_worker-case",
                case.name,
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
                "--device",
                args.device,
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
                output = completed.stdout.rstrip()
                if output:
                    print(output)
                return_code = completed.returncode
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                output = stdout + stderr
                if output:
                    print(output.rstrip())
                print(f"[TIMEOUT] exceeded {args.timeout} seconds")
                return_code = 124

            status = "PASS" if return_code == 0 else "FAIL"
            print(f"[{status}] backend={backend_name} case={case.name} exit_code={return_code}", flush=True)
            results.append((backend_name, case.name, return_code))

    print("\n" + "=" * 88)
    print("SUMMARY")
    for backend_name, case_name, return_code in results:
        status = "PASS" if return_code == 0 else "FAIL"
        print(f"  {status:4s}  backend={backend_name:9s} case={case_name}")
    failed = [item for item in results if item[2] != 0]
    print(f"\nPassed {len(results) - len(failed)}/{len(results)} isolated runs.")
    return 1 if failed else 0


def main() -> int:
    args = parse_args()
    if args.list_cases:
        list_cases()
        return 0
    if args._worker:
        if args._worker_backend is None or args._worker_case is None:
            raise ValueError("internal worker requires --_worker-backend and --_worker-case")
        return worker(args)
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
