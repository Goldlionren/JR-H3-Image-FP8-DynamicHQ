#!/usr/bin/env python3
"""Convert Comfy-Org MiniMax H3-Image pruned BF16 DiT to ComfyUI FP8 E4M3FN.

Single-pass, low-host-RAM output writer. Unlike the original experimental
converter, this preserves the ORIGINAL safetensors metadata/config, including
H3-Image PDD steps and sigmas, and measures quantization error for ALL 200
converted matrices. No source file is overwritten.

This script does not calibrate activations. High-quality W8A8 inference needs
JR_H3_FP8_Dynamic (or another independently validated runtime implementation).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open

CORE = re.compile(r"^blocks\.(\d+)\.(attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))\.weight$")
EXPECTED = 200
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
QUANT_SETTINGS = {"format": "float8_e4m3fn", "full_precision_matrix_mult": False}
MARKER = json.dumps(QUANT_SETTINGS, separators=(",", ":")).encode("utf-8")
DTYPE_NBYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "U64": 8, "I32": 4, "U32": 4, "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def read_header(path: Path):
    """Read safetensors header only, not the tensor payload."""
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors length prefix")
        size = struct.unpack("<Q", raw)[0]
        if not 2 <= size <= 128 * 1024 * 1024:
            raise ValueError("Unreasonable safetensors header length")
        data = f.read(size)
        if len(data) != size:
            raise ValueError("Truncated safetensors header")
    header = json.loads(data)
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header")
    return header, size + 8


def validate_source_header(header, source_bytes, data_base):
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise ValueError("Source contains no tensors")
    if any(k.endswith(".comfy_quant") or k.endswith(".weight_scale") or k.endswith(".input_scale") for k in tensors):
        raise ValueError("Source already appears quantized. Start from official PRUNED BF16, never re-quantize an FP8/INT8 file.")

    targets = [k for k in tensors if CORE.fullmatch(k)]
    if len(targets) != EXPECTED or {int(CORE.fullmatch(k).group(1)) for k in targets} != set(range(50)):
        raise ValueError(f"Expected 200 core linears across blocks 0..49; found {len(targets)}")
    for key in targets:
        desc = tensors[key]
        if desc["dtype"] not in ("BF16", "F16", "F32") or len(desc["shape"]) != 2:
            raise ValueError(f"Not a floating 2D weight: {key}: {desc}")

    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or not isinstance(metadata.get("config"), str):
        raise ValueError("Missing source metadata['config']: cannot recover PDD schedule. Do not create a broken checkpoint.")
    config = json.loads(metadata["config"])
    if not isinstance(config, dict) or not isinstance(config.get("transformer"), dict):
        raise ValueError("Missing config.transformer")
    transformer = config["transformer"]
    steps = transformer.get("pdd_steps")
    sigmas = transformer.get("pdd_sigmas")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("Missing/invalid config.transformer.pdd_steps")
    if not isinstance(sigmas, list) or len(sigmas) == 0:
        raise ValueError("Missing/invalid config.transformer.pdd_sigmas")

    ordered = sorted(tensors.items(), key=lambda kv: kv[1]["data_offsets"][0])
    previous = 0
    for key, desc in ordered:
        offsets = desc["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"Invalid offsets for {key}")
        start, stop = offsets
        if start != previous or stop < start or data_base + stop > source_bytes:
            raise ValueError(f"Noncontiguous or out-of-bounds source tensor {key}: {offsets}")
        previous = stop
        if desc["dtype"] not in DTYPE_NBYTES:
            raise ValueError(f"Unknown dtype for {key}: {desc['dtype']}")
        elements = math.prod(desc["shape"])
        if stop - start != elements * DTYPE_NBYTES[desc["dtype"]]:
            raise ValueError(f"Tensor storage size mismatch for {key}")
    if data_base + previous != source_bytes:
        raise ValueError("Source has unexpected trailing bytes")
    return ordered, targets, metadata, steps, sigmas


def plan_output(ordered, metadata):
    out_header = {"__metadata__": dict(metadata)}
    out_header["__metadata__"].update({
        "jr_converter": "JR-H3-Image-FP8-DynamicHQ streaming PTQ v2",
        "jr_activation_scaling": "requires JR_H3_FP8_Dynamic runtime plugin; not encoded as input_scale",
        "jr_quantized_linear_count": "200",
    })
    cursor = 0
    plan = []

    def append_entry(name, dtype, shape, action, original=None):
        nonlocal cursor
        nbytes = math.prod(shape) * DTYPE_NBYTES[dtype]
        out_header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [cursor, cursor + nbytes]}
        plan.append((name, action, original, nbytes))
        cursor += nbytes

    for key, desc in ordered:
        if CORE.fullmatch(key):
            append_entry(key, "F8_E4M3", desc["shape"], "quant", key)
            prefix = key[:-len(".weight")]
            append_entry(prefix + ".weight_scale", "F32", [], "scale", key)
            append_entry(prefix + ".comfy_quant", "U8", [len(MARKER)], "marker", key)
        else:
            append_entry(key, desc["dtype"], desc["shape"], "copy", key)
    encoded = json.dumps(out_header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    return encoded, plan, cursor


def copy_exact(src, dst, offset, nbytes, chunk=16 * 1024 * 1024):
    src.seek(offset)
    remaining = nbytes
    while remaining:
        block = src.read(min(chunk, remaining))
        if not block:
            raise IOError("Unexpected end of source tensor data")
        dst.write(block)
        remaining -= len(block)


def run(args):
    src = args.source.resolve()
    dst = args.out.resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    if src == dst:
        raise ValueError("Refusing to replace original BF16 checkpoint")
    if src.suffix.lower() != ".safetensors" or dst.suffix.lower() != ".safetensors":
        raise ValueError("Input and output must both be .safetensors files")
    if dst.exists():
        raise FileExistsError(f"Output already exists: {dst}")
    partial = dst.with_name(dst.stem + ".partial.safetensors")
    if partial.exists():
        raise FileExistsError(f"Partial output already exists: {partial}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in this Python environment. Select --device cpu or a CUDA-enabled PyTorch venv")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    src_header, src_base = read_header(src)
    ordered, targets, source_meta, pdd_steps, pdd_sigmas = validate_source_header(src_header, src.stat().st_size, src_base)
    header_bytes, plan, payload_bytes = plan_output(ordered, source_meta)
    dst.parent.mkdir(parents=True, exist_ok=True)
    required = payload_bytes + len(header_bytes) + 8 + (512 << 20)
    free = shutil.disk_usage(dst.parent).free
    if free < required:
        raise OSError(f"Not enough disk space: need >= {required/1e9:.2f} GB, free {free/1e9:.2f} GB")

    print(f"Source: {src}\nTarget: {dst}\nCompute device: {device}")
    print(f"Input tensors: {len(ordered)}; FP8 target linears: {len(targets)}; PDD steps: {pdd_steps}")
    print(f"PDD sigmas: {pdd_sigmas}")
    print(f"Expected output size: {(payload_bytes+len(header_bytes)+8)/1e9:.3f} GB", flush=True)

    stats = []
    scales = {}
    src_d = {k: v for k, v in ordered}
    try:
        with safe_open(str(src), framework="pt", device="cpu") as reader, src.open("rb") as rf, partial.open("xb") as wf:
            wf.write(struct.pack("<Q", len(header_bytes)))
            wf.write(header_bytes)
            for name, action, original, nbytes in plan:
                if action == "copy":
                    start, end = src_d[original]["data_offsets"]
                    assert end - start == nbytes
                    copy_exact(rf, wf, src_base + start, nbytes)
                elif action == "quant":
                    t = reader.get_tensor(original)
                    w = t.to(device=device, dtype=torch.float32)
                    if not torch.isfinite(w).all().item():
                        raise ValueError(f"NaN/Inf in source: {original}")
                    amax = w.abs().max().item()
                    scale = amax / FP8_MAX if amax > 0 else 1.0
                    if not math.isfinite(scale) or scale <= 0:
                        raise ValueError(f"Invalid scale for {original}")
                    # Use the *stored* FP32 scalar for quantization, not a higher-precision Python float.
                    scale = struct.unpack("<f", struct.pack("<f", scale))[0]
                    if not math.isfinite(scale) or scale <= 0:
                        raise ValueError(f"FP32 weight_scale is not representable for {original}")
                    scales[original] = scale
                    q = (w / scale).clamp(min=-FP8_MAX, max=FP8_MAX).to(torch.float8_e4m3fn)
                    err = (q.float() * scale - w)
                    denom = w.square().mean().sqrt().clamp_min(1e-20)
                    rel_rmse = float((err.square().mean().sqrt() / denom).item())
                    if not math.isfinite(rel_rmse):
                        raise ValueError(f"Non-finite quantization error for {original}")
                    qbytes = q.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    if len(qbytes) != nbytes:
                        raise ValueError(f"Wrong FP8 storage size for {original}")
                    wf.write(qbytes)
                    stats.append({"layer": original, "relative_rmse": round(rel_rmse, 8), "weight_scale": scale})
                    print(f"[{len(stats):3d}/{EXPECTED}] FP8 {original}  rel_RMSE={rel_rmse:.6f}", flush=True)
                    del qbytes, err, denom, q, w, t
                elif action == "scale":
                    wf.write(struct.pack("<f", scales[original]))
                elif action == "marker":
                    wf.write(MARKER)
                else:
                    raise AssertionError(action)
            wf.flush()
            os.fsync(wf.fileno())

        if partial.stat().st_size != 8 + len(header_bytes) + payload_bytes:
            raise ValueError("Output file size mismatch")
        with safe_open(str(partial), framework="pt", device="cpu") as verify:
            all_keys = list(verify.keys())
            f8_keys = [k for k in all_keys if CORE.fullmatch(k)]
            if len(f8_keys) != EXPECTED:
                raise ValueError("Missing FP8 core linears")
            for k in f8_keys:
                if verify.get_slice(k).get_shape() != src_d[k]["shape"]:
                    raise ValueError(f"Changed layer shape: {k}")
                prefix = k[:-len(".weight")]
                if not all(v in all_keys for v in (prefix + ".weight_scale", prefix + ".comfy_quant")):
                    raise ValueError(f"Missing scale/config: {k}")
            md = verify.metadata() or {}
            if md.get("config") != source_meta["config"]:
                raise ValueError("Original PDD config not preserved byte-for-byte")
            if md.get("jr_quantized_linear_count") != "200":
                raise ValueError("Missing JR FP8 metadata")
        os.replace(partial, dst)
    except Exception:
        print(f"Conversion failed. Partial file (if present) left for inspection: {partial}", file=sys.stderr)
        raise

    report = dst.with_suffix(".validation.json")
    report.write_text(json.dumps({
        "converter": "JR-H3-Image-FP8-DynamicHQ streaming PTQ v2",
        "source_path": str(src), "output_path": str(dst),
        "source_bytes": src.stat().st_size, "output_bytes": dst.stat().st_size,
        "pdd_steps": pdd_steps, "pdd_sigmas": pdd_sigmas,
        "quantized_linears": EXPECTED, "format": QUANT_SETTINGS,
        "activation_scaling": "DYNAMIC PLUGIN REQUIRED; checkpoint contains no input_scale",
        "all_layer_errors": stats,
        "relative_rmse_max": max(s["relative_rmse"] for s in stats),
        "relative_rmse_mean": sum(s["relative_rmse"] for s in stats) / len(stats),
        "validation_limits": "File, tensor metadata, weights quantization checks only; no H3 generation or GEMM profiling",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUCCESS: {dst}\nSize: {dst.stat().st_size / 1e9:.3f} GB\nFull 200-layer report: {report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Original Comfy-Org MiniMax H3-Image PRUNED BF16 checkpoint")
    parser.add_argument("--out", required=True, type=Path, help="NEW output path (will never overwrite existing file)")
    parser.add_argument("--device", default="cuda:0", help="Device for per-layer conversion (cuda:0 or cpu)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
