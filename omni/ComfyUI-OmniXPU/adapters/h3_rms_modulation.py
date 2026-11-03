"""Route the exact MiniMax H3 RMSNorm/modulation boundary to Omni XPU."""

from __future__ import annotations

import inspect
import logging
import os
from functools import wraps
from typing import Any, Callable

import torch

from ..patches.debug import log_debug_event, trace_patch


log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "__omnixpu_h3_rms_modulation_original__"
_HIDDEN_SIZE = 5376
_MODULATION_EXPAND = 6
_MAX_SEGMENTS = 8
_ATTENTION_SIZE = 7168
_FFN_SIZE = 14336
# Projections whose inputs the h3fuse sidecar rotates and quantizes in one pass.
_FUSED_WEIGHT_SHAPES = {
    "qkv_proj": (3 * _ATTENTION_SIZE, _HIDDEN_SIZE),
    "out_proj": (_HIDDEN_SIZE, _ATTENTION_SIZE),
    "fc1": (2 * _FFN_SIZE, _HIDDEN_SIZE),
    "fc2": (_HIDDEN_SIZE, _FFN_SIZE),
}
_omni_norm = None
_omni_int8 = None
_fused_ops = None
_routed_calls = 0
_fallback_calls = 0
_fallback_reasons: dict[str, int] = {}
_fused_block_calls = 0
_fused_block_fallbacks: dict[str, int] = {}
# B6：融合块检查之前，把 lowvram 留在 CPU 的 INT8 权重临时搬到 XPU，让原有检查照常通过
_FUSED_STAGE = os.environ.get("OMNIXPU_H3_FUSED_STAGE", "0") == "1"
_fused_block_staged = 0
# B5-2：attention 被替换（2 卡 USP、外部 attention 覆盖）时，attention 当黑盒照常调用，只融合 MLP 半块。
# OMNIXPU_H3_FUSED_HALF=0 关闭；OMNIXPU_H3_FUSED_HALF_FORCE=1 时即使整块融合可用也走半融合（1 卡上做对比用）
_FUSED_HALF = os.environ.get("OMNIXPU_H3_FUSED_HALF", "1") != "0"
_FUSED_HALF_FORCE = os.environ.get("OMNIXPU_H3_FUSED_HALF_FORCE", "0") == "1"
_fused_half_calls = 0
_fused_half_fallbacks: dict[str, int] = {}
_int8_attn_ops = None
_int8_attn_bf16_blocks: frozenset[int] = frozenset()
_int8_attn_check_left = 0
_int8_attn_calls = 0
# B4 诊断：OMNIXPU_H3_FUSED_DIAG=1 时记录 _fused_int8_weight 第一个失败的条件，默认关闭、行为不变
_FUSED_DIAG = os.environ.get("OMNIXPU_H3_FUSED_DIAG", "0") == "1"
_FUSED_DIAG_MAX_LOGS = int(os.environ.get("OMNIXPU_H3_FUSED_DIAG_MAX_LOGS", "20"))
_fused_diag: dict[str, int] = {}                # "<层>:<第一个失败条件或 ok>" -> 次数
_fused_diag_detail_logged: set[str] = set()     # 每种 "<层>:<条件>" 只打印一次详细信息
_fused_diag_block_index = None                  # 当前 block 下标，由 _run_fused_block 设置
_fused_diag_forward: dict[str, list] = {}       # 本次 forward 内 "<层>:<条件>" -> 失败的 block 下标
_fused_diag_last_summary = None
_fused_diag_logs = 0


def _record_fallback(reason: str) -> None:
    global _fallback_calls
    if torch.compiler.is_compiling():
        return
    _fallback_calls += 1
    _fallback_reasons[reason] = _fallback_reasons.get(reason, 0) + 1


def get_stats() -> dict[str, Any]:
    return {
        "routed": _routed_calls,
        "fallback": _fallback_calls,
        "reasons": dict(_fallback_reasons),
        "fused_block": _fused_block_calls,
        "fused_block_fallback": dict(_fused_block_fallbacks),
        "fused_block_staged": _fused_block_staged,
        "fused_half": _fused_half_calls,
        "fused_half_fallback": dict(_fused_half_fallbacks),
        "int8_attention": _int8_attn_calls,
        **({"fused_diag": dict(_fused_diag)} if _FUSED_DIAG else {}),
    }


def _forward_api_reason(function: Callable[..., Any]) -> str:
    # Registration needs the supported call API, not a particular source-file
    # spelling. Installed-source tests separately check the upstream model math.
    try:
        parameters = tuple(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return "MiniMax H3 DiTBlock.forward signature is unavailable"
    names = (
        "self", "x", "t_emb", "mod_segments", "rope_freqs",
        "transformer_options",
    )
    if len(parameters) == len(names) + 1:
        attention = parameters[-1]
        if (
            attention.name != "attention"
            or attention.default is not None
            or attention.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
        ):
            return "MiniMax H3 DiTBlock.forward API is unsupported"
        parameters = parameters[:-1]
    if (
        tuple(parameter.name for parameter in parameters) != names
        or any(
            parameter.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in parameters
        )
        or any(
            parameter.default is not inspect.Parameter.empty
            for parameter in parameters[:-1]
        )
        or not isinstance(parameters[-1].default, dict)
        or parameters[-1].default
    ):
        return "MiniMax H3 DiTBlock.forward API is unsupported"
    return ""


def _segment_reason(segments: Any, rows: int, modulation_rows: int) -> str:
    if not isinstance(segments, (tuple, list)):
        return "segment_type"
    if not 1 <= len(segments) <= _MAX_SEGMENTS:
        return "segment_count"
    previous_stop = 0
    for segment in segments:
        if not isinstance(segment, (tuple, list)) or len(segment) != 3:
            return "segment_shape"
        if any(type(value) is not int for value in segment):
            return "segment_value_type"
        start, stop, modulation_row = segment
        if start != previous_stop or stop <= start:
            return "segment_coverage"
        if not 0 <= modulation_row < modulation_rows:
            return "segment_modulation_row"
        previous_stop = stop
    if previous_stop != rows:
        return "segment_coverage"
    return ""


def _block_input_reason(block: Any, x: Any, segments: Any) -> str:
    if not isinstance(x, torch.Tensor):
        return "input_type"
    if x.dtype != torch.bfloat16:
        return "input_dtype"
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] != _HIDDEN_SIZE:
        return "input_shape"
    if not x.is_contiguous():
        return "input_layout"
    if x.requires_grad:
        return "requires_grad"
    for name in ("norm1", "norm2"):
        layer = getattr(block, name, None)
        if layer is None:
            return f"{name}_missing"
        normalized_shape = getattr(layer, "normalized_shape", None)
        try:
            normalized_shape = tuple(normalized_shape)
        except TypeError:
            return f"{name}_shape"
        if normalized_shape != (_HIDDEN_SIZE,):
            return f"{name}_shape"
        weight = getattr(layer, "weight", None)
        if weight is None or tuple(weight.shape) != (_HIDDEN_SIZE,):
            return f"{name}_weight"
    return _segment_reason(segments, x.shape[0], 1 << 30)


def _modulation_reason(values: Any, x: torch.Tensor, segments: Any) -> str:
    if not isinstance(values, (tuple, list)) or len(values) != _MODULATION_EXPAND:
        return "adaln_outputs"
    expected_stride = (_MODULATION_EXPAND * _HIDDEN_SIZE, 1)
    rows = None
    for value in values:
        if not isinstance(value, torch.Tensor):
            return "modulation_type"
        if value.device != x.device or value.dtype not in (
            torch.bfloat16, torch.float32
        ):
            return "modulation_device_dtype"
        if value.ndim != 2 or value.shape[1] != _HIDDEN_SIZE:
            return "modulation_shape"
        if tuple(value.stride()) != expected_stride:
            return "modulation_layout"
        if value.requires_grad:
            return "modulation_requires_grad"
        if rows is None:
            rows = value.shape[0]
        elif value.shape[0] != rows:
            return "modulation_rows"
    return _segment_reason(segments, x.shape[0], int(rows))


def _weight_reason(weight: Any, bias: Any, x: torch.Tensor) -> str:
    if bias is not None:
        return "norm_bias"
    if not isinstance(weight, torch.Tensor):
        return "norm_weight_type"
    if weight.device != x.device or weight.dtype != torch.bfloat16:
        return "norm_weight_device_dtype"
    if weight.ndim != 1 or weight.numel() != _HIDDEN_SIZE:
        return "norm_weight_shape"
    if not weight.is_contiguous():
        return "norm_weight_layout"
    return ""


def _run_fused(
    comfy_ops: Any,
    layer: Any,
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    segments: list[tuple[int, int, int]],
) -> tuple[torch.Tensor | None, str]:
    comfy_ops.run_every_op()
    context = comfy_ops.CastBiasWeightContext(
        layer if layer.weight is not None else None,
        x,
        offloadable=True,
    )
    with context as (weight, bias):
        reason = _weight_reason(weight, bias, x)
        if reason:
            return None, reason
        eps = layer.eps if layer.eps is not None else 1e-6
        # Curve-based H3 checkpoints produce FP32 modulation. Match the
        # original _mod_scale_shift: cast before 1 + scale and before shift
        # addition, while leaving the original gate path untouched.
        output = _omni_norm.rms_norm_segmented_modulation(
            weight,
            x,
            scale.to(dtype=x.dtype),
            shift.to(dtype=x.dtype),
            segments,
            eps=eps,
        )
    return output, ""


def _rms_modulation(
    comfy_ops: Any,
    original_modulate: Callable[..., Any],
    layer: Any,
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    segments: list[tuple[int, int, int]],
    reason: str,
) -> tuple[torch.Tensor, int]:
    global _routed_calls

    # Cast helpers execute these hooks. Let the original norm own that call
    # once instead of probing a hook result and then invoking the norm again.
    if not reason and len(getattr(layer, "weight_function", ())) > 0:
        reason = "norm_weight_function"
    if not reason and len(getattr(layer, "bias_function", ())) > 0:
        reason = "norm_bias_function"
    if not reason:
        output, reason = _run_fused(comfy_ops, layer, x, scale, shift, segments)
        if output is not None:
            # Eager diagnostics must not become mutable Dynamo guards.
            if not torch.compiler.is_compiling():
                _routed_calls += 1
            return output, 1

    # Once AdaLN or a residual update has run, keep fallback at this operator
    # boundary. Replaying the block would repeat projection or in-place gates.
    _record_fallback(reason)
    return original_modulate(layer(x), shift, scale, segments), 0


def _sidecar_path(name: str, env: str) -> str:
    """Return the H3 sidecar path: ``env`` override, else the omni_xpu_kernel build."""
    path = os.environ.get(env, "")
    if path:
        return path
    try:
        from omni_xpu_kernel import h3
    except ImportError:
        return ""
    return h3.find_extension(name)


def _load_fused_sidecar() -> str:
    """Load the h3fuse ops shipped as omni_xpu_kernel.h3.h3_fused."""
    global _omni_int8, _fused_ops
    if os.environ.get("OMNIXPU_H3_FUSED", "1") == "0":
        return "disabled by env"
    path = _sidecar_path("h3_fused", "OMNIXPU_H3_FUSED_SO")
    if not os.path.isfile(path):
        return f"h3_fused not built ({path or 'omni_xpu_kernel.h3'})"
    from omni_xpu_kernel import int8 as omni_int8

    native = omni_int8._get_native()
    if native is None or not hasattr(native, "int8_linear_prequantized"):
        return "native int8_linear_prequantized is unavailable"
    try:
        torch.ops.load_library(path)
    except (OSError, RuntimeError) as error:
        return f"cannot load {path}: {error}"
    _omni_int8 = native
    _fused_ops = torch.ops.h3fuse
    return ""


def _load_int8_attention_sidecar() -> str:
    """Load the INT8 QK^T attention ops shipped as omni_xpu_kernel.h3.h3_int8_attn."""
    global _int8_attn_ops, _int8_attn_bf16_blocks, _int8_attn_check_left
    if os.environ.get("OMNIXPU_H3_INT8_ATTN", "1") == "0":
        return "disabled by env"
    path = _sidecar_path("h3_int8_attn", "OMNIXPU_H3_INT8_ATTN_SO")
    if not os.path.isfile(path):
        return f"h3_int8_attn not built ({path or 'omni_xpu_kernel.h3'})"
    try:
        torch.ops.load_library(path)
    except (OSError, RuntimeError) as error:
        return f"cannot load {path}: {error}"
    _int8_attn_ops = torch.ops.h3i8attn
    # INT8 QK^T flips some tokens of these blocks' sharp heads (worst cos < 0.81 over 20 steps); keep them in bf16.
    _int8_attn_bf16_blocks = frozenset(
        int(b) for b in os.environ.get("OMNIXPU_H3_INT8_ATTN_BF16_BLOCKS", "45,48,49").split(",") if b
    )
    _int8_attn_check_left = int(os.environ.get("OMNIXPU_H3_INT8_ATTN_CHECK", "0"))
    return ""


def _fused_int8_weight(linear: Any, x: torch.Tensor, name: str = ""):
    """Return resident (qdata, scale) of a G256 ConvRot TensorWise INT8 linear."""
    weight = getattr(linear, "weight", None)
    params = getattr(weight, "_params", None)
    qdata = getattr(weight, "_qdata", None)
    scale = getattr(params, "scale", None)
    # 条件和顺序与原来的单个 or 表达式完全一致，只是拆开以便知道是哪一条失败
    checks = (
        ("layout", lambda: getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout"),
        ("bias", lambda: getattr(linear, "bias", None) is not None),
        ("weight_function", lambda: len(getattr(linear, "weight_function", ())) > 0),
        ("bias_function", lambda: len(getattr(linear, "bias_function", ())) > 0),
        ("force_cast", lambda: getattr(linear, "comfy_force_cast_weights", False)),
        ("full_precision_mm", lambda: getattr(linear, "_full_precision_mm", False)),
        ("qdata_type", lambda: not isinstance(qdata, torch.Tensor)),
        ("scale_type", lambda: not isinstance(scale, torch.Tensor)),
        ("qdata_device", lambda: qdata.device != x.device),
        ("scale_device", lambda: scale.device != x.device),
        ("scale_dtype", lambda: scale.dtype != torch.float32),
        ("orig_dtype", lambda: getattr(params, "orig_dtype", None) != x.dtype),
        ("transposed", lambda: getattr(params, "transposed", False)),
        ("convrot", lambda: not getattr(params, "convrot", False)),
        ("convrot_groupsize", lambda: getattr(params, "convrot_groupsize", 256) != 256),
    )
    failure = next((check for check, failed in checks if failed()), "")
    if _FUSED_DIAG and name and not torch.compiler.is_compiling():
        _fused_diag_record(name, failure, linear, weight, params, qdata, scale, x)
    if failure:
        return None
    return qdata, scale


def _fused_diag_record(name, failure, linear, weight, params, qdata, scale, x) -> None:
    key = f"{name}:{failure or 'ok'}"
    _fused_diag[key] = _fused_diag.get(key, 0) + 1
    if not failure:
        return
    _fused_diag_forward.setdefault(key, []).append(_fused_diag_block_index)
    if key in _fused_diag_detail_logged:
        return
    _fused_diag_detail_logged.add(key)
    # worker 里的 logging 级别是 WARNING，INFO 看不到，所以用 print
    print(
        f"H3_FUSED_DIAG/ first {key} block={_fused_diag_block_index} x.device={x.device} "
        f"weight={type(weight).__name__} layout={getattr(weight, '_layout_cls', None)} "
        f"qdata.device={getattr(qdata, 'device', None)} qdata.shape={tuple(getattr(qdata, 'shape', ()))} "
        f"scale.device={getattr(scale, 'device', None)} scale.shape={tuple(getattr(scale, 'shape', ()))} "
        f"comfy_cast_weights={getattr(linear, 'comfy_cast_weights', None)} "
        f"lowvram_marked={hasattr(linear, 'prev_comfy_cast_weights')} "
        f"force_cast={getattr(linear, 'comfy_force_cast_weights', None)} "
        f"weight_function={len(getattr(linear, 'weight_function', ()))} "
        f"orig_dtype={getattr(params, 'orig_dtype', None)} "
        f"convrot={getattr(params, 'convrot', None)}/{getattr(params, 'convrot_groupsize', None)}",
        flush=True,
    )


def _fused_diag_flush_forward() -> None:
    """上一次 forward 的失败分布和上次打印的不同时打印一行（交替模式也能看到），最多 MAX_LOGS 行。"""
    global _fused_diag_forward, _fused_diag_last_summary, _fused_diag_logs
    summary = dict(sorted(_fused_diag_forward.items()))
    _fused_diag_forward = {}
    if not summary or summary == _fused_diag_last_summary or _fused_diag_logs >= _FUSED_DIAG_MAX_LOGS:
        return
    _fused_diag_last_summary = summary
    _fused_diag_logs += 1
    print("H3_FUSED_DIAG/ forward_failures " + " ".join(
        f"{key}={len(blocks)}@{blocks}" for key, blocks in summary.items()), flush=True)


def _stage_offloaded_weights(block: Any, x: torch.Tensor, h3_model: Any, rope_freqs: Any) -> tuple[list, dict]:
    """把不在 x.device 上的 4 个 INT8 线性层权重临时换成 XPU 副本，返回恢复用的 (linear, 原 Parameter) 列表。

    原路径（ComfyUI 的 CastBiasWeightContext）在每层计算前同样要 H2D，这里只是提前到检查之前，
    搬运量相同。只替换 linear._parameters["weight"] 这个引用，检查结束立即恢复；
    之后 XPU 副本由 _fused_block_reason 返回的 (qdata, scale) 持有，每层 GEMM 用完就释放。
    """
    global _fused_block_staged
    # 这两种情况 _fused_block_reason 一定回落，提前搬运只会让原路径再搬一遍
    if rope_freqs is None or any(
        type(module).forward.__module__ != h3_model.__name__ for module in (block.attn, block.mlp)
    ):
        return [], {}
    return _stage_modules(
        block, x, h3_model,
        (block.attn.qkv_proj, block.attn.out_proj, block.mlp.fc1, block.mlp.fc2), ("norm1", "norm2"),
    )


def _stage_modules(block: Any, x: torch.Tensor, h3_model: Any, linears: tuple, norm_names: tuple) -> tuple[list, dict]:
    """_stage_offloaded_weights 的搬运部分：整块融合搬 4 个线性层 + 2 个 norm，半融合只搬 fc1/fc2/norm2。"""
    global _fused_block_staged
    staged = []
    norms = {}
    for linear in linears:
        weight = getattr(linear, "weight", None)
        qdata = getattr(weight, "_qdata", None)
        if (
            not isinstance(qdata, torch.Tensor)
            or qdata.device == x.device
            # 动态 VRAM（vbar）有自己的搬运和预取，不在这里处理
            or hasattr(linear, "_v")
            # 有 LoRA 等 weight_function 时检查一定不过，搬了也白搬
            or len(getattr(linear, "weight_function", ())) > 0
            or len(getattr(linear, "bias_function", ())) > 0
        ):
            continue
        # 同步拷贝，不走 offload stream 的 cast buffer：同一 block 可能同时持有 3 层，共用 buffer 会互相覆盖
        resident = h3_model.comfy.model_management.cast_to(weight, None, x.device, non_blocking=False)
        staged.append((linear, linear._parameters["weight"]))
        linear._parameters["weight"] = torch.nn.Parameter(resident, requires_grad=False)
    # norm1/norm2 的 gamma（bf16 [5376]，约 10.5KB）同样会被 lowvram 留在 CPU（R7 实测 47/50 块卡在 norm1_weight）。
    # 和线性层不同，_run_fused_block 在检查之后还要直接把 norm 权重交给 kernel，所以副本要返回给调用方继续用。
    for name in norm_names:
        layer = getattr(block, name)
        weight = getattr(layer, "weight", None)
        if (
            not isinstance(weight, torch.Tensor)
            or weight.device == x.device
            or hasattr(layer, "_v")
            or len(getattr(layer, "weight_function", ())) > 0
        ):
            continue
        resident = h3_model.comfy.model_management.cast_to(weight, None, x.device, non_blocking=False)
        staged.append((layer, layer._parameters["weight"]))
        layer._parameters["weight"] = torch.nn.Parameter(resident, requires_grad=False)
        norms[name] = resident
    if staged and not torch.compiler.is_compiling():
        _fused_block_staged += len(staged)
    return staged, norms


def _restore_staged_weights(staged: list) -> None:
    for linear, original in staged:
        linear._parameters["weight"] = original


def _fused_block_reason(block: Any, x: torch.Tensor, h3_model: Any, rope_freqs: Any):
    if rope_freqs is None:
        return "no_rope", None
    # A replaced Attention/MLP forward (e.g. sparse attention) must keep running.
    for module in (block.attn, block.mlp):
        if type(module).forward.__module__ != h3_model.__name__:
            return "patched_submodule", None
    weights = {}
    first_bad = ""
    for name, linear in (
        ("qkv_proj", block.attn.qkv_proj),
        ("out_proj", block.attn.out_proj),
        ("fc1", block.mlp.fc1),
        ("fc2", block.mlp.fc2),
    ):
        weight = _fused_int8_weight(linear, x, name)
        if weight is None or tuple(weight[0].shape) != _FUSED_WEIGHT_SHAPES[name]:
            first_bad = first_bad or f"{name}_weight"
            if not _FUSED_DIAG:
                break
            continue
        weights[name] = weight
    if first_bad:
        return first_bad, None
    for name in ("norm1", "norm2"):
        layer = getattr(block, name)
        if _weight_reason(layer.weight, layer.bias, x) or len(getattr(layer, "weight_function", ())) > 0:
            return f"{name}_weight", None
    return "", weights


def _module_hooked(module: Any) -> bool:
    """实例上被换了 forward，或挂了 forward hook：类上的 forward 已经不代表实际执行的逻辑。

    raylight 有 FSDP LoRA sidecar 时在实例上替换 mlp.forward（usp.py:420），
    comfy 的 BypassInjectionManager 把 LoRA 注入成 fc1/fc2 实例上的 forward（bypass.py:200）。
    融合路径直接拿 qdata 做 GEMM，会悄悄丢掉这些逻辑，所以一律回落。
    """
    return (
        "forward" in vars(module)
        or bool(getattr(module, "_forward_hooks", None))
        or bool(getattr(module, "_forward_pre_hooks", None))
    )


def _fused_half_reason(block: Any, x: torch.Tensor, h3_model: Any):
    """半融合块的检查：只看 MLP 半块（mlp 原版、fc1/fc2 驻留 INT8、norm2 权重可用），attention 不管。"""
    mlp = block.mlp
    if type(mlp).forward.__module__ != h3_model.__name__ or _module_hooked(mlp):
        return "patched_mlp", None
    weights = {}
    for name in ("fc1", "fc2"):
        linear = getattr(mlp, name)
        if _module_hooked(linear):
            return f"{name}_hooked", None
        weight = _fused_int8_weight(linear, x, f"half_{name}")
        if weight is None or tuple(weight[0].shape) != _FUSED_WEIGHT_SHAPES[name]:
            return f"{name}_weight", None
        weights[name] = weight
    layer = block.norm2
    if _weight_reason(layer.weight, layer.bias, x) or len(getattr(layer, "weight_function", ())) > 0:
        return "norm2_weight", None
    return "", weights


def _run_fused_half_block(
    comfy_ops, h3_model, block, x, modulation, segments, rope_freqs, transformer_options, attention,
):
    """norm1 走逐算子 kernel，attention 当黑盒调用，MLP 半块和 _run_fused_block 后半段相同。

    检查在动 x 之前做完；不合格时返回 None，调用方照旧走逐算子路径。
    """
    global _fused_half_calls
    staged, staged_norms = (
        _stage_modules(block, x, h3_model, (block.mlp.fc1, block.mlp.fc2), ("norm2",))
        if _FUSED_STAGE and not _module_hooked(block.mlp) else ([], {})
    )
    try:
        reason, weights = _fused_half_reason(block, x, h3_model)
    finally:
        _restore_staged_weights(staged)
    del staged
    if reason:
        if not torch.compiler.is_compiling():
            _fused_half_fallbacks[reason] = _fused_half_fallbacks.get(reason, 0) + 1
        return None

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
    h, _ = _rms_modulation(
        comfy_ops, h3_model._mod_scale_shift, block.norm1, x, scale_msa, shift_msa, segments, "")
    attn_out = (block.attn if attention is None else attention)(
        h, rope_freqs=rope_freqs, transformer_options=transformer_options)
    del h
    if attn_out.shape != x.shape or attn_out.dtype != x.dtype:
        # 黑盒 attention 的输出不符合 norm_rot_quant 残差模式的要求；x 还没动，按原版补完这个 block
        if not torch.compiler.is_compiling():
            _fused_half_fallbacks["attn_out_layout"] = _fused_half_fallbacks.get("attn_out_layout", 0) + 1
        x = h3_model._mod_gate(x, gate_msa, attn_out, segments)
        h, _ = _rms_modulation(
            comfy_ops, h3_model._mod_scale_shift, block.norm2, x, scale_mlp, shift_mlp, segments, "")
        return h3_model._mod_gate(x, gate_mlp, block.mlp(h), segments)

    starts, stops, rows = (list(column) for column in zip(*segments))
    # x += gate_msa * attn_out in place (as _mod_gate), fused with norm2.
    comfy_ops.run_every_op()
    q, q_scale = _fused_ops.norm_rot_quant(
        x, staged_norms.get("norm2", block.norm2.weight), scale_mlp.to(x.dtype), shift_mlp.to(x.dtype),
        starts, stops, rows, 1e-6 if block.norm2.eps is None else block.norm2.eps,
        attn_out.contiguous(), gate_msa.to(x.dtype))
    del attn_out
    hidden = _fused_linear(q, q_scale, weights.pop("fc1"))
    q, q_scale = _fused_ops.silu_mul_rot_quant(hidden)
    del hidden
    mlp_out = _fused_linear(q, q_scale, weights.pop("fc2"))
    del q, q_scale
    if not torch.compiler.is_compiling():
        _fused_half_calls += 1
    return h3_model._mod_gate(x, gate_mlp, mlp_out, segments)


def _fused_linear(q, q_scale, weight):
    return _omni_int8.int8_linear_prequantized(q, q_scale, weight[0], weight[1], None, 2)


def _bf16_attention(h3_model, attn, qkv, rope_freqs, q_weight, k_weight, transformer_options):
    """Attention.forward after qkv_proj: rms_rope in place on qkv, then the selected attention."""
    s = qkv.shape[0]
    q, k, v = qkv.split(attn.heads * attn.head_dim, dim=-1)
    q = q.view(1, s, attn.heads, attn.head_dim)
    k = k.view(1, s, attn.heads, attn.head_dim)
    v = v.view(s, attn.heads, attn.head_dim)
    h3_model.comfy.quant_ops.ck.rms_rope_split_half_(
        q, k, rope_freqs, q_weight, k_weight, epsilon=attn.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
    container = h3_model.AttentionTensorContainer
    return h3_model.optimized_attention(
        container(q[0].transpose(0, 1).unsqueeze(0)),
        container(k[0].transpose(0, 1).unsqueeze(0)),
        container(v.transpose(0, 1).unsqueeze(0)),
        attn.heads, mask=None, skip_reshape=True, transformer_options=transformer_options,
    ).squeeze(0)


def _int8_attention(attn, qkv, rope_freqs, q_weight, k_weight, transformer_options):
    """INT8 QK^T attention with rms_rope fused into the Q/K quantization; None keeps the bf16 route."""
    global _int8_attn_calls
    block = transformer_options.get("block_index")
    if (
        _int8_attn_ops is None
        or torch.compiler.is_compiling()
        or block is None
        or block in _int8_attn_bf16_blocks
        or rope_freqs.dtype != torch.bfloat16
        or rope_freqs.shape[-3] != 48
    ):
        return None
    s = qkv.shape[0]
    q, k, v = qkv.split(attn.heads * attn.head_dim, dim=-1)
    q8, k8, q_scale, k_scale = _int8_attn_ops.rms_rope_quant(
        q.view(1, s, attn.heads, attn.head_dim), k.view(1, s, attn.heads, attn.head_dim),
        rope_freqs, q_weight, k_weight, attn.q_norm.eps)
    v = v.view(1, s, attn.heads, attn.head_dim).transpose(1, 2)
    _int8_attn_calls += 1
    return _int8_attn_ops.fmha(q8, k8, v, q_scale, k_scale).transpose(1, 2).reshape(s, -1)


def _check_int8_attention(h3_model, attn, qkv, rope_freqs, q_weight, k_weight, transformer_options, out):
    """Log the INT8 attention error against the bf16 route on a copy of qkv."""
    global _int8_attn_check_left
    _int8_attn_check_left -= 1
    ref = _bf16_attention(h3_model, attn, qkv.clone(), rope_freqs, q_weight, k_weight, transformer_options).float()
    err = ((out.float() - ref).norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(
        out.float().view(-1, attn.head_dim), ref.view(-1, attn.head_dim), dim=-1).min().item()
    log.info("[OmniXPU] norm: H3 int8 attention check block=%d rel_err=%.4f min_cos=%.4f",
             transformer_options["block_index"], err, cos)


def _run_fused_block(comfy_ops, h3_model, block, x, modulation, segments, rope_freqs, transformer_options):
    """Run DiTBlock.forward with norm/ConvRot/quant fused ahead of each INT8 projection.

    Returns None, before the residual stream is touched, when the block does not
    match the sidecar contract.
    """
    global _fused_block_calls
    staged, staged_norms = _stage_offloaded_weights(block, x, h3_model, rope_freqs) if _FUSED_STAGE else ([], {})
    try:
        reason, weights = _fused_block_reason(block, x, h3_model, rope_freqs)
    finally:
        # 模块状态只在检查期间被替换；回落时原路径看到的仍是原来的 CPU 权重
        _restore_staged_weights(staged)
    del staged
    if reason:
        if not torch.compiler.is_compiling():
            _fused_block_fallbacks[reason] = _fused_block_fallbacks.get(reason, 0) + 1
        return None

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
    starts, stops, rows = (list(column) for column in zip(*segments))
    attn = block.attn
    comfy_ops.run_every_op()
    q, q_scale = _fused_ops.norm_rot_quant(
        x, staged_norms.get("norm1", block.norm1.weight), scale_msa.to(x.dtype), shift_msa.to(x.dtype),
        starts, stops, rows, 1e-6 if block.norm1.eps is None else block.norm1.eps, None, None)
    qkv = _fused_linear(q, q_scale, weights.pop("qkv_proj"))
    del q, q_scale

    s = x.shape[0]
    q_weight = h3_model.comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
    k_weight = h3_model.comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
    out = _int8_attention(attn, qkv, rope_freqs, q_weight, k_weight, transformer_options)
    if out is None:
        out = _bf16_attention(h3_model, attn, qkv, rope_freqs, q_weight, k_weight, transformer_options)
    elif _int8_attn_check_left > 0:
        _check_int8_attention(h3_model, attn, qkv, rope_freqs, q_weight, k_weight, transformer_options, out)
    del qkv
    q, q_scale = _fused_ops.rot_quant(out.reshape(s, -1))
    del out
    attn_out = _fused_linear(q, q_scale, weights.pop("out_proj"))

    # x += gate_msa * attn_out in place (as _mod_gate), fused with norm2.
    comfy_ops.run_every_op()
    q, q_scale = _fused_ops.norm_rot_quant(
        x, staged_norms.get("norm2", block.norm2.weight), scale_mlp.to(x.dtype), shift_mlp.to(x.dtype),
        starts, stops, rows, 1e-6 if block.norm2.eps is None else block.norm2.eps,
        attn_out, gate_msa.to(x.dtype))
    del attn_out
    hidden = _fused_linear(q, q_scale, weights.pop("fc1"))
    q, q_scale = _fused_ops.silu_mul_rot_quant(hidden)
    del hidden
    mlp_out = _fused_linear(q, q_scale, weights.pop("fc2"))
    del q, q_scale
    if not torch.compiler.is_compiling():
        _fused_block_calls += 1
    return h3_model._mod_gate(x, gate_mlp, mlp_out, segments)


def apply():
    global _omni_norm

    try:
        from omni_xpu_kernel import norm as candidate
    except ImportError:
        return False, "omni_xpu_kernel segmented RMS modulation is unavailable"
    capability = getattr(
        candidate, "supports_rms_norm_segmented_modulation", None
    )
    if not callable(capability) or not capability():
        return False, "native segmented RMS modulation is unavailable"
    policy_supported = getattr(
        candidate, "rms_norm_segmented_modulation_supported", None
    )
    if not callable(policy_supported):
        return False, "native segmented RMS modulation policy is unavailable"
    if not callable(getattr(candidate, "rms_norm_segmented_modulation", None)):
        return False, "native segmented RMS modulation operation is unavailable"

    try:
        import comfy.model_management
        import comfy.ops as comfy_ops
        import comfy.ldm.minimax.model as h3_model
    except ImportError:
        return False, "MiniMax H3 ComfyUI model is unavailable"
    if not all(callable(getattr(comfy_ops, name, None)) for name in (
        "CastBiasWeightContext", "run_every_op",
    )):
        return False, "ComfyUI cast/offload helpers are unavailable"
    if not all(callable(getattr(h3_model, name, None)) for name in (
        "_mod_scale_shift", "_mod_gate",
    )):
        return False, "MiniMax H3 modulation helpers are unavailable"

    target = getattr(h3_model, "DiTBlock", None)
    if target is None or not callable(getattr(target, "forward", None)):
        return False, "MiniMax H3 DiTBlock.forward is unavailable"
    original = target.forward
    if hasattr(original, _PATCH_MARKER):
        return True, "already patched"
    reason = _forward_api_reason(original)
    if reason:
        return False, reason

    _omni_norm = candidate
    fused_reason = _load_fused_sidecar()
    if fused_reason:
        log.info("[OmniXPU] norm: H3 fused block route off (%s)", fused_reason)
    else:
        log.info("[OmniXPU] norm: H3 fused block route enabled")
        int8_reason = _load_int8_attention_sidecar()
        if int8_reason:
            log.info("[OmniXPU] norm: H3 int8 attention off (%s)", int8_reason)
        else:
            log.info("[OmniXPU] norm: H3 int8 attention enabled")
    if _FUSED_DIAG:
        log.warning("[OmniXPU] norm: H3 fused block diagnostics enabled (OMNIXPU_H3_FUSED_DIAG=1)")
    if _fused_ops is not None and (_FUSED_HALF or _FUSED_HALF_FORCE):
        log.warning("[OmniXPU] norm: H3 fused half block %s",
                    "forced (OMNIXPU_H3_FUSED_HALF_FORCE=1)" if _FUSED_HALF_FORCE else "enabled for replaced attention")
    if _FUSED_STAGE:
        log.warning("[OmniXPU] norm: H3 fused block stages offloaded INT8 weights (OMNIXPU_H3_FUSED_STAGE=1)")
    accepts_attention = "attention" in inspect.signature(original).parameters

    @wraps(original)
    def patched(
        self,
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options={},
        attention=None,
    ):
        attention_kwargs = {"attention": attention} if accepts_attention else {}
        if attention is not None and not accepts_attention:
            raise TypeError("this MiniMax H3 forward API has no attention override")
        if not policy_supported(x):
            return original(
                self,
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
                **attention_kwargs,
            )

        reason = _block_input_reason(self, x, mod_segments)
        if reason or comfy.model_management.in_training:
            _record_fallback(reason or "training")
            return original(
                self,
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
                **attention_kwargs,
            )

        modulation = self.adaln_proj(t_emb)
        reason = _modulation_reason(modulation, x, mod_segments)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation

        if not reason and _fused_ops is not None:
            global _fused_diag_block_index
            if _FUSED_DIAG:
                # 整块和半融合共用一次 block 下标 / 每次 forward 的汇总
                _fused_diag_block_index = transformer_options.get("block_index")
                if _fused_diag_block_index == 0:
                    _fused_diag_flush_forward()
            if attention is None and not _FUSED_HALF_FORCE:
                output = _run_fused_block(
                    comfy_ops,
                    h3_model,
                    self,
                    x,
                    modulation,
                    mod_segments,
                    rope_freqs,
                    transformer_options,
                )
                if output is not None:
                    return output
            if _FUSED_HALF or _FUSED_HALF_FORCE:
                output = _run_fused_half_block(
                    comfy_ops,
                    h3_model,
                    self,
                    x,
                    modulation,
                    mod_segments,
                    rope_freqs,
                    transformer_options,
                    attention,
                )
                if output is not None:
                    return output

        h, first_fused = _rms_modulation(
            comfy_ops,
            h3_model._mod_scale_shift,
            self.norm1,
            x,
            scale_msa,
            shift_msa,
            mod_segments,
            reason,
        )

        x = h3_model._mod_gate(
            x,
            gate_msa,
            (self.attn if attention is None else attention)(
                h,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            ),
            mod_segments,
        )
        h, second_fused = _rms_modulation(
            comfy_ops,
            h3_model._mod_scale_shift,
            self.norm2,
            x,
            scale_mlp,
            shift_mlp,
            mod_segments,
            reason,
        )
        output = h3_model._mod_gate(
            x,
            gate_mlp,
            self.mlp(h),
            mod_segments,
        )

        fused_calls = first_fused + second_fused
        if fused_calls:
            log_debug_event(
                "kernel",
                "h3_rms_norm_segmented_modulation",
                {"input": x, "output": output},
                details={
                    "backend": "bmg_sycl",
                    "segments": len(mod_segments),
                    "fused_calls": fused_calls,
                },
            )
        return output

    patched = trace_patch(
        "norm.H3DiTBlock.rms_modulation",
        ("self", "x", "t_emb", "mod_segments", "rope_freqs"),
        stage="dispatch",
        verbose_only=True,
    )(patched)
    setattr(patched, _PATCH_MARKER, original)
    target.forward = patched
    log.info("[OmniXPU] norm: patched MiniMax H3 segmented RMS modulation")
    return True, ""


__all__ = ["apply", "get_stats"]
