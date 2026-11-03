# MiniMax H3 fused block and INT8 attention (preview)

This branch adds two optional fast paths for the MiniMax H3 DiT in
ComfyUI-OmniXPU. Both are enabled by default when their kernels are present
and fall back to the regular path block by block when a precondition is not
met.

| Path | Kernel (built into `omni_xpu_kernel`) | What changes |
|---|---|---|
| Fused block | `omni_xpu_kernel.h3.h3_fused` (`torch.ops.h3fuse`) | RMSNorm/modulation, ConvRot rotation and INT8 quantization in front of each INT8 GEMM run as one kernel per GEMM; the attention residual `addcmul` is fused too. The INT8 GEMM itself is still `int8_linear_prequantized`. |
| INT8 attention | `omni_xpu_kernel.h3.h3_int8_attn` (`torch.ops.h3i8attn`) | Inside the fused block, Q/K `rms_rope` and INT8 quantization run as one kernel, and attention computes QK^T with INT8 DPAS (P·V stays bf16). |

The kernels are built only for BMG on Linux (AOT `bmg-g21,bmg-g31`). They
have been tuned and validated on Arc Pro B70 only; the `bmg-g21` binary
(Arc Pro B60) is built but untested.

## When the paths are used

The fused block is used for a DiT block only if all four projections are
resident TensorWise INT8 weights with G256 ConvRot, there is no LoRA
`weight_function`, the block's Attention/MLP `forward` is not replaced, and
rope is present. Otherwise the whole block takes the regular path and the
residual stream is untouched.

INT8 attention additionally requires the fused block, bf16 rope frequencies
with `rot_dim` 96, and a block index outside `OMNIXPU_H3_INT8_ATTN_BF16_BLOCKS`.

The ComfyUI log shows which paths are active:

```
[OmniXPU] norm: H3 fused block route enabled
[OmniXPU] norm: H3 int8 attention enabled
```

`get_stats()` of the H3 adapter reports `fused_block` and `int8_attention`
block counts.

## Switches

| Variable | Default | Effect |
|---|---|---|
| `OMNIXPU_H3_FUSED` | `1` | `0` disables the fused block (and therefore INT8 attention). |
| `OMNIXPU_H3_INT8_ATTN` | `1` | `0` keeps the fused block but uses bf16 attention. |
| `OMNIXPU_H3_INT8_ATTN_BF16_BLOCKS` | `45,48,49` | Blocks kept in bf16 attention. INT8 QK^T flips the direction of some tokens in these blocks. |
| `OMNIXPU_H3_INT8_ATTN_CHECK` | `0` | For the first N INT8 attention calls, also run the bf16 path and log `rel_err` / `min_cos`. |
| `OMNIXPU_H3_FUSED_SO`, `OMNIXPU_H3_INT8_ATTN_SO` | unset | Load the kernel from this path instead of `omni_xpu_kernel.h3`. |

Building `omni_xpu_kernel` with `OMNI_XPU_BUILD_H3=0` leaves both kernels out;
the adapter then logs the paths as off and uses the regular path.

## Accuracy

The output is not bit-identical to the regular path, but differences are
within the run-to-run noise of the pipeline (two regular runs with the same
seed differ by 22–24 dB PSNR).

- Fused block: quantization scales match exactly, 99.999% of INT8 values
  match (the rest differ by 1 from fp32 Hadamard accumulation order), and the
  residual stream is bit-identical. Fused vs regular video: 20 dB PSNR.
- INT8 attention: over 20 steps × 47 INT8 blocks, relative error is at most
  1.65% (mean 0.83%) and the worst token cosine is 0.980. INT8 vs bf16
  attention video: 20.7 dB PSNR.

## Performance (Arc Pro B70, `video_minimax_h3_t2v2.json`, 20 steps)

| | Steady-state H3 forward |
|---|---|
| Regular | 5.640 s |
| Fused block | 5.323 s (−5.6%) |
| Fused block + INT8 attention | −6.3% further vs fused block alone, measured on one card |

Per layer (L = 15787), attention goes from 56.3 ms (bf16) to 43.0 ms, and
Q/K `rms_rope` + quantization takes 1.76 ms vs 2.66 ms for bf16 `rms_rope`
alone.
