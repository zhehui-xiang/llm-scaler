// H3 int8 GEMM 前的激活预处理融合 kernel。
//
// 生产链路里每个 int8 GEMM 前都是 3 个独立 kernel，每个都把整块激活读写一遍显存：
//   QKV / fc1 : rms_norm_segmented_modulation -> rotate_convrot (bf16 mm) -> quantize_int8_rowwise
//   out_proj  :                                  rotate_convrot            -> quantize_int8_rowwise
//   fc2       : fused_silu_mul_exact_bf16      -> rotate_convrot            -> quantize_int8_rowwise
// 这里一个 kernel 做完，只读一次输入、写一次 int8 + scale，再交给原来的 int8_linear_prequantized GEMM。
//
// 数值上逐步复刻生产链路的 bf16 落地点（norm 的 4 次 bf16、silu 的 2 次 bf16、旋转结果 bf16），
// 旋转用 fp32 快速 Hadamard 变换代替 bf16 GEMM（矩阵相同：H4⊗H4⊗H4⊗H4/16，只有累加顺序不同）。
//
// 线程映射：一个 work-group 处理一行；一个 sub-group(16 lane) 处理一个 256 列的旋转组，
// 每个 lane 持有组内连续 16 个元素（组内下标 = lane*16 + e）。
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>
#include <torch/library.h>

#include <sycl/sycl.hpp>

namespace {

constexpr int SG = 16;           // sub-group 大小
constexpr int VEC = 16;          // 每 lane 元素数
constexpr int GROUP = SG * VEC;  // ConvRot 组大小 256
constexpr int MAX_SEGMENTS = 8;

using bf16 = sycl::ext::oneapi::bfloat16;

inline float bf2f(uint16_t b) { return sycl::bit_cast<float>(static_cast<uint32_t>(b) << 16); }
inline uint16_t f2bf(float f) { return sycl::bit_cast<uint16_t>(bf16(f)); }  // RNE
inline float round_bf16(float f) { return bf2f(f2bf(f)); }

struct Segments {
  int count;
  int starts[MAX_SEGMENTS];
  int stops[MAX_SEGMENTS];
  int rows[MAX_SEGMENTS];
};

inline int segment_row(int row, const Segments& s) {
  int r = 0;
#pragma unroll
  for (int i = 0; i < MAX_SEGMENTS; ++i)
    if (i < s.count && row >= s.starts[i] && row < s.stops[i]) r = s.rows[i];
  return r;
}

// 对 4 个元素做一次 H4 (对称矩阵): y_b = S - 2*x_{3-b}
inline void h4(float& a0, float& a1, float& a2, float& a3) {
  const float s = a0 + a1 + a2 + a3;
  const float b0 = s - 2.f * a3, b1 = s - 2.f * a2, b2 = s - 2.f * a1, b3 = s - 2.f * a0;
  a0 = b0; a1 = b1; a2 = b2; a3 = b3;
}

// 跨 lane 的一位 base-4 数字：同组 4 个 lane 的下标只在该位不同（xor m, 2m, 3m）
inline float h4_lanes(sycl::sub_group sg, float x, int m) {
  float s = x + sycl::permute_group_by_xor(sg, x, m);
  s += sycl::permute_group_by_xor(sg, s, 2 * m);
  return s - 2.f * sycl::permute_group_by_xor(sg, x, 3 * m);
}

// 256 点 ConvRot 旋转，结果按生产链路落成 bf16
inline void rotate256(sycl::sub_group sg, float (&v)[VEC]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) h4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);  // e 的低位
#pragma unroll
  for (int i = 0; i < 4; ++i) h4(v[i], v[i + 4], v[i + 8], v[i + 12]);                  // e 的高位
#pragma unroll
  for (int e = 0; e < VEC; ++e) {
    v[e] = h4_lanes(sg, v[e], 1);  // lane 的低位
    v[e] = h4_lanes(sg, v[e], 4);  // lane 的高位
    v[e] = round_bf16(v[e] * 0.0625f);
  }
}

// 与 h3_silu_mul.cpp 的 silu_mul_exact_bf16 相同
inline float silu_mul_exact(float x, float up) {
  const float d = 1.0f + sycl::exp(-x);
  float silu = x / d;
  if (x > 0.0f && sycl::isfinite(x)) silu += sycl::fma(-silu, d, x) / d;
  return round_bf16(round_bf16(silu) * up);
}

enum Mode { kNorm, kNormResidual, kSilu, kPlain };

template <int Mode, int Groups>
class FusedRotQuantKernel;

struct Params {
  const uint16_t* in;      // kNorm*/kPlain: [M,K]; kSilu: [M,2K] (gate|up)
  uint16_t* resid;         // kNormResidual: x，原地更新 x += other*gate
  const uint16_t* other;   // kNormResidual: attn/mlp 输出 [M,K]
  const uint16_t* gate;    // kNormResidual: gate 表
  const uint16_t* weight;  // RMSNorm weight [K]
  const uint16_t* scale;   // modulation scale 表
  const uint16_t* shift;   // modulation shift 表
  int64_t gate_stride, scale_stride, shift_stride, in_stride;
  int8_t* q;
  float* q_scale;
  float eps;
  Segments seg;
};

template <int Mode, int Groups>
void launch(const Params p, int64_t rows, sycl::queue& queue) {
  constexpr int K = Groups * GROUP;
  constexpr int WG = Groups * SG;
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<FusedRotQuantKernel<Mode, Groups>>(
        sycl::nd_range<1>(rows * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
          auto sg = it.get_sub_group();
          auto wg = it.get_group();
          const int64_t row = it.get_group(0);
          const int lane = sg.get_local_linear_id();
          const int col = sg.get_group_linear_id() * GROUP + lane * VEC;
          using V = sycl::vec<uint16_t, VEC>;
          float v[VEC];

          if constexpr (Mode == kSilu) {
            const V g = *reinterpret_cast<const V*>(p.in + row * p.in_stride + col);
            const V u = *reinterpret_cast<const V*>(p.in + row * p.in_stride + K + col);
#pragma unroll
            for (int e = 0; e < VEC; ++e) v[e] = silu_mul_exact(bf2f(g[e]), bf2f(u[e]));
          } else {
            const V x = *reinterpret_cast<const V*>(p.in + row * p.in_stride + col);
#pragma unroll
            for (int e = 0; e < VEC; ++e) v[e] = bf2f(x[e]);
          }

          if constexpr (Mode == kNorm || Mode == kNormResidual) {
            const int mrow = segment_row(row, p.seg);
            if constexpr (Mode == kNormResidual) {
              // 与 x[a:b].addcmul_(other, gate.to(bf16)) 相同：fp32 算 x + o*g，落一次 bf16
              const V o = *reinterpret_cast<const V*>(p.other + row * K + col);
              const V g = *reinterpret_cast<const V*>(p.gate + mrow * p.gate_stride + col);
              V out;
#pragma unroll
              for (int e = 0; e < VEC; ++e) {
                out[e] = f2bf(v[e] + bf2f(o[e]) * bf2f(g[e]));
                v[e] = bf2f(out[e]);
              }
              *reinterpret_cast<V*>(p.resid + row * K + col) = out;
            }
            float ss = 0.f;
#pragma unroll
            for (int e = 0; e < VEC; ++e) ss += v[e] * v[e];
            ss = sycl::reduce_over_group(wg, ss, sycl::plus<float>());
            const float rs = sycl::rsqrt(ss / K + p.eps);
            const V w = *reinterpret_cast<const V*>(p.weight + col);
            const V sc = *reinterpret_cast<const V*>(p.scale + mrow * p.scale_stride + col);
            const V sh = *reinterpret_cast<const V*>(p.shift + mrow * p.shift_stride + col);
            // 与 rms_norm_segmented_modulation 相同的 4 个 bf16 落地点
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
              const float n = round_bf16(v[e] * rs * bf2f(w[e]));
              const float m = round_bf16(n * round_bf16(bf2f(sc[e]) + 1.0f));
              v[e] = round_bf16(m + bf2f(sh[e]));
            }
          }

          rotate256(sg, v);

          // 与 quantize_int8_rowwise 相同：scale = max(absmax/127, 1e-30)，rint(x * (1/scale))
          float amax = 0.f;
#pragma unroll
          for (int e = 0; e < VEC; ++e) amax = sycl::fmax(amax, sycl::fabs(v[e]));
          amax = sycl::reduce_over_group(wg, amax, sycl::maximum<float>());
          float s = amax / 127.0f;
          if (s < 1e-30f) s = 1e-30f;
          const float inv = 1.0f / s;
          sycl::vec<int8_t, VEC> q;
#pragma unroll
          for (int e = 0; e < VEC; ++e)
            q[e] = static_cast<int8_t>(sycl::fmax(-128.f, sycl::fmin(127.f, sycl::rint(v[e] * inv))));
          *reinterpret_cast<sycl::vec<int8_t, VEC>*>(p.q + row * K + col) = q;
          if (it.get_local_linear_id() == 0) p.q_scale[row] = s;
        });
  });
}

template <int Mode>
void dispatch(const Params& p, int64_t rows, int64_t k, sycl::queue& queue) {
  switch (k) {  // H3 的三种宽度：hidden 5376 / attn 7168 / ffn 14336
    case 5376: return launch<Mode, 21>(p, rows, queue);
    case 7168: return launch<Mode, 28>(p, rows, queue);
    case 14336: return launch<Mode, 56>(p, rows, queue);
    default: TORCH_CHECK(false, "unsupported width ", k);
  }
}

const uint16_t* bits(const at::Tensor& t) { return reinterpret_cast<const uint16_t*>(t.data_ptr()); }

void check_bf16_rows(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_xpu() && t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.stride(1) == 1,
              name, " must be a row-major bf16 XPU matrix");
}

std::tuple<at::Tensor, at::Tensor> alloc_out(const at::Tensor& like, int64_t rows, int64_t k) {
  // scale 形状与生产的 quantize_int8_rowwise_fused 一致: [M, 1]
  return {at::empty({rows, k}, like.options().dtype(at::kChar)), at::empty({rows, 1}, like.options().dtype(at::kFloat))};
}

Segments make_segments(at::IntArrayRef starts, at::IntArrayRef stops, at::IntArrayRef mrows, int64_t rows) {
  TORCH_CHECK(!starts.empty() && starts.size() <= MAX_SEGMENTS && starts.size() == stops.size() &&
              starts.size() == mrows.size(), "1..8 segments");
  Segments s{};
  s.count = starts.size();
  for (int i = 0; i < s.count; ++i) {
    s.starts[i] = starts[i];
    s.stops[i] = stops[i];
    s.rows[i] = mrows[i];
  }
  TORCH_CHECK(starts[0] == 0 && stops.back() == rows, "segments must cover all rows");
  return s;
}

// RMSNorm + segmented modulation (+ 可选 gated residual) + ConvRot + int8 rowwise quant
std::tuple<at::Tensor, at::Tensor> norm_rot_quant(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& scale, const at::Tensor& shift,
    at::IntArrayRef starts, at::IntArrayRef stops, at::IntArrayRef mod_rows, double eps,
    const std::optional<at::Tensor>& other, const std::optional<at::Tensor>& gate) {
  check_bf16_rows(x, "x");
  check_bf16_rows(scale, "scale");
  check_bf16_rows(shift, "shift");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && weight.numel() == x.size(1), "bad x/weight");
  const int64_t rows = x.size(0), k = x.size(1);
  auto [q, s] = alloc_out(x, rows, k);
  Params p{};
  p.in = bits(x);
  p.in_stride = k;
  p.weight = bits(weight);
  p.scale = bits(scale);
  p.shift = bits(shift);
  p.scale_stride = scale.stride(0);
  p.shift_stride = shift.stride(0);
  p.q = q.data_ptr<int8_t>();
  p.q_scale = s.data_ptr<float>();
  p.eps = eps;
  p.seg = make_segments(starts, stops, mod_rows, rows);
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  if (other.has_value()) {
    TORCH_CHECK(gate.has_value(), "residual mode needs gate");
    check_bf16_rows(*gate, "gate");
    TORCH_CHECK(other->is_contiguous() && other->sizes() == x.sizes(), "other must match x");
    p.resid = reinterpret_cast<uint16_t*>(x.data_ptr());
    p.other = bits(*other);
    p.gate = bits(*gate);
    p.gate_stride = gate->stride(0);
    dispatch<kNormResidual>(p, rows, k, queue);
  } else {
    dispatch<kNorm>(p, rows, k, queue);
  }
  return {q, s};
}

// fc2 输入：[M, 2N] 的 gate|up -> silu_mul_exact -> ConvRot -> int8
std::tuple<at::Tensor, at::Tensor> silu_mul_rot_quant(const at::Tensor& gu) {
  check_bf16_rows(gu, "gu");
  const int64_t rows = gu.size(0), k = gu.size(1) / 2;
  auto [q, s] = alloc_out(gu, rows, k);
  Params p{};
  p.in = bits(gu);
  p.in_stride = gu.stride(0);
  p.q = q.data_ptr<int8_t>();
  p.q_scale = s.data_ptr<float>();
  dispatch<kSilu>(p, rows, k, at::xpu::getCurrentXPUStream().queue());
  return {q, s};
}

// out_proj 输入：ConvRot -> int8
std::tuple<at::Tensor, at::Tensor> rot_quant(const at::Tensor& x) {
  check_bf16_rows(x, "x");
  const int64_t rows = x.size(0), k = x.size(1);
  auto [q, s] = alloc_out(x, rows, k);
  Params p{};
  p.in = bits(x);
  p.in_stride = x.stride(0);
  p.q = q.data_ptr<int8_t>();
  p.q_scale = s.data_ptr<float>();
  dispatch<kPlain>(p, rows, k, at::xpu::getCurrentXPUStream().queue());
  return {q, s};
}

}  // namespace

TORCH_LIBRARY(h3fuse, m) {
  m.def("norm_rot_quant(Tensor(a!) x, Tensor weight, Tensor scale, Tensor shift, int[] starts, int[] stops, "
        "int[] mod_rows, float eps, Tensor? other=None, Tensor? gate=None) -> (Tensor, Tensor)");
  m.def("silu_mul_rot_quant(Tensor gu) -> (Tensor, Tensor)");
  m.def("rot_quant(Tensor x) -> (Tensor, Tensor)");
  m.impl("norm_rot_quant", c10::DispatchKey::XPU, &norm_rot_quant);
  m.impl("silu_mul_rot_quant", c10::DispatchKey::XPU, &silu_mul_rot_quant);
  m.impl("rot_quant", c10::DispatchKey::XPU, &rot_quant);
}
