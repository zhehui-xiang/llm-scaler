// MiniMax H3 attention 的 INT8 QKᵀ 路径，注册为 torch.ops.h3i8attn.{rms_rope_quant, fmha}。
//
// rms_rope_quant(q, k, freqs, q_weight, k_weight, eps) -> (q8, k8, q_scale, k_scale)
//   q, k     bf16 [1, L, H, 128] 视图（H3 里是 qkv buffer 的视图，rms_rope 之前，不会被修改）
//   freqs    bf16 [1, L, 1, 48, 2, 2]；q_weight / k_weight 为 q_norm / k_norm 的 bf16 [128] 权重
//   q8, k8   int8 [1, H, L, 128]，BLHD 存储
//   q_scale  fp32 [H, L]                逐 token
//   k_scale  fp32 [H, ceil(L/kKvTile)]  每个 KV tile 一个
// fmha(q8, k8, v, q_scale, k_scale) -> o
//   v        bf16 [1, H, L, 128] 视图；o 为 BLHD-backed 的 [1, H, L, 128]（同 cute sdp_bhld_d128）
//
// FMHA 配置在 B70 上扫过 40 个：Q256 / KV32 / V32 / MmaK64 / 16 subgroup / 2 stage。
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/device_kernel.h"
#include "cute/util/compat.hpp"

#include "int8_qk_fmha.hpp"

using namespace cute;

namespace {

constexpr int kHeadDim = 128;
constexpr int kQTile = 256, kKvTile = 32, kVTile = 32, kMmaK = 64, kSgQ = 16, kStages = 2;

// ---- rms_rope + Q/K 量化：一个 work-group 处理一个 (2 个 head, KV tile) -------------------------------
// 数学同 omni_xpu_kernel kitchen_rms_rope_sycl.cpp 的 launch_minimax_h3_rms_rope（逐 head RMSNorm × weight
// 落 bf16，前 96 维 split-half rope 再落 bf16），只是不把 bf16 Q/K 写回 qkv buffer，直接量化成 int8。
constexpr int kSg = 16;                   // sub-group 大小
constexpr int kVec = kHeadDim / kSg;      // 每 lane 8 个元素
constexpr int kSgPerWg = 16;
constexpr int kRowsPerSg = kKvTile / kSgPerWg;
constexpr int kRotPairs = 48;             // rope 只转前 96 维：第 c 维与第 c+48 维成对
constexpr int kRopeLanes = kRotPairs / 4; // 12 个 lane 各持有 4 对
constexpr int kHeadsPerWg = 2;            // 同一 token 的 freqs 读一次给 2 个 head 用

using bf16x8 = sycl::vec<uint16_t, kVec>;  // 每 lane 一次 16 字节读
using i8x8 = sycl::vec<int8_t, kVec>;

inline float bf2f(uint16_t b) { return sycl::bit_cast<float>(static_cast<uint32_t>(b) << 16); }
inline float round_bf16(float f) { return float(sycl::ext::oneapi::bfloat16(f)); }

// lane l < 12 持有第 [4l, 4l+4) 维和第 [48+4l, 48+4l+4) 维（e<4 与 e+4 是 rope 的一对，不用跨 lane），
// lane 12..15 持有第 [96+8(l-12), +8) 维（不转）。y 是落过 bf16 的 float
inline void rms_rope_row(sycl::sub_group sg, bool rope, const float (&x)[kVec], const float (&w)[kVec],
                         const float (&f)[16], float eps, float (&y)[kVec]) {
  float ss = 0.f;
#pragma unroll
  for (int e = 0; e < kVec; ++e) ss += x[e] * x[e];
  const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / kHeadDim + eps);
  float n[kVec];
#pragma unroll
  for (int e = 0; e < kVec; ++e) n[e] = round_bf16(x[e] * inv * w[e]);
  if (rope) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      y[p] = round_bf16(f[p * 4] * n[p] + f[p * 4 + 1] * n[p + 4]);
      y[p + 4] = round_bf16(f[p * 4 + 2] * n[p] + f[p * 4 + 3] * n[p + 4]);
    }
  } else {
#pragma unroll
    for (int e = 0; e < kVec; ++e) y[e] = n[e];
  }
}

using bf16x4 = sycl::vec<uint16_t, 4>;
using i8x4 = sycl::vec<int8_t, 4>;

// 按上面的映射读一行 8 个元素 / 写 8 个 int8
inline void load_row(const uint16_t* row, int lane, float (&x)[kVec]) {
  if (lane < kRopeLanes) {
    const bf16x4 a = *reinterpret_cast<const bf16x4*>(row + 4 * lane);
    const bf16x4 b = *reinterpret_cast<const bf16x4*>(row + 48 + 4 * lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) { x[e] = bf2f(a[e]); x[e + 4] = bf2f(b[e]); }
  } else {
    const bf16x8 a = *reinterpret_cast<const bf16x8*>(row + 96 + 8 * (lane - kRopeLanes));
#pragma unroll
    for (int e = 0; e < kVec; ++e) x[e] = bf2f(a[e]);
  }
}

inline void store_row(int8_t* row, int lane, const float (&y)[kVec], float inv) {
  if (lane < kRopeLanes) {
    i8x4 a, b;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      a[e] = int8_t(sycl::clamp(sycl::rint(y[e] * inv), -127.f, 127.f));
      b[e] = int8_t(sycl::clamp(sycl::rint(y[e + 4] * inv), -127.f, 127.f));
    }
    *reinterpret_cast<i8x4*>(row + 4 * lane) = a;
    *reinterpret_cast<i8x4*>(row + 48 + 4 * lane) = b;
  } else {
    i8x8 a;
#pragma unroll
    for (int e = 0; e < kVec; ++e) a[e] = int8_t(sycl::clamp(sycl::rint(y[e] * inv), -127.f, 127.f));
    *reinterpret_cast<i8x8*>(row + 96 + 8 * (lane - kRopeLanes)) = a;
  }
}

inline void load_weight(const uint16_t* w, int lane, float (&x)[kVec]) {
#pragma unroll
  for (int e = 0; e < kVec; ++e)
    x[e] = bf2f(w[lane < kRopeLanes ? (e < 4 ? 4 * lane + e : 48 + 4 * lane + e - 4) : 96 + 8 * (lane - kRopeLanes) + e]);
}

class RmsRopeQuantKernel {};

void rms_rope_quant_launch(const uint16_t* q, const uint16_t* k, int64_t q_seq, int64_t q_head, int64_t k_seq,
                           int64_t k_head, const uint16_t* freqs, const uint16_t* q_weight, const uint16_t* k_weight,
                           float eps, int8_t* q8, int8_t* k8, float* q_scale, float* k_scale, int H, int L) {
  constexpr int G = kHeadsPerWg;
  const int blocks = (L + kKvTile - 1) / kKvTile;
  const int head_groups = H / G;
  const int64_t row8 = int64_t(H) * kHeadDim;
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  queue.parallel_for<RmsRopeQuantKernel>(
      sycl::nd_range<1>(int64_t(head_groups) * blocks * kSgPerWg * kSg, kSgPerWg * kSg),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
        // 相邻 work-group 是同一批 token 的不同 head，读写的 qkv / q8 / k8 行和 freqs 在内存里连续
        const int hg = it.get_group_linear_id() % head_groups, blk = it.get_group_linear_id() / head_groups;
        auto sg = it.get_sub_group();
        const int lane = sg.get_local_linear_id();
        const bool rope = lane < kRopeLanes;
        float wq[kVec], wk[kVec];
        load_weight(q_weight, lane, wq);
        load_weight(k_weight, lane, wk);
        float ky[kRowsPerSg][G][kVec];
        float kmax[G];
#pragma unroll
        for (int g = 0; g < G; ++g) kmax[g] = 0.f;
#pragma unroll
        for (int r = 0; r < kRowsPerSg; ++r) {
          const int t = blk * kKvTile + sg.get_group_linear_id() * kRowsPerSg + r;
          const int tc = t < L ? t : L - 1;  // 越界行照常算（sub-group 内归约要全员参与），不写出
          float f[16] = {};
          if (rope) {  // 第 4l..4l+3 对的 2x2 矩阵，32 字节
            const auto* src = reinterpret_cast<const bf16x8*>(freqs + (int64_t(tc) * kRotPairs + 4 * lane) * 4);
            const bf16x8 f0 = src[0], f1 = src[1];
#pragma unroll
            for (int e = 0; e < kVec; ++e) { f[e] = bf2f(f0[e]); f[e + 8] = bf2f(f1[e]); }
          }
#pragma unroll
          for (int g = 0; g < G; ++g) {
            const int h = hg * G + g;
            float x[kVec], qy[kVec];
            load_row(q + tc * q_seq + h * q_head, lane, x);
            rms_rope_row(sg, rope, x, wq, f, eps, qy);
            load_row(k + tc * k_seq + h * k_head, lane, x);
            rms_rope_row(sg, rope, x, wk, f, eps, ky[r][g]);
            float qmax = 0.f;
#pragma unroll
            for (int e = 0; e < kVec; ++e) {
              qmax = sycl::fmax(qmax, sycl::fabs(qy[e]));
              if (t < L) kmax[g] = sycl::fmax(kmax[g], sycl::fabs(ky[r][g][e]));
            }
            const float qs = sycl::fmax(sycl::reduce_over_group(sg, qmax, sycl::maximum<float>()), 1e-8f) / 127.f;
            if (t < L) {
              store_row(q8 + t * row8 + h * kHeadDim, lane, qy, 1.f / qs);
              if (lane == 0) q_scale[int64_t(h) * L + t] = qs;
            }
          }
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
          const int h = hg * G + g;
          const float ks = sycl::fmax(sycl::reduce_over_group(it.get_group(), kmax[g], sycl::maximum<float>()), 1e-8f) / 127.f;
#pragma unroll
          for (int r = 0; r < kRowsPerSg; ++r) {
            const int t = blk * kKvTile + sg.get_group_linear_id() * kRowsPerSg + r;
            if (t < L) store_row(k8 + t * row8 + h * kHeadDim, lane, ky[r][g], 1.f / ks);
          }
          if (it.get_local_linear_id() == 0) k_scale[int64_t(h) * blocks + blk] = ks;
        }
      });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> rms_rope_quant(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& freqs, const at::Tensor& q_weight,
    const at::Tensor& k_weight, double eps) {
  TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(3) == kHeadDim && k.sizes() == q.sizes(),
              "rms_rope_quant expects matching [1, L, H, 128] q/k");
  const int L = q.size(1), H = q.size(2);
  for (const at::Tensor* t : {&q, &k, &freqs, &q_weight, &k_weight})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16, "rms_rope_quant expects bf16 inputs");
  for (const at::Tensor* t : {&q, &k})  // 16 字节向量读
    TORCH_CHECK(t->stride(3) == 1 && reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0 &&
                    t->stride(1) % 8 == 0 && t->stride(2) % 8 == 0,
                "rms_rope_quant expects 16-byte aligned q/k rows");
  TORCH_CHECK(freqs.is_contiguous() && freqs.sizes() == at::IntArrayRef({1, L, 1, kRotPairs, 2, 2}),
              "rms_rope_quant expects contiguous [1, L, 1, 48, 2, 2] freqs (rot_dim 96)");
  TORCH_CHECK(q_weight.is_contiguous() && k_weight.is_contiguous() && q_weight.numel() == kHeadDim &&
                  k_weight.numel() == kHeadDim, "rms_rope_quant expects [128] norm weights");
  TORCH_CHECK(H % kHeadsPerWg == 0, "rms_rope_quant expects an even head count");
  const int blocks = (L + kKvTile - 1) / kKvTile;
  auto i8 = q.options().dtype(at::kChar), f32 = q.options().dtype(at::kFloat);
  at::Tensor q8 = at::empty({1, L, H, kHeadDim}, i8), k8 = at::empty({1, L, H, kHeadDim}, i8);
  at::Tensor q_scale = at::empty({H, L}, f32), k_scale = at::empty({H, blocks}, f32);
  auto u16 = [](const at::Tensor& t) { return reinterpret_cast<const uint16_t*>(t.data_ptr()); };
  rms_rope_quant_launch(u16(q), u16(k), q.stride(1), q.stride(2), k.stride(1), k.stride(2), u16(freqs),
                        u16(q_weight), u16(k_weight), float(eps), q8.data_ptr<int8_t>(), k8.data_ptr<int8_t>(),
                        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(), H, L);
  return {q8.transpose(1, 2), k8.transpose(1, 2), q_scale, k_scale};
}

// ---- INT8 QKᵀ FMHA ------------------------------------------------------------------------------
template <typename Kernel>
class Int8QKKernelTag {};

template <typename Kernel>
void launch_on_torch_queue(typename Kernel::Params params) {
  compat::dim3 const block = Kernel::get_block_shape();
  compat::dim3 const grid = Kernel::get_grid_shape(params);
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  compat::experimental::launch_properties launch_props{syclex::work_group_scratch_size(Kernel::SharedStorageSize)};
  compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
  compat::experimental::launch_policy policy{grid, block, launch_props, kernel_props};
  syclex::launch_config config(policy.get_range(), policy.get_launch_properties());
  auto cgf = [&](::sycl::handler& cgh) {
    auto functor =
        compat::experimental::detail::build_kernel_functor<cutlass::device_kernel<Kernel>>(cgh, policy, params);
    syclex::detail::LaunchConfigAccess<sycl::nd_range<3>, decltype(policy.get_launch_properties())> access(config);
    cgh.parallel_for<Int8QKKernelTag<Kernel>>(access.getRange(), access.getProperties(), functor);
  };
  at::xpu::getCurrentXPUStream().queue().submit(cgf);
}

struct Int8QKTile {
  using ShapeQK = Shape<Int<kQTile>, Int<kKvTile>, Int<kMmaK>>;
  using ShapePV = Shape<Int<kQTile>, Int<kVTile>, Int<kKvTile>>;
  using ShapeOutput = Shape<Int<kQTile>, Int<kHeadDim>>;
  using SubgroupLayoutQK = Layout<Shape<Int<kSgQ>, _1, _1>>;
  static constexpr int AtomM = cute::gcd(kQTile / kSgQ, 8);

  // Batch stride is int64: V is a view into the packed qkv buffer, so it spans
  // L * 3 * H * 128 elements and passes 2^31 above ~99.8k tokens (H3 15 s video).
  using StrideQ = Stride<int, _1, int, int64_t>;
  using StrideK = Stride<int, _1, int, int64_t>;
  using StrideV = Stride<_1, int, int, int64_t>;
  using StrideO = Stride<int, _1, int, int64_t>;

  using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<AtomM, int32_t, int8_t, int8_t>>,
                                             Layout<ShapeQK>, SubgroupLayoutQK>::TiledMMA;
  using TiledMMAQKf = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<AtomM, float, cutlass::bfloat16_t>>,
                                              Layout<ShapeQK>, SubgroupLayoutQK>::TiledMMA;
  using SubgroupLayoutPV = decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));
  using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<AtomM, float, cutlass::bfloat16_t>>,
                                             Layout<ShapePV>, SubgroupLayoutPV>::TiledMMA;

  template <class T, class S>
  using GTensor = decltype(make_tensor(make_gmem_ptr((T*)nullptr), make_layout(repeat<rank_v<S>>(1), S{})));
  using TensorQ = GTensor<int8_t, StrideQ>;
  using TensorK = GTensor<int8_t, StrideK>;
  using TensorV = GTensor<cutlass::bfloat16_t, StrideV>;
  using TensorO = GTensor<cutlass::bfloat16_t, StrideO>;

  using Mainloop = cutlass::fmha::collective::Int8QKMainloop<
      kStages, TiledMMAQK, TiledMMAQKf, TiledMMAPV, kHeadDim / kVTile, TensorQ, TensorK, TensorV>;
  using Epilogue = cutlass::fmha::collective::FMHAFwdEpilogue<Mainloop, ShapeOutput, TensorO, void>;
  using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<false>;
  using Kernel = cutlass::fmha::kernel::XeFMHAFwdInt8QKKernel<
      ProblemShape, Mainloop, Epilogue, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>;
};

int checked_int(int64_t value, const char* label) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), label, " exceeds int32: ", value);
  return static_cast<int>(value);
}

template <typename S>
S seq_major_stride(const at::Tensor& t) {
  return cute::make_stride(checked_int(t.stride(2), "seq stride"), _1{},
                           checked_int(t.stride(1), "head stride"), static_cast<int64_t>(t.stride(0)));
}

at::Tensor fmha(const at::Tensor& q8, const at::Tensor& k8, const at::Tensor& v,
                const at::Tensor& q_scale, const at::Tensor& k_scale) {
  using KT = Int8QKTile;
  using K = KT::Kernel;
  TORCH_CHECK(q8.dim() == 4 && q8.size(0) == 1 && q8.size(3) == kHeadDim, "fmha expects [1, H, L, 128]");
  TORCH_CHECK(q8.scalar_type() == at::kChar && k8.scalar_type() == at::kChar && v.scalar_type() == at::kBFloat16,
              "fmha expects int8 q/k and bf16 v");
  TORCH_CHECK(k8.sizes() == q8.sizes() && v.sizes() == q8.sizes(), "fmha expects self-attention shapes");
  TORCH_CHECK(q8.stride(3) == 1 && k8.stride(3) == 1 && v.stride(3) == 1, "fmha expects unit head_dim stride");
  const int H = q8.size(1), L = q8.size(2);
  TORCH_CHECK(q_scale.is_contiguous() && q_scale.sizes() == at::IntArrayRef({H, L}) &&
                  k_scale.is_contiguous() && k_scale.size(0) == H && k_scale.size(1) == (L + kKvTile - 1) / kKvTile,
              "fmha scales must come from rms_rope_quant");

  at::Tensor o = at::empty({1, L, H, kHeadDim}, v.options()).permute({0, 2, 1, 3});

  typename KT::ProblemShape shape;
  shape.batch = 1;
  shape.num_heads_q = shape.num_heads_kv = H;
  shape.seq_len_qo = shape.seq_len_kv = L;
  shape.seq_len_kv_cache = 0;
  shape.head_size_qk = shape.head_size_vo = kHeadDim;

  typename KT::StrideV stride_v = cute::make_stride(
      _1{}, checked_int(v.stride(2), "V seq stride"), checked_int(v.stride(1), "V head stride"),
      static_cast<int64_t>(v.stride(0)));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename K::Arguments arguments{
      {shape,
       static_cast<const int8_t*>(q8.data_ptr()), seq_major_stride<KT::StrideQ>(q8),
       static_cast<const int8_t*>(k8.data_ptr()), seq_major_stride<KT::StrideK>(k8),
       static_cast<const cutlass::bfloat16_t*>(v.data_ptr()), stride_v,
       static_cast<cutlass::bfloat16_t*>(o.data_ptr()), seq_major_stride<KT::StrideO>(o)},
      {1.0f / std::sqrt(float(kHeadDim)), q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
       L, checked_int(k_scale.size(1), "k_scale blocks")},
      {},
      hw_info};
  TORCH_CHECK(K::can_implement(arguments), "h3i8attn: can_implement failed");
  launch_on_torch_queue<K>(K::to_underlying_arguments(arguments, nullptr));
  return o;
}

}  // namespace

TORCH_LIBRARY(h3i8attn, m) {
  m.def("rms_rope_quant(Tensor q, Tensor k, Tensor freqs, Tensor q_weight, Tensor k_weight, float eps) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("fmha(Tensor q8, Tensor k8, Tensor v, Tensor q_scale, Tensor k_scale) -> Tensor");
  m.impl("rms_rope_quant", c10::DispatchKey::XPU, &rms_rope_quant);
  m.impl("fmha", c10::DispatchKey::XPU, &fmha);
}
