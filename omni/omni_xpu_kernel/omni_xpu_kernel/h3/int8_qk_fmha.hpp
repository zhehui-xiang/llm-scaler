// INT8 QKᵀ 的 FMHA：Q/K 是预先量化好的 int8，S = Q·Kᵀ 走 int8 DPAS（int32 累加），
// 反量化 S = s32 · q_scale[行] · k_scale[KV tile] 后做 online softmax，P·V 仍是 bf16 DPAS。
// 反量化并进 softmax 的 FMA；head_dim 方向的 MMA-K 循环是编译期常量；最后一个 KV tile 单独处理越界 mask。
//
// 基于 sycl-tla applications/flash_attention_v2 的 XeDefault mainloop 和 XeFMHAFwdKernel 改写
// （非 causal、无 KV cache / paged / varlen），不修改 sycl-tla 本身；epilogue 直接复用 FMHAFwdEpilogue。
// 量化约定（h3_int8_attn.cpp 的 rms_rope_quant 准备）：
//   Q: 逐 token 对称 int8，q_scale [H, Lq]
//   K: 每个 KV tile（KvTile 个 token）一个 scale，k_scale [H, ceil(Lkv/KvTile)]
#pragma once

#include "cutlass/util/packed_stride.hpp"
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"  // 只用其中的 FMHAProblemShape / epilogue / tile scheduler

namespace cutlass::fmha::collective {

using namespace cute;

template <int Stages,
          class TiledMMAQK_,    // int8 x int8 -> int32
          class TiledMMAQKf_,   // 同样 tile / subgroup 布局的 bf16 atom：只用来给 S 提供 float 分片和坐标布局
          class TiledMMAPV_,
          int VTiles_,
          class TensorQ_, class TensorK_, class TensorV_>
struct Int8QKMainloop {
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAQKf = TiledMMAQKf_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(take<1,4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;
  using TensorQ2D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_,_),0)));
  using TensorK2D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_,_),0)));
  using TensorV2D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_,_),0)));

  using TiledCopyQ = decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{}));
  using TiledCopyK = decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{}));
  using TiledCopyV = decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{}));

  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
                           make_identity_tensor(select<0,1>(TiledMMA{}.tile_mnk()))));

  using FragSi = FragC<TiledMMAQK>;
  using FragS = FragC<TiledMMAQKf>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQKf::ValTypeD;
  static_assert(std::is_same_v<ElementS, float>, "S must be float");

  using SingleFragA = FragC<TiledMMAPV>;
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  struct Arguments {
    float scale;                 // softmax scale（1/sqrt(d)）
    const float* q_scale;        // [H, Lq]
    const float* k_scale;        // [H, ceil(Lkv/KvTile)]
    int q_scale_head_stride;
    int k_scale_head_stride;
  };
  using Params = Arguments;
  struct SharedStorage {};

  Params params;

  Int8QKMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void*) {
    Params p = args;
    p.scale = args.scale * float(1.4426950408889634074);  // exp -> exp2
    return p;
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) { return true; }

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(TensorQ2D const& Q_2D,   // (q,d) int8
                                 TensorK2D const& K_2D,   // (k,d) int8
                                 TensorV2D const& V_2D,   // (d,k) bf16
                                 FragA& tArA, FragARow& tA_max, FragARow& tA_sum,
                                 QVCoord blk_qv, int blk_k0, int blk_k1,
                                 int thr_id, int seq_len, int head) {
    using namespace sycl::ext::oneapi::this_work_item;

    auto tile_shape_v = make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    Tensor cQ = make_identity_tensor(Q_2D.shape());             // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());             // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());             // (v,k)
    Tensor cP = make_identity_tensor(take<0,2>(TileShapeQK{})); // (q,k)，tile 内坐标

    Tensor gQ       = local_tile(cQ, TileShapeQK{}, append(blk_qv,_),  Step<_1,X,_1>{});   // (q,d,D)
    Tensor gK       = local_tile(cK, TileShapeQK{}, make_coord(_,_,_), Step<X,_1,_1>{});   // (k,d,K,D)
    Tensor gV       = local_tile(cV, tile_shape_v,  make_coord(get<1>(blk_qv),_));          // (v,k,K)
    Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_,_,0), Step<X,_1,_1>{});   // (v,k,VV,K)

    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};

    TiledMMAQK mma_qk{};
    TiledMMAQKf mma_qkf{};
    TiledMMAPV mma_pv{};

    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_qkf = mma_qkf.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)

    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_,_,0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_,_,0));
    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_,_,0,0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_,_,0,0));

    auto tSrSi = thr_mma_qk.partition_sg_fragment_C(cP);   // int32
    auto tSrS = thr_mma_qkf.partition_sg_fragment_C(cP);   // float，同布局
    static_assert(sizeof(tSrSi) == sizeof(tSrS), "int8/bf16 S fragments must match");  // int32 与 float 同为 4 字节
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_,_,0,0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_,_,0,0));

    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v = make_block_2d_prefetch(copy_v);
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_split);

    // S 元素在 tile 内的 (q,k) 坐标；Q 行 scale 只和行有关，循环外取一次
    auto cS_thread = thr_mma_qkf.partition_C(cP);
    const float* q_scale = params.q_scale + head * params.q_scale_head_stride;
    const float* k_scale = params.k_scale + head * params.k_scale_head_stride;
    const int q0 = get<0>(blk_qv) * get<0>(TileShapeQK{});
    FragS tQs;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tQs.size(); i++) {
      int row = q0 + get<0>(cS_thread(i));
      tQs(i) = row < seq_len ? q_scale[row] : 1.0f;   // 越界行读到的 Q 是 0；scale 不能是 0，否则乘 -inf 得 NaN
    }
    // 同一行的 tQs 都相同，按行归约成 FragSRow，并入 softmax scale
    auto tQsRow = reduce<1>(tQs, sycl::maximum{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tQsRow.size(); i++) tQsRow(i) *= params.scale;

    constexpr int NumD = get<1>(TileShapePV{}) * VTiles / get<2>(TileShapeQK{});   // head_dim 方向的 MMA-K 步数

    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_,_,_,D));
    }
    for (int D = 0; D < size<4>(pKgK); D++) {
      CUTLASS_PRAGMA_UNROLL
      for (int K = 0; K < Stages; K++) {
        prefetch(prefetch_k, pKgK(_,_,_,K,D));
      }
    }
    if (blk_k0 == 0) {
      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);
    }

    const bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    // Mask: 这一轮是否可能是不满的最后一个 KV tile
    auto body = [&](int K, auto Mask) {
      barrier_arrive(ScopeWorkgroup);

      /* GEMM 1: S = Q · Kᵀ，int8 -> int32 */
      clear(tSrSi);
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < NumD; D++) {
        copy(copy_q, tQgQ(_,_,_,D), tQrQ);
        copy(copy_k, tKgK(_,_,_,K,D), tKrK);
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);
        cute::gemm(mma_qk, tSrQ, tSrK, tSrSi);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        prefetch(prefetch_v, pVgV(_,_,_,VV,K));
      }

      /* 反量化：行 scale × 当前 KV tile 的 scale，在 softmax 的 FMA 里乘 */
      FragSRow rowscale;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < rowscale.size(); i++) rowscale(i) = tQsRow(i) * k_scale[K];
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tSrS.size(); i++) tSrS(i) = float(tSrSi(i));

      /* 最后一个不满的 KV tile：越界的列置 -inf（坐标取法同 BMG overlay） */
      if (decltype(Mask)::value && check_remainder_k) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          if (K * get<1>(TileShapeQK{}) + get<1>(cS_thread(i)) >= seq_len) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      auto rescale = softmax(K == blk_k0, tSrS, rowscale, tA_max, tA_sum);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P · V，bf16 */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v, tVgV(_,_,_,VV,K), tVrV);
        reorder(tVrV, tArV);
        if (K != blk_k0) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArA.size() / VTiles; i++)
            tArA(_,_,_,VV)(i) *= broadcast<0>(rescale, tArA, i);
        }
        cute::gemm(mma_pv, tArP, tArV, tArA(_,_,_,VV));
      }

      for (int D = 0; D < size<4>(pKgK); D++) {
        prefetch(prefetch_k, pKgK(_,_,_,K + Stages,D));
      }
      barrier_wait(ScopeWorkgroup);
    };

    for (int K = blk_k0; K < blk_k1 - 1; K++) body(K, std::false_type{});
    if (blk_k1 > blk_k0) body(blk_k1 - 1, std::true_type{});
  }

  // 与 XeDefault mainloop 相同的 online softmax，只是 scale 按行给出（rowscale > 0）
  CUTLASS_DEVICE FragSRow softmax(bool first_block, FragS& tS, FragSRow const& rowscale,
                                  FragSRow& tS_max, FragSRow& tS_sum) {
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});
    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), rowscale(i) * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(broadcast<0>(rowscale, tS, i) * tS(i) - broadcast<0>(tS_max, tS, i));
    if (!first_block) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) tS_sum(i) *= rescale(i);
    }
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_sum.size(); i++) tS_sum(i) += tS_bsum(i);
    return rescale;
  }
};

}  // namespace cutlass::fmha::collective

namespace cutlass::fmha::kernel {

using namespace cute;

// XeFMHAFwdKernel 的精简版：非 varlen、无 KV cache，额外把 head 传给 mainloop（取 Q/K scale 用）
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeFMHAFwdInt8QKKernel {
 public:
  using ProblemShape = ProblemShape_;
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  using TiledMMAQKf = typename CollectiveMainloop::TiledMMAQKf;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using SubgroupLayoutQK = typename CollectiveMainloop::SubgroupLayoutQK;
  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;
  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));
  using SGPerWG = typename CollectiveMainloop::SGPerWG;
  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;

  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };
  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  struct KernelArguments {
    ProblemShape shape;
    const ElementQ* Q; StrideQ dQ;
    const ElementK* K; StrideK dK;
    const ElementV* V; StrideV dV;
    ElementO* O;       StrideO dO;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return {args.kernel,
            CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
            TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{})};
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.mainloop) && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(SGPerWG::value * intel::sg_size, 1, 1); }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    auto& p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;
    int thr_id = int(ThreadIdxX());

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_q, idx_b] = tile_scheduler.get_block_coord();
      auto blk_qv = make_coord(blk_q, blk_v);
      int head = head_q / head_group_q;
      if (blk_q * get<0>(TileShapeQK{}) >= s.seq_len_qo) continue;
      const int seq_len = s.seq_len_kv;
      const int k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));

      auto shape_Q = make_shape(s.seq_len_qo, s.head_size_qk, s.num_heads_q, s.batch);
      auto shape_K = make_shape(s.seq_len_kv, s.head_size_qk, s.num_heads_kv, s.batch);
      auto shape_V = make_shape(s.head_size_vo, s.seq_len_kv, s.num_heads_kv, s.batch);
      auto shape_O = make_shape(s.seq_len_qo, s.head_size_vo, s.num_heads_q, s.batch);

      Tensor Q = make_tensor(make_gmem_ptr(const_cast<ElementQ*>(p.Q)), make_layout(shape_Q, p.dQ));
      Tensor K = make_tensor(make_gmem_ptr(const_cast<ElementK*>(p.K)), make_layout(shape_K, p.dK));
      Tensor V = make_tensor(make_gmem_ptr(const_cast<ElementV*>(p.V)), make_layout(shape_V, p.dV));
      Tensor O = make_tensor(make_gmem_ptr(p.O), make_layout(shape_O, p.dO));

      FragA tArA;
      FragARow tA_max, tA_sum;

      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      mainloop(Q(_,_,head_q,idx_b), K(_,_,head,idx_b), V(_,_,head,idx_b),
               tArA, tA_max, tA_sum, blk_qv, 0, k_blocks, thr_id, seq_len, head_q);

      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      epilogue(O(_,_,head_q,idx_b), tArA, tA_max, tA_sum, blk_qv, thr_id);
    }
  }
};

}  // namespace cutlass::fmha::kernel
