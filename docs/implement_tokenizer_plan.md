# JointSceneTokenizer 设计研究

> 配套文档: `docs/research_plan.md` §3.3
> 目标: 为 FlowDGGT 的 joint scene latent `z_scene` 给出一个可落地、可消融、对扩散友好的 tokenizer 设计。
> 范围: 仅涵盖 tokenizer (T0 阶段) 的架构、损失、训练策略; SceneFlow (T1) 与联合微调 (T2) 不在本文件。

---

## 1. Context

`research_plan.md` §3.3 中的 JointSceneTokenizer 参数是初步草拟, 存在两类明确风险需要重新设计:

1. **"图像 VAE 式"设计不适配 ViT 中间特征**。

   SD-VAE 针对 RGB 图像设计, 以 pixel 重建 + LPIPS 为主损失, 瓶颈维度极低 (4-channel)。我们要编码的是 DGGT Aggregator 产出的四层 `image_tokens` (`[B,S,1369,3072]`), 是已经经过 24 层 Transformer 的深度 token。这类特征的统计特性 (近似高斯化、通道相关性、跨层冗余) 与 RGB pixel 完全不同, 直接照搬 VAE 训练配方会导致:

   - 特征域 MSE 低, 但解码后进 `gs_head / depth_head / instance_head` 时输出崩坏 (**catastrophic amplification**)。
   - 扩散模型在 `z_scene` 上损失收敛, 但对应高斯几何/外观严重失真 (**latent-head mismatch**)。

2. **四层联合压缩比过大 + 多头异构消费**未被正视。

   `image_tokens_list[l]` 内部已是 `concat(DINO(1024), Frame(1024), Global(1024))`, 不同下游 head 取的是不同通道切片。若 tokenizer 把 `4 × 12288 → 768` 一把塞进单一 latent (16× 压缩), 且 encoder 内部对三路通道等同处理, 很难同时维持 `gs_head / depth_head / instance_head` 三类精度。

本文件在**代码验证过的特征事实 (§2)** 与**相关文献 (§3)** 的基础上, 针对 8 个核心设计问题 (§4 a–h) 给出具体方案, 并配套架构伪代码 (§5)、训练策略 (§6)、风险分析 (§7) 与验证计划 (§8)。

---

## 2. 代码验证后的特征事实 (Ground Truth)

以下结论来自对 `dggt/models/aggregator.py`, `dggt/heads/dpt_head.py`, `dggt/models/vggt.py`, `dggt/layers/block.py`, `dggt/utils/gs.py` 的直接阅读, 作为后续设计的硬约束。

### 2.1 Aggregator 与 token 组织

- **Patch backbone**: DINOv2 ViT-L/14 with 4 register tokens, **完全 frozen** (`aggregator.py:185`, `mask_token.requires_grad_(False)`)。DINO 输出是 `x_norm_patchtokens` (layer-normed, `aggregator.py:216`)。
- **每层 token 布局**: `[camera_token(1), register_tokens(4), patch_tokens(1369)]`, `patch_start_idx = 5`, 每帧 1374 个 token。
- **24 层 frame_blocks + 24 层 global_blocks** (`aggregator.py:78-109`), 都是 `dim=1024, num_heads=16` 的标准 ViT Block (LN → MHSA → LN → MLP, with residual + layer-scale)。
- **Block 内部有 LN, 但层级之间没有额外 LN**, 拼接时是 raw concat (`aggregator.py:279`):

```python
concat_inter_with_tokens = torch.cat(
    [dino_token_list[i], frame_intermediates[i], global_intermediates[i]], dim=-1
)
```

- 因此 `image_tokens_list[l]` 的通道结构确实是 `[DINO(1024) | Frame(1024) | Global(1024)]`, 且**没有 post-concat LayerNorm**。

### 2.2 DPT Head 消费方式

所有 dense head 入口的**第一操作是** `LayerNorm(dim_in)` (`dggt/heads/dpt_head.py:516`)。这意味着: tokenizer 输出只需匹配方向/协方差结构即可, LN 会吃掉整体尺度差异。但通道级分布若失真, LN 之后仍会放大错误。

| Head | 输入 | `dim_in` | 取什么层 | 是否内部 split? |
|------|------|----------|----------|-----------------|
| `gs_head` | `image_tokens_list` | 3072 | [4,11,17,23] patch | **不拆**, 整块 3072 过 LN → 4 × Conv → DPT fuse |
| `depth_head` / `point_head` | `aggregated_tokens_list` | 2048 | [4,11,17,23] patch | 不拆 (已是 frame+global) |
| `instance_head` | `dino_token_list` | 1024 | [4,11,17,23] patch | 不拆 (已是 DINO only) |
| `camera_head` | `aggregated_tokens_list[-1]` | 2048 | 仅 camera token | 不用 patch |

关键事实: **`gs_head` 不会内部 split DINO/Frame/Global**。只要重建完整 3072-d image_tokens, 下游三路 head 直接按通道 slice (`split([1024,1024,1024])`) 就能拿到各自需要的 sub-stream。这是 `research_plan.md` §3.6 已经指出的复用方式, 代码验证无误。

### 2.3 形状与训练模式

- 输入图像 `[B, S, 3, 518, 518]`, patch 14×14 → **P=1369 (37×37)**。
- 每层特征 `[B, S, 1369, 3072]` (忽略 special token)。
- `global_blocks` 在 `[B, S*P, C]` 上做时空混合 attention, 因此特征**本身已含跨帧信息**, tokenizer 不必重做 aggressive 跨帧融合, 但仍需跨帧一致性。
- 典型 S=4 (train) / ≥10 (inference mode=3), **latent 设计对 S 不做时间压缩**。
- `train.py:84-86` 显示当前 fine-tune 阶段只训 `gs_head / instance_head / sky_model`, aggregator 默认冻结 — 这和 RAE "frozen encoder + trained decoder" 思路天然兼容。

### 2.4 渲染链条

- `gs_head` 产出 `gs_map`: `[B, S, 1, H, W]` 共 **12 通道** = `RGB(3) + opacity(1) + log_scale(3) + quat(4) + SH(1)` (可配置)。
- `get_split_gs` (`dggt/utils/gs.py:71-76`) 按 mask 提取 GS 参数, 喂入 `gsplat.rendering.rasterization()`。
- 因此 tokenizer 的 "downstream consistency loss" 有两个天然钩子: **(a) gs_map 参数空间** L2; **(b) 渲染图像** L2/LPIPS。

### 2.5 与 research_plan.md 的偏差说明

| 项 | research_plan | 代码实际 | 本设计影响 |
|----|---------------|---------|-----------|
| image_tokens 通道结构 | 3072 = 1024×3 | ✅ 确认 | 无 |
| patch 数 P | 1369 | ✅ 确认 (37×37) | 无 |
| special token | 未明说 | 代码里 `patch_start_idx=5`, 必须处理 | tokenizer 只处理 1369 patch, special token 由 `reattach_special_tokens` 复用 edited-pass 原值 |
| post-concat LN | 未说 | **无** | tokenizer 输入是 raw concat, 内部自己做 LN |
| Aggregator 冻结状态 | 未强调 | T0 阶段应冻结, 对齐 RAE | 影响 T0 损失设计与初始化 |

---

## 3. 相关文献

对"对非 pixel 的 latent 做生成"这个问题, 搜索后锁定两篇最相关的近期工作做深入研读, 另附若干辅助参考。

### 3.1 主参考: RAE — Diffusion Transformers with Representation Autoencoders

- **论文**: "Diffusion Transformers with Representation Autoencoders", Zheng et al., 2024 (arXiv 2510.11690)

为什么它最相关: RAE 处理的正是 "pretrained ViT 特征 → latent → decoder" 这条路径, 而且明确否定了 SD-VAE 那种低维 bottleneck + KL 的图像 VAE 配方, 这正是本项目担心的风险。

**要点**:

- **冻结 pretrained 编码器 (DINO / SigLIP / MAE) 作为 encoder**, 只训练一个轻量 decoder。
- **latent 不做 aggressive bottleneck**, 维度直接就是编码器 token 维度 (768–1024), **不用 KL**, 依靠 pretrained 特征自带的语义结构避免 diffusion 学习塌缩。
- **dimension-dependent noise schedule shift**: 高维 latent 需要把 noise schedule 按 `sqrt(d / d_ref)` 缩放, 否则扩散损失看似低但生成差 (ImageNet-256 FID 23 → 4.81)。
- **decoder 训练时加噪声增广**, 让 decoder 对 diffusion 轨迹上的 noisy latent 鲁棒。
- 结果: rFID 0.16 vs SD-VAE 0.79 (16× 更好的重建), DiT 训练收敛快 40×+。

**对本项目的映射**:

- DGGT 的 Aggregator + 四层 image_tokens 就扮演 RAE 的 "frozen pretrained encoder" 角色。
- `JointSceneTokenizerEncoder` 实际上只是一个**轻量的 cross-layer 压缩器** (3072×4 → 768), 不用学习"语义"。
- `JointSceneTokenizerDecoder` 是真正需要训练的部分, 要把 768 稳定扩回 3072×4。
- **不要在 latent 上加 KL**; 用 LayerNorm + 通道 std 归一化即可。
- **Decoder 训练时应该加噪声增广** (见 §4.8 的 `L_denoise_recon`), 因为 SceneFlow 后期会喂进 noisy `z_hat`。

### 3.2 次参考: REPA — Representation Alignment for Generation

- **论文**: "Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think", Yu et al., ICLR 2025 Oral (arXiv 2410.06940)

为什么相关: REPA 证明了"对齐到 frozen pretrained 特征"本身就是极强的生成监督 — 比学习重建 pixel 快 17.5×。

**要点**:

- 在 diffusion 模型的中间激活上, 加一个 projection, 和 **frozen DINO 特征做 L2 对齐**。
- 不需要 pixel 重建损失, 纯对齐就能到 FID 1.42。
- 这说明特征域的生成监督是稳定的, 前提是目标特征来自 frozen encoder。

**对本项目的映射**:

- 我们的 `z_scene` 可以直接用 "frozen Aggregator 产出的 image_tokens" 作为 alignment target。
- 如果 RAE 风格纯 MSE 不稳定, 可以退化到 REPA 风格的 projection alignment (把 decoder 中间层对齐到 clean image_tokens)。
- 这也解释了为什么我们**不应该用 pixel loss 作为主 tokenizer 损失** — 特征域监督更直接、更稳。

### 3.3 辅助参考

- **latentSplat** (ECCV 2024, arXiv 2403.16292): 把 3D 场景 tokenize 成 "variational feature Gaussians", 说明在几何特征 latent 上做 uncertainty modeling 是可行的, 对本项目主要是 conceptual 背书。
- **DiffGS** (NeurIPS 2024, arXiv 2410.19657) / **GaussianAnything** (arXiv 2411.08033): 都在 3DGS / 多视角 latent 上做扩散, 共同信号是 "几何/任务驱动的重建目标 比 pixel RGB 更稳"。
- **Perceiver IO**: 对变长输入/输出的处理方式对本项目 P=1369 固定 patch 帮助有限, 但 cross-attention bottleneck 的做法可供比较。

**结论**: RAE 是主骨架参考, REPA 提供"特征域监督可以替代 pixel 重建"的信心, latentSplat/DiffGS 提供 "geometry-aware 重建目标" 的信心。

---

## 4. 设计决策 (对应 8 个核心问题)

### 4.1 (a) 四层怎么处理 / 相关性怎么保留

**选定方案: 单一 latent, 但 encoder 内部显式保留四层身份**, 而不是独立 4 个 latent, 也不是直接拼成 12288-d 黑箱。

推荐结构:

```
输入: image_tokens_list[[4,11,17,23]] → 4 × [B, S, 1369, 3072]

Per-layer projection (层间不共享权重, 流间共享):
  x_l = LN(x_l) + layer_embed_l   # 显式层身份
  x_l' = MLP_l(x_l)   # 3072 → 768

Cross-layer fusion (共享权重):
  X = stack([x_0', x_1', x_2', x_3'], dim=layer)  # [B, S, 1369, 4, 768]
  # 在层维度做 self-attention, 保留层间相关性, 同时允许冗余压缩
  X' = LayerAttnBlock(X)           # over L=4
  z  = LayerPool(X')               # 4 × 768 → 768 (learned query)

z ∈ [B, S, 1369, 768]
```

**理由**:

- DGGT 的 4 层是 DPT 所需的多尺度信号, 不能像图像 VAE 那样只用最后一层。
- 层间有强冗余 (同一 ViT 的相邻 block), 联合压缩比独立压缩参数效率高。
- layer-attention 比 "直接 concat + MLP" 更能选择性保留每层独特信息, 并给 decoder 提供足够解耦能力。
- 最终输出还是**单一 latent**, 便于 SceneFlow 做统一扩散, 不会让 flow 同时处理 4 个不同分布的 latent。

**被拒选项 A** (4 × [B,S,P,192] 分层 latent): flow 侧每层分布不一致, 容易 diverge, 且 layer-attention 已承担跨层融合能力。

**被拒选项 C** (直接 concat 12288 → 768 单层 MLP): 参数大 (9.4M 一层), 丢失层身份, 对 DPT 不友好。

### 4.2 (b) 不同 head 取不同特征 — 要不要分流处理

**不分流**。保留 `research_plan.md` 的 "只 tokenize image_tokens (3072), 解码后按通道 split" 思路, 原因:

- `gs_head` 就是直接吃 3072 整块过 LN, 如果在 tokenizer 里强行拆, 反而要重新实现 head。
- `image_tokens` 已经包含了 `aggregated_tokens` 和 `dino_tokens` 的全部信息 (`aggregated = [frame|global]`, `dino = dino-stream`), 三路 head 的差异只是 slice 不同通道, 不需要 tokenizer 知道。
- **但 loss 必须对三路 head 的输出都施加约束** (§4.8), 否则 tokenizer 会偏向某一路而忽视其它。

工程上在 encoder 入口做 **per-stream sub-projection**:

```python
dino_l, frame_l, global_l = x_l.split([1024, 1024, 1024], dim=-1)
e_dino_l   = LN_d(dino_l)   → MLP_d → 256
e_frame_l  = LN_f(frame_l)  → MLP_f → 256
e_global_l = LN_g(global_l) → MLP_g → 256
x_l_token = concat(e_dino_l, e_frame_l, e_global_l)  # [B, S, P, 768]
```

好处: encoder 显式尊重 DGGT 的三路通道语义, 不用 1024×3072 大 linear 一锅烩; 解码侧对应做 inverse split, 方便 head consistency loss 按通道算。

### 4.3 (c) 帧间注意力 / 时间维度

**保留跨帧 attention, 但不做时间压缩**。

- Aggregator 本身已做跨帧 attention, 四层特征已含全局时间信息。但 tokenizer 压缩到 768 后若完全独立处理每帧, 会在 latent 上丢失帧间几何一致。
- 推荐: encoder / decoder 各含 **frame-attn + cross-frame-attn 交替块** (类似 Aggregator 风格但更浅):
  - frame-attn: reshape 到 `[B*S, P, d]`, 每帧内 attention
  - cross-frame-attn: reshape 到 `[B, S*P, d]`, 跨帧 attention
- **不做时间降采样** (`S` 保留), 因为 SceneFlow 需要 per-frame latent 和 camera pose 对齐。
- 位置编码: `axial(frame_idx, patch_y, patch_x)` RoPE (和 DiT 3D 变体一致)。

### 4.4 (d) 输入形状 N×D 还是 H×W×D

**主路径 N×D (token sequence), 保留 P=1369 patch 作为 token**。

- 输入就是 ViT token, 直接走 attention 最自然。
- DPT heads 会在 `patch_start_idx` 之后把 `(P,)` reshape 成 `(37, 37)` 再走 conv, 所以 N×D 与 H×W×D 的转换在 head 侧本来就要做。
- 加一个 **lightweight 2D conv "detail branch"** 并行: 在每层投影后 reshape 成 `[B*S, 128, 37, 37]` 走 2-3 层 3×3 conv, 再 reshape 回 token concat — 只对"高频局部细节"补足。这是 Swin / ConvFormer 的常见做法, 参数少 (~1M), 对 Gaussian 细节很有帮助。

**被拒**: 纯 H×W×D CNN tokenizer — 会丢失长程 token 关系, 与 Aggregator 的 attention 风格不一致。

### 4.5 (e) 如何让 latent 对扩散友好

5 条具体措施, 全部来自 RAE / REPA 的经验:

1. **Latent 出口加 LN (无 KL)**。给 `z_scene` 一个确定的尺度 (std≈1), 而不把它变成随机变量。KL 在 768 维会退化为全体方差压制, 反而磨平语义。
2. **通道 std 归一化 target**。训练前在 Waymo 训练集上跑一遍 Aggregator, 统计每层每通道的 `mean_c, std_c` (维度 `4 × 3072`), 重建 loss 里除以 std, 避免高方差通道吃掉梯度。
3. **Decoder noise-augmented pre-training (RAE 关键 trick)**。T0 训练 decoder 时, 以 50% 概率对 `z` 加高斯噪声 `z_noisy = α·z + β·ε`, `α,β` 从 VP 调度中采样。让 decoder 对 SceneFlow 出来的 `z_hat` 鲁棒。
4. **Dimension-aware flow schedule (后续 T1)**。tokenizer 本身不管 flow schedule, 但设计时要记一笔: 768-d latent 需要 `σ_shift = sqrt(768 / 256)` 的 schedule shift, 不然 SceneFlow 会学崩。
5. **重建 loss 在标准化特征域, 但 task/render loss 在原尺度**。feature MSE 数值稳定, 同时 gs_head / gsplat 拿到的是正确尺度的输入。

### 4.6 (f) 模型结构与参数

参考 RAE decoder (MAE-B 级别) + 当前 VGGT Block 风格, 推荐配置:

```
JointSceneTokenizerEncoder
├─ per-stream LN + linear: {dino, frame, global} × 1024 → 256
├─ cat 3 × 256 = 768 (per-layer token)
├─ detail branch: [B*S, 128, 37, 37] conv×2, → 128 token feature, concat → 896
├─ per-layer layer-embed (learnable [4, 896])
├─ layer-attention block ×2 (over 4 layers, dim 896)
├─ layer-pool (learned query, 4 → 1)
├─ trunk: {frame-attn, global-attn} × 3 pairs (dim 896, heads 14)
├─ final LN → 768 projection
└─ output z_scene [B, S, 1369, 768]

JointSceneTokenizerDecoder
├─ input LN + 768 → 896
├─ trunk: {frame-attn, global-attn} × 3 pairs (dim 896, heads 14)
├─ layer-unpool (1 → 4, learned query tokens)
├─ layer-attention block ×2
├─ per-layer expansion: dim 896 → 3072 via (linear → 1 transformer block → linear)
└─ output 4 × [B, S, 1369, 3072]
```

参数估计: encoder ≈ 28M, decoder ≈ 34M, 总 ≈ 62M。合理, 远小于 SceneFlow 本体 (~200M)。

**与 research_plan.md §3.3 的差异**:

- research_plan 给 encoder=640, decoder=896, detail=128。本方案把 encoder 也提到 896 与 decoder 对称, 更稳。
- detail branch 128 保持一致。
- 加入了显式 **layer-attention**, research_plan 里没有但对四层非常关键。

### 4.7 (g) 基于什么权重训练

**从头训练, 但使用结构感知的初始化**。

- Aggregator 权重无法直接复用: 维度不匹配 (1024 vs 896 vs 768), 结构也不同 (本设计含 layer-attention)。
- 从头训练可行, 原因:
  - RAE 证明了 lightweight decoder 从头训即可, 关键是 encoder (Aggregator) frozen 提供稳定信号。
  - 本 tokenizer 是压缩器 + 扩张器, 不是语义抽取器, 不需要 ImageNet 级预训练。
- **初始化细节**:
  - 线性层 `trunc_normal_(std=0.02)`。
  - per-stream projection 的偏置初始化为 0, 保持 LN 后分布稳定。
  - transformer block 复用 DGGT `dggt/layers/block.py` 的 `Block` (带 LayerScale, 初始化值 1e-6), 与主干一致。
- **可选实验**: encoder 的 frame-attn/global-attn block 用 Aggregator 的最后 2 层 block 做参数正交投影初始化 (1024 → 896), 预期能加速收敛, 但不是必须。

### 4.8 (h) 损失函数

核心主张: **多层级一致性 + 通道归一化重建 + 下游任务闭环**, 不做纯特征 MSE。完整 loss:

```
L_tokenizer(T0) =
    λ_recon     * L_recon          # 特征重建 (std-normalized, per-layer, per-channel)
  + λ_layer     * L_layer_align    # 每层 cosine 对齐, 防止 layer-pool 退化
  + λ_head      * L_head_consist   # 三路 head 输出一致 (gs_map / depth / dynamic_conf)
  + λ_render    * L_render         # rasterize 后图像一致 (L2 + LPIPS)
  + λ_denoise   * L_denoise_recon  # 加噪 latent 的重建 (RAE 风格)
  + λ_lat_reg   * L_latent_stats   # latent 通道 std 接近 1, 防 collapse
```

各项具体定义:

**1. `L_recon` — 通道归一化特征重建**

```
L_recon = Σ_l Σ_c w_c · (x_l_hat[..., c] - x_l[..., c])² / (σ_l_c² + ε)
```

其中 `w_c` 对 DINO/Frame/Global 分通道加权 (初始 1:1:1, 可做 ablation)。`σ_l_c` 是 Waymo 训练集上预统计的每层每通道标准差。

**2. `L_layer_align` — 每层 cosine 对齐**

```
for l in [4, 11, 17, 23]:
  L_layer_l = 1 - cosine(x_l_hat, x_l).mean()
L_layer_align = mean_l(L_layer_l)
```

防止 layer-pool 把某层彻底吃掉。

**3. `L_head_consist` — 下游 head 一致性 (关键, 防 catastrophic amplification)**

```python
with torch.no_grad():
    g_ref = gs_head(image_tokens_orig, ...)
    d_ref = depth_head(agg_orig, ...)
    m_ref = instance_head(dino_orig, ...)

g_hat = gs_head(image_tokens_hat, ...)      # 重建 token 进 frozen head
d_hat = depth_head(agg_hat, ...)
m_hat = instance_head(dino_hat, ...)

L_head_consist = w_gs    * L2(g_hat.gs_map,  g_ref.gs_map)
               + w_depth * L2(d_hat.depth,   d_ref.depth)
               + w_dyn   * L2(m_hat.dyn,     m_ref.dyn)
```

所有 head **冻结** (`requires_grad_=False`), 梯度只经它们反传到 tokenizer。

**4. `L_render` — 渲染一致性**

用 `get_split_gs` + `gsplat.rasterization` 渲染重建 GS, 和原 GS 渲染结果比:

```
I_hat = render(gs_map_hat, cam_orig)
I_ref = render(gs_map_ref, cam_orig)
L_render = L2(I_hat, I_ref) + 0.1 · LPIPS(I_hat, I_ref)
```

**只在观测视角上算, 不做 novel view**。避免引入 pseudo-GT 风险。

**5. `L_denoise_recon` — RAE 风格加噪鲁棒化**

```
t ~ U[0, T]
z_noisy = α_t · z + σ_t · ε,     ε ~ N(0, I)
x_l_hat_n = Decoder(z_noisy)
L_denoise_recon = Σ_l ||x_l_hat_n - x_l||² / σ_l²
```

只在后半程训练启用 (T0.d), 让 decoder 对 SceneFlow 输出鲁棒。

**6. `L_latent_stats` — latent 归一化 (代替 KL)**

```
L_latent_stats = |std(z, dim=(B, S, P)) - 1|.mean()
               + |mean(z, dim=(B, S, P))|.mean()
```

**推荐权重** (T0 初始):

```
λ_recon    = 1.0
λ_layer    = 0.3
λ_head     = 0.5
λ_render   = 0.2   (warm up 5k 步后再开)
λ_denoise  = 0.1   (20k 步后再开)
λ_lat_reg  = 0.01
```

---

## 5. 最终架构伪代码

```python
import torch
import torch.nn as nn
from dggt.layers.block import Block   # 复用 Aggregator 同款 Block


class FrameGlobalBlockPair(nn.Module):
    """frame-attn → global-attn 的组合块, 参考 Aggregator 的 space-time 风格。"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.frame_block  = Block(dim=dim, num_heads=num_heads)
        self.global_block = Block(dim=dim, num_heads=num_heads)

    def forward(self, x):
        # x: [B, S, P, D]
        B, S, P, D = x.shape
        # frame attention
        x = x.view(B * S, P, D)
        x = self.frame_block(x)
        x = x.view(B, S, P, D)
        # cross-frame attention
        x = x.view(B, S * P, D)
        x = self.global_block(x)
        x = x.view(B, S, P, D)
        return x


class LayerAttnStack(nn.Module):
    """在 layer 维度 (size=4) 做 self-attention。"""
    def __init__(self, dim, depth=2, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim=dim, num_heads=num_heads) for _ in range(depth)])

    def forward(self, X):
        # X: [B, S, P, L, D]
        B, S, P, L, D = X.shape
        X = X.view(B * S * P, L, D)
        for blk in self.blocks:
            X = blk(X)
        return X.view(B, S, P, L, D)


class LearnedQueryPool(nn.Module):
    """learned-query cross-attention, 用于 layer-pool (L → 1) 和 layer-unpool (1 → L)。"""
    def __init__(self, dim, n_query, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, dim) * 0.02)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm  = nn.LayerNorm(dim)

    def forward(self, X):
        # X: [B, S, P, L, D] (输入)  →  [B, S, P, n_query, D] (输出)
        B, S, P, L, D = X.shape
        q = self.query.unsqueeze(0).expand(B * S * P, -1, -1)
        kv = X.view(B * S * P, L, D)
        out, _ = self.attn(q, kv, kv)
        out = self.norm(out)
        out = out.view(B, S, P, -1, D)
        if out.shape[-2] == 1:
            out = out.squeeze(-2)   # [B, S, P, D]
        return out


class JointSceneTokenizerEncoder(nn.Module):
    """
    Input : 4 × [B, S, P=1369, 3072] 来自 aggregator.image_tokens_list[[4,11,17,23]]
            (已剥掉 patch_start_idx 之前的 5 个 special tokens)
    Output: z_scene [B, S, 1369, D_z=768]
    """
    def __init__(self, D_z=768, D_hid=896, num_layers=4,
                 num_blocks_pairs=3, num_heads=14, P=1369, patch_hw=37):
        super().__init__()
        self.num_layers = num_layers
        self.patch_hw = patch_hw
        # per-stream LN + linear, 层间不共享
        self.stream_ln = nn.ModuleList([
            nn.ModuleDict({
                "d": nn.LayerNorm(1024),
                "f": nn.LayerNorm(1024),
                "g": nn.LayerNorm(1024),
            }) for _ in range(num_layers)
        ])
        self.stream_proj = nn.ModuleList([
            nn.ModuleDict({
                "d": nn.Linear(1024, 256),
                "f": nn.Linear(1024, 256),
                "g": nn.Linear(1024, 256),
            }) for _ in range(num_layers)
        ])
        # detail conv branch (shared, per-layer apply)
        self.detail_conv = nn.Sequential(
            nn.Conv2d(768, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        # layer-aware embedding & fusion
        self.layer_embed = nn.Parameter(torch.zeros(num_layers, D_hid))
        self.layer_attn  = LayerAttnStack(D_hid, depth=2)
        self.layer_pool  = LearnedQueryPool(D_hid, n_query=1)
        # trunk
        self.blocks = nn.ModuleList([
            FrameGlobalBlockPair(D_hid, num_heads)
            for _ in range(num_blocks_pairs)
        ])
        self.out_ln   = nn.LayerNorm(D_hid)
        self.out_proj = nn.Linear(D_hid, D_z)

    def forward(self, image_tokens_list_4):
        per_layer_tokens = []
        for l, x_l in enumerate(image_tokens_list_4):
            B, S, P, _ = x_l.shape
            d, f, g = x_l.split([1024, 1024, 1024], dim=-1)
            e = torch.cat([
                self.stream_proj[l]["d"](self.stream_ln[l]["d"](d)),
                self.stream_proj[l]["f"](self.stream_ln[l]["f"](f)),
                self.stream_proj[l]["g"](self.stream_ln[l]["g"](g)),
            ], dim=-1)                                      # [B, S, P, 768]
            # detail conv branch
            e_conv = e.view(B * S, self.patch_hw, self.patch_hw, 768).permute(0, 3, 1, 2)
            e_conv = self.detail_conv(e_conv)               # [B*S, 128, 37, 37]
            e_conv = e_conv.permute(0, 2, 3, 1).reshape(B, S, P, 128)
            e = torch.cat([e, e_conv], dim=-1)              # [B, S, P, 896]
            e = e + self.layer_embed[l]
            per_layer_tokens.append(e)
        X = torch.stack(per_layer_tokens, dim=-2)           # [B, S, P, 4, 896]
        X = self.layer_attn(X)
        x = self.layer_pool(X)                              # [B, S, P, 896]
        for blk in self.blocks:
            x = blk(x)
        z = self.out_proj(self.out_ln(x))                   # [B, S, P, 768]
        return z


class JointSceneTokenizerDecoder(nn.Module):
    """
    Input : z_scene [B, S, 1369, 768]
    Output: 4 × [B, S, P, 3072]
    """
    def __init__(self, D_z=768, D_hid=896, num_layers=4,
                 num_blocks_pairs=3, num_heads=14):
        super().__init__()
        self.num_layers = num_layers
        self.in_proj = nn.Linear(D_z, D_hid)
        self.in_ln   = nn.LayerNorm(D_hid)
        self.blocks = nn.ModuleList([
            FrameGlobalBlockPair(D_hid, num_heads)
            for _ in range(num_blocks_pairs)
        ])
        self.layer_unpool = LearnedQueryPool(D_hid, n_query=num_layers)
        self.layer_attn   = LayerAttnStack(D_hid, depth=2)
        self.layer_embed  = nn.Parameter(torch.zeros(num_layers, D_hid))
        # per-layer 3 stream expansion: 896 → 1024 per stream
        self.stream_exp = nn.ModuleList([
            nn.ModuleDict({
                "d": nn.Sequential(nn.LayerNorm(D_hid), nn.Linear(D_hid, 1024)),
                "f": nn.Sequential(nn.LayerNorm(D_hid), nn.Linear(D_hid, 1024)),
                "g": nn.Sequential(nn.LayerNorm(D_hid), nn.Linear(D_hid, 1024)),
            }) for _ in range(num_layers)
        ])
        self.per_layer_refine = nn.ModuleList([
            Block(dim=3072, num_heads=16) for _ in range(num_layers)
        ])

    def forward(self, z):
        x = self.in_ln(self.in_proj(z))                     # [B, S, P, 896]
        for blk in self.blocks:
            x = blk(x)
        X = self.layer_unpool(x)                            # [B, S, P, 4, 896]
        X = X + self.layer_embed
        X = self.layer_attn(X)
        outs = []
        for l in range(self.num_layers):
            h = X[..., l, :]                                # [B, S, P, 896]
            d = self.stream_exp[l]["d"](h)
            f = self.stream_exp[l]["f"](h)
            g = self.stream_exp[l]["g"](h)
            x_l = torch.cat([d, f, g], dim=-1)              # [B, S, P, 3072]
            x_l = self.per_layer_refine[l](x_l)
            outs.append(x_l)
        return outs


class JointSceneTokenizer(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        self.encoder = JointSceneTokenizerEncoder(**cfg)
        self.decoder = JointSceneTokenizerDecoder(**cfg)

    def encode(self, image_tokens_4):
        return self.encoder(image_tokens_4)

    def decode(self, z):
        return self.decoder(z)
```

**工具函数/复用说明**:

- `Block` 直接复用 `dggt/layers/block.py`, 保持与主干一致 (LayerScale, drop path 支持)。
- `select_patch_pyramid(image_tokens_all, levels, patch_start_idx)`、`reattach_special_tokens(template_tokens, levels, patch_start_idx, patch_tokens)`、`replace_selected_levels(all_levels, levels, new_values)` 三个工具函数已由 `research_plan.md` §6.2 描述, 新增到 `dggt/utils/tokens.py`。
- `get_split_gs` 可直接在 `L_head_consist` / `L_render` 里复用 (`dggt/utils/gs.py`)。

---

## 6. 训练策略 (T0: Tokenizer Pretraining)

| Phase | 步数 | 冻结 | 开启的 loss | 目的 |
|-------|------|------|-------------|------|
| T0.a  | 0–5k   | Aggregator + 所有 heads | `L_recon + L_layer_align + L_latent_stats` | 先拿到稳定 recon |
| T0.b  | 5k–20k | Aggregator + heads     | + `L_head_consist` | 闭环 head 输出 |
| T0.c  | 20k–40k| Aggregator + heads     | + `L_render` (低权重 0.1) | 对齐最终像素域 |
| T0.d  | 40k–60k| Aggregator + heads     | + `L_denoise_recon` | 为 T1 扩散做 decoder 鲁棒化 |

**训练配置**:

- 优化器: AdamW, lr = 2e-4, cosine decay 到 1e-5, weight decay 0.05
- Batch: 4 clips × 4 frames / GPU, BF16, 8 × A100 (与 research_plan §5.5 一致)
- 数据: Waymo Open 训练集, 无需 Trellis 资产 (只需要 clean 原视频)
- 预处理: 训练前跑一遍 Aggregator, 统计每层每通道 σ, 缓存为 `dggt/utils/feature_stats.py` 可加载的 `.pt`

**监控指标 (每 2k 步)**:

- `L_recon`, 三个 head 的 per-pixel 误差, render PSNR / SSIM, latent std / mean
- 每层 cosine similarity
- 每 10k 步跑一次完整 `L_denoise_recon` (不开启加噪训练的分支也要测) 作为 leading indicator

**Early stop criterion**:

- `L_head_consist` 进入 plateau 且 render PSNR 与 "直接 image_tokens 过 heads" baseline 差 < 0.5 dB

---

## 7. 风险与替代方案

### 7.1 主要风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| 16× 压缩过大, 768 latent 不够 | gs_map / depth 细节丢 | 消融 `D_z ∈ {512, 768, 1024}`; 如果 768 过边缘, 扩到 1024 (flow 侧同步调 schedule shift) |
| detail branch 无法弥补高频 | GS 边缘模糊 | 后续可加 `L_gs_edge`: 在 `gs_map` 梯度域加 Sobel loss |
| head consistency loss 使 tokenizer 过拟合某一 head | 其他 head 变差 | 动态 loss 重加权: 若某 head 误差 < 0.05, 降权到 0.1×; 其它 head 优先 |
| frame-attn + global-attn 参数过多 | T0 训练慢 | 先只用 frame-attn + 1 个 global-attn, 后面加回 |
| Aggregator 输出未来会 fine-tune (T2) | 分布漂移 | 设计 `tokenizer.refit(agg_new)` 接口, 在 T2 联合微调时重走 T0.a+b 短流程 |
| gs_head LN 会吸收小尺度错 | 看似 MSE 好但 render 差 | 必须保留 `L_render`, 不能只看特征 MSE |
| 渲染 loss 开销大 | T0.c 每步慢 2-3× | 可抽 20% 样本做 render loss, 其余只跑 head consistency |

### 7.2 被放弃的替代方案

- **纯 VAE + KL**: KL 在 768 维退化为方差压制, RAE 已证明对 pretrained 特征无益。
- **VQ-VAE 离散化**: 与扩散 / 流匹配不兼容, 除非切换到 AR 生成。
- **4 个独立 latent**: flow 侧分布不一致, 失去 research_plan 的 "单一 `z_scene`" 核心主张。
- **Pixel LPIPS 主损失**: 绕一圈回到图像 VAE 风险, 不如直接 head + render loss。
- **冻结 Aggregator 权重初始化 tokenizer**: 维度不匹配 (1024 vs 896 vs 768), 需要正交投影, 收益不大。
- **全局 attention 一把梭 (不做 frame/global 分解)**: `S × P = 4 × 1369 = 5476` token 直接做 self-attention, 显存 / 算力双爆, 不可取。

---

## 8. 验证计划

### 8.1 单元级

- `JointSceneTokenizer` 前向 shape 检查:
  - 输入 4 × [1, 4, 1369, 3072] → `z [1, 4, 1369, 768]` → decode 回 4 × [1, 4, 1369, 3072]
- 梯度可传: `torch.autograd.grad(L_recon, encoder.parameters())` 全非零
- 参数量与 FLOPs 估算与本文设计吻合

### 8.2 Identity baseline (sanity upper bound)

- 构造 passthrough 的 fake tokenizer (encode/decode 都是 identity), 跑完整 pipeline, 记录 `gs_head / depth_head / instance_head` 的 GT 输出作为上限
- 真 tokenizer 的每项指标应逼近这个上限 (render PSNR 差 < 0.5 dB)

### 8.3 特征重建指标

- per-layer feature MSE (std-normalized)
- per-layer cosine similarity
- per-channel 相关系数
- DINO / Frame / Global 三路单独的 MSE, 防止某一路被忽略

### 8.4 下游 head 一致性

- `gs_map` L2 < 5e-3 (std-normalized), render PSNR ≥ baseline − 0.5 dB
- `depth` abs-rel < 0.02
- `instance dyn_conf` AUROC drop < 0.01

### 8.5 Latent 统计

- `z_scene` 每通道 std ∈ [0.8, 1.2], mean ≈ 0
- collapse 通道比例 (std < 0.1) < 1%

### 8.6 为扩散做准备 (RAE 标准)

- 加 VP 噪声 `σ_t ∈ {0.1, 0.3, 0.5}` 后 decode, render PSNR 降幅 < 1.5 dB
- 没加 `L_denoise_recon` 的 ablation 应该在这一项上显著差 (> 3 dB 降幅)

### 8.7 端到端 smoke test

- 用 `inference.py --mode 2` 的一段 Waymo clip, 走:
  `Aggregator → Tokenizer.encode → Tokenizer.decode → reattach special tokens → heads → gsplat rasterization`
- 对比基线: 直接 `Aggregator → heads → rasterization`
- 保存左右对比图到 `vis/tokenizer_smoke/`

### 8.8 关键消融 (T0 完成后)

| 实验 | 验证目标 |
|------|----------|
| `D_z ∈ {512, 768, 1024}` | latent 容量权衡 |
| 去掉 layer-attention | 四层是否需要显式融合 |
| 去掉 detail conv branch | 2D conv 是否对细节有帮助 |
| 去掉 `L_head_consist` | catastrophic amplification 是否出现 |
| 去掉 `L_render` | render 质量是否退化 |
| 去掉 `L_denoise_recon` | SceneFlow (T1) 收敛是否变慢 |
| frame-attn only (无 global-attn) | 跨帧 attention 是否必要 |
| 加 KL 替代 `L_latent_stats` | 验证 RAE 结论在本场景成立 |

---

## 9. 参考文献

### 核心

- **RAE**: Zheng et al., "Diffusion Transformers with Representation Autoencoders", 2024. arXiv:2510.11690
- **REPA**: Yu et al., "Representation Alignment for Generation", ICLR 2025. arXiv:2410.06940

### 3D / 3DGS latent 扩散

- **latentSplat**: Wewer et al., ECCV 2024. arXiv:2403.16292
- **DiffGS**: Zhou et al., NeurIPS 2024. arXiv:2410.19657
- **GaussianAnything**: NeurIPS 2024. arXiv:2411.08033

### 基础

- **DGGT / VGGT**: 本项目基础
- **DINOv2**: Oquab et al., TMLR 2024
- **DPT**: Ranftl et al., "Vision Transformers for Dense Prediction", ICCV 2021
- **gsplat**: Ye et al., "gsplat: An Open-Source Library for Gaussian Splatting", 2024
