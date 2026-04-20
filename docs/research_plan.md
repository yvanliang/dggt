# FlowDGGT: Feature-Splat 驱动的 4D 驾驶场景目标编辑研究计划

> 目标会议: CVPR / NeurIPS 2027
> 核心约束: 单一模型; 目标编辑精确; 有限训练资源; 不依赖 SceneDirector 权重
> 范围: 仅目标编辑 (删除/插入/替换/重定位), 暂不考虑轨迹编辑
> **废弃"编辑后渲染 -> 再过 Aggregator"的 Pass 2**, 改为直接用 Feature Splatting 从 3D 高斯场构造 `z_edited`, 配合连续软掩码、per-token 噪声调度、DPT 高分旁路和 gs_map late fusion。

---

## 一、核心结论

FlowDGGT 的核心设计:

1. **精确几何编辑仍在 3DGS 空间完成**: 删除、插入、替换、重定位全部由 3D bbox + 高斯操作严格控制。
2. **生成变量是 DGGT 上游的 joint scene latent `z_scene`**; `z_edited` 由 3D 高斯场**直接 splat** 构造, 不再经过"渲染 -> 图像 -> Aggregator"的信息瓶颈。
3. **每粒高斯的特征 `F_g` 来自真实 aggregator token**: 保留高斯复用 Pass-1 的 token, 新资产高斯来自**资产独立 DGGT Pass** 的 token (per-object per-frame, 在 **Waymo 真值相机 + Waymo bbox** 下渲染, 2D 像素与 GT 逐像素对齐)。Asset Pass 的 2D 空间 (Waymo) 与 Feature Splat 的 3D 空间 (DGGT) 由语义指针 `(object_id, view_n, patch_idx)` 桥接, 两侧坐标系不需要强制一致。
4. **连续软掩码替代二值 mask**: `K_map / D_map / I_map` 由 3 类高斯分别渲染, 面积下采样到 37×37 得到三路 `[0,1]` 归一化分布 `M_preserve_soft / M_source_soft / M_dest_soft`, 并驱动 **per-token 噪声调度**。
5. **小目标三重保真**: 高分 splat + DPT 多尺度 side-injection + gs_map late fusion, 让确定性 3D 信息绕开 37×37 latent 瓶颈。
6. **两种训练模式保留**: 协调训练 (insert/replace/reposition 的 dest) + 补全训练 (delete/reposition 的 source)。**新增 T0.5 splatted-token 自一致阶段**, 让 tokenizer 适配 splat 输入分布。
7. **跨视角监督不使用 pseudo novel-view GT**, 只用真实观测视图之间的深度引导重投影一致性, 再辅以 unseen-view 几何正则。

**关键澄清**

- 补全训练不是"删掉车辆后再把车辆恢复回来", 而是在无车背景上合成 deletion-shaped holes, GT 始终是原始背景本身。
- 不再有"第二次 scene Aggregator 前向"。仍然有 **一次 Asset Aggregator 前向** (资产 isolated render), 但输入是干净渲染, 没有场景编辑伪影。
- z_clean 仍由真实原图过 Aggregator 得到, 作为 clean context 条件始终提供给 flow。

---

## 二、调研结论

### 2.1 扩散空间分类

| 类别 | 代表方法 | 扩散空间 | 对本问题的结论 |
|------|----------|----------|----------------|
| 像素或 latent 后处理 | Difix3D+ | SD latent | 可借助强 2D 先验, 但多视角一致性弱 |
| GS 参数空间扩散 | DiffGS, GaussianAnything | GS 参数 latent | 原生 3D 一致, 但训练成本高 |
| 多视角 latent 联合去噪 | DSplats, DiffSplat | 多视角图像 latent | 兼顾 2D 先验与 3D 一致, 但仍需大量 3D 数据 |
| 前馈重建加扩散增强 | GIFSplat, Leveling3D | ViT 特征或渲染图像 | 前馈高效, 但主要面向重建增强而非可控编辑 |

### 2.2 对本方案最关键的启发

1. **DriveEditor** 证明了"重建式训练可统一覆盖多种编辑"。训练时只学重建, 推理时通过输入构造切换删除、插入、替换、重定位。
2. **RegNeRF / FreeNeRF / 3DGIC**: 不要把模型自己渲染出的 novel view 当 GT, 用真实观测视图之间的对应关系 + 几何正则更稳。
3. **LDM / Latent Flow Matching / RAE / DINO-SAE / Perceiver IO / DUNE** 共同支持: 生成应发生在可解码、可抗噪、可服务多下游头的 latent 上, 而不是任意 hidden state。
4. **Feature Splatting (Pixel-NeRF / Instant-NGP feature grids / gsplat feature rendering)**: gsplat 原生支持任意通道特征的 α-blend 栅格化, 为"3D → 2D token 空间"提供直接通道, 不必经过 RGB 瓶颈。这是 v4 的技术基石。

### 2.3 与 DGGT 直接相关的结构结论

当前 DGGT 中, `Aggregator` 已经产出了天然的联合场景状态:

| 流 | 通道 | 代码来源 | 下游 head |
|----|------|----------|-----------|
| `dino_tokens[l]` | `1024` | `dino_token_list[l]` | `instance_head`, `semantic_head` |
| `frame_tokens[l]` | `1024` | `frame_intermediates[l]` | 几何与局部外观 |
| `global_tokens[l]` | `1024` | `global_intermediates[l]` | 时序与全局上下文 |
| `image_tokens[l]` | `3072` | `concat(dino, frame, global)` | `gs_head` |

而 `gs_head` 输出的 `gs_map` 是**像素对齐**的: 每个像素 `(view, y, x)` 生成一粒高斯, `means` 由 depth 反投影得到。这天然建立了 **"高斯 <-> 源 patch token"** 的对应 (一个 14×14 patch 包含 196 粒高斯, 它们共享同一个 aggregator token)。

结论: **每粒高斯可以零成本地继承它源 patch 的 4 层 aggregator token 作为 F_g**, 编辑任务无需学习新的 "Gaussian → feature" 编码器。

---

## 三、方案设计

### 3.1 设计原则

1. 精确几何编辑在 3DGS 空间完成, 位置由 3D bbox 严格确定。
2. **不做 Pass 2**: 编辑场景不再被渲染回 RGB 去过 Aggregator。
3. `z_edited` 通过 Feature Splatting 从 3D 高斯场直接构造, 保留资产深度、形状、appearance 的精确 3D 信息。
4. 整个方法仍保持**单一模型**: `DINO + Aggregator + tokenizer + scene_flow + dense heads` 主干共享。
5. 用连续软掩码 + per-token 噪声调度 + 编辑类型 coverage 指示编辑意图, 而不是二值 mask。
6. 解码后复用原有 dense heads, 但对 M_dest 区提供高分 DPT 旁路和 gs_map late fusion。
7. 新增模块尽量小且可独立训练: `FeatureSplatter`, `SoftMaskBuilder`, `HighResBypass`, `PerTokenNoiseScheduler` 都是薄封装。

### 3.2 整体流程

```text
Pass 1 (scene): images -> Aggregator -> img_tok_clean_4, agg_clean_4, dino_clean_4
                       -> z_clean = tokenizer.encode(img_tok_clean_4)
                       -> gs_map_clean, depth_clean, cameras
                       -> G_original (pixel-aligned 高斯, 每粒带 source_view + source_patch 指针)

3DGS Edit: G_original, edit_instructions -> G_edited_kept, G_deleted, G_edited_asset
           每粒高斯带 edit-type tag

Asset Pass (per-object, per-frame, Waymo-coord 渲染):
  对每个资产 k ∈ {0..K-1}:
    用 Waymo GT 相机 C_waymo[n] + Waymo bbox B_k[n] 在 N 帧上 rasterize G_asset_k
      -> I_asset[k] [N, 3, H, W] (纯黑背景, 2D 与 GT 精确对齐)
    Aggregator(I_asset[k], batch=N) -> F_g_lut_asset[k] [N, 1369, 3072] × 4 层
  T_w2d = umeyama(C_waymo, cameras_dggt)  # Waymo→DGGT 全局对齐
  G_asset_placed_dggt[k] = transform(G_k_per_frame, T_w2d)  # 后续 splat / 覆盖图用

F_g 赋值 (指针语义, 不存 3072-d 特征本身):
  kept 高斯   : (src_kind='scene', object_id=-1, view_n=source_view_n, patch_idx=source_patch_idx)
                 -> lookup F_g_lut_scene[view_n, patch_idx]
  asset 高斯 k: (src_kind='asset', object_id=k,  view_n=asset_view_n,  patch_idx=asset_patch_idx)
                 -> lookup F_g_lut_asset[k][view_n, patch_idx]
  deleted 高斯: 不参与 F_g splat, 仅用于渲染 D_map

Feature Splat:
  splatted_tok_4_low  = FeatureSplatter(G_kept∪G_asset, F_g, cameras, H=148, W=148, pool_to=37)
  splatted_tok_4_high = FeatureSplatter(..., H=296, W=296)  # 高分旁路

Coverage Maps (3 类 α 渲染, 全分辨率 518×518):
  K_map = render_alpha(G_kept)
  D_map = render_alpha(G_deleted_in_target_view)  # 删除目标在目标视角的 alpha
  I_map = render_alpha(G_asset)

Soft Masks (面积下采样到 37×37, 归一化):
  M_preserve_soft = K/(K+D+I+ε)
  M_source_soft   = D/(K+D+I+ε)
  M_dest_soft     = I/(K+D+I+ε)

Scaffold: depth_edited, alpha_edited, K/D/I_map, dynamic_prior -> pack to scaffold_feat

z_edited 构造:
  z_splat = tokenizer.encode(splatted_tok_4_low)
  z_init = M_preserve_soft·z_clean + M_source_soft·ε_noise + M_dest_soft·(z_splat + σ·ε_noise)

SceneFlow (cond: z_clean, scaffold_feat, F_asset_tokens via cross-attn, soft_masks, t_tok):
  z_hat = scene_flow(z_init, ...)

Decode: z_hat -> tokenizer.decode -> img_tok_hat_4 -> reattach_special -> split {dino,frame,global}
                                                                       -> img_tok_hat_all, agg_hat_all, dino_hat_all

Dense Heads (with high-res bypass for M_dest):
  gs_map_flow     = gs_head(img_tok_hat_all, bypass=splatted_tok_4_high, bypass_mask=I_map)
  gs_map_asset    = rasterize_asset_direct(G_edited_asset, cameras, 518×518)
  gs_map_final    = gs_map_flow·(1-τ) + gs_map_asset·τ    # τ = M_dest_hires · σ(gs_conf)·羽化
  depth, depth_c  = depth_head(agg_hat_all, bypass=scaffold_hires, bypass_mask=K+I)
  dynamic_conf    = instance_head(dino_hat_all)
  pose_enc        = camera_head(agg_clean_all)  # 锚定 Pass-1

Render for loss: 3DGS -> RGB, 与 GT 图像算 render loss
```

### 3.3 Feature Splatting 与 F_g

**核心定义**: `F_g ∈ R^{4 × 3072}` 是每粒高斯在**源 aggregator pass 的 patch token** 的直接复本。

| 高斯类型 | 源 | F_g 来源 |
|---------|----|----|
| Kept | Pass-1 scene | `img_tok_clean_4[layer][source_view_n, source_patch_idx]` |
| Inserted (object k) | Asset Pass (object k) | `F_g_lut_asset[k][layer][asset_view_n, asset_patch_idx]` |
| Deleted | 无 | 不参与 splat, 仅贡献 `D_map` |

**存储**: 不按高斯独立存, 而是**源 patch 共享**。每粒高斯只存 `(source_kind, object_id?, view_n, patch_idx)` 指针。F_g LUT 由两部分组成:
- Scene LUT: `img_tok_clean_4`, 大小 `N × 1369 × 4 × 3072 × 2 B ≈ 130 MB` (N = 帧数)。
- Per-object Asset LUT dict `{k: F_g_lut_asset[k]}`, 每个约 `130/N × N = 130 MB`? 实际上每资产 `N × 1369 × 4 × 3072 × 2 B`, K 资产合计 `K × 130 MB`。K ≤ 3 时 ≤ 400 MB, 可驻留; 超过时按目标 chunk。

Splat 阶段按指针 gather (scene 和 asset 用同一套 gather kernel, 仅 LUT 不同)。

---

**Asset Pass 细节 (Waymo-coord 渲染, per-frame, per-object)**:

相对 v4.0 的三个关键修订:

1. **用 Waymo 真值相机 + Waymo-coord 3D bbox 直接 rasterize**, 绕开 "Waymo→DGGT 坐标系转换 + DGGT 预测相机" 的两级近似。`I_asset` 在 2D 像素空间与 GT 场景图像**逐像素精确对齐**, 不被 DGGT `camera_head` / `depth_head` 的误差放大。
2. **Per-frame N 帧渲染 (取消 v4.0 的 `S_a = 1~2` 限制)**: 场景 N 帧, Asset Pass 也渲 N 帧, 每帧用该时刻的 Waymo 相机 + 资产该时刻 3D bbox (支持动态目标)。
3. **Per-object 独立画布**: 场景含 K 个资产 → K 套 N 帧渲染, 各自独立过 Aggregator, F_g LUT 按 `object_id` 组织。

---

**两空间解耦 (回答为什么 Waymo 渲染可行)**:

| 空间 | 位置 / 相机 | 用途 |
|---|---|---|
| **Waymo GT 空间** | Waymo GT 相机 `C_waymo[n]` + Waymo GT bbox `B_k[n]` | Asset Pass **渲染 + Aggregator 输入**, 2D 精确对齐 GT 图像 |
| **DGGT 空间** | `camera_head` 预测相机 + 资产 Gaussian 在 DGGT 坐标 | **Feature Splat + I_map / K_map / D_map coverage**, 与场景 kept 高斯同坐标系 |

二者由**语义指针** `(object_id, asset_view_n, asset_patch_idx)` 桥接:

- 指针决定"用哪一个 3072-d 资产外观 token" (在 Waymo 渲染空间里计算)。
- Feature Splat 决定"把这个 token 放到目标视角的哪个 2D 位置" (在 DGGT 空间里计算)。
- 两个 2D 空间的几像素偏差对 Aggregator 抽取的**资产外观语义**不构成威胁: 指针仍指向正确资产 token, splat 仍按 DGGT-coord Gaussian 几何定位。
- 副产物: Asset Pass 不再依赖 DGGT 自预测相机的精度, Asset Pass 的 2D 质量上限等于 GT Waymo 渲染质量, 与 v3 render-encode 或 v4.0 DGGT-cam 渲染相比明显更高。

**Asset Gaussian 的 DGGT-coord 位置**: 通过 Waymo↔DGGT 全局相似变换 `T_w2d` 从 Waymo bbox 变换得到。T_w2d 用 Pass-1 预测相机 vs Waymo GT 相机做 Umeyama 对齐 (SE(3) + scale), 每条序列估算一次。训练时 `C_waymo` 直接可用; 推理时若无 GT, 仍可用 `camera_head` 预测相机 + 资产源视角假定, 或 `T_w2d = I` 直接重合 (高精度场景下 DGGT 相机通常已与 Waymo 相机近似同参数)。

---

**流程** (Waymo 相机 `C_waymo[n]` 为每帧姿态, 资产 `k` 的 3D bbox `B_k[n]` 对应 N 帧, 资产 3DGS `G_asset_k`):

```python
I_asset        = {}   # dict[k] -> [N, 3, H, W]  (Waymo 相机渲染, 纯背景)
A_asset        = {}   # dict[k] -> [N, 1, H, W]
F_g_lut_asset  = {}   # dict[k] -> 4 × [N, 1369, 3072]
ptr_asset      = {}   # dict[k] -> per-Gaussian (view_n, patch_idx, visible_mask)

for k in range(K):   # per-object, 独立画布
    G_k_per_frame = place_asset_at_bbox_waymo(G_asset_k, B_k, N)   # N 帧 Waymo-coord 姿态
    rgbs, alphas = [], []
    for n in range(N):
        rgb_n, alpha_n, _ = gsplat.rasterization(
            means=G_k_per_frame[n].means,     # Waymo-coord
            quats=..., scales=..., opacities=..., colors=G_asset_k.rgb,
            viewmats=C_waymo[n].viewmat, Ks=C_waymo[n].K,
            width=W, height=H, render_mode="RGB+ED",
        )
        rgbs.append(rgb_n * alpha_n)   # 纯黑背景
        alphas.append(alpha_n)
    I_asset[k] = torch.stack(rgbs)        # [N, 3, H, W]
    A_asset[k] = torch.stack(alphas)      # [N, 1, H, W]
    # Aggregator per object (batch=N)
    _, img_tok_asset_all_k, _, _, _ = self.aggregator(I_asset[k].unsqueeze(0))
    F_g_lut_asset[k] = select_patch_pyramid(img_tok_asset_all_k, levels, patch_start_idx)
    # 指针: 每粒 Gaussian 在每帧 Waymo 相机下投影, 记录 patch idx 与可见性
    ptr_asset[k] = annotate_asset_patches_per_frame(
        G_asset_k, B_k, C_waymo, N, patch_grid=(37, 37), occlusion_test=True,
    )

# 资产 Gaussian 进 DGGT-coord, 供后续 Feature Splat / I_map coverage 使用
T_w2d = estimate_umeyama_align(C_waymo, cameras_dggt)       # 训练: GT 相机; 推理: 预测相机
G_asset_placed_dggt = {
    k: transform_gaussians_per_frame(G_k_per_frame, T_w2d)
    for k in range(K)
}
```

---

**为什么每帧独立渲染 (per-frame N)**:

- 动态目标: 资产在每帧的位置 (Waymo 3D bbox) 不同 (车辆轨迹), 必须按每帧 bbox 渲染。
- 静态资产: 相机运动也让资产跨帧落在不同 patch, per-frame 投影给出正确指针。
- 跨帧一致性由 Aggregator 自身 **global cross-frame attention** 处理, 与 scene Pass-1 行为对称 —— tokenizer/flow 下游逻辑零改动。
- 取代 v4.0 的 `S_a = 1~2`: Waymo clip 短 (N ≈ 4-8), 帧间相机角度变化小; self-replacement 训练中 Trellis 资产有 Waymo 多视角 + 跨帧监督, 360° 重建质量较高; 剩余 off-axis 伪影由 `M_dest_soft` + flow harmonization 吸收。
- 新资产 (单视角源) 推理时, 可按 "本帧视角 vs 资产源视角余弦" 给每帧 token 附 confidence weight (不阻断渲染, 只降权跨帧聚合), 作为可选稳健性机制。

---

**为什么每目标分画布 (per-object K 次 Aggregator)**:

- 纯净: 目标 A 与 B 不在同一 Aggregator self-attention 内混杂, 避免 "A 的轮子 + B 的车身" 被 global attn 串成一个共同表示。
- 遮挡隔离: 2D 上两资产互相遮挡时, 分画布保证各自**完整**被编码, 不因互相遮挡丢失特征。
- 与 Trellis 独立重建一致: 资产间本无共同物理耦合 (阴影 / 反射 / 光照交互) —— 这些交互应由 flow 的 harmonization 生成, 不该在 Aggregator token 里预编码。
- 成本: 每资产 1 次 Aggregator 前向 (batch=N, 图相对小), K ≤ 3 时总共 K 次; 工程上可 batch 成 `[K·N, 3, H, W]` 单次 forward 再按 K 切回, K 较大时也不会阻塞。
- 工程: `F_g_lut_asset` 按 dict 存; 指针扩展 `(object_id, asset_view_n, asset_patch_idx)`; 下游 `F_asset_kv` (cross-attn K/V) 按 object 维度 concat 后 flatten 成 `[B, K·N·1369, 3072]`。

---

**I_map 与覆盖图 (仍在 DGGT 空间)**:

- `I_map` / `K_map` / `D_map` 三者需要同一 2D 空间做归一化, 因此仍用 DGGT 相机渲染:
  - `I_map = render_alpha_per_object(G_asset_placed_dggt, cameras_dggt, H=518, W=518).sum(K)` (跨对象并合 α)。
  - 单对象 α-map 也保留 `I_map_k`, 供 per-object 的 flow cross-attn bias 细粒度使用。
- 此处不用 Waymo 相机: 若用 Waymo 渲染 I_map, 它就与 K/D_map 对不齐, 归一化无意义。
- Asset Pass 的 Waymo 渲染只用于**Aggregator 特征抽取**; 覆盖图 / Feature Splat / 最终 3DGS 渲染损失**全部仍在 DGGT 空间**。

**Feature Splatting 实现** (统一处理 scene kept + 多 object asset):

```python
def feature_splat(
    gaussians,       # 每粒带 (src_kind, object_id, view_n, patch_idx) 指针, DGGT-coord 几何
    F_g_lut_scene,   # [N, 1369, 4, 3072]           Pass-1 scene tokens
    F_g_lut_asset,   # dict[k] -> [N, 1369, 4, 3072]  per-object asset tokens
    cameras_dggt,    # DGGT 相机 (camera_head 预测), 与 scene kept Gaussian 同坐标系
    H, W,            # 中间分辨率 (148 低 / 296 高)
    pool_to,         # 37 for tokenizer input, None for DPT bypass
):
    out_per_layer = []
    for layer in range(4):
        # 统一 gather: src_kind 分派到 scene LUT 或 object LUT
        colors = gather_features(
            gaussians.src_kind,        # 'scene' or 'asset'
            gaussians.object_id,       # -1 for scene; k for asset
            gaussians.view_n,
            gaussians.patch_idx,
            F_g_lut_scene[layer],
            {k: F_g_lut_asset[k][layer] for k in F_g_lut_asset},
        )   # [N_gauss, 3072]
        feat_map = gsplat.rasterization(
            means=gaussians.means_dggt, quats=gaussians.quats_dggt,   # DGGT-coord 几何
            scales=gaussians.scales, opacities=gaussians.opacities,
            colors=colors,                     # 3072 通道
            viewmats=cameras_dggt.viewmats, Ks=cameras_dggt.Ks,
            width=W, height=H, render_mode="RGB",
        )                                      # [N_frames, H, W, 3072]
        if pool_to is not None:
            feat_map = F.avg_pool2d(
                feat_map.permute(0,3,1,2),
                kernel_size=H // pool_to, stride=H // pool_to,
            ).permute(0,2,3,1)                 # [N_frames, pool_to, pool_to, 3072]
        out_per_layer.append(feat_map)
    return out_per_layer
```

- 3072 通道一次 splat 内存压力大, 实际实现**按 512 通道 chunk**, 串行 6 次, 峰值内存 = N × 512 × 2 B ≈ 1 GB (N=1M)。
- 梯度只反传到 `colors` (即 F_g LUT, 进而到 Aggregator), **不反传到高斯几何参数** (它们由 DGGT 已训练的 head 给定, 视作常量)。

**低覆盖 / 空洞 token**:

- 某些 patch 完全无高斯覆盖 (如 delete 区, 或 disocclusion 区), α≈0, splat 输出≈0。
- 这些 patch 自动被 `M_source_soft` 捕获 (K+D+I ≈ D 或全 0 时归一化分母加 ε)。
- z_init 在这些 token 用纯噪声, 由 flow 生成。

### 3.4 JointSceneTokenizer

**架构保持 v3 不变**: encoder `4 × [B,S,P,3072] -> [B,S,P,768]`, decoder 反向。

**关键变化**: 新增 **T0.5 splatted-token 自一致训练阶段**。

**T0.5 训练流程** (无需任何编辑标注):

1. 取无编辑场景样本, 过 Pass-1 得 `img_tok_clean_4` 和 `gs_map_clean`。
2. 从 `gs_map_clean` 组装 `G_original`, 不做任何编辑。
3. 给每粒高斯赋 F_g = 它源 patch 对应的 `img_tok_clean_4` token。
4. Feature Splat (同 3.3) -> `splatted_tok_4` (无编辑情况下应高度接近 `img_tok_clean_4`)。
5. 训练目标:
   - `L_self`: `tokenizer.decode(tokenizer.encode(splatted_tok_4)) ≈ splatted_tok_4`
   - `L_anchor`: 经 tokenizer 往返后过 dense heads (冻结), 输出 `gs_map / depth / dynamic` 应与 `gs_map_clean / depth_clean / ...` 一致

T0.5 完成后 tokenizer 在 splatted 和 real token 两个分布上都有良好表现, 消除 OOD 风险。

**配置**: `C_scene=768, hidden_dim=896, num_block_pairs=3, num_heads=14` (不变)。

### 3.5 SceneFlowMatching (重设计)

#### 3.5.1 输入接口

| 输入 | 形状 | 用法 |
|------|------|------|
| `z_t` | `[B, S, P, 768]` | 当前噪声 latent |
| `t_tok` | `[B, S, P, 1]` | per-token 噪声级别 |
| `z_clean` | `[B, S, P, 768]` | clean context, AdaLN 或 residual 条件 |
| `scaffold_feat` | `[B, S, P, 768]` | 编辑 scaffold (见 3.6) 压缩 |
| `soft_masks` | `[B, S, P, 3]` | (M_preserve, M_source, M_dest)_soft |
| `F_asset_tokens` | `[B, S·P_a, 3072] -> K/V` | cross-attention K/V, asset DGGT pass 输出 |

#### 3.5.2 Block 结构

```text
for each block_pair in flow_stack (num_block_pairs=3):
    x = x + frame_self_attn(x, pos=2D_RoPE, t_tok_modulation)
    x = x + global_cross_frame_attn(x)
    x = x + masked_cross_attn_to_asset(x, F_asset_tokens, bias=M_dest_soft)
    x = x + cond_mlp(x, z_clean, scaffold_feat, soft_masks)
```

- **Masked cross-attention**: flow token 作 Q, asset tokens 作 K/V。attention logits 加上 `log(M_dest_soft + ε)` 偏置, preserve/source 主导 token 的 attention 被压低, 避免它们误被资产拉偏。
- **AdaLN 模调**: `t_tok` 和 `soft_masks` 通过 MLP 生成 scale/shift, 作用在每个 block 的 LN 之后, 实现 per-token 可变噪声强度。

推荐配置:
- `token_dim=768, hidden_dim=1024, num_block_pairs=3, num_heads=16`
- Sampling: Flow Matching, 6 步 Euler (可调)

#### 3.5.3 Per-token 噪声调度

```text
base_t ∈ [0, 1]   # 全局时间
edit_weight[i] = M_source_soft[i] + M_dest_soft[i] · γ_dest       # γ_dest ∈ [0.3, 0.5]
preserve_weight[i] = M_preserve_soft[i]
t_tok[i] = clip( base_t · edit_weight[i] + (1 - preserve_weight[i]) · ε_floor, 0, 1 )
```

- `γ_dest` 控制 dest 区的噪声强度 (SDEdit 风格部分加噪)。
- `ε_floor` 让 preserve 区也有**微量噪声**扰动, 避免梯度完全消失; 但远小于 edit 区。

**训练策略**: 前 5K 步用全局 `t` warm-up; 之后平滑过渡到 per-token `t`, 避免冷启动不稳。

#### 3.5.4 z_edited 初始化

```text
ε_noise ~ N(0, I)
z_init = M_preserve_soft · z_clean
       + M_source_soft  · ε_noise
       + M_dest_soft    · (z_splat + σ_partial · ε_noise)
```

`σ_partial ≈ 0.3` (dest 保资产粗结构, 允许协调 noise)。

#### 3.5.5 Edit-State Routing (沿用 v3 三专家)

三类 residual experts 作用在每个 block 之后:

| expert | 主激活 | 更新强度 |
|--------|--------|----------|
| `preserve` | M_preserve_soft 主导 | 0.25 |
| `harmonize` | M_dest_soft 主导 | 1.0 |
| `generate` | M_source_soft 主导 | 1.5 |

Router 直接使用 `soft_masks` + `scaffold` 的 K/D/I coverage 差异作为 state 特征, **不需要额外学习 routing map** (软路由权重即 `soft_masks` 的归一化变体)。

### 3.6 Coverage Maps, 软掩码与 Scaffold

**全分辨率 Coverage (渲染自 G_edited, 518×518)**:

| Map | 来源高斯 | 语义 |
|-----|---------|------|
| `K_map` | G_edited_kept (保留) | 保留结构的 α 覆盖 |
| `D_map` | G_deleted (仅用于 coverage 渲染) | 原目标的 α 覆盖, 在**目标视角**重投影 |
| `I_map` | G_edited_asset (插入) | 新资产的 α 覆盖 |

三者对应**三类编辑语义**, 相加不一定等于 1 (重叠区可 > 1, 可正确归一化)。

**软掩码 (37×37, 供 flow 使用)**:

```text
K_soft = area_pool(K_map, to=37×37)
D_soft = area_pool(D_map, to=37×37)
I_soft = area_pool(I_map, to=37×37)
N = K_soft + D_soft + I_soft + ε
M_preserve_soft = K_soft / N
M_source_soft   = D_soft / N
M_dest_soft     = I_soft / N
```

连续 `[0,1]`, 和为 1, 子 patch 精度由面积池化保留。

**高分版 (供 DPT 旁路和 gs_map late fusion)**:

`K_map / D_map / I_map` 保留 518×518, 在 DPT refinenet 和 gs_map 融合阶段使用。

**Scaffold Pack** (37×37, 供 flow 使用):

```text
scaffold_channels = cat(
    D_edited_lowres,         # 1ch, edited depth 下采样
    A_edited_lowres,         # 1ch, edited alpha
    K_soft, D_soft, I_soft,  # 3ch
    dynamic_prior_lowres,    # 1ch, instance_head 输出
)                            # 7 ch 总计
scaffold_feat = MLP(scaffold_channels) -> [B, S, P, 768]
```

高分 scaffold (518×518) 同时保留供下游用。

### 3.7 Dense Heads 高分旁路 与 gs_map Late Fusion

#### 3.7.1 DPT 旁路

`gs_head` 和 `depth_head` 是 DPT 结构: 4 个 refinenet 从低分到高分递进融合 (37×37 -> 74 -> 148 -> 296 -> 518)。

为小目标保真, 在 refinenet1 和 refinenet2 (高分侧) 插入**资产高分 splat 特征作为残差**:

```python
# refinenet1 输入 (148×148 分辨率)
rn1_input = dpt_upsample(prev_layer) + α1 · gate(I_map_148) · splatted_tok_high_148
# refinenet2 输入 (256×256 左右)
rn2_input = dpt_upsample(rn1_output) + α2 · gate(I_map_256) · splatted_tok_high_256
```

- `gate(I_map)` = soft sigmoid around I_map 阈值, 只在资产区激活。
- `α1, α2` 为学习标量 (初始化为小值, 让 DPT 先起主导)。
- 仅对 `gs_head` 开启 (appearance); `depth_head` 的 bypass 用 scaffold depth。

#### 3.7.2 gs_map Late Fusion

```text
gs_map_flow      = gs_head(img_tok_hat_all, bypass, ...)            # [B, S, 518, 518, 14]
gs_map_asset_dir = rasterize_asset_gs_params(G_edited_asset, cameras, 518×518)
                                                                     # 直接 α-blend 资产 gs 参数
τ = I_map_hires · sigmoid(gs_conf_flow) · feather_kernel              # 羽化避免硬边
gs_map_final     = gs_map_flow · (1 - τ) + gs_map_asset_dir · τ
```

- M_dest 高置信度处用 asset 直出 (几何纹理精确)。
- 边界 / 阴影 / 交互区用 flow 输出 (harmonization)。
- 羽化核 (高斯 blur on I_map_hires, σ≈3 px) 避免硬边撕裂。

#### 3.7.3 depth_head / point_head / instance_head

- `depth_head`: 用 scaffold depth 做 bypass (I_map + K_map 区域)。
- `point_head`: 不做 bypass (由 depth_head 衍生)。
- `instance_head`: 直接从 dino_hat_all 输出, 不做 bypass (dynamic_conf 来自资产的 dynamic tag, 融入 scaffold)。

### 3.8 必须满足的约束

1. **四层联合处理**: flow 和 tokenizer 必须覆盖 `[4, 11, 17, 23]` 四层, 与 DPT 多尺度对齐。
2. **Asset DGGT Pass 不可省**: 它提供 asset F_g 和 F_asset_tokens。
3. **Asset Pass 必须在 Waymo 真值相机 + Waymo bbox 下渲染** (训练时直接可用; 推理退化为 DGGT 预测相机), 与 Feature Splat 的 DGGT-空间几何由语义指针桥接, 不由像素对齐桥接。
4. **Asset Pass per-frame per-object**: N 帧每帧单独渲染 (动态目标需要), K 个对象各自独立画布 (避免特征串扰)。
5. **覆盖图 (K/D/I_map) 和 Feature Splat 统一在 DGGT 相机下执行**, 保证归一化软掩码一致。
6. **连续软掩码**: 不使用硬二值 `M_source / M_dest`。
7. **Per-token 噪声调度**: 不使用全局 t。
8. **高分 scaffold + gs_map late fusion 是小目标保真的硬性保障**, 不能只靠 37×37 flow。
9. **F_g 存储用指针 + LUT**, 不按高斯独立存, 否则显存爆炸。
10. **Feature Splat 梯度只反传到 F_g**, 不反传到高斯几何参数。

---

## 四、3DGS 编辑接口

### 4.1 编辑指令

```python
class EditInstruction:
    action: str        # "delete" | "insert" | "replace" | "reposition"
    bbox: Tensor       # [T, 8, 3], 原位置 / 目标位置
    bbox_new: Tensor   # [T, 8, 3], reposition 时使用
    asset: Asset       # insert / replace / reposition 的资产 (3DGS + 参考图)
```

### 4.2 四种操作的统一表示

| 操作 | 3DGS 操作 | `G_kept` | `G_deleted` | `G_asset` |
|------|-----------|---------|-------------|-----------|
| 删除 | 删除目标高斯 | 剩余原高斯 | 被删的原高斯 | 无 |
| 插入 | 放入新资产 | 全部原高斯 | 无 | 新资产高斯 |
| 替换 | 删旧目标 + 放新资产 | 剩余原高斯 | 旧目标高斯 | 新资产高斯 |
| 重定位 | 原目标高斯迁移到新位置 | 剩余 + 迁移后的原目标 (F_g 用 Pass-1) | 旧位置空壳 (仅用于 D_map) | 视同 kept 但同时激活 I_map 触发 dest 协调, F_g 仍用 Pass-1, F_asset 为迁移后 isolated render |

统一原则:

1. `M_source_soft` 自动来自 `D_map`, 覆盖背景补全任务。
2. `M_dest_soft` 自动来自 `I_map`, 覆盖外观协调任务。
3. 重定位在 source 和 dest 区同时出现, 由三路高斯天然处理。

---

## 五、训练设计

### 5.1 两种训练模式

| 模式 | 输入构造 | GT | 覆盖推理任务 |
|------|---------|----|-------------|
| 协调训练 | 删除动态车辆, 用同一车辆的 Trellis 资产重放 | 原始视频与原始场景状态 | insert / replace / reposition 的 dest |
| 补全训练 | 在无车背景区域合成 deletion-shaped holes | 原始视频与原始场景状态 | delete / reposition 的 source |

**协调训练**:

1. 原始视频 Pass 1, 得 `G_original`、相机位姿、动态目标掩码。
2. 选一个动态目标, 用**它自己的** Trellis 资产替换 (self-replacement, 保证 GT 可比)。
3. 资产独立渲染 + Asset Pass 获得 `img_tok_asset_4`。
4. M_dest 训练外观协调能力。

**补全训练**:

1. 只选 vehicle-free 背景区域。
2. 合成与真实删除任务同分布的 holes (见 5.2)。
3. 删除对应高斯并渲染 `D_map`。
4. M_source 训练背景补全能力。

### 5.2 hole 合成 (不变)

| 策略 | 比例 | 目的 |
|------|------|------|
| vehicle track replay mask | 70% | 洞的形状/尺度/透视/时序接近真实车辆删除 |
| core + ring mask | 20% | 边界、残影、接地区域鲁棒性 |
| generic irregular / rectangle | 10% | 避免过拟合单一形状 |

要求: 显式覆盖 truncation、partial occlusion、far-object、multi-hole; 以 3DGS cutout 为主; 时间维度平滑。

### 5.3 损失函数

```text
L_total =
    λ_flow       * L_flow           # latent flow matching 主损失
  + λ_render     * L_render         # 原始观测视角的渲染重建
  + λ_lpips      * L_lpips          # 感知质量
  + λ_xview      * L_xview          # 真实视图间深度引导重投影
  + λ_auxgeom    * L_auxgeom        # 虚拟视角几何正则 + floater 抑制
  + λ_3d         * L_3d             # pixel-aligned scene state 直接 3D 约束
  + λ_route      * L_route          # 反向路由惩罚 (弱监督)
  + λ_preserve   * L_preserve       # 非编辑区保持
  + λ_tok_recon  * L_tok_recon      # tokenizer 真 token 重建 (T0)
  + λ_splat_cons * L_splat_cons     # tokenizer splatted token 自一致 (T0.5+)
  + λ_asset_id   * L_asset_id       # M_dest 区资产身份保持
  + λ_attn_ent   * L_attn_ent       # cross-attention entropy 正则
```

**新增损失**:

- `L_tok_recon` (T0 主): tokenizer 往返真 token 的 L2 + head-anchor。
- `L_splat_cons` (T0.5): tokenizer 对 splatted token 的自一致重建 + head-anchor。
- `L_asset_id` (T1+): `LPIPS(render(gs_map_final) restricted to I_map, I_asset_gt)`, 防 flow 把资产平滑。
- `L_attn_ent` (T1+): masked cross-attn 的 attention 分布 entropy 下限, 避免注意力坍塌。

**监督原则**:
- `L_xview` 只在真实视图对之间计算, 不用 pseudo novel-view GT。
- `L_auxgeom` 只提供几何正则, 不对虚拟视角做 RGB 回归。
- `L_3d` 约束 pixel-aligned scene state, 不只是高斯外观。

**推荐权重**:

```text
λ_flow = 1.0
λ_render = 1.0
λ_lpips = 0.1
λ_xview = 0.25
λ_auxgeom = 0.05
λ_3d = 0.1
λ_route = 0.05
λ_preserve = 0.5
λ_tok_recon = 1.0   (仅 T0)
λ_splat_cons = 1.0  (仅 T0.5)
λ_asset_id = 0.2
λ_attn_ent = 0.01
```

### 5.4 训练阶段

**Stage T0: Tokenizer 真 token 预训练** (已完成或沿用 v3)

- 冻结 Aggregator + dense heads。
- 训练 `tokenizer.encoder / decoder`。
- 目标: `L_tok_recon` (真 token 往返) + head anchors + noisy decoding。

**Stage T0.5: Splatted-token 自一致适配 (新增, 必做)**

- 冻结 Aggregator + dense heads。
- Tokenizer 全部解冻 (小学习率 1e-4)。
- 用 Pass-1 的 gs_map 重建 G_original (无编辑)。
- 每粒高斯赋 F_g = Pass-1 对应 patch 的 token。
- Feature Splat -> splatted_tokens。
- 目标: `L_splat_cons` + head anchors on splatted。
- 3-5 epochs 足够 (预期 splatted ≈ real, 适配成本小)。
- **关键**: 这一步让 tokenizer 在 T1 开始时不因 splat 分布偏移而崩溃。

**Stage T1: SceneFlow 训练**

- 冻结 Aggregator + tokenizer encoder。
- 前期只解冻 tokenizer decoder 的 `layer_heads + local_refine`, 后期可放开。
- 训练 `scene_flow + FeatureSplatter (极少参数) + HighResBypass + AssetEncoder`。
- 编辑模式: 协调训练 + 补全训练 (按 5.1 混合, 50/50)。
- Warm-up 策略:
  - 前 2K 步用 GT 3D bbox 生成 oracle soft mask (避免早期 scaffold 渲染噪声)。
  - 前 5K 步用全局 t, 之后过渡到 per-token t_tok。
  - 前 3K 步关闭 masked cross-attn (所有 attention), 只靠 condition concat, 避免 attn 冷启动。
- Router 在 2K 步后开启, `L_route` warm-up。

**Stage T2: 小学习率联合微调**

- tokenizer 最后一层 cross-scale + scene_flow + decoder + `gs_head` 最后一层联合微调。
- 学习率 5e-6, 各任务平衡采样。
- `L_asset_id` 权重适度提升, 保证资产身份。

### 5.5 推荐训练配置

| 项 | 推荐值 |
|----|--------|
| 优化器 | AdamW + cosine decay, β=(0.9, 0.95) |
| T0.5 学习率 | 1e-4 (tokenizer) |
| SceneFlow (T1) 学习率 | 2e-4 |
| Dense heads (T1) 学习率 | 5e-6 |
| T2 学习率 | 5e-6 for all |
| Batch | 4 clips/GPU × 5 帧/clip × S=4 views |
| 精度 | BF16 |
| 训练资源 | 8 × A100/H100 |
| T0.5 时长 | 3-5 epochs |
| T1 时长 | 30-50 epochs |
| T2 时长 | 5-10 epochs |

---

## 六、代码落点

### 6.1 `dggt/models/vggt.py`

需要新增/调整的组件:

1. `self.scene_tokenizer` (保留, 仅新增 T0.5 训练 hook)
2. `self.scene_flow` (相对 v3 重写, 支持 cross-attn to asset + per-token t + soft mask AdaLN)
3. `self.asset_encoder` — **注意**: 这里不是 "仅 DINO", 而是**在 Asset Pass 中直接复用 `self.aggregator`**。无需独立模块, 只需调用 `self.aggregator(I_asset)` 即可。
4. `self.feature_splatter` (新增, 薄 gsplat 封装 + chunked gather)
5. `self.soft_mask_builder` (新增, 渲染 K/D/I_map + 面积池化 + 归一化)
6. `self.scaffold_packer` (新增, pack 7 通道 scaffold 到 768 维)
7. `self.high_res_bypass` (新增, DPT refinenet 残差注入 + gs_map late fusion 封装)
8. `self.per_token_scheduler` (新增, 构造 t_tok 和 z_init)

原有 `forward()` **保持不变**, 仅新增 `forward_edit()`。

### 6.2 方案级伪代码

```python
def forward_edit(
    self,
    images,                    # [B, S, 3, H, W], 原始场景
    edit_instructions,         # EditInstruction list
    asset_gaussians=None,      # 预加载的资产 3DGS (optional)
    mode="inference",
):
    levels = [4, 11, 17, 23]

    # === Pass 1: 唯一的 scene Aggregator 前向 ===
    agg_clean_all, img_tok_clean_all, dino_clean_all, _, patch_start_idx = \
        self.aggregator(images)
    img_tok_clean_4 = select_patch_pyramid(img_tok_clean_all, levels, patch_start_idx)

    # clean latent + 原始高斯场
    z_clean = self.scene_tokenizer.encode(img_tok_clean_4)
    gs_map_clean, _ = self.gs_head(img_tok_clean_all, images, patch_start_idx)
    cameras = pose_enc_to_cameras(self.camera_head(agg_clean_all)[-1])
    G_original = assemble_gaussians_with_pointers(
        gs_map_clean, depth=self.depth_head(...)[0], cameras=cameras,
    )  # 每粒高斯带 (source_view, source_patch_idx)

    # === 3DGS 确定性编辑 ===
    G_kept, G_deleted, G_asset_placed, edit_meta = self.gs_editor(
        G_original, edit_instructions, cameras, asset_gaussians,
    )

    # === Asset Pass (per-object, per-frame, Waymo-coord 渲染) ===
    # 入参 edit_instructions 应提供每帧每目标的 Waymo bbox 和 Waymo GT 相机
    cams_waymo = get_waymo_cameras(edit_instructions, N)            # training: GT; inference: 用预测兜底
    F_g_lut_asset = {}   # dict[k] -> list of 4 × [B, N, 1369, 3072]
    I_asset       = {}
    ptr_asset     = {}
    G_asset_dggt  = {}
    for k, asset_info in enumerate(edit_instructions.assets):       # K ≤ 3 typical
        G_k_waymo = place_asset_per_frame_waymo(
            asset_info.gaussians, asset_info.bbox_per_frame, N,
        )
        rgb_list, alpha_list = [], []
        for n in range(N):
            rgb_n, alpha_n, _ = gsplat_rasterize(
                G_k_waymo[n], cams_waymo[n], H=W_img, W=W_img, render_mode="RGB+ED",
            )
            rgb_list.append(rgb_n * alpha_n)                        # 纯黑背景
            alpha_list.append(alpha_n)
        I_asset[k] = torch.stack(rgb_list)                          # [N, 3, H, W]
        A_asset_k  = torch.stack(alpha_list)
        _, img_tok_asset_all_k, _, _, _ = self.aggregator(I_asset[k].unsqueeze(0))
        F_g_lut_asset[k] = select_patch_pyramid(
            img_tok_asset_all_k, levels, patch_start_idx,
        )  # 4 × [B, N, 1369, 3072]
        # per-frame 指针 (Waymo 相机下投影), 含可见性
        ptr_asset[k] = annotate_asset_patches_per_frame(
            asset_info.gaussians, asset_info.bbox_per_frame, cams_waymo, N,
            patch_grid=(37, 37), occlusion_test=True,
        )
        # Waymo→DGGT 变换, 供 Feature Splat / I_map coverage
        T_w2d = estimate_umeyama_align(cams_waymo, cameras)
        G_asset_dggt[k] = transform_gaussians_per_frame(G_k_waymo, T_w2d)
    if len(F_g_lut_asset) == 0:
        F_g_lut_asset = None
        G_asset_dggt  = None

    # === F_g 指针统一 (scene + 多 object asset, 全部在 DGGT-coord 几何下) ===
    all_gaussians = merge_pointers(
        G_kept,                          # ptr: (scene, -1, src_view_n, src_patch_idx)
        {k: (G_asset_dggt[k], ptr_asset[k]) for k in F_g_lut_asset} if F_g_lut_asset else {},
    )  # 每粒 Gaussian 含 (src_kind, object_id, view_n, patch_idx) + DGGT-coord 几何
    F_g_lut_scene = img_tok_clean_4     # 4 × [B, N, 1369, 3072]
    # F_g_lut_asset 保留 dict 形式, 按 object_id 索引

    # === Feature Splatting (两个分辨率, 用 DGGT 相机) ===
    splatted_tok_4_low = self.feature_splatter(
        all_gaussians, F_g_lut_scene, F_g_lut_asset, cameras,
        H=148, W=148, pool_to=37,
    )  # 4 × [B, N, 1369, 3072]
    splatted_tok_4_high = self.feature_splatter(
        all_gaussians, F_g_lut_scene, F_g_lut_asset, cameras,
        H=296, W=296, pool_to=None,
    )  # 4 × [B, N, 296, 296, 3072]  (高分旁路, 用最深两层即可)

    # === Coverage maps + soft masks + scaffold (仍在 DGGT 空间) ===
    # I_map 按 object 合并, 并保留 per-object I_map_k 供 flow 细粒度 cross-attn 使用
    K_map, D_map, I_map, I_map_per_obj = self.soft_mask_builder.render_coverage(
        G_kept, G_deleted, G_asset_dggt, cameras, H=518, W=518,
    )
    M_preserve_soft, M_source_soft, M_dest_soft = self.soft_mask_builder.pool_and_normalize(
        K_map, D_map, I_map, target_grid=37,
    )  # [B, S, 1369, 1]  each
    scaffold_hires = render_edited_scaffold(all_gaussians, cameras)
    scaffold_feat = self.scaffold_packer(
        scaffold_hires, K_map, D_map, I_map, dynamic_prior=..., target_grid=37,
    )  # [B, S, 1369, 768]

    # === z_init 软掩码加权 ===
    z_splat = self.scene_tokenizer.encode(splatted_tok_4_low)
    z_init = self.per_token_scheduler.compose_z_init(
        z_clean, z_splat, M_preserve_soft, M_source_soft, M_dest_soft,
        sigma_partial=0.3,
    )

    # === F_asset_tokens for cross-attention K/V (跨对象 concat) ===
    if F_g_lut_asset is not None:
        # 每对象 [B, N, 1369, 3072] × 4 层, 拼接成 [B, K·N·1369, 3072]
        F_asset_kv = concat_then_flatten_asset_tokens(
            F_g_lut_asset, layer_select="topmost",
        )  # [B, K*N*1369, 3072]
    else:
        F_asset_kv = None

    # === Per-token noise schedule ===
    t_tok = self.per_token_scheduler.build_t_tok(
        base_t=None if mode == "train" else 0.0,
        M_preserve_soft=M_preserve_soft,
        M_source_soft=M_source_soft,
        M_dest_soft=M_dest_soft,
    )

    # === SceneFlow ===
    z_hat = self.scene_flow(
        z_init,
        t_tok=t_tok,
        cond_clean=z_clean,
        cond_scaffold=scaffold_feat,
        cond_asset_kv=F_asset_kv,
        soft_masks=torch.cat([M_preserve_soft, M_source_soft, M_dest_soft], dim=-1),
        mode=mode,
    )

    # === Tokenizer decode + 重连 special tokens + 通道拆分 ===
    img_tok_hat_4_patch = self.scene_tokenizer.decode(z_hat)
    img_tok_hat_4 = reattach_special_tokens(
        template=img_tok_clean_all, levels=levels,
        patch_start_idx=patch_start_idx, patches=img_tok_hat_4_patch,
    )
    dino_hat_4, frame_hat_4, global_hat_4 = split_joint_channels(
        img_tok_hat_4, dims=[1024, 1024, 1024],
    )
    agg_hat_4 = [torch.cat([frame_hat_4[i], global_hat_4[i]], dim=-1)
                 for i in range(len(levels))]
    img_tok_hat_all = replace_selected_levels(img_tok_clean_all, levels, img_tok_hat_4)
    dino_hat_all    = replace_selected_levels(dino_clean_all,    levels, dino_hat_4)
    agg_hat_all     = replace_selected_levels(agg_clean_all,     levels, agg_hat_4)

    # === Dense heads with high-res bypass ===
    predictions = {}
    gs_map_flow, gs_conf_flow = self.high_res_bypass.run_gs_head(
        self.gs_head, img_tok_hat_all, images, patch_start_idx,
        bypass_tokens=splatted_tok_4_high, bypass_mask_hires=I_map,
    )
    gs_map_asset_dir = rasterize_asset_gs_params_full(G_asset_placed, cameras, 518, 518) \
                       if G_asset_placed is not None else None
    gs_map_final = self.high_res_bypass.late_fuse_gs_map(
        gs_map_flow, gs_map_asset_dir, I_map, gs_conf_flow, feather_sigma=3,
    )
    predictions["gs_map"] = gs_map_final
    predictions["gs_conf"] = gs_conf_flow

    depth, depth_conf = self.high_res_bypass.run_depth_head(
        self.depth_head, agg_hat_all, images, patch_start_idx,
        bypass_scaffold=scaffold_hires, bypass_mask_hires=K_map + I_map,
    )
    predictions["depth"] = depth
    predictions["depth_conf"] = depth_conf

    pts3d, pts3d_conf = self.point_head(agg_hat_all, images, patch_start_idx)
    predictions["world_points"] = pts3d
    predictions["world_points_conf"] = pts3d_conf

    dynamic_conf, _ = self.instance_head(dino_hat_all, images, patch_start_idx)
    predictions["dynamic_conf"] = dynamic_conf

    # 相机锚定 Pass-1 (编辑不改变 ego pose)
    predictions["pose_enc"] = self.camera_head(agg_clean_all)[-1]

    return predictions
```

### 6.3 对原有 dense heads 的影响

- `GaussianHead`, `DPTHead` **主干不改**, `gs_head` 和 `depth_head` 通过 `HighResBypass` 薄包装接收 bypass 信号。
- `JointSceneTokenizer` 架构不变, 新增 T0.5 训练 hook。
- `SceneFlowMatching` 相对 v3 大幅重写 (cross-attention, per-token t, soft mask AdaLN)。
- 新增独立模块: `FeatureSplatter`, `SoftMaskBuilder`, `ScaffoldPacker`, `HighResBypass`, `PerTokenNoiseScheduler`。

---

## 七、推理流程

### 7.1 删除

1. Pass 1 得 `G_original`、相机位姿。
2. 删除目标高斯 -> `G_kept`, `G_deleted`。
3. **无 Asset Pass** (没有新插入), F_g 只来自 Pass-1。
4. `D_map` 渲染出来, 得 `M_source_soft`; `I_map = 0`, `M_dest_soft = 0`。
5. Feature Splat + flow: source 区纯噪声初始化, flow 生成背景。

### 7.2 插入

1. 资产 3DGS 放到场景目标位置, 得 `G_asset_placed`。
2. **Asset Pass**: 资产 isolated render -> Aggregator -> `img_tok_asset_4` + F_asset_kv。
3. 资产高斯赋 asset_patch 指针, F_g 来自 asset tokens。
4. `I_map` 渲染, 得 `M_dest_soft`; `D_map = 0`。
5. Flow: dest 区 SDEdit partial noise, 做 harmonization。

### 7.3 替换

- delete 旧目标 + insert 新资产。
- `D_map` 和 `I_map` 同时出现, `M_source_soft` + `M_dest_soft` 都激活。

### 7.4 重定位

- 提取原目标高斯, 迁移到新位置 (几何上视为 "kept-moved", F_g **仍用 Pass-1 token** —— 目标身份与内在外观未变, 无需再过一次 Asset Aggregator)。
- 旧位置由 `D_map` 提供 coverage; 新位置**同时进入 `K_map` 与 `I_map`**:
  - `K_map` (作为迁移后"kept"): 让保留损失和高分 scaffold 感知新几何。
  - `I_map` (作为 dest): 触发 `M_dest_soft`, 激活光照/接地/阴影 harmonization。
- `M_source_soft` 在旧位置 (背景补全), **`M_dest_soft` 在新位置启用** (光照/接地/周边遮挡的协调式 harmonization)。这是相对 v3 的关键变化: 即便身份未变, 新位置的**几何-光照耦合仍需要 flow 做协调**, 否则会出现"目标是对的, 但与新环境的阴影/反射不一致"。
- Cross-attn 条件: 把**迁移后目标的 isolated render** (直接用 Pass-1 高斯子集 + 新位姿 rasterize, 无需新 Asset DGGT Pass) 构成 `F_asset_tokens`, 让 flow 在 dest 区参考目标的外观细节做微调。
- 可选阀门: `edit_instruction.reposition_harmonize_strength ∈ [0, 1]` 缩放 `M_dest_soft` 的强度 (默认 1.0; 光照变化弱的场景下可降至 0.3)。

> 与 insert/replace 的区别:
> - F_g 不来自 Asset DGGT Pass, 而是来自 Pass-1 —— 身份像素级保留。
> - F_asset_tokens 的作用从"提供新目标身份" 退化为"提供迁移后视角的外观参考"。
> - 即便 flow 在 dest 上"协调过度", 也不会破坏目标身份, 因为 gs_map late fusion 会用 asset-direct rasterize (即迁移后目标) 把身份主结构盖回。

### 7.5 推理开销

| 方法 | 推理时间 (N=4, K=1) | 前馈次数 |
|------|----------------------|----------|
| DGGT 纯重建 | ~0.5s | 1 scene DGGT |
| FlowDGGT v3 (render+encode) | ~1.5s | 2 scene DGGT + 1 flow |
| **FlowDGGT (splat, Waymo-coord asset)** | **~1.3s** | 1 scene DGGT + K asset DGGT (batch=N per object) + 1 flow |
| SceneDirector | ~15min | - |
| DriveEditor | ~2min | - |

Asset DGGT 输入是 isolated render (单目标+透明背景), 每对象 batch=N 帧, attention 复杂度远低于全场景; K 较大时多对象可 batch 成 `[K·N, 3, H, W]` 单次 forward。Feature Splat 在 DGGT 相机下统一执行, 综合开销相对 v3 略降。

---

## 八、实验方案

### 8.1 数据与评估

- 训练: Waymo Open 训练集 + Trellis 资产
- 验证: WOD validation
- 泛化: nuScenes 零样本

| 任务 | 指标 |
|------|------|
| 删除 | FID, FVD, CLIP-I, 多视角一致性 |
| 插入 | FID, FVD, 多视角一致性, 3D bbox 精度 |
| 重定位 | FID, FVD, ATE |
| 无编辑重建 | PSNR, SSIM, LPIPS |
| 下游任务 | 3D 检测 mAP |
| **小目标插入专项** | 远距车辆 (10-20 px) 插入成功率, LPIPS on asset region |

### 8.2 关键消融

v3 继承的:

| 实验 | 验证目标 |
|------|----------|
| 单掩码 vs 双掩码 | 双掩码是否提升删除与重定位 |
| `gs-only` latent vs `joint scene` latent | 联合场景隐空间是否必要 |
| `C_scene = 512 / 640 / 768 / 896` | latent 容量权衡 |
| 无 `F_asset` | 资产条件是否必要 |
| 无 `L_xview` / `L_auxgeom` / `L_3d` | 各类 3D 监督是否必要 |
| 无 router / hard vs soft router | 软路由是否更稳 |
| 只最深层 vs 四层联合 | DPT 多层特征是否必须 |
| 去掉跨帧 attention | 时序一致模块是否必要 |

v4 新增/关键:

| 实验 | 验证目标 |
|------|----------|
| **v3 render-encode vs v4 Feature Splat** | **核心对比, 验证 splat 避免渲染瓶颈** |
| **F_g 来源: aggregator token vs DINO-only vs learned MLP** | F_g 定义最关键 |
| **Asset Pass (isolated) vs 资产直接放场景渲染再编码** | 验证 asset 特征域一致性 |
| **硬二值 mask vs 连续软 mask** | 软掩码是否提升边界与小目标 |
| **全局 t vs per-token t_tok** | per-token noise 是否必要 |
| **有无 DPT 高分旁路** | 小目标是否被恢复 |
| **有无 gs_map late fusion** | 资产身份保真 |
| **T0.5 splat 适配 vs 跳过 T0.5** | tokenizer 域适配是否必要 |
| **Cross-attn to F_asset vs conditional concat** | attention 比 concat 是否更精细 |

### 8.3 关键定性: 小目标专项

远距车辆 (10-20 px) 插入:
- v3 预期成功率 < 50% (渲染伪影 + 37×37 编码丢失)。
- v4 预期成功率 > 80% (DPT 旁路 + gs_map late fusion 绕开 latent 瓶颈)。

---

## 九、风险与应对

### 9.1 训练风险

| 阶段/步骤 | 风险 | 应对 |
|-----------|------|------|
| F_g LUT 显存 | scene + K asset 各 4 层 × N × 1369 × 3072 BF16 ≈ (1 + K) × 130 MB | K ≤ 3 下 ≤ 520 MB 可驻留; K 更多时按 object chunk |
| Feature Splat 峰值显存 | 3072 通道 × N_gauss=1M 一次 splat 太大 | 按 512 通道 chunk, 6 次串行, 峰值 ~1 GB |
| Splat 反向传播不稳 | 3072 通道梯度易爆 | 梯度只到 F_g LUT, 不到几何; chunk 内独立 backward |
| Splatted token OOD | tokenizer 在 splat 分布上崩溃 | **T0.5 自一致训练强制执行** |
| Asset token 域偏移 | 资产在黑背景下的 token 与真实场景光照不同 | T1 warm-up 用 "资产插入 vehicle-free 场景" 样本 |
| Waymo↔DGGT 坐标对齐漂移 | `T_w2d` 估计不准导致 asset Gaussian 在 DGGT 空间位置偏移 → splat 错位 | 训练时用 GT Waymo cams 解析对齐; 推理时 Umeyama 拟合全序列 (rigid+scale); 少量残差由 M_dest_soft + flow harmonization 吸收; 监控 `mean reprojection error` > 2 px 时告警 |
| 多对象 Aggregator 多次前向 | K 对象 × N 帧 forward 开销 | batch 成 `[K·N, 3, H, W]` 单次 forward, 最后按 K 切回 |
| Per-frame 指针在遮挡帧无效 | 资产 Gaussian 在某帧被自遮挡 | `occlusion_test` 打 `visible_mask=False`, 该帧 splat 不 gather 该 patch, 回退最近可见帧 token (ptr `view_n` 改指向邻近可见帧) |
| Off-axis 资产渲染伪影 (通用插入) | 推理时新资产帧角度远离源视角, 渲染糊 | 按 "本帧视角 vs 源视角余弦" 给每帧 token 附 confidence weight; M_dest_soft partial noise 吸收伪影 |
| Per-token t 训练不稳 | preserve 区 t≈0 梯度弱 | 前 5K 步全局 t, 平滑过渡; ε_floor 保微量噪声 |
| Cross-attn 被资产主导 | flow 忽略 z_clean 全抄资产 | `L_attn_ent` 正则 + `L_preserve` 强约束 + M_dest_soft bias 限制范围 |
| Soft mask 源自 gs_map | G_edited 渲染质量差 -> mask 噪声 | T1 前 2K 步用 GT 3D bbox oracle mask; 之后过渡 |
| T2 全联合微调过拟合 | 多目标联合训练易漂 | 小学习率 5e-6, 平衡采样, 定期 val 检查 |
| 小目标 gradient 信号弱 | 1 个 token 的 loss 被淹没 | `L_asset_id` on I_map 区域 + `L_render` 空间加权 (小目标区域权重 up) |

### 9.2 推理风险

| 步骤 | 风险 | 应对 |
|------|------|------|
| 3D 编辑定位不准 | bbox 漂移导致 mask 错 | 沿用 SceneBoxRefiner + AssetPoseRefiner |
| 资产 Trellis 纹理差 | F_asset 信息密度不足 | Asset 渲染后做 anti-alias + perceptual upsample |
| 大 disocclusion | splat 无覆盖 -> splatted token 全 0 | 该区自动归入 M_source_soft, flow 从噪声生成 (正确行为) |
| 小目标 splat 采样不足 | 远距资产 < 1 patch | 高分 splat 148/296 + 面积 pool + DPT 旁路 + gs_map late fusion |
| Multi-edit 重叠 | 同 token 既 source 又 dest | 软掩码归一化天然处理; 训练时禁止生成重叠样本 |
| gs_map late fusion 边界撕裂 | τ 硬切 | I_map 高斯羽化 (σ≈3 px) |
| Camera pose 与 Asset Pass 不一致 | 资产 patch 指针错 | Asset Pass 用 Waymo GT 相机渲染 + 投影指针 (训练); 推理时 `T_w2d = I` 或 Umeyama 拟合。所有 DGGT-空间 splat / coverage / render 统一用 `camera_head` 预测相机 |
| Waymo cams 推理不可得 | 部署场景没有 Waymo GT cameras | 回退: Asset Pass 用 `camera_head` 预测相机渲染; 已知精度损失会被 M_dest_soft 吸收, 作为 graceful degradation 而非硬失败 |
| Asset 光照不匹配场景 | flow 协调不够 | M_dest 用 SDEdit partial noise (σ=0.3), 留足协调空间 |
| 动态 scene 跨帧不一致 | flow 每帧独立生成导致闪烁 | global cross-frame attn + `L_xview` 保证时序一致 |

### 9.3 数据风险

| 风险 | 应对 |
|------|------|
| Trellis 资产与真实车辆域差 | 协调训练 **大比例 self-replacement** (同 ID 资产), F_asset 渲染扰动增强 |
| 补全训练 hole 分布偏 | 70/20/10 策略 (见 5.2) |
| 预处理慢 | 离线缓存 asset 3DGS + asset DGGT tokens (per asset × canonical view set) |
| Router 伪标签有噪 | 只监督高置信 core, 先 warm-up 再开启 `L_route` |

---

## 十、论文主张

本文的核心主张可以压缩为五点:

1. **Render-Free Editing via Feature Splatting**: 编辑信息从 3D 高斯场直接通过 Feature Splatting 抵达 token 空间, **不经过 RGB 渲染瓶颈**, 从物理上消除 "Pass-2 渲染伪影 → 编码污染" 的信息损失链路。
2. **Joint Scene-Latent Correction**: 生成变量是 DGGT 的 joint scene latent `z_scene`, 不是图像域后处理, 也不是单 head 修补。
3. **Soft-Mask Per-Token Flow Matching**: 用连续软掩码和 per-token 噪声调度替代二值 mask, 解决低分辨率 latent 无法指示 sub-patch 编辑意图的问题。
4. **Multi-Resolution Fidelity Loop**: 高分 splat + DPT 多尺度 side-injection + gs_map late fusion 的三重机制保障小目标在 37×37 latent 瓶颈外恢复。
5. **Self-Supervised 3D Consistency**: 训练依靠原始视频、真实视图间重投影一致性和几何正则, 不依赖 pseudo novel-view GT。

---

## 十一、参考文献

### 核心

- DGGT
- VGGT
- SceneDirector
- DriveEditor

### 扩散与 3DGS

- DiffGS
- DiffSplat
- GaussianAnything
- GIFSplat
- Leveling3D
- GaussianEditor

### latent 与 flow

- LDM
- Flow Matching in Latent Space
- RAE
- DINO-SAE
- Perceiver IO
- DUNE

### Feature Rendering / Splatting

- gsplat (官方特征 rasterize 支持)
- Pixel-NeRF (pixel-aligned features for neural radiance)
- Feature-3DGS (feature field rendering with Gaussians)

### 3D 一致监督

- RegNeRF
- FreeNeRF
- 3DGIC
