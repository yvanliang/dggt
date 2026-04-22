# FlowDGGT 实施方案 v4（Feature-Splat 版，分步 LLM 可执行）

> 对齐 [docs/research_plan.md](/home/dancer/code/dm/dggt/docs/research_plan.md) v4 架构：Pass-1 + Asset Pass + Feature Splatting + 软掩码 + per-token 噪声 + DPT 高分旁路 + gs_map late fusion。**不再做 v3 的 "render edited scene → Pass 2" 链路**。
>
> 本文件面向"分步实施"：每个 Phase 给出【目标 / 输入状态 / 输出状态 / 要改的文件 / 要新建的文件 / 对外 API / smoke test / 通过标准】，后续 LLM 可按 Phase 顺序 bring-up，不需要再回读 research_plan 即可动手。

---

## 0. 当前已完成的基线（直接复用，不改动）

| 条目 | 位置 | 状态 |
|---|---|---|
| DGGT 预训练权重 | `/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt` | done |
| Waymo 预处理（images/depth/instances/appearances/object_id_map） | `/data/disk2/lyy_dataset/waymo_dggt_edit_processed/{training,validation}` | done |
| Sky mask / fine dynamic mask | 同上目录 `sky_masks/`、`fine_dynamic_masks/{all,human,vehicle}` | done |
| Edit metadata / manifest / 资产缓存 | `/data/disk2/lyy_dataset/waymo_edit_cache/` + `reference_images_selected/` + `asset_gaussians/<raw_id>.ply` | done |
| `WaymoEditDataset` | [datasets/waymo_edit_dataset.py](/home/dancer/code/dm/dggt/datasets/waymo_edit_dataset.py) | done（支持 views=1/3、online mode-A/B） |
| `JointSceneTokenizer` 模块 | [dggt/models/joint_scene_tokenizer.py](/home/dancer/code/dm/dggt/dggt/models/joint_scene_tokenizer.py) 已挂到 [dggt/models/vggt.py:27](/home/dancer/code/dm/dggt/dggt/models/vggt.py) | done |
| Tokenizer T0 训练脚本 | [train_tokenizer.py](/home/dancer/code/dm/dggt/train_tokenizer.py) `TokenizerTrainWrapper` + `L_tok_rec / L_tok_cos / L_head_anchor / L_render_anchor / L_noisy / L_lat_stat` | done（T0 checkpoint 已产出） |
| Token utilities | [dggt/utils/tokens.py](/home/dancer/code/dm/dggt/dggt/utils/tokens.py) — `select_patch_pyramid / reattach_special_tokens / replace_selected_levels / split_joint_channels / split_special_and_patch` | done |
| 确定性 GS 编辑器基线 | [dggt/utils/gaussian_edit.py](/home/dancer/code/dm/dggt/dggt/utils/gaussian_edit.py) — `build_clean_scene_state / estimate_scene_alignment / localize_objects / apply_mode_a`（Sim3 对齐 / ObjectLocalizer / SceneBoxRefiner / AssetPoseRefiner 已内置） | done |
| Mode-A 推理脚本（baseline） | [inference_mode_a.py](/home/dancer/code/dm/dggt/inference_mode_a.py) — Pass-1 → align → localize → yaw refine → apply_mode_a → 渲染 clean/deleted/asset/edited | done（**仅 views=1**；作为编辑正确性金标准保留） |

**硬约束**：

- `VGGT.forward()` 必须保持不变；新链路以 `forward_edit()` 形式叠加。
- T0 tokenizer 权重必须可直接加载进新路径，不允许重新训练。
- 所有新模块默认读取 `/data/disk2/lyy_dataset/waymo_dggt_edit_processed/` 与 `waymo_edit_cache/`。
- 新训练脚本以 [train_tokenizer.py](/home/dancer/code/dm/dggt/train_tokenizer.py) 的 DDP / autocast / logging 骨架为模板克隆，不重造轮子。
- `inference_mode_a.py` 作为 reference baseline 冻结；Feature-Splat 分支另开文件，不破坏它。

## 0.5 当前 Mode-A 数据语义约定

为避免再次混淆，Waymo 编辑数据里的“目标在这一帧是否有 2D 框”和“这一帧是否适合作为编辑锚点”必须彻底分开。

### 0.5.1 两套语义

| 语义 | 含义 | 来源 | 用途 |
|---|---|---|---|
| `bbox present` | 该目标在该帧/该视角确实有有效投影框，`boxes_by_view_transfer/raw/model` 都存在 | metadata 里的 `boxes_by_view_*` | 几何定位、语义匹配、protected boxes、逐帧执行编辑 |
| `bbox editable` | 该目标在该帧/该视角满足 metadata 的“适合作为编辑锚点”规则 | `bbox_editable_by_view` | 训练采样、决定一个 object 是否进入本条样本的 editable 集合 |

### 0.5.2 为什么要拆开

`bbox_editable_by_view` 包含了额外过滤，不等价于“目标不可见”：

- 小框过滤：目标虽然在画面中，但 transfer box 太小，会被置为 `False`
- 元数据侧遮挡过滤：某些帧会被标成“不适合作为锚点”，但目标仍然在图中

因此：

- `bbox_editable_by_view=False` 不能解释成“这一帧没有框”
- 推理删除阶段如果拿 `bbox_editable_by_view` 驱动逐帧定位，会漏掉本来应该继续编辑的 follower frame

### 0.5.3 Dataset 导出约定

`WaymoEditDataset` 必须同时导出下面两组字段：

- `object_bbox_present_mask[_selected]`
- `object_front_bbox_present_mask[_selected]`
- `object_bbox_editable_mask[_selected]`
- `object_front_bbox_editable_mask[_selected]`

硬约束：

- 不再保留 `object_bbox_valid_mask` / `object_front_bbox_valid_mask` 这类旧字段
- 新代码必须只读 `present/editable` 命名的字段
- metadata / manifest cache 不做旧 schema 兼容；字段切换后必须整包重建

### 0.5.4 Anchor Frame / Follower Frame 策略

对一条 4 帧 inference / training sample，object-level 编辑资格与 frame-level 执行资格分开处理：

1. object 是否进入 `editable_object_indices`

- 仍按 `editable` 语义决定
- 只要该 object 在当前 sample 的任意一帧、任一视角存在 `bbox_editable=True`，它就进入本条样本的 editable object 集合

2. 某一帧是否对这个 object 执行删除/插入

- 按 `present` 语义决定，而不是按 `editable`
- 一旦 object 已经进入本条样本的 editable 集合，则对该 sample 内所有 `bbox_present=True` 的帧都尝试执行编辑

3. 非锚点帧的角色

- `bbox_editable=True` 的帧是 anchor frame：用于保证这个 object 在本条样本中“值得编辑”
- `bbox_present=True && bbox_editable=False` 的帧是 follower frame：只要目标仍在画面中，就应该继续编辑，以保证时序一致性

4. 安全约束

- follower frame 不做“强制编辑”
- 如果该帧虽然 `present=True`，但语义/深度/几何证据不足，允许该帧局部定位失败并跳过，不能为了追求全帧一致而盲删背景

### 0.5.5 Inference 侧强制规则

Mode-A 删除 / 插入链路必须遵守：

- object 选择：看 `editable_object_indices`
- 逐帧 view 选择：看 `bbox_present_mask`
- protected boxes：看 `bbox_present_mask`
- 调试图里如果要画“目标在哪”，默认画 `present` 框；如果要画“哪些帧是锚点”，单独画 `editable` 标记，不允许再共用一个 mask 字段

---

## 1. 整体实施路径

Phases 按依赖拓扑排序，每 Phase 产出可独立 smoke test 的代码切片：

```
P1 编辑器库化（views=3 扩展）
      │
      ├── P2 FeatureSplatter            ── 独立数值测试
      │
      ├── P3 SoftMaskBuilder + ScaffoldPacker
      │
      ├── P4 Asset Aggregator Pass（Waymo-coord, per-frame, per-object）
      │
      ├── P5 T0.5 splatted-token 自一致训练（新脚本）
      │
      ├── P6 SceneFlowMatching（cross-attn + AdaLN + per-token t）
      │
      ├── P7 HighResBypass + gs_map Late Fusion
      │
      ├── P8 FlowDGGT 顶层 forward_edit() 串联
      │
      ├── P9 T1 SceneFlow 训练（Mode A + Mode B 混合）
      │
      ├── P10 T2 联合微调
      │
      └── P11 推理脚本（inference_flow_edit.py）+ 评估
```

每 Phase 的 "输入状态"都等于前置 Phase 的"输出状态"的并集。后续 LLM 可以只读相应 Phase 即可开工。

---

## 2. 模块坐标系与命名约定

| 名字 | 目的 | 绝对路径 |
|---|---|---|
| `FlowDGGT` | 顶层 wrapper，对外暴露 `forward_edit()` | `dggt/models/flow_dggt.py`（新增） |
| `FeatureSplatter` | gsplat 通道 splat + pooling 的薄封装 | `dggt/models/feature_splatter.py`（新增） |
| `SoftMaskBuilder` | 渲染 K/D/I_map + 面积池化 + 归一化 | `dggt/models/soft_mask.py`（新增） |
| `ScaffoldPacker` | 7 通道 scaffold → 768-d token | `dggt/models/scaffold.py`（新增） |
| `PerTokenNoiseScheduler` | `t_tok` + `z_init` 构造 | `dggt/models/per_token_noise.py`（新增） |
| `SceneFlowMatching` | flow 主体 + 内置 router | `dggt/models/scene_flow.py`（新增） |
| `HighResBypass` | DPT 残差注入 + late fuse | `dggt/models/high_res_bypass.py`（新增） |
| `AssetAggregatorPass` | 调度 Asset 渲染 + aggregator 前向 | `dggt/models/asset_pass.py`（新增） |
| `GaussianSceneEditor` | 把 `gaussian_edit.py` 已有函数打包成 nn-friendly 的 `nn.Module`（仅 forward-only，无参数） | `dggt/models/gaussian_scene_editor.py`（新增，薄封装） |
| `tools/` 训练 / 推理脚本 | `train_tokenizer_t05.py / train_scene_flow_t1.py / train_joint_t2.py / inference_flow_edit.py` | 项目根目录 |

**禁止**：

- 不要新增与 `JointSceneTokenizer` 并列的 "gs-only latent" 分支。
- 不要修改 `dggt/heads/` 下的 dense heads 主干，`HighResBypass` 必须是**外部包装**，只接收 head 的中间张量做残差注入。
- 不要在 `VGGT.forward()` 里加任何编辑逻辑，所有编辑相关代码走 `FlowDGGT`。

---

## 3. Phase 详述

### Phase 1: 编辑器库化 + views=3 打通

**目标**：把 `inference_mode_a.py` 里散落的"Pass-1 → clean scene state → align → localize → apply_mode_a → render"流程收敛进一个可被训练 loop 调用的 `GaussianSceneEditor` 模块，并补齐 `views=3`。

**输入状态**：P0 完成。`inference_mode_a.py` 仅支持 `views=1`。

**输出状态**：
- `dggt/models/gaussian_scene_editor.py` 提供一个 `GaussianSceneEditor(nn.Module)`，`forward(sample, predictions, asset_bank, edit_instruction) → EditedSceneBundle`。
- `EditedSceneBundle` 统一字段：`G_kept, G_deleted, G_asset_per_object, cameras_dggt, cameras_waymo, T_w2d, per_gauss_pointers, edit_meta`，`views=1` 与 `views=3` 一致。
- `inference_mode_a.py` 改造为调用该模块（保持 baseline 行为，输出 diff 可接受 ≤ 1e-4）。

**要改的文件**：
- [inference_mode_a.py](/home/dancer/code/dm/dggt/inference_mode_a.py)：`_render_clean_with_dggt / _render_edited_sequence_with_dggt / _render_asset_sequence_local / _refine_asset_local_yaw_offsets / _composite_asset_over_scene` 里通用的"Pass-1 → editor → 渲染"段落迁到新模块。
- [datasets/waymo_edit_dataset.py](/home/dancer/code/dm/dggt/datasets/waymo_edit_dataset.py)：如有 views=1 硬编码的 return shape 假设，提升到 `views in {1, 3}`。

**要新建的文件**：
- `dggt/models/gaussian_scene_editor.py`
- `dggt/utils/asset_bank.py`（资产 `.ply` 加载 + LRU 缓存，迁出 `_load_asset_gaussians`）
- `tests/test_gaussian_scene_editor.py`（smoke）

**对外 API**：
```python
class GaussianSceneEditor(nn.Module):
    def build_clean_bundle(self, sample, predictions) -> CleanSceneState:   # 透传 build_clean_scene_state
    def align(self, sample, clean_state) -> Sim3Transform:                  # 透传 estimate_scene_alignment
    def localize(self, sample, clean_state, alignment) -> list[LocalizedFrameObject]:
    def apply_mode_a(self, clean_state, localized) -> EditedSceneState:
    def forward(self, sample, predictions, asset_bank, edit_instruction) -> EditedSceneBundle:
```
`EditedSceneBundle` 在 `EditedSceneState` 基础上补 `cameras_waymo / T_w2d / per_gauss_pointers`（pointer 形式见 Phase 2）。

**smoke test**：
- `views=1` 下对 10 个 Mode A 样本，新模块与 baseline `inference_mode_a.py` 的 `edited.rgb / delete_mask / asset_bbox_proj` 数值 diff ≤ 1e-4。
- `views=3` 下同样 10 个样本能跑通，无 exception；每帧都拿到 `G_kept / G_deleted / G_asset`；`T_w2d` 的 `mean_alignment_error < 2 px`（否则打印 warning）。

**通过标准**：
- baseline inference_mode_a 在 `views=1` 上视觉输出无回归。
- `views=3` 能在单张 A100 上跑通 sequence_length=4。

---

### Phase 2: FeatureSplatter

**目标**：实现"每粒高斯带指针 → 按指针从 LUT gather 3072-d token → gsplat rasterize → pool to 37×37"的主干，作为 Feature Splatting 的唯一通道。

**输入状态**：P1 的 `per_gauss_pointers`（每粒高斯 `(src_kind, object_id, view_n, patch_idx, visible_mask)`）。LUT 暂用 `F_g_lut_scene = img_tok_clean_4` 桩输入。

**输出状态**：`FeatureSplatter(...)` 返回 `4 × [B, S, P=1369, 3072]`（低分 pool_to=37）与可选 `4 × [B, S, H_high, W_high, 3072]`（高分）。梯度只到 LUT，不到几何。

**要新建的文件**：
- `dggt/models/feature_splatter.py`
- `tests/test_feature_splatter.py`

**对外 API**：
```python
class FeatureSplatter(nn.Module):
    def __init__(self, channels=3072, chunk_channels=512):
    def forward(
        self,
        gaussians_dggt,          # dict: means, quats, scales, opacities (DGGT-coord)
        pointers,                # (src_kind, object_id, view_n, patch_idx) per-gauss
        lut_scene,               # 4 × [B, N_scene, 1369, 3072]
        lut_asset_dict,          # dict[k] -> 4 × [B, N_asset_frames, 1369, 3072]
        cameras_dggt,            # viewmats, Ks
        H, W,                    # 148 or 296
        pool_to=37,              # None = no pooling
    ) -> list[torch.Tensor]:     # len=4
```

**关键实现细节**：
- 3072 通道按 `chunk_channels=512` 切 6 次 rasterize，串行累加，峰值显存 ~1 GB。
- gather kernel 不分 `src_kind`：提前把 `lut_scene` 与 `lut_asset_dict` flatten 为一张全局 LUT + offset，指针改写为全局 index。避免 kernel 内 branch。
- `pool_to=37` 时用 `F.avg_pool2d(kernel=H//37)` 做面积池化（不是 interpolate）。
- `visible_mask=False` 的高斯被赋 `opacity=0`（或从 gather batch 中剔除），不贡献 splat。
- 梯度屏蔽：`means/quats/scales/opacities` 全部 `detach()`；`colors` 来自 LUT 保持梯度。

**smoke test**：
- T0.5 预实验：用无编辑场景，splat 得到的 `splatted_tok_4` 与 `img_tok_clean_4` 的 cosine ≥ 0.9。
- backward：`loss = splatted.pow(2).mean()`；`lut_scene.grad.abs().max() > 0`；`gaussians_dggt.means.grad is None`。
- chunked vs full 通道一次 splat 数值 diff < 1e-4。

**通过标准**：
- 单次 forward + backward 显存峰值 ≤ 2 GB（`B=1, S=4, N_gauss=1M, 3072ch`）。
- P40ms < latency < 200ms / clip（A100，`views=3, S=4`）。

---

### Phase 3: SoftMaskBuilder + ScaffoldPacker

**目标**：从 `G_kept / G_deleted / G_asset` 三组高斯渲染 `K/D/I_map` 并导出软掩码 + 7 通道 scaffold。

**输入状态**：P1 的 `EditedSceneBundle`、Pass-1 的 `depth / alpha / dynamic_conf`。

**输出状态**：
- `K_map, D_map, I_map, I_map_per_obj`：`[B, S, 518, 518, 1]`
- `M_preserve_soft, M_source_soft, M_dest_soft`：`[B, S, 1369, 1]`
- `scaffold_feat`：`[B, S, 1369, 768]`
- `scaffold_hires`：`[B, S, 518, 518, 7]`（留给 Phase 7 DPT 旁路）

**要新建的文件**：
- `dggt/models/soft_mask.py`
- `dggt/models/scaffold.py`
- `tests/test_soft_mask.py`

**对外 API**：
```python
class SoftMaskBuilder(nn.Module):
    def render_coverage(self, G_kept, G_deleted, G_asset_dggt_dict, cameras_dggt, H=518, W=518):
        return K_map, D_map, I_map, I_map_per_obj
    def pool_and_normalize(self, K_map, D_map, I_map, target_grid=37, eps=1e-4):
        return M_preserve_soft, M_source_soft, M_dest_soft

class ScaffoldPacker(nn.Module):
    def __init__(self, in_channels=7, out_dim=768):
    def forward(self, scaffold_hires, target_grid=37) -> torch.Tensor:   # [B,S,1369,768]
```

**关键实现细节**：
- `K/D/I_map` 用 gsplat 的 `alpha` 通道单独渲染；`D_map` 渲染的是被删目标在**目标视角**（不是原始视角）的 α。
- `I_map_per_obj` 保留每对象独立 map（供 Phase 6 cross-attn 偏置），`I_map = sum_k I_map_per_obj[k].clamp(0,1)`。
- 归一化：`N = K_soft + D_soft + I_soft + eps`；三路各自 `X_soft / N`。
- 面积池化 kernel=14、stride=14（518 → 37），**不用 interpolate**。
- scaffold 7 通道：`D_edited_lowres, A_edited_lowres, K_soft, D_soft, I_soft, dynamic_prior_lowres, time_index` → MLP 到 768。

**smoke test**：
- 无编辑场景：`K_soft ≈ 1, D_soft ≈ 0, I_soft ≈ 0`（逐 token L1 差 < 0.05）。
- 纯删除：`D_soft > 0` 区与 GT 车辆 mask IoU > 0.6。
- 纯插入：`I_soft > 0` 区与投影 bbox IoU > 0.6。

**通过标准**：三路软掩码和 ≈ 1（逐 token 误差 < 1e-3，除 ε 平滑区外）。

---

### Phase 4: Asset Aggregator Pass（Waymo-coord, per-frame, per-object）

**目标**：按 research_plan §3.3 的"Waymo 真值相机 + Waymo bbox"方案，把资产在 N 帧上独立 rasterize、过 aggregator、抽 4 层 token LUT，并给每粒资产高斯打 `(object_id, asset_view_n, asset_patch_idx, visible_mask)` 指针。

**输入状态**：P1 的 `asset_bank`（预加载的资产 3DGS）、`edit_instruction.bbox_per_frame`（Waymo 坐标）、`cams_waymo[n]`（训练：GT；推理：`T_w2d=I` 或 Umeyama 拟合）。

**输出状态**：
- `F_g_lut_asset: dict[int_k → 4 × [B, N, 1369, 3072]]`
- `ptr_asset: dict[int_k → per-gauss (view_n, patch_idx, visible_mask)]`
- `G_asset_dggt: dict[int_k → per-frame Gaussian in DGGT-coord]`（经 `T_w2d`）
- `I_asset, A_asset: dict[int_k → [B, N, 3/1, H, W]]`（纯背景），保留给 Phase 6 的 F_asset_kv 与 Phase 11 调试

**要新建的文件**：
- `dggt/models/asset_pass.py`
- `tests/test_asset_pass.py`

**对外 API**：
```python
class AssetAggregatorPass(nn.Module):
    def __init__(self, aggregator):      # 共享 self.aggregator 引用
    def forward(
        self, edit_instruction, asset_bank, cams_waymo, cameras_dggt, T_w2d,
        patch_grid=(37, 37), occlusion_test=True,
    ) -> dict:  # {F_g_lut_asset, ptr_asset, G_asset_dggt, I_asset, A_asset}
```

**关键实现细节**：
- **batch 化**：`K` 个对象 × `N` 帧 拼成 `[K*N, 3, H, W]` 单次过 aggregator，然后 reshape 回 dict。
- **指针**：对每粒资产高斯，在每帧 Waymo 相机下做 `P = K @ [R | t] @ means_waymo`，落到 `37×37` 网格；对自遮挡做 depth-test（rasterize 出 α>τ 才算 visible），`visible_mask=False` 的帧回退到最近可见帧的 `(view_n, patch_idx)`。
- **T_w2d**：训练调用 `estimate_scene_alignment` 的底层 Umeyama（已在 `gaussian_edit.py`）；推理可退化为 `T_w2d = I`。
- **资产无时默认空返回**：delete-only 样本里 `K=0`，所有 dict 返回空，下游 `FeatureSplatter` 跳过 asset LUT。

**smoke test**：
- Mode A self-replacement 样本：`I_asset[k]` 与原视频动态车辆区域的像素 alpha 有意义的交叠（IoU>0.3）。
- `T_w2d` reproject 误差：把 `cams_waymo` 的 `ego_pose` 变到 DGGT 后与 `cameras_dggt` 的平移差 < 2 m、旋转差 < 3°。
- Aggregator 输出每层通道 3072、patch 数 1369。

**通过标准**：K=1, N=4 时，Asset Pass 总耗时 < 300 ms（A100, BF16）。

---

### Phase 5: T0.5 Splatted-Token 自一致训练

**目标**：让已训好的 T0 tokenizer 在 "splatted token" 分布上也不崩溃。这是 T1 flow 训练能稳住的必要前置。

**输入状态**：T0 tokenizer 权重 + P2 `FeatureSplatter` + P1 `GaussianSceneEditor`。**不涉及任何编辑语义**，只做无编辑重建链路。

**输出状态**：T0.5 checkpoint（tokenizer 小幅更新，encoder+decoder 皆可训，lr=1e-4）。

**要新建的文件**：
- `train_tokenizer_t05.py`（克隆 [train_tokenizer.py](/home/dancer/code/dm/dggt/train_tokenizer.py) 骨架）

**要复用的文件**：
- `TokenizerTrainWrapper`、`extract_levels`、`load_model_checkpoint`、`reduce_per_sample`、`normalized_token_reconstruction_loss`、`token_cosine_loss`、`latent_stat_loss`、DDP/autocast 部分全部直接 import。

**训练 pipeline（每 step）**：
1. 从 `WaymoEditDataset(edit_mode='clean')` 取 clean clip。
2. 跑 `VGGT` 前向得到 `img_tok_clean_all / gs_map / depth / cameras`。
3. 按 Pass-1 `gs_map + depth` 装配 `G_original`；每粒赋指针 `(scene, -1, source_view, source_patch_idx)`。
4. `FeatureSplatter` 得 `splatted_tok_4`（低分 148→37）。
5. `z = tokenizer.encode(splatted_tok_4)`；`decoded = tokenizer.decode(z)`。
6. 损失：
   - `L_splat_cons = normalized_token_reconstruction_loss(decoded, splatted_tok_4) + 0.2 * token_cosine_loss(decoded, splatted_tok_4)`
   - `L_anchor_splat`：`gs_head / depth_head / point_head / instance_head(decoded重组后)` 输出 vs Pass-1 head 输出（冻结 head）
   - `L_lat_stat`
7. `L = L_splat_cons + 0.5 * L_anchor_splat + 0.01 * L_lat_stat`

**训练时长**：3–5 epochs on training set（Waymo ~700 scenes × sequence_length=4）。

**smoke test**：
- 16 个 clean 样本过拟合：`L_splat_cons < 0.02` 在 <500 步内达到。
- T0.5 checkpoint 在 val split 上 `L_tok_recon`（真 token）不能退化 > 5%（否则说明适配过头）。

**通过标准**：splatted token 的 head anchor L1 相比 T0（未做 splat 适配）下降 ≥ 50%。

---

### Phase 6: SceneFlowMatching

**目标**：实现 research_plan §3.5 的 flow，带 cross-attention to asset、per-token t、soft-mask AdaLN、三专家 routing。

**输入状态**：P2–P4 全部就绪（`z_clean / z_splat / scaffold_feat / soft_masks / F_asset_kv`）。

**输出状态**：`SceneFlowMatching` 可独立单元测试；默认 6 步 Euler 推理，训练按标准 Flow Matching。

**要新建的文件**：
- `dggt/models/scene_flow.py`
- `dggt/models/per_token_noise.py`
- `tests/test_scene_flow.py`

**对外 API**：
```python
class PerTokenNoiseScheduler(nn.Module):
    def build_t_tok(self, base_t, M_preserve, M_source, M_dest, gamma_dest=0.4, eps_floor=0.05) -> Tensor
    def compose_z_init(self, z_clean, z_splat, M_preserve, M_source, M_dest, sigma_partial=0.3) -> Tensor

class SceneFlowMatching(nn.Module):
    def __init__(self, token_dim=768, hidden_dim=1024, num_block_pairs=3, num_heads=16):
    def forward(self, z_t, t_tok, cond_clean, cond_scaffold, cond_asset_kv, soft_masks, mode) -> z_hat
    def sample(self, z_init, cond_*, num_steps=6) -> z_hat    # 推理用
```

**block 结构**（伪代码）：
```python
for _ in range(num_block_pairs):
    x = x + frame_self_attn(x, rope=2d, modulation=AdaLN(t_tok, soft_masks))
    x = x + global_cross_frame_attn(x)
    if cond_asset_kv is not None:
        x = x + masked_cross_attn(x, cond_asset_kv, bias=log(M_dest + eps))
    x = x + cond_mlp(cat([x, z_clean, scaffold_feat]))
    x = x + router_residual(x, soft_masks)   # 三专家按 soft_masks 加权
```

**训练损失（本 Phase 不直接训练，只定义接口）**：`L_flow = MSE(v_pred, v_gt)`，`v_gt` 按 Flow Matching 标准构造。

**smoke test**：
- 单 step forward shape：`[B,S,1369,768] → [B,S,1369,768]`。
- `cond_asset_kv=None` 时 cross-attn 路径被 short-circuit；`num_block_pairs=1` 时参数量 ~ 15M。
- 16 sample 过拟合：构造恒等任务（`z_target=z_clean`），`L_flow` 能降到 < 0.01。

**通过标准**：单 clip 推理（6 步 Euler）< 150 ms；training step（fwd+bwd）< 400 ms。

---

### Phase 7: HighResBypass + gs_map Late Fusion

**目标**：`gs_head` 的 refinenet1/refinenet2 输入注入 `splatted_tok_4_high`，`gs_map_final = gs_map_flow·(1-τ) + gs_map_asset_direct·τ`（τ = feathered I_map · sigmoid(gs_conf)）。

**要新建的文件**：
- `dggt/models/high_res_bypass.py`
- `tests/test_high_res_bypass.py`

**要改的文件**：
- [dggt/heads/dpt_head.py](/home/dancer/code/dm/dggt/dggt/heads/dpt_head.py)（如需暴露 refinenet 中间张量，只加**可选** hook，不改主前向）。

**对外 API**：
```python
class HighResBypass(nn.Module):
    def run_gs_head(self, gs_head, img_tok_hat_all, images, patch_start_idx,
                    bypass_tokens, bypass_mask_hires):
    def run_depth_head(self, depth_head, agg_hat_all, images, patch_start_idx,
                       bypass_scaffold, bypass_mask_hires):
    def late_fuse_gs_map(self, gs_map_flow, gs_map_asset_direct, I_map_hires,
                         gs_conf_flow, feather_sigma=3.0) -> Tensor
```

**关键实现细节**：
- `α1 / α2` 初始化 0.05，确保早期训练 DPT 主导。
- `gate(I_map) = sigmoid(5*(I_map - 0.3))`。
- 羽化：`gauss_blur(I_map_hires, σ=feather_sigma)`。
- `depth_head` 旁路仅用 scaffold depth，不用 splatted tokens。

**smoke test**：
- 无编辑场景：`τ ≈ 0`，`gs_map_final ≈ gs_map_flow`（逐像素 L1 < 1e-3）。
- 纯资产区：`τ > 0.9`，`gs_map_final` 与 asset direct rasterize 差 < 0.02。

---

### Phase 8: FlowDGGT 顶层 forward_edit()

**目标**：把 P1–P7 串起来，实现 research_plan §6.2 伪代码。

**要新建的文件**：
- `dggt/models/flow_dggt.py`

**对外 API**：
```python
class FlowDGGT(nn.Module):
    def __init__(self, vggt: VGGT):
        super().__init__()
        self.backbone = vggt    # 直接引用，不复制权重
        self.gs_editor          = GaussianSceneEditor(...)
        self.asset_pass         = AssetAggregatorPass(vggt.aggregator)
        self.feature_splatter   = FeatureSplatter(...)
        self.soft_mask_builder  = SoftMaskBuilder(...)
        self.scaffold_packer    = ScaffoldPacker(...)
        self.per_token_sched    = PerTokenNoiseScheduler(...)
        self.scene_flow         = SceneFlowMatching(...)
        self.high_res_bypass    = HighResBypass(...)

    def forward(self, *a, **kw):               # 保持 VGGT.forward 行为
        return self.backbone.forward(*a, **kw)

    def forward_edit(self, images, edit_instruction, asset_bank, mode="inference"):
        ...   # 精确按 research_plan §6.2 实现
```

**关键细节**：
- `VGGT.forward()` 保持零改动；`FlowDGGT.forward()` 直接转发，允许在训练脚本里用 `DDP(FlowDGGT)` 同时训 clean 与 edited 两条路。
- `levels = [4, 11, 17, 23]` 写常量，不从外部读。
- 任何 edit-free 采样（`edit_instruction.action == 'clean'`）走最小路径：只跑 aggregator + tokenizer encode/decode + heads，不调 asset pass、不渲 K/D/I_map。

**smoke test**：
- `edit_instruction.action='clean'`：`forward_edit()` 的 `gs_map / depth / dynamic_conf` 与 `VGGT.forward()` 差 < 1e-3。
- `action='replace'`, `views=1`：编辑后 `gs_map_final` 在 asset 区与 `G_asset` 直接 rasterize 结果 LPIPS < 0.05（对比 baseline inference_mode_a.py 的 asset-only render）。
- `action='delete'`：`M_source_soft` 在目标 bbox 内 > 0.5。

**通过标准**：`views=3, S=4, K=1` 推理总耗时 ≤ 1.5 s（A100, BF16）。

---

### Phase 9: T1 SceneFlow 训练

**目标**：在 T0.5 基础上训练 `scene_flow + feature_splatter + scaffold_packer + per_token_sched + high_res_bypass + soft_mask_builder`，tokenizer encoder 冻结、decoder 仅 layer_heads + local_refine 可训。

**要新建的文件**：
- `train_scene_flow_t1.py`（以 [train_tokenizer.py](/home/dancer/code/dm/dggt/train_tokenizer.py) 骨架为模板）
- `dggt/losses/flow_losses.py`（`L_flow / L_render / L_lpips / L_xview / L_auxgeom / L_3d / L_state / L_route / L_preserve / L_asset_id / L_attn_ent`）

**训练策略**（对齐 research_plan §5.4）：
- **Warm-up 1**: 前 2K 步用 `edit_spec.bbox` oracle soft mask 代替渲染 K/D/I_map，避免早期渲染噪声；2K 后切到真渲染。
- **Warm-up 2**: 前 5K 步用全局 `t`，5K 后过渡到 per-token `t_tok`。
- **Warm-up 3**: 前 3K 步关闭 masked cross-attn，条件仅靠 concat。
- **Mode 混比**: 先 Mode B 10K 步 bring-up（背景补全），再 Mode A : Mode B = 1:1 混合 30–40K 步。
- **损失权重**:
  ```
  L = 1.0*L_flow + 1.0*L_render + 0.1*L_lpips + 0.25*L_xview + 0.05*L_auxgeom
      + 0.1*L_3d + 0.1*L_state + 0.05*L_route + 0.5*L_preserve
      + 0.2*L_asset_id + 0.01*L_attn_ent
  ```
- **优化器**: AdamW, β=(0.9, 0.95), cosine decay。`scene_flow lr=2e-4`, `dense heads lr=5e-6`（默认关）。
- **batch**: 4 clips × S=4 × views=3 on 8×A100 BF16。
- **T0.5 warmup not reset**: 从 T0.5 checkpoint 加载 tokenizer，flow/bypass/splatter/mask_builder 从头训。

**smoke test**：
- 16 Mode A 过拟合 1k 步：`L_render` 降 > 50%。
- 16 Mode B 过拟合 1k 步：`L_source_completion` 降 > 50%。
- per-token `t_tok` 切换后前 500 步 `L_flow` 波动 < 2×。

**通过标准**：
- val FID（替换任务）相比 "固定 asset direct rasterize + 原 gs_map 拼接" baseline 降 ≥ 10%。
- 非编辑区 PSNR 不下降 > 0.3 dB。

---

### Phase 10: T2 联合微调

**目标**：小学习率联合微调 `scene_flow + tokenizer decoder 最后层 + gs_head 最后层`。

**要新建的文件**：
- `train_joint_t2.py`（从 `train_scene_flow_t1.py` 克隆，改冻结表与 lr）

**训练策略**：
- `scene_flow / tokenizer_decoder_last / gs_head_last`: lr=5e-6，AdamW + 极短 cosine。
- `L_asset_id` 权重提到 0.3。
- 5–10 epochs，early stop by val FID。

**通过标准**：
- Val FID 再降 ≥ 3%；asset-region LPIPS 再降 ≥ 5%。
- Preserve 区 PSNR 不退化。

---

### Phase 11: 推理脚本与评估

**目标**：产出最终推理入口 `inference_flow_edit.py`，并跑完 research_plan §8.1 指标。

**要新建的文件**：
- `inference_flow_edit.py`（以 [inference_mode_a.py](/home/dancer/code/dm/dggt/inference_mode_a.py) 为骨架，替换编辑链路为 `FlowDGGT.forward_edit()`）
- `tools/eval_edit_metrics.py`（FID / FVD / CLIP-I / 多视角一致性 / ATE / 3D mAP downstream）

**功能点**：
- 支持 `--action {delete, insert, replace, reposition}`；`--views {1, 3}`；`--ckpt_path` 接收 T2 权重。
- 保留 `inference_mode_a.py` 的可视化（`_save_mask_overlay_grid / _save_target_vs_asset_boxes / _composite_asset_over_scene`）。
- 输出：`{rgb_flow, rgb_asset_direct, rgb_final, K_map, D_map, I_map, M_*, ply}`。
- 推理时若无 `cams_waymo`：回退 `T_w2d = I` + `cameras_dggt` 直接作为 Asset Pass 相机，并记录一条 `degrade=cams_waymo_missing`。

**smoke test**：
- 对 10 个 val 样本，每个 action 跑通；输出目录 structure 与 `inference_mode_a.py` 兼容。

**通过标准**：end-to-end 推理 ≤ 2 s / clip（A100）；指标报告满足 research_plan §8.1 表格每行都有数。

---

## 4. 训练节奏总表

| 阶段 | 数据 | 训练模块 | 冻结 | 步数 | lr |
|---|---|---|---|---|---|
| **T0** | clean clip | tokenizer.enc/dec | 其余全冻 | 已完成 | 5e-4 |
| **T0.5** | clean clip | tokenizer.enc/dec | aggregator + heads | 3–5 ep | 1e-4 |
| **T1** | Mode A+B | flow + splatter + mask_builder + scaffold + bypass + per_token_sched | aggregator + tokenizer.enc；tokenizer.dec 仅 layer_heads + local_refine | 40–50k | 2e-4 |
| **T2** | Mode A+B 均衡 | flow + tokenizer.dec last + gs_head last | aggregator 永冻 | 10–20k | 5e-6 |

---

## 5. 损失别名与计算点对照

| 名称 | 定义 | 计算位置 |
|---|---|---|
| `L_tok_rec` | 真 token 重建 | T0（已完成） |
| `L_splat_cons` | splatted token 自一致 | T0.5 `train_tokenizer_t05.py` |
| `L_head_anchor` | 冻结 head 输出差 L1 | T0 / T0.5 |
| `L_flow` | Flow Matching velocity MSE | T1 / T2 |
| `L_render` | RGB L1 + L2 on 原视角 | T1 / T2 |
| `L_lpips` | LPIPS on 原视角 | T1 / T2 |
| `L_xview` | depth-guided reproject between real views | T1 / T2 |
| `L_auxgeom` | novel-view geometry reg | T1 / T2 |
| `L_3d` | pixel-aligned scene state L1 on 3D points | T1 / T2 |
| `L_preserve` | `M_preserve_soft` 区 token L1 | T1 / T2 |
| `L_asset_id` | LPIPS on `I_map` 区 vs `I_asset_gt` | T1 后期 / T2 |
| `L_attn_ent` | masked cross-attn entropy 下限 | T1 / T2 |
| `L_state`, `L_route` | router state + 路由权重正则 | T1（2K 步后启用） |

---

## 6. Smoke test 清单（CI 守护）

存放位置：`tests/`。每个 Phase 的 test 以 Phase 号前缀命名（`test_p2_feature_splatter.py` 等）。CI 在合并到 dev 前需全部通过。

- [P1] `test_gaussian_scene_editor.py`：views=1 baseline diff、views=3 no-exception。
- [P2] `test_feature_splatter.py`：gather 正确性、chunked 数值、backward mask。
- [P3] `test_soft_mask.py`：归一化 ≈ 1、无编辑 K≈1。
- [P3] `test_scaffold.py`：输出 shape + 梯度。
- [P4] `test_asset_pass.py`：batch 化 K×N、T_w2d reproject 精度。
- [P5] 过拟合 smoke：`test_t05_overfit.py` 构造 2-clip toy，跑 200 步。
- [P6] `test_scene_flow.py`：恒等任务、cross-attn off path。
- [P7] `test_high_res_bypass.py`：τ≈0 / τ≈1 的极限情形。
- [P8] `test_flow_dggt_forward_edit.py`：clean action 与 VGGT.forward 对齐。
- [P9] `test_flow_losses.py`：损失 shape + backward。
- [P11] `test_inference_flow_edit.py`：端到端 action=replace 跑通。

---

## 7. 默认假设（不经用户追问即使用）

- 类别只处理 `Vehicle`。
- `views=1` 仅用于 bring-up 和 smoke；正式训练 / 评测 `views=3`。
- Mode A 主线是 self-replacement；cross-object replacement 留给 T2 后期 fine-grained 实验。
- 不再使用 `final_info_all.json`。
- Hunyuan3D-2 单图模式 + Mesh2Splat 是默认资产管线；失败资产通过 smoke 过滤。
- `semantic_head` 不阻塞主线，如启用则镜像 `dino_tokens'`，使用现有权重不再训。
- 推理时 `cams_waymo` 不可得 → `T_w2d = I` + DGGT 预测相机，作为 graceful degradation，不作为硬失败。

---

## 8. 与 research_plan.md 的强对齐点

| 实施 Phase | research_plan 对应 | 关键约束 |
|---|---|---|
| P2 FeatureSplatter | §3.3 | 梯度仅到 F_g；按通道 chunk；LUT 指针式 |
| P3 SoftMask/Scaffold | §3.6 | 连续软掩码；面积池化；7ch scaffold |
| P4 Asset Pass | §3.3 "Waymo-coord per-frame per-object" | Waymo 真值相机；K×N batch；遮挡 ptr 回退 |
| P5 T0.5 | §5.4 / §3.4 | splatted token OOD 消除 |
| P6 SceneFlow | §3.5 | cross-attn bias=log(M_dest)；per-token AdaLN；三专家 |
| P7 HighResBypass | §3.7 | DPT 残差注入；gs_map 羽化 late fusion |
| P8 FlowDGGT | §6.1, §6.2 | 保 VGGT.forward 不变；levels=[4,11,17,23] |
| P9 T1 | §5.4 | Warm-up 1/2/3；Mode A/B 1:1；损失权重 |
| P10 T2 | §5.4 | 5e-6 小 lr；asset_id 提权 |
| P11 Inference | §7.* | action 四种；graceful degrade |

---

## 9. 风险与应对（实施期）

| 风险 | 触发期 | 应对 |
|---|---|---|
| T0.5 让 tokenizer 漂移，真 token 重建退化 | P5 | `val L_tok_recon` 上升 > 5% 时回退 lr=5e-5 或混入 20% 真 token batch |
| Asset Pass K×N batch OOM | P4 | 按 K 切 2 次 forward；保留 `--asset_chunk` flag |
| FeatureSplatter 反传爆显存 | P2/P9 | chunked backward；`chunk_channels` 降到 256 |
| flow 早期 cross-attn 崩 | P9 | Warm-up 3 关 cross-attn 3K 步，`L_attn_ent` 5K 后启 |
| gs_map late fusion 硬边 | P7/P11 | 提高 `feather_sigma`；降 α1/α2 |
| Waymo cams 在 val 某些 scene 不全 | P9/P11 | 加载时跳过该 frame 或退化为 `T_w2d=I` + 日志告警 |
| inference_mode_a baseline 回归 | P1 | CI 守护 `test_gaussian_scene_editor.py` 数值 diff |

---

## 10. 里程碑

| 里程碑 | 时间预估 | 交付 |
|---|---|---|
| M1 编辑器库化 + FeatureSplatter 可用 | P1–P2 | `GaussianSceneEditor` + `FeatureSplatter` + 2 个 smoke test |
| M2 Asset Pass + 软掩码 | P3–P4 | `AssetAggregatorPass` + `SoftMaskBuilder` |
| M3 T0.5 完成 | P5 | T0.5 ckpt，val L_splat_cons 达标 |
| M4 FlowDGGT 串联跑通 | P6–P8 | `forward_edit()` smoke 全绿 |
| M5 T1 收敛 | P9 | T1 ckpt，FID baseline 对比 |
| M6 T2 收敛 + 最终推理 | P10–P11 | 最终 ckpt + 完整评估报告 |

---

**本文件取代 v3 实施计划。v3 的 "render edited → Pass 2" 链路、`raw_ffm / z_ffm / gs-only latent` 等过渡命名**全部废弃**。后续 LLM 按 Phase 顺序开工，每 Phase 完成后需提交对应 smoke test 通过证据再进入下一 Phase。**
