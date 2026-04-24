# Mode B 数据生成方案（FlowDGGT 补全训练）

## Context

FlowDGGT 训练需要两类数据：Mode A（协调训练，覆盖 insert/replace/reposition dest）与 **Mode B（补全训练，覆盖 delete / reposition source）**。Mode A 的数据生成脚本 `inference_scene_editor.py` 已经跑通（`--views 1 --dataset_mode 2`）并产出了 `vis/1` 之类的调试样本；Mode A 的样本 manifest 放在 `waymo_edit_cache/manifests/training/training_mode_a_views{1,3}.jsonl`（views1=2907 条、views3=3356 条，candidates 总共 3731）。

Mode B 的目标是：在"目标很少或没有目标"的场景里**假想**若干个车辆目标，直接把 clean scene 在假想目标 bbox 区域的 Gaussian 当作"被删除的背景"抠掉，再 rasterize 回 2D 得到 Mode-A 风格的 deletion-shaped hole。GT 始终是原始 clean 场景（29 帧全量）。这个数据喂给 T1 flow 的补全训练分支，让模型学会在 "deleted 洞"处生成正确背景（符合 research_plan §5.1 的 "补全训练：在无车背景区域合成 deletion-shaped holes"）。

关键用户约束（已对齐）：
- **planning space = DGGT 坐标系**：Waymo↔DGGT 的 Sim3 scale 每 clip 不同且没有解析值，直接在 DGGT 里规划可跳过这层误差。
- **deletion = 真正删除背景高斯**：在 bbox 区域挑选 clean_state 的 Gaussians（3D means 落入 bbox）打 `delete_mask=True`，然后 rasterize 剩下的 Gaussians。不注入额外黑色 Gaussian——否则 T1 flow 的 latent 分布与 Mode A 不一致。
- 每 clip **全 29 帧**都生成 metadata 和 D_map（与 Phase 4.5 offline cache 的"29 帧上下文"约定对齐）。训练时 dataloader 再在 29 帧里随机选 20 帧窗 → 4–8 帧子集。
- 每个 29 帧 clip 里至少 15 帧存在假想目标的有效 2D 投影（避免自车跑远后目标全部出画面）。

## 目录与落点总览

| 项 | 路径 | 说明 |
|---|---|---|
| Mode B 候选池 | `waymo_edit_cache/metadata/training/mode_b_candidates.jsonl` | 全量 Waymo 29 帧 clip（798 scene × 6 clip = ~4788），附是否在 Mode A 被采用、是否满足 Mode B 可用性 |
| Mode B manifest | `waymo_edit_cache/manifests/training/training_mode_b_views{1,3}.jsonl` | 实际入训 sample 列表 |
| 每 clip 规划结果 | `waymo_edit_cache/mode_b/training/{scene_dir}/{clip_index}_views{1,3}.json` | 假想目标 bbox 轨迹（DGGT 坐标） + 动机模式 + 验证指标 |
| 每 clip deletion map | `waymo_edit_cache/mode_b/training/{scene_dir}/{clip_index}_views{1,3}_dmap.pt` | `D_map [29,H,W]` + `delete_indices` |
| debug 可视化 | `vis/mode_b/{index}/` | 单样本 dry-run 输出（同 `vis/1` 结构） |

写新代码时 `vis/1` 是 Mode A baseline 视觉金标准——Mode B 的 deletion_render 应在视觉风格上向它看齐。

## 三、设计细节

### 3.1 样本池构建

总池 = Waymo 798 scene × 6 clip，每 clip 29 连续帧。沿用 `build_clean_clip_records`（`datasets/waymo_edit_dataset.py:207`）的枚举方式，仅把 `final_info.json` 替换为直接遍历 `processed_root/training/*/images/`。

对每个 29 帧 clip，打两个标记：

1. `in_mode_a_views{1,3}`：是否在 Mode A manifest 里出现。通过读取 `clip_name_to_record_index.json` + `training_mode_a_views{1,3}.jsonl` 比对得到。
2. `mode_b_eligible`：是否满足 Mode B 的场景条件：
   - views=1：**前视角**在 29 帧内平均 editable 目标数 ≤ 2（严格按 `object_front_bbox_editable_mask` 每帧的 `.sum(dim=0)` 求均值）
   - views=3：**前向三路视角合并**后平均 editable 目标数 ≤ 2

满足 Mode B 的 sample = `mode_b_eligible ∧ (¬in_mode_a_views{1,3} ∨ front_editable_count ≤ 2)`。用户要求的"Mode A 没有用上的所有样本 + Mode A 里场景干净的样本"直接由这组布尔运算得到。

实现：`tools/build_mode_b_manifest.py`（新建）。输出 `mode_b_candidates.jsonl` 每条字段：

```json
{
  "record_index": 0,
  "scene_id": 0, "scene_dir": "000", "scene_name": "...", "clip_name": "...", "clip_index": 0,
  "scene_frame_indices": [0,1,...,28],
  "source": "unused" | "mode_a_eligible",
  "in_mode_a_views1": true, "in_mode_a_views3": true,
  "front_editable_count_per_frame": [1,1,1,...],
  "front3_editable_count_per_frame": [2,2,2,...],
  "existing_objects": [  // 只拷来自 Mode A candidates 的字段，用于 3D 冲突检测
    { "scene_raw_object_id": "...", "obj_to_world_waymo": [[...],...], "box_size_waymo": [...], "present_mask": [true,...] }
  ]
}
```

### 3.2 假想目标规划（DGGT 坐标系）

每条 sample 规划时的输入：
- 29 帧 clean scene（`build_clean_scene_state` 的 `CleanSceneState`），含：
  - `means` `[N_gauss, 3]`, `source_image_ids`, `depth [29,H,W]`, `camera_to_world [29,4,4]`, `intrinsics [29,3,3]`
  - `semantic_vehicle_mask [29,H,W]`（predictions 的 semantic head argmax == 4）
  - existing 编辑目标的 DGGT-空间 refined box（若有）
- Mode A candidates 中的 `refined_size_dggt` 经验分布（用于 canonical car size 先验）

规划算法：

1. **Ground plane 估计**（DGGT 坐标 "up" 轴向 = DGGT camera y 轴负方向）：
   - 取所有 `clean_state.means` 中满足 `dynamic_prob < 0.5 ∧ sky_mask==False` 的点
   - 按 y 坐标取 5% 分位作 ground y。对 29 帧每一帧局部也估一个 ground_y（使用当帧视角下的 valid 投影点）。
   - 输出：`ground_y_per_frame [29]`
   - 实现复用 `clean_state.point_map_world` + `valid_mask`；不引入新依赖。

2. **Canonical car size（DGGT units）**：
   - 预处理阶段扫描 Mode A candidates 所有 `refined_size_dggt`（Sim3×Waymo box_size），算每个 scene 的均值+分位；缺省用全局分布 `L ~ U(q25, q75)` 的采样。
   - per-clip：如果 `existing_objects` 非空，优先用该 clip 已观测到的 refined_size 做先验；否则用全局分布。
   - yaw 采样：从 `clean_state.camera_to_world[i, :3, 2]`（第 i 帧相机 forward）出发，yaw ~ U(-30°, +30°) 对 forward 轴做扰动（车朝前/朝后有约 50/50 概率）。

3. **运动模式采样**（每个假想目标独立）：
   | 模式 | 概率 | 生成方式 |
   |---|---|---|
   | `static` | 0.5 | 所有帧使用同一 DGGT 世界坐标的 center |
   | `slow` | 0.3 | 沿 yaw 方向线速度 ~ U(0.3, 1.2) × canonical_length / 29_frames，在 29 帧线性插值 |
   | `ego_matched` | 0.2 | `center_frame_i = center_frame_0 + (ego_pose_dggt[i] − ego_pose_dggt[0]) + δ`；δ 是小随机位移，模拟并排跟车 |
   - ego pose in DGGT = `camera_to_world[:, :3, 3]`（第 0 列即 view=0 那路 cam）
   - 严禁"帧间随机跳变"（对应用户强调）：轨迹必须是 DGGT 世界系下的一条连续/不变曲线。

4. **候选目标数**：
   - views=1：每 clip 随机选 1–3 个假想目标（服从 U{1,2,3}）
   - views=3：每 clip 随机选 3–5 个（服从 U{3,4,5}），且约束至少前、前左、前右三视角每视角有 ≥1 目标在 ≥15 帧内可见。

5. **启发式搜索 + 验证**（对每个假想目标最多试 `max_trials=80` 次）：
   对采样到的 `(center_seq, size, yaw_seq)` 做以下**全部**验证：
   - (a) 落在地面：`center_y_t ≈ ground_y_per_frame[t]` ±10% canonical height。
   - (b) **2D 视角内**：把 8 个 3D bbox 角点投影到每一帧每一视角（`clean_state.world_to_camera / intrinsics`），如果帧上至少 4 个角点可见 + 投影框面积 ≥ 64 px² → 记 `visible_in_frame=True`。要求 `visible_in_frame.sum() ≥ 15`。
   - (c) **与已有语义车辆 mask 冲突**：对每帧每视角，取投影 bbox 的多边形填充，和 `clean_state.semantic_vehicle_mask[image_idx]` 做逐像素交集。**全 clip 全视角的总交集像素数 ≤ 50**（用户硬约束）。
   - (d) **与已有 3D bbox 冲突**：把 existing_objects 的 DGGT bbox（Sim3×Waymo）与候选 bbox 做 3D AABB 相交测试，禁止相交。
   - (e) **不挡进已有墙/建筑**：取投影 bbox 的中心像素，查 `clean_state.depth[image_idx, v_c, u_c]`，要求 `depth_at_center + 1.0 > bbox_depth`（bbox 中心不能深于 DGGT depth → 避免插在墙内）。canonical_length 用作 tolerance。
   - (f) **目标之间不冲突**：已接受的假想目标之间做 AABB 相交测试。

   `max_trials` 后仍失败 → 该目标缩小 canonical size × 0.8 重试一次；还失败 → 从当前 sample 的目标数里剔除。最终接受 count 低于 min_count（views=1 的 min=1，views=3 的 min=3）→ 整个 sample 标记 `eligibility=false` 不入 manifest。

6. **写出规划结果**：
   ```json
   {
     "scene_name": "...", "clip_name": "...", "clip_index": 0, "views": 1,
     "num_imagined_objects": 2,
     "imagined_objects": [
       {
         "slot": 0,
         "motion_mode": "static",
         "size_dggt": [L, W, H],
         "center_dggt_per_frame": [[x,y,z], ... 29],
         "yaw_dggt_per_frame": [...29],
         "visible_in_frame_per_view": [[true,...29], ... num_views],  // 按 frame-major view-minor 展开
         "bbox_2d_per_view": [...29, num_views, 4],
         "semantic_overlap_px": 12, "existing_box_iou_3d": 0.0
       }
     ],
     "rng_seed": 12345
   }
   ```

### 3.3 Deletion 仿真（3D-删除，**不注入新高斯**）

对每个假想目标、每一帧：

1. 构造**时变 3D bbox**（center_dggt_per_frame[t], size_dggt, yaw_dggt_per_frame[t]）。
2. 在 `clean_state.means` 里挑出 `_points_in_box(means, center, R_yaw, size, scale=1.0)` 的索引 → `delete_core_indices`。
3. 附加 shell：`_points_in_box(..., scale=1.05)` 的索引减去 core → `delete_shell_indices`（匹配 Mode A 的 `core_scale=0.85, shell_scale=1.05`；这里 core 用 1.0 是因为 Mode B 里没有 Waymo-pose refine 误差，直接用 GT 尺寸）。
4. 合并所有假想目标的索引 → `delete_mask [N_gauss]`。
5. **渲染**：调 `_render_edited_sequence_with_dggt`（`inference_scene_editor.py:548` 已有的函数），传入 `delete_mask=mode_b_delete_mask` 即可，视觉风格会自然和 Mode A deletion 一致，因为用的是同一条渲染链路。
6. **D_map**：
   - `D_map[image_idx]` = 单独渲染 `delete_mask` 指向的那部分 Gaussians 的 alpha channel（透过 `_rasterize_scene` 取 alpha 分量）。
   - Phase 3 的 SoftMaskBuilder 将来从这个 D_map 下采样到 37×37 得到 `M_source_soft`。

用户修正的关键点：**我们不创造新的 dark Gaussian，而是直接把 clean scene 自己的 Gaussian 按 bbox 删掉**。这保证了：
- Gaussian feature 域的分布与 Mode A 一致（flow 训练时看到的 `G_edited_kept = clean_gs \ deleted` 与 Mode A 同分布）
- deleted 区域的视觉效果完全由 "真实 DGGT 在该位置形成的 Gaussian 被删掉" 决定，和 vis/1 风格同构

### 3.4 数据落盘

每 clip 每 views 一份 payload（`mode_b/{scene_dir}/{clip_index}_views{V}_dmap.pt`）：
```
{
  "config": <3.2 节 json>,
  "delete_mask_per_frame": bool[29, N_gauss_max_for_clip]    # 按 frame-slot 联合，方便子集裁剪
    # 落盘压缩：用 int32 indices + RLE
  "delete_core_indices": int32[K],
  "delete_shell_indices": int32[K'],
  "D_map": fp16[29, num_views, H, W]              # 将来 SoftMaskBuilder 会消费
  "D_map_low37": fp16[29, num_views, 37, 37]      # 预下采样，训练直接取用
  "pseudo_deleted_render": uint8[29, num_views, 3, H, W]   # debug 期保留，正式训练可裁剪字段
}
```

大小估算：views=1、H=W=518、29 帧时，`D_map` ≈ 15 MB/clip，`pseudo_deleted_render` ≈ 23 MB/clip，合计 < 50 MB。4788 clip × 50 MB ≈ 230 GB；正式训练把 render 字段裁掉后可压到 80 GB。

训练期的 `WaymoFlowCacheDataset`（Phase 4.5）在原有 `pass1` cache 之外，Mode B 样本额外挂 `mode_b/.../*_dmap.pt`——训练 step 不需要 gsplat 在线。

### 3.5 Debug 脚本（用户要求的 Phase 1 交付）

新建 `inference_mode_b.py`（参考 `inference_scene_editor.py` 结构）：

```
python -u inference_mode_b.py \
  --views 1 --dataset_mode 2 --split training \
  --output_dir vis/mode_b/31 --index 31 \
  --mode_b_manifest /data/disk2/.../training_mode_b_views1.jsonl
```

脚本流水线：
1. 用 `WaymoEditDataset`（复用，**加一个可选 manifest_path 指向 mode_b manifest**；Dataset 本身 schema 不变，只是读取不同 manifest）。
2. VGGT 前向 → `build_clean_scene_state` → semantic_vehicle_mask。
3. `ModeBPlanner.plan(clean_state, existing_objects, num_objects_target, views)` → `imagined_objects` 列表。
4. `apply_mode_b(clean_state, imagined_objects)` → `delete_mask, delete_core, delete_shell`（和 Mode A 的 `apply_mode_a` 对称）。
5. 调用 `_render_edited_sequence_with_dggt(..., delete_mask=mode_b_mask)` 得 `pseudo_deleted_render`。
6. 渲染 `D_map`（per-object + 合并）。
7. 落盘到 `output_dir`：
   - `clean_render_grid.jpg`（baseline）
   - `imagined_boxes_overlay.jpg`（在 clean 上画每个假想目标的投影 bbox，颜色区分）
   - `semantic_vehicle_mask_overlay.jpg`（vehicle mask 上画假想目标，显示交集像素统计）
   - `deleted_render_grid.jpg`（rasterize 结果——视觉上应该和 vis/1 风格同构）
   - `d_map_grid.jpg`（D_map 热力图）
   - `mode_b_summary.json`（imagined_objects 全量配置 + 验证指标）

### 3.6 代码落点

新建 / 修改：

| 文件 | 状态 | 作用 |
|---|---|---|
| `dggt/utils/mode_b_planner.py` | 新建 | `ModeBPlanner`（规划算法 + 启发式搜索）、`apply_mode_b`（生成 delete_mask）、`ImaginedObject` dataclass |
| `dggt/utils/ground_plane.py` | 新建 | `estimate_ground_plane_per_frame(clean_state)` |
| `tools/build_mode_b_manifest.py` | 新建 | 枚举全 Waymo clips，生成 `mode_b_candidates.jsonl` 和 `training_mode_b_views{1,3}.jsonl` |
| `inference_mode_b.py` | 新建 | 单样本 debug 脚本（§3.5） |
| `dggt/utils/gaussian_edit.py` | 轻微改动 | 把 `_points_in_box` 从私有改为暴露（`points_in_box`），复用到 planner |
| `tests/test_mode_b_planner.py` | 新建 | 单元测试（规划器的冲突检测、motion trajectory、≥15 帧约束） |

**不要改动**：`WaymoEditDataset.__getitem__`、`GaussianSceneEditor`、`AssetAggregatorPass`——Mode B 数据生成是 Mode A 管线的**外围扩展**，不改已跑通的数据流。

### 3.7 默认超参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `min_visible_frames` | 15 | §3.2 (b) |
| `max_semantic_overlap_px` | 50 | §3.2 (c) |
| `max_trials_per_object` | 80 | §3.2 retry |
| `canonical_size_jitter` | ±15% | size 采样扰动 |
| `yaw_jitter_deg` | ±30 | yaw 采样扰动 |
| `motion_probs` | (0.5, 0.3, 0.2) | static / slow / ego_matched |
| `core_scale / shell_scale` | 1.0 / 1.05 | Gaussian 子集选择（Mode A 是 0.85 / 1.05；Mode B 用 1.0 因无 pose refine 误差） |
| `depth_tolerance_m` | canonical_length | §3.2 (e) |
| `num_imagined_objects_views1` | U{1, 2, 3} | |
| `num_imagined_objects_views3` | U{3, 4, 5} | |

### 3.8 和 Mode A 的接口对齐

- Mode A `edit_summary.json` 有 `localized_objects` 数组；Mode B `mode_b_summary.json` 保持同构，字段名改为 `imagined_objects`，少两个字段（`asset_object_id`、`match_score`）。
- Mode A 的 `delete_mask` 和 Mode B 的 `delete_mask` shape 都是 `[N_gauss]`；都能直接喂给 `_render_edited_sequence_with_dggt`。
- Phase 3 的 `SoftMaskBuilder.pool_and_normalize(K, D, I)` 对 Mode B 的 `I_map = 0, K_map = render_alpha(G_kept), D_map = render_alpha(deleted subset)` 自然成立。

## 四、验证 / 通过标准（执行阶段使用）

1. **可视化对齐**：Mode B debug 样本 `vis/mode_b/{idx}/deleted_render_grid.jpg` 和 Mode A `vis/1/deleted_render_grid.jpg` 的深色洞风格肉眼同构（洞中心黑、边缘有 shell smear）。
2. **规划器过拟合**：对固定 seed 的 10 个 clip，规划得到的 `imagined_objects` 全部满足 §3.2 (a)–(f)，`semantic_overlap_px ≤ 50` 严格通过。
3. **轨迹连续性**：静止目标所有帧 `center_dggt` 逐元素方差 < 1e-6；slow/ego_matched 在相邻帧的位移 ≤ `canonical_length / 29 × 2`，无跳变。
4. **15 帧约束**：至少 15 帧 `visible_in_frame=True`，否则该 slot 不入 summary。
5. **D_map 面积**：D_map 大于 bbox 投影多边形的填充面积、但小于 1.5 × 多边形面积（允许 shell 扩散）。
6. **不改 Mode A**：Mode A 的 `inference_scene_editor.py --views 1 --dataset_mode 2 --output_dir vis/31 --index 31` smoke 输出字节级 diff = 0（证明 §3.6 "不改已跑通路径" 被遵守）。

---

## 五、后续连接（超出本 plan，仅备注）

- Phase 4.5 offline cache 写入侧要加 Mode B 样本的读取路径（`mode_b_manifest` flag）。
- T1 训练（Phase 9）在 Mode A : Mode B = 1:1 混合时，Mode B 样本的 F_g_lut_scene 和 Pass-1 heads 完全复用；只需再读 `D_map` 和 `delete_indices`。
- 本 plan **不**触及 Phase 2+（FeatureSplatter、SoftMaskBuilder、SceneFlow 等模块）。
