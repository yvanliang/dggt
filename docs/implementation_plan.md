# DGGT 编辑训练实施计划 v3（joint scene-state 对齐版）

## 摘要

- 原始数据根目录固定为 `/data/disk2/lyy_dataset/waymo/{training,validation}`；新的 DGGT 处理结果固定写到 `/data/disk2/lyy_dataset/waymo_dggt_edit_processed/{training,validation}`；编辑缓存固定写到 `/data/disk2/lyy_dataset/waymo_edit_cache/`。
- 数据链路分成四层：`保留 DGGT 现有 waymo_preprocess_.py 主链 -> 最小增补 object sidecar / class-separated dynamic masks -> sky/fine dynamic mask 提取 -> edit metadata / asset cache`。
- 模型链路更新为：`image_tokens_[4,11,17,23] -> JointSceneTokenizer -> SceneFlowMatching -> decoder 输出 [dino', frame', global'] -> 重组 {image_tokens', aggregated_tokens', dino_tokens'} -> {gs_head, depth_head/point_head, instance_head}`。
- 确定性编辑链路固定为：`clean pass -> GaussianSceneEditor(ObjectLocalizer + SceneBoxRefiner + AssetPoseRefiner) -> render edited scaffold -> edited pass`；`scene_flow` 不负责删高斯或摆资产。
- Waymo 3D 标注只作为 coarse prior；训练时先做 clip 级 `Sim(3)` 对齐，再在 DGGT clean-pass 世界系中完成对象定位、box refine 和资产局部 `SE(3)+scale` 细化。
- 编辑模式只新增 `forward_edit()`；[dggt/models/vggt.py](/home/dancer/code/dm/dggt/dggt/models/vggt.py) 的原始 `forward()` 保持不变。
- 旧公开命名不再沿用：`raw_ffm`、`z_ffm`、`FeatureTokenizer`、`ESRF` 不再作为主线模块名；统一改为 `scene_tokenizer`、`scene_flow` 和 `scene_flow.router`。
- 最终视角配置仍与 DGGT 代码对齐，只支持 `views=1` 和 `views=3`；最终模型固定训练 `views=3`，`views=1` 只用于早期 bring-up。

## 1. 数据处理与缓存

### 1.1 保留现有预处理主链，不切换到 `waymo_preprocess.py`

- 保留 [datasets/preprocess_waymo.py](/home/dancer/code/dm/dggt/datasets/preprocess_waymo.py) 继续调用 [datasets/waymo/waymo_preprocess_.py](/home/dancer/code/dm/dggt/datasets/waymo/waymo_preprocess_.py)。
- 不切换到 `waymo_preprocess.py` 的原因：
  - `_` 版本已经稳定产出 DGGT 当前依赖的 `make_json`、`images_4`、`depth_flows_*`、`ground_label_*`
  - 直接切换会改变 `dynamic_masks` 结构，并且丢掉 scene annotation json
  - 这不符合“最小修改代码”的要求
- 最小修改策略是：只在 `waymo_preprocess_.py` 上补我们缺的 object-level sidecar，不动旧产物的主格式。

### 1.2 在 `waymo_preprocess_.py` 上做最小增补

- 修改 [datasets/waymo/waymo_preprocess_.py](/home/dancer/code/dm/dggt/datasets/waymo/waymo_preprocess_.py)，只增加以下能力：
  - 支持 `process_keys` 中的 `objects`
  - 新增 `save_objects(dataset)`，输出：
    - `instances/instances_info.json`
    - `instances/frame_instances.json`
    - `instances/object_id_map.json`
  - 扩展 `save_dynamic_mask()`，同时写：
    - 兼容旧逻辑的 `dynamic_masks/{frame}_{cam}.png`
    - 新增 `dynamic_masks/human/{frame}_{cam}.png`
    - 新增 `dynamic_masks/vehicle/{frame}_{cam}.png`
    - 新增 `appearances_all.json`
    - 新增 `appearances_human.json`
    - 新增 `appearances_vehicle.json`
  - 扩展 `create_folder()`，为 `instances/` 和 `dynamic_masks/{all,human,vehicle}` 建目录
- `instances_info.json` 和 `frame_instances.json` 保持 DGGT 当前兼容的连续整数 id 格式，不改主 schema。
- 每个 instance 条目里新增 `raw_object_id` 字段，内容来自原始 Waymo `label.id`。
- `object_id_map.json` 固定包含：
  - `contig_to_raw`
  - `raw_to_contig`
- 这样可以同时满足：
  - 原 DGGT loader 继续按连续 id 读取
  - 编辑数据构造脚本通过 `raw_to_contig` 和参考图目录做 join

### 1.3 重新跑 DGGT Waymo 预处理

- 训练集命令固定为：

```bash
python datasets/preprocess_waymo.py \
  --data_root /data/disk2/lyy_dataset/waymo \
  --target_dir /data/disk2/lyy_dataset/waymo_dggt_edit_processed \
  --dataset waymo \
  --split training \
  --scene_list_file data/waymo_train_list.txt \
  --num_workers 8 \
  --process_keys images lidar calib pose ground dynamic_masks objects \
  --json_folder_to_save /data/disk2/lyy_dataset/waymo_edit_cache/annotations
```

- 验证集命令固定为：

```bash
python datasets/preprocess_waymo.py \
  --data_root /data/disk2/lyy_dataset/waymo \
  --target_dir /data/disk2/lyy_dataset/waymo_dggt_edit_processed \
  --dataset waymo \
  --split validation \
  --scene_list_file data/waymo_val_list.txt \
  --num_workers 8 \
  --process_keys images lidar calib pose ground dynamic_masks objects \
  --json_folder_to_save /data/disk2/lyy_dataset/waymo_edit_cache/annotations
```

- 处理结果只从这个新目录读取：
  - `images/`
  - `images_4/`
  - `intrinsics/`
  - `extrinsics/`
  - `ego_pose/`
  - `depth_flows_4/`
  - `ground_label_4/`
  - `dynamic_masks/`
    - `dynamic_masks/human/`
    - `dynamic_masks/vehicle/`
  - `instances/instances_info.json`
  - `instances/frame_instances.json`
  - `instances/object_id_map.json`
  - `appearances_vehicle.json`
  - `annotations/<scene_name>.json`

### 1.4 生成 sky mask 和 fine dynamic mask

- 按 [datasets/Waymo.md](/home/dancer/code/dm/dggt/datasets/Waymo.md) 再跑 [datasets/tools/extract_masks.py](/home/dancer/code/dm/dggt/datasets/tools/extract_masks.py)。
- 输入根目录固定为：
  - `/data/disk2/lyy_dataset/waymo_dggt_edit_processed/training`
  - `/data/disk2/lyy_dataset/waymo_dggt_edit_processed/validation`
- 输出必须具备：
  - `sky_masks/`
  - `fine_dynamic_masks/all`
  - `fine_dynamic_masks/human`
  - `fine_dynamic_masks/vehicle`
- 编辑训练优先读取 `fine_dynamic_masks/vehicle`；不存在时退回 `dynamic_masks/vehicle`。
- 主线不修改 `extract_masks.py`。
- 为了最小改动，调用时优先使用 `--start_idx/--num_scenes` 或 `--scene_ids`，不依赖 `split_file` 解析逻辑。

### 1.5 构建 edit metadata

- 新增一个离线脚本 `tools/build_edit_metadata.py`，输入只允许：
  - `/data/disk2/lyy_dataset/waymo_dggt_edit_processed/{training,validation}`
  - `/data/disk2/lyy_dataset/test_transfer/reference_images_selected/`
- 这个脚本负责生成：
  - `scene_name_to_index.json`
  - `object_scene_index.json`
  - `asset_candidate_ids.json`
  - `vehicle_track_library.jsonl`
  - `scene_vehicle_visibility.jsonl`
  - `mode_a_candidates.jsonl`
  - `mode_b_replay_library.jsonl`
- 不再使用旧 `track/trajectory.pkl`、`track_camera_visible.pkl` 等自定义结构。
- 不使用 `final_info_all.json`。所需 object 信息全部从新 processed 数据中获取。
- 轨迹库的字段固定为：
  - `scene_id`
  - `contig_instance_id`
  - `raw_object_id`
  - `class_name`
  - `frame_indices`
  - `obj_to_world`
  - `box_size`
  - `visible_cameras`
  - `bbox2d_by_frame_camera`
  - `speed_flag`
- `visible_cameras` 和 `bbox2d_by_frame_camera` 当前实现直接来自 `appearances_vehicle.json`。
- `asset_candidate_ids.json` 的来源是：
  - 先扫描 processed scenes 中所有 `Vehicle` 的 `raw_object_id`
  - 再与 `reference_images_selected/` 下实际存在的目录名取交集
- `mode_a_candidates.jsonl` 由 `asset_candidate_ids.json` 和轨迹可见性共同生成，不依赖任何外部标注 json。

### 1.6 资产选择与 3D 资产缓存

- 继续使用你现有的 `/data/disk2/lyy_dataset/test_transfer/reference_images_selected/`，但只把它当参考图源，不当训练图源。
- 只对 `asset_candidate_ids.json` 中出现的 raw object id 生成资产缓存，不扫全量目录。
- 每个 object id 只选 1 张最佳 RGBA 图。评分公式固定为：
  - `0.30 * sqrt(alpha_area)`
  - `0.20 * sharpness`
  - `0.20 * fill_ratio`
  - `0.15 * non_truncation`
  - `0.15 * camera_prior`
- `camera_prior` 固定：
  - `front=1.00`
  - `front_left/front_right=0.92`
  - `side_left/side_right=0.80`
- 输出：
  - `asset_image_selection.jsonl`
- 然后离线生成资产：
  1. RGBA 预处理到 `1024x1024`
  2. Hunyuan3D-2 生成 `.glb`
  3. Mesh2Splat 转成 gaussian
- 缓存目录固定为：
  - `asset_meshes/<raw_object_id>.glb`
  - `asset_gaussians/<raw_object_id>.ply`
- 当前数据处理阶段只缓存文件路径，不在这里额外重打包成 `npz`。

### 1.7 训练样本 manifest

- 不离线生成编辑后的图片，只离线生成采样 manifest。
- 新增 `tools/build_edit_manifest.py`，输出：
  - `manifests/train_mode_a.jsonl`
  - `manifests/train_mode_b.jsonl`
  - `manifests/val_mode_a.jsonl`
  - `manifests/val_mode_b.jsonl`
- `Mode A` 只做 `self-replacement`：
  - 当前 scene 里存在动态车辆
  - 该 object 的 `raw_object_id` 在资产缓存里存在
  - 至少在 4 帧窗口里有 1 或 3 视角可见
- `Mode B` 从 replay library 采样 vehicle-shaped hole：
  - 与当前 clip 的真实动态区域重叠 < `2%`
  - 最多重试 10 次
  - 否则退化成 generic irregular mask

## 2. 训练数据集与在线编辑构造

### 2.1 新 dataset，不复用旧目录假设

- 新增 `WaymoEditDataset`，不使用旧 [datasets/dataset.py](/home/dancer/code/dm/dggt/datasets/dataset.py) 的硬编码扫描逻辑。
- 读取根目录固定为：
  - `/data/disk2/lyy_dataset/waymo_dggt_edit_processed/training`
  - `/data/disk2/lyy_dataset/waymo_dggt_edit_processed/validation`
- 支持：
  - `views=1` 取相机 `0`
  - `views=3` 取相机 `0,1,2`
- 时间采样固定与 DGGT 训练一致：
  - `sequence_length=4`
  - 从 20 帧窗口内随机抽 4 帧
- 返回字段固定为：
  - `images_clean`
  - `sky_mask`
  - `dynamic_mask`
  - `timestamps`
  - `scene_id`
  - `frame_indices`
  - `cam_ids`
  - `edit_mode`
  - `edit_spec`
  - `asset_meta`
  - `object_meta`
- `edit_spec` 中与几何编辑相关的 `obj_to_world / box_size / heading / bbox2d_by_frame_camera` 只作为 coarse prior，不作为最终删除框或资产位姿的直接来源。

### 2.2 在线构造 Mode A

- 训练循环里先跑 clean clip 的 `Pass 1`，得到 `G_original`、`pose_enc_clean`、`world_points`、`dynamic_conf`，以及可选的 `semantic_logits`。
- 不直接使用 `edit_spec` 里的 Waymo `obj_to_world / box_size / heading` 作为最终编辑框；这些字段只作为 coarse prior。
- 在线编辑固定由 `GaussianSceneEditor` 完成，工作顺序为：
  - `estimate_scene_alignment(...)`：用 Waymo GT 相机外参与 clean pass 的相机预测估计 clip 级 `T_waymo_to_dggt`；默认使用 `Sim(3)`，允许尺度漂移。
  - `ObjectLocalizer(...)`：将 Waymo track 变换到 DGGT 世界系后只作为 proposal，再结合多视角 2D 可见框、`dynamic_conf`、可选 `semantic vehicle`、深度一致性和跨帧连续性选出待编辑高斯集合。
  - `SceneBoxRefiner(...)`：基于已选高斯重新拟合 `B_dggt`；后续 delete / insert / mask projection 全都使用 `B_dggt`，不再直接使用原始 Waymo box。
  - 删除策略固定为“core 高斯硬删除，boundary shell 保守处理”；shell 区域优先交给后续 `M_source / M_dest` 与 flow 做残影清理和协调。
  - `AssetPoseRefiner(...)`：资产初始位姿来自 `B_dggt`，再做小范围 `SE(3) + scale` 局部优化，使资产单独渲染的 alpha / depth / silhouette 与原目标支持尽量对齐。
- 然后执行：
  - 从 `G_original` 中删除 refine 后目标高斯
  - 把缓存资产高斯按 refined asset pose 放入场景
- 在线渲染得到：
  - `I_edited`
  - `I_asset`
  - `A_edited`
  - `D_edited`
  - `A_asset`
  - `D_asset`
- 掩码固定：
  - `M_dest = projected refined asset support + dilation`
  - `M_source` 默认 `0`；若 refined 资产 footprint 小于被删目标并产生新暴露背景，则把 disocclusion ring 记入 `M_source`

### 2.3 在线构造 Mode B

- clean clip 先跑 `Pass 1`
- 在线编辑固定由 `GaussianSceneEditor` 的 `mask-driven cutout` 完成：
  - replay mask 只作为 hole 分布先验，不直接当最终 cutout 区域
  - 结合 clean pass 的 `depth / world_points / visibility` 把 replay mask lift 到 DGGT 世界系
  - 在 DGGT 世界系里选出真实 cutout 高斯集合，并输出 realized source support
- 在线渲染得到 `I_deleted`
- 掩码固定：
  - `M_source = realized source support + dilation`
  - `M_dest = 0`

### 2.4 patch mask 与像素 mask

- 所有 pixel mask 先在 `H x W` 上构建，再下采样成 `37 x 37` patch mask。
- 固定保留两份：
  - `mask_px` 给渲染监督、深度/alpha scaffold、scene-state 监督
  - `mask_patch` 给 tokenizer、scene_flow 和 router
- 默认 dilation：
  - pixel 空间 `6 px`
  - patch 空间 `1 patch`

## 3. 模型修改顺序

### 3.0 `GaussianSceneEditor` 与坐标对齐

- 3DGS 编辑不在 `scene_tokenizer`、`scene_flow` 或原始 `VGGT.forward()` 内实现；固定新增一个确定性模块 `GaussianSceneEditor`，位置在 clean pass 和 edited pass 之间。
- `GaussianSceneEditor` 不是生成模型的一部分；它由训练 / 推理 driver 调用，负责：
  - 根据 `edit_spec` 和 clean-pass 场景状态找到待编辑高斯
  - 执行 delete / insert / cutout
  - 渲染 `images_edited / I_asset / D_edited / A_edited`
  - 构造 `M_source / M_dest / edit_scaffold`
- 子模块固定拆成：
  - `ObjectLocalizer`
  - `SceneBoxRefiner`
  - `AssetPoseRefiner`
- 坐标对齐策略固定为：
  - 先利用 Waymo GT 相机和 clean pass 相机预测估计 clip 级 `T_waymo_to_dggt`
  - 默认使用 `Sim(3)` 而不是死板 `SE(3)`，显式吸收 DGGT 世界系的尺度漂移
  - 变换后的 Waymo box track 只作为 proposal，不直接用于删高斯或摆资产
  - 最终编辑 anchor 必须来自 DGGT 场景内部重新定位后的 `B_dggt`
- 删除 / 插入策略固定为：
  - 高置信 core 高斯硬删除
  - 边界 shell 高斯保守处理，并通过 `M_source / M_dest` 留给后续 flow 做补全和协调
  - 资产插入先由 `B_dggt` 初始化，再做局部 `SE(3)+scale` refine

### 3.1 Phase 1: `FlowDGGT` 包装与 `forward_edit()` 主链

- 保持 [dggt/models/vggt.py](/home/dancer/code/dm/dggt/dggt/models/vggt.py) 的 `forward()` 不变。
- 新增包装模型 `FlowDGGT`，内部复用 `VGGT` 的 `aggregator / camera_head / gs_head / depth_head / point_head / instance_head`。
- 新增三个主组件：
  - `self.scene_tokenizer`
  - `self.scene_flow`
  - `self.asset_encoder`
- 新增的辅助接口固定为：
  - `select_patch_pyramid(image_tokens_list, layer_indices, patch_start_idx)`
  - `pack_scaffold(edit_scaffold)`
  - `build_edit_state(M_source, M_dest, edit_scaffold)`
  - `reattach_special_tokens(template_tokens, layer_indices, patch_start_idx, patch_tokens)`
  - `split_joint_channels(joint_tokens, dims=[1024,1024,1024])`
  - `replace_selected_levels(full_list, layer_indices, replacement_list)`
- `forward_edit()` 不直接执行删高斯或资产放置；`images_edited / edit_scaffold / M_source / M_dest` 默认来自外部 `GaussianSceneEditor`。
- `forward_edit()` 主链固定为：
  1. clean pass：`images -> aggregator -> agg_clean_all / image_tok_clean_all / dino_clean_all`
  2. edited pass：`images_edited -> aggregator -> agg_edit_all / image_tok_edit_all / dino_edit_all`
  3. 仅抽取 `[4,11,17,23]` 四层的 `image_tokens` patch token 作为统一 scene state
  4. `scene_tokenizer.encode()` 得到 `z_clean / z_edit`
  5. `pack_scaffold + asset_encoder + build_edit_state` 组装条件
  6. `scene_flow.initialize()` 按 `M_source / M_dest` 构造初始 latent
  7. `scene_flow()` 产出 `z_hat`
  8. `scene_tokenizer.decode()` 输出编辑后的四层 joint token
  9. 固定 split 成 `dino' / frame' / global'`
  10. 重组出 `image_tokens' / aggregated_tokens' / dino_tokens'`
  11. 把这三路结果统一送回 `gs_head + depth_head/point_head + instance_head`
  12. `pose_enc` 继续来自 clean pass 的 `camera_head(agg_clean_all)[-1]`
- 不再保留单独的 `decode_gs_from_tokens()` 主线接口，也不再保留“只改 `gs_head` 输入”的过渡实现。

### 3.2 Phase 2: `JointSceneTokenizer`

- 输入固定为 `image_tokens_[4,11,17,23]` 的 patch token：
  - `4 × [B, S, P=1369, C=3072]`
- latent 固定为：
  - `[B, S, P=1369, C_scene=768]`
- decoder 输出固定为：
  - `4 × [B, S, P=1369, C=3072]`
- encoder 结构固定为：
  - 先按通道拆出 `dino / frame / global`
  - 各层分别做轻量投影
  - 在 joint space 中做 cross-scale fusion
  - 保留 shallow detail branch
- 推荐配置：
  - `C_scene = 768`
  - `encoder_width = 640`
  - `detail_branch = 128`
  - `decoder_width = 896`
- 不再单独并联 `aggregated_tokens` 或 `dino_tokens` 给 encoder；统一以 `image_tokens` 作为 scene tokenizer 的唯一主输入。
- decoder 的输出语义必须与 DGGT 原生三路 token 严格对齐，不能再引入 `gs-only latent` 分支。

### 3.3 Phase 3: `SceneFlowMatching` 与 reliability routing

- `scene_flow` 在 `z_scene` 上做条件流匹配，不直接生成 RGB。
- 条件输入固定为：
  - `z_clean`
  - `D_edited / A_edited / dynamic_prior`
  - `F_asset`
  - `M_source / M_dest`
- 推荐配置：
  - `token_dim = 768`
  - `hidden_dim = 1024`
  - `num_block_pairs = 3`
  - `num_heads = 16`
  - `num_steps = 6`
  - `state_dim = 96`
- 初始化策略固定为：
  - `M_source` 从纯噪声开始
  - `M_dest` 对 `z_edit` 做 `t_start=0.3` 的 SDEdit 式部分加噪
- routing 逻辑作为 `scene_flow.router` 的内部子模块实现，不再单独作为外部模块命名。
- `route_state` 固定包含：
  - `M_source`
  - `M_dest`
  - `|A_edited - A_original|`
  - `clip(|D_edited - D_original| / d0, 0, 1)`
  - `vis_support`
  - `boundary_flag`
- expert 配置固定为：
  - `preserve`
  - `harmonize`
  - `generate`
- 路由强度固定为：
  - `alpha_p = 0.25`
  - `alpha_h = 1.00`
  - `alpha_g = 1.50`

### 3.4 Phase 4: 三路 token 重组与 dense heads 复用

- decoder 输出的四层 joint token 先按通道拆分：

```python
dino_l, frame_l, global_l = joint_l.split([1024, 1024, 1024], dim=-1)
```

- 然后固定重组为：

```python
image_tokens_l = torch.cat([dino_l, frame_l, global_l], dim=-1)
aggregated_tokens_l = torch.cat([frame_l, global_l], dim=-1)
dino_tokens_l = dino_l
```

- 整个四层金字塔的重组规则固定为：
  - `image_tokens' = [dino' | frame' | global']`
  - `aggregated_tokens' = [frame' | global']`
  - `dino_tokens' = dino'`
- 只替换 `[4,11,17,23]` 四层；其余层默认保留 edited pass 的原始结果，以兼容 DGGT 现有的 DPT-style heads。
- 统一喂给下游：
  - `gs_head(image_tokens')`
  - `depth_head(aggregated_tokens')`
  - `point_head(aggregated_tokens')`
  - `instance_head(dino_tokens')`
- `semantic_head` 如需保留，可镜像复用 `dino_tokens'`，但不作为主线阻塞项。
- `camera_head` 不参与 edit correction，保持 clean pass world anchor。

## 4. 训练计划

### 4.1 Phase P0: 数据与资产准备

- 先重新跑 DGGT 官方 Waymo 预处理
- 再跑 `extract_masks.py`
- 再跑 `build_edit_metadata.py`
- 再跑 Hunyuan3D-2 + Mesh2Splat
- 最后生成 manifest
- P0 完成标准：
  - train/val processed 数据目录完整
  - `instances/object_id_map.json` 和 `appearances_vehicle.json` 正常生成
  - asset cache 覆盖率可用
  - manifest 能抽样出 Mode A / Mode B

### 4.2 Stage T0: `JointSceneTokenizer` 预训练

- bring-up 可先 `views=1` 跑通，正式 checkpoint 固定使用 `views=3`。
- 输入只用 clean scene。
- 训练模块：
  - `scene_tokenizer.encoder`
  - `scene_tokenizer.decoder`
- 冻结模块：
  - `aggregator`
  - `camera_head`
  - `gs_head`
  - `depth_head`
  - `point_head`
  - `instance_head`
  - `semantic_head`
  - `track_head`
  - `sky_model`
- 目标：
  - joint token reconstruction
  - multi-head anchors
  - noisy latent decoding
- 训练步数：
  - `60k-100k steps`

### 4.3 Stage T1: `SceneFlowMatching` 训练

- 加载 T0 checkpoint。
- 初期固定 `scene_tokenizer.encoder`，避免 latent 分布漂移。
- `scene_tokenizer.decoder` 前期只保留最小必要更新，后期再放开输出头与局部 refine 模块。
- 主训练模块：
  - `scene_flow`
  - `scene_flow.router`
- 训练顺序：
  1. 先用 Mode B 做背景补全 bring-up，建议 `10k steps`
  2. 再切到 Mode A : Mode B = `1 : 1` 的混合训练，建议 `30k-40k steps`
- `L_state / L_route` 前 `2k` steps 默认关闭，shared trunk 稳定后再开启。
- `M_source` 固定纯噪声初始化，`M_dest` 固定 `t_start=0.3` 的部分加噪初始化。
- 正式训练视角固定为 `views=3`。

### 4.4 Stage T2: 小学习率联合微调

- 在 T1 收敛后开启联合微调。
- 训练模块：
  - `scene_flow`
  - `scene_flow.router`
  - `scene_tokenizer.decoder`
  - `scene_tokenizer.encoder` 的最后一层 cross-scale block
- 默认不动 `camera_head`；它继续只负责 clean pass 的 pose anchor。
- 如 head anchor 漂移明显，可选地以极低学习率放开：
  - `gs_head`
  - `depth_head`
  - `point_head`
  - `instance_head`
- 推荐学习率：
  - `scene_flow = 2e-4`
  - `dense_heads = 5e-6`（仅在需要微调时启用）
- 最终模型固定为：
  - `views=3`
  - `scene_tokenizer on`
  - `scene_flow on`
  - `router on`

## 5. 损失、测试与验收

### 5.1 损失

- T0:
  - `L_tok_rec`
  - `0.2 * L_tok_cos`
  - `0.5 * L_head_anchor`
  - `0.25 * L_render_anchor`
  - `0.5 * L_noisy`
  - `0.01 * L_lat_stat`
- `L_head_anchor` 固定同时约束：
  - `L_gs_anchor`
  - `L_geom_anchor`
  - `L_dyn_anchor`
- 其中：
  - `L_gs_anchor` 约束 `gs_head(image_tokens')`
  - `L_geom_anchor` 约束 `depth_head/point_head(aggregated_tokens')`
  - `L_dyn_anchor` 约束 `instance_head(dino_tokens')`
- T1/T2:

```text
L_total =
    1.0  * L_flow
  + 1.0  * L_render
  + 0.1  * L_lpips
  + 0.25 * L_xview
  + 0.05 * L_auxgeom
  + 0.10 * L_3d
  + 0.10 * L_state
  + 0.05 * L_route
  + 0.5  * L_preserve
```

- 监督原则固定为：
  - `L_xview` 只在真实视图对之间计算，不使用 pseudo novel-view GT
  - `L_auxgeom` 只做虚拟视角几何正则，不做 RGB 回归
  - `L_3d` 约束的是 pixel-aligned scene state，不再退化成只看 `gs_head` 的外观参数

### 5.2 smoke test

- 数据 smoke test：
  - 官方预处理输出完整
  - `fine_dynamic_masks` 和 `sky_masks` 存在
  - `instances_info.json` 仍使用连续 id，但每个条目都带 `raw_object_id`
  - `instances/object_id_map.json` 的 `raw_to_contig` 和 `contig_to_raw` 可双向验证
  - `reference_images_selected/` 目录名能和 `raw_object_id` 正确取交集
- 资产 smoke test：
  - 每个 cached gaussian 能 turntable 渲染
  - 失败资产被过滤
- 编辑器 smoke test：
  - `GaussianSceneEditor` 在 Mode A self-replacement 上能稳定输出非空的 edited gaussian set
  - `T_waymo_to_dggt` 能把 Waymo track 粗对齐到 DGGT 世界系，且 refine 后 `B_dggt` 的多视角投影落在目标支持内
  - `AssetPoseRefiner` 输出的资产 alpha / silhouette 与原目标支持基本对齐，不出现明显悬浮或穿地
- 模型 smoke test：
  - `forward_edit()` 在 `views=1` 和 `views=3` 都能跑通
  - `image_tokens'` 的通道维保持 `3072`
  - `aggregated_tokens'` 的通道维保持 `2048`
  - `dino_tokens'` 的通道维保持 `1024`
  - `gs_head / depth_head / point_head / instance_head` 都能直接吃重组后的 token，不需要单独分支
- 训练 smoke test：
  - 16 个样本 Mode A 过拟合
  - 16 个样本 Mode B 过拟合
  - tokenizer `1k steps`
  - scene_flow `1k steps`
  - joint finetune `1k steps`

### 5.3 验收标准

- 不允许任何训练代码读取 `/data/disk2/lyy_dataset/waymo_processed/`
- 所有训练样本都来自：
  - 新 processed DGGT root
  - 新 edit metadata
  - 新 asset cache
- 主线实现里不再保留 `raw_ffm / z_ffm / gs-only edit path` 作为默认路径。
- 最终 `views=3` 模型在以下方面优于 direct edit baseline：
  - replacement 画质
  - hole completion
  - 多视图一致性
  - 非编辑区域 preserve error

## 6. 默认假设

- 只处理 `Vehicle`
- `views=1` 仅用于早期实验，最终训练与推理固定 `views=3`
- `Mode A` 主线只做 self-replacement，不做 cross-object replacement 主训练
- 不使用 `final_info_all.json` 参与训练数据构造
- Hunyuan3D-2 输入固定单图，不用多图模式
- Mesh2Splat 是默认 mesh->3DGS 转换器；若个别资产失败，只做失败过滤，不改变主训练链路
- `semantic_head` 不阻塞主线；若启用，默认复用 `dino_tokens'`
