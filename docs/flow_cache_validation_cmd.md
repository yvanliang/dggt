# Validation Flow 缓存流水线（Mode A 语义对齐）

本文档介绍 FlowDGGT **validation** 离线缓存的生成。它与 `docs/flow_cache_cmd.md`
的当前 training Mode A 共享**完全相同的逻辑与物理格式**：

* 逻辑 schema：`schema_version=8`、`mode_kind="mode_a"`；
* 物理格式：chunked-zstd SQLite container，文件扩展名仍为 `.pt`；
* chunked format：当前为 `format_version=2`，Mode-A asset LUT 保存完整 4 levels；
* 读取入口：`dggt.utils.flow_cache_io`，不要直接假定是普通 `torch.save` 文件。

因此下游 `WaymoFlowCacheDataset` / `FlowFeatureAssembler` / `SceneFlowMatching`
可零改动消费。

v8 同时包含旧版本问题的修复：

* v7 修复：`pass1.gs_conf` 以 finite fp32 保存，避免 v6 的 fp16 `inf` 溢出；
* v8 修复：pass2 splat 使用与当前 training 一致的动态 Gaussian 生命周期阈值
  `sigmoid(0.5)`，不再使用旧的 `0.5` 概率阈值；
* validation 生成器只复用已有的当前 v8 chunked cache。已有 v6/v7、旧 chunked
  format v1 或 monolithic v8 文件会默认自动重算，避免静默混用旧数据。

与训练 Mode A 的区别：

| 维度 | 训练 Mode A | Validation |
|---|---|---|
| 编辑配置 | `final_info.json` + `build_edit_metadata.py` | `data/final_info_validation.json`（33 条） |
| 删除/资产 | 同一 Waymo 目标：删除后用其自身资产重渲染 | **解耦**：删 3 个源目标，插 3 个外部/复用资产 |
| 3D 框来源 | `instances_info.json` | `data/validation_info/all_object_info*` 的 tar（**不读 tfrecord**） |
| 每条产物 | 1 个 `.pt` | **5 个 `.pt`**：`combined`、`deletion`、`insertion`、`replacement`、`repositioning` |

每条 validation entry 的 4 个编辑：

* **删除（deletion）**：从高斯中删 `origin_object_dict.deletion`、`.replacement`、`.repositioning` 三个源目标。
* **添加（insertion）**：在 `all_object_info_insertion`（`insertion_0`）的逐帧目标框处插入 `insertion_candidates` 资产。
* **替换（replacement）**：删 `replacement` 源目标，在 `all_object_info_replacement`
  中该源 ID 对应的逐帧目标框处插入 `replacement_candidates` 资产。
* **移动（repositioning）**：删 `repositioning` 源目标，从 `--asset_root` 加载
  `{repositioning_id}.ply`，在 `all_object_info_reposition` 的逐帧目标框处插入。
  该 tar 已经保存“原框沿物体局部轴按 `action_for_reposition` 平移 3.0m”的结果，
  生成 cache 时不能再次平移。这里的 PLY 是
  validation 数据准备阶段提供的权威 move 资产，不要求从当前 clean scene 的删除
  Gaussian 中重建。

`combined` = 同时做以上 4 个编辑；其余四个名称表示对应的单一编辑类型。

删除路径 100% 复用经过验证的 `localize_objects`（`load_asset=False` ⇒ 资产为空，只贡献删除），
含 Sim3 对齐 + 角点投影位姿精修 + 语义 mask 连通域；资产路径用与 Mode A 相同的
`_transform_asset_gaussians_simple` 放置，产出 schema 一致的 `LocalizedFrameObject`。

> **坐标系**：已验证 `all_object_info` tar 的 `object_to_world` 与处理后数据集的
> `instances_info`/`ego_pose` 同处一个绝对 Waymo 世界系（平移差 ~3cm），**不做任何
> center_point 归一化**；`estimate_scene_alignment` 用 `camera_to_world_corrected`
> （来自 `ego_pose`）与 DGGT 预测相机求 Sim3，把 tar 框搬到 DGGT 系。

## 0. 前置条件

```bash
conda activate dggt
```

* 处理后数据集：`/data/disk2/lyy_dataset/waymo_processed_dggt/validation/{NNN}`
  （段名→`NNN` 由 `pointcloud_validation/toolkits/waymo_name_index.py:val_name2index` 映射）。
* 标注：`{processed_root}/waymo_edit_cache/annotations/validation/segment-{段名}_with_camera_labels.json`。
* 3D 框 tar（已复制到仓库内）：
  `all_object_info` 提供三个删除源框；`all_object_info_insertion`、
  `all_object_info_replacement`、`all_object_info_reposition` 分别提供 add、replace、
  move 的逐帧目标框。四类 tar 都参与生成。tar member 的编号按
  `scene_frame_idx * 3` 解释；不能把排序后的 member 序号直接当 scene frame，
  因为部分目标 tar 从场景中段开始。
* 资产 `.ply`：单一目录 `--asset_root`（默认
  `/data/disk2/lyy_dataset/test_transfer/objects_ply_transformed`，文件名 `{id}.ply`）。
  insertion 使用 `{insertion_candidates}.ply`，replacement 使用
  `{replacement_candidates}.ply`，move 使用 `{repositioning_id}.ply`。
  **缺失资产** → 该 entry 跳过并记入 `_errors.jsonl`（把缺失 id 复制进 `--asset_root` 后重跑即可）。

## 1. 生成 validation 缓存

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run -n dggt --no-capture-output \
    python -u tools/precompute_flow_features_validation.py \
    --ckpt_path /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
    --asset_root /data/disk2/lyy_dataset/test_transfer/objects_ply_transformed \
    --out_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_validation \
    --save_compression chunked_zstd \
    --gzip_level 1 \
    --max_save_threads 1
```

输出使用与 training 相同的六位数字补零，再接单下划线和完整编辑名称：

```
{out_root}/validation/{entry_index:06d}_{edit_name}.pt
{out_root}/validation/_errors.jsonl           # {index, clip_name, reason, missing_asset_ids:[...]}
```

例如 entry 12：

```text
000012_combined.pt
000012_deletion.pt
000012_insertion.pt
000012_replacement.pt
000012_repositioning.pt
```

manifest 中仍保留逻辑数字 `index = entry_index*5 + variant_ord`，只用于保持
下游采样和 `--index` 兼容，不再作为文件名。文件名、cache metadata、manifest 和
`--variants` 统一使用
`combined/deletion/insertion/replacement/repositioning`。生成器会自动迁移合法的
历史命名文件；manifest builder 也保留历史读取兼容。

每条 entry **只跑一次 VGGT Pass-1 + 一次解耦定位（含位姿精修）**，再按 5 个 variant
切分已定位对象、分别跑 asset pass + phase1 打包 + splat/blend，组装与
`precompute_one_clip` 完全一致的 payload。

常用参数（`--help` 查看全部）：

* `--variants combined,deletion` —— 只生成指定 variant（调试 / 省时）。
* `--start N --end M` —— 只处理 entry 索引 `[N, M)`（0..32）。
* `--force_overwrite` —— 无条件覆盖已存在的 `.pt`。默认只跳过合法的当前 v8
  chunked-zstd cache；v6/v7、旧 chunked format v1 或非 chunked 文件会自动重算。
* `--save_compression chunked_zstd` —— 默认值，与 training 一致。`gzip/zstd/none`
  仅保留用于调试，不应作为正式 validation cache 格式。
* `--asset_yaw_correction_deg 180 --max_pose_refine_yaw_deg 15` —— 与 Mode A 默认一致。
* `--sync_save` —— 同步落盘（默认后台线程异步写）。

> **显存**：与 Mode A 预计算一致，29 帧 VGGT-L + asset pass + splat 需要一块较空的 GPU
> （建议 ≳25GB 空闲）。共享机上若只有几 GB 空闲会 OOM（脚本会捕获并以 `err` 计数继续）。

## 2. 构建 validation manifest

```bash
python tools/build_flow_validation_manifest.py \
    --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_validation \
    --split validation \
    --out_path  /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/validation/validation_manifest.jsonl
```

每行（与 `build_flow_train_manifest.py` 同形，下游可直接消费）：

```json
{"index": 60, "mode_kind": "mode_a", "split": "validation",
 "scene_name": "validation_027", "clip_name": "12496433400137459534_120_000_140_000_3",
 "variant": "combined", "validation_entry_index": 12,
 "clip_start": 0, "num_frames": 29, "num_objects": 2,
 "cache_path": ".../flow_cache_validation/validation/000012_combined.pt"}
```

`index = entry_index*5 + variant_ord`
（`combined/deletion/insertion/replacement/repositioning` = 0..4），保证唯一。
也会写 `*.summary.json`（各 variant 计数）。

## 3. 校验

```bash
# 3.1 Smoke：entry 0 的四种编辑在 29 帧内均有目标，生成 5 个 .pt
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u tools/precompute_flow_features_validation.py \
    --ckpt_path /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
    --asset_root /data/disk2/lyy_dataset/test_transfer/objects_ply_transformed \
    --out_root /tmp/valcache_smoke --start 0 --end 1 --sync_save
ls /tmp/valcache_smoke/validation/   # 000000_combined.pt ... 000000_repositioning.pt

# 3.2 Schema 与 Mode A 逐字段对齐
PYTHONPATH=. python -c "
from dggt.utils.flow_cache_io import is_chunked_flow_cache, load_flow_cache
a=load_flow_cache('/data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_a/training/000190.pt')
b=load_flow_cache('/tmp/valcache_smoke/validation/000000_combined.pt')
assert set(a)==set(b), (set(a)^set(b))
assert is_chunked_flow_cache('/tmp/valcache_smoke/validation/000000_combined.pt')
assert a['schema_version']==b['schema_version']==8 and b['mode_kind']=='mode_a'
assert b['pass1']['gs_conf'].dtype == __import__('torch').float32
for k in ('pass1','phase1_localized','pass2_splatted_tok_low'):
    assert set(a[k])==set(b[k]), (k, set(a[k])^set(b[k]))   # binding blocks identical
print('schema OK')"
# 注：meta 多了 variant / validation_edit 等附加键（仅追加，下游用 .get 读取，安全）。

# 3.3 下游闸门：用 WaymoFlowCacheDataset 直接消费 validation manifest
PYTHONPATH=. python tools/build_flow_validation_manifest.py --cache_root /tmp/valcache_smoke \
    --split validation --out_path /tmp/valcache_smoke/validation_manifest.jsonl
PYTHONPATH=. python -c "
from datasets.waymo_flow_cache_dataset import WaymoFlowCacheDataset
ds=WaymoFlowCacheDataset(manifest_path='/tmp/valcache_smoke/validation_manifest.jsonl')
for i in range(len(ds)): ds[i]
print('downstream OK', len(ds), 'rows')"

# 3.4 WYSIWYG（整 29 帧，Mode A 同工具可直接读取）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run -n dggt --no-capture-output \
    python -u tools/verify_flow_cache_wysiwyg.py \
    --cache_path /tmp/valcache_smoke/validation/000000_combined.pt \
    --output_dir runs/val_wysiwyg_combined --skip_live_compare --splat_pca --nrow 6

# 3.5 对齐 sanity：tar Waymo 3D 框投影 vs DGGT 精修框（确认同一世界系）
#     需要 .tfrecord 复算时用：conda activate waymo160
```

通过标准：

* entry 0 的 `000000_{combined,deletion,insertion,replacement,repositioning}.pt` 5 个齐全。
* 3.2 打印 `schema OK`（顶层 key 全等；`pass1`/`phase1_localized`/`pass2_splatted_tok_low`
  绑定块与 Mode A 子 key 全等；`meta` 仅多出 `variant`/`validation_edit` 等附加键）。
* 3.3 `WaymoFlowCacheDataset` 5 行全部取样不抛错（过 `_validate_v6_payload` + `_subset_phase1_localized`
  + `_build_asset_pass` + `_subset_pass2_splatted_tok_low`）。
* 3.4 WYSIWYG 在 `runs/val_wysiwyg_combined/` 下生成 mask/coverage/scaffold/depth 可视化网格。

> 历史 smoke 使用 entry 12，其中 move 是无目标的合法 no-op。现在建议使用 entry 0，
> 因为 deletion/insertion/replacement/repositioning 的目标在 29 帧窗口内都存在，
> 可覆盖完整路径。

## 3.6 完整可视化（RGB + flow_features，WYSIWYG）

`verify_flow_cache_wysiwyg.py` 只产出 `flow_features/{masks,coverage,scaffold,depth}`，**没有 RGB
渲染结果**。用 `tools/visualize_flow_cache.py` 产出与 `docs/flow_cache_cmd.md` 的 Mode A
`inference_scene_editor.py --dump_features` **完全一致**的可视化集合，且全部从 `.pt` 反读
（走训练同款 `WaymoFlowCacheDataset → build_clean_scene_state → FlowFeatureAssembler` 路径，
保证所见即所得）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run -n dggt --no-capture-output \
    python -u tools/visualize_flow_cache.py \
    --cache_path /tmp/valcache_smoke/validation/000000_combined.pt \
    --output_dir runs/flow_cache_vis --splat_pca --nrow 6
# 显存充足时加 --ckpt_path <dggt.pt> 用真实 scene_tokenizer 存真 flow_features.pt
```

entry 0 的五个文件依次是：

```text
000000_combined.pt
000000_deletion.pt
000000_insertion.pt
000000_replacement.pt
000000_repositioning.pt
```

检查删除是否正确时，至少分别可视化 `000000_deletion.pt`、
`000000_replacement.pt`、`000000_repositioning.pt`：
`deleted_render_grid.jpg` 应分别显示三个源目标并集、replacement 源目标、move 源目标被移除；
`edited_grid.jpg` 则应在 replace/move 目标位置重新出现对应 PLY。

输出 `runs/flow_cache_vis/{name}/`：

```
input_grid.jpg            # 缓存原始输入帧（raw.images_u8）
clean_render_grid.jpg     # 全部场景高斯（编辑前）合成在输入上
deleted_render_grid.jpg   # 保留高斯（= 删除目标后的场景）          ← 对齐 docs/flow_cache_cmd.md
asset_image_grid.jpg      # 各资产槽缓存 I_asset 拼接               ← 对齐 docs/flow_cache_cmd.md
asset_slot{03,04,05}_grid.jpg / asset_alpha_slot{XX}_grid.jpg
edited_grid.jpg           # 删除渲染 + 资产合成（最终编辑结果）
flow_features/{flow_features.pt, masks/, coverage/, scaffold/, depth/}  ← 对齐 docs/flow_cache_cmd.md
visualize_summary.json
```

RGB 用缓存权威的 `cameras_dggt` + 缓存高斯（保留=clean 去掉
`phase1_localized.delete_mask`；资产=`asset_pass.G_asset_dggt`/`I_asset`）直接 gsplat 渲染，
不需要模型；`flow_features/` 由训练同款 `FlowFeatureAssembler` + `dump_flow_features` 产出，
与 `verify_flow_cache_wysiwyg` / 训练消费的 bundle 逐值对齐。正式 validation
产物统一为 v8 chunked cache。

## 4. 实现要点（与 Mode A 的复用关系）

* `datasets/waymo_validation_edit_dataset.py:WaymoValidationEditDataset`
  —— entry → 6 槽位 sample（0–2 删除源 / 3–5 资产目标）。删除槽
  `object_valid_mask=True`；资产槽 `object_valid_mask=False`（使
  `_collect_protected_boxes` 跳过，避免 slot4=slot1 框反而“保护”了要删的替换源）。
* `dggt/utils/validation_edit_localize.py:localize_validation_objects`
  —— 删除槽走 `editor.localize(..., load_asset=False)`；资产槽自定义放置
  —— insertion/replacement/reposition 统一读取各自目标 tar，执行
  `_transform_track_box`、2D bbox/depth center refine 和跨帧 median translation
  stabilization。reposition tar 已含 3m 位移，不再调用额外 shift。三个资产槽都通过
  `object_asset_paths` 加载 PLY；其中 slot 5 正确读取
  `--asset_root/{repositioning_id}.ply`，这是当前 move 设计的标准实现。
* `tools/precompute_flow_features_validation.py` —— 复用
  `_pack_pass1_tokens / _build_training_predictions_from_cache_payload /
  _pack_mode_a_asset_pass_result / _pack_phase1_localized /
  _compute_and_pack_pass2_splatted_tok_low / _build_object_meta /
  AsyncFlowCacheWriter`（均未改动）。
* `tools/build_flow_validation_manifest.py` —— 扫描产物写 JSONL（含 `variant` 字段）。

## 5. 与 training Mode A 的算法关系

不是“只换输入，所有步骤逐行完全相同”。准确关系如下：

| 阶段 | 与 training 是否相同 | 说明 |
|---|---|---|
| VGGT Pass-1、v8 量化与 chunked 保存 | 相同 | 复用 training 的 packer、finite-fp32 `gs_conf` 和 chunked writer |
| clean scene、Waymo→DGGT Sim3 | 相同 | 同一个 `GaussianSceneEditor.build_clean_bundle/align` |
| 删除目标定位 | 相同 | slot 0/1/2 直接调用 training 的 `editor.localize(..., load_asset=False)` |
| 三类资产目标位置 | 相同 placement primitive | 各自逐帧目标 tar 框经 Sim3，再执行 training 同款 fixed-rotation 2D bbox/depth center refine，并使用同款跨帧 median translation stabilization |
| replacement/move 目标来源 | validation 专用输入 | 分别来自 `all_object_info_replacement` / `all_object_info_reposition`；不复用删除定位框，move 不重复施加 3m shift |
| PLY 尺寸拟合与变换 | 相同 primitive | 复用 `_compute_asset_scale_factors` 和 `_transform_asset_gaussians_simple` |
| AssetAggregatorPass | 相同 | 同样的 DGGT-fitted render、asset LUT、pointer、遮挡测试 |
| 删除 mask 打包、pass2 splat/blend | 相同 | 复用 `_pack_phase1_localized` 和 `_compute_and_pack_pass2_splatted_tok_low` |

删除语义：

* `deletion`/`combined` 对 slot 0、1、2 的删除 mask 做并集；
* `replacement` 只保留 slot 1 的删除 mask，`repositioning` 只保留 slot 2，
  `insertion` 不删除；
* core 和 shell Gaussian 都由 `apply_mode_a` 标为删除；
* 三个删除源同时进入 stock localizer，彼此会作为 protected box，避免一个目标的
  删除簇侵入另一个目标。

需要注意：删除不是按 Waymo 3D 框直接硬裁剪，而是 training 同款的“2D 语义连通域 +
深度/3D 框约束 + Gaussian 连通簇”定位。若某帧语义或可见点不足，localizer 会跳过该帧；
因此正式全量生成后必须查看 `_errors.jsonl`、manifest 数量和 RGB/mask 可视化，不能只以
脚本退出码判断删除质量。

资产目标框与删除定位已经解耦：即使某帧删除语义定位失败，资产仍可按目标 tar 放置。
这不代表删除质量自动合格；`replacement`/`repositioning` 的
`deleted_render_grid.jpg` 仍是正式验收项。
目标 tar 中没有该对象记录的帧会保持 no-op，不会从相邻帧或删除框臆造目标位置。
缓存 metadata 中记录
`validation_localization_policy=target_tar_member_index_sim3_bbox_depth_shared_delta_v3`，
旧定位策略生成的
v8 cache 会自动重算。

测试环境：`conda activate dggt`；如需读 `.tfrecord` 做对齐复核：`conda activate waymo160`。
