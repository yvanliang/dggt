# Validation Flow 缓存流水线（Mode A 语义对齐）

本文档介绍 FlowDGGT **validation** 离线缓存的生成。它与 `docs/flow_cache_cmd.md`
的 Mode A 共享**完全相同的磁盘 schema**（`schema_version=6`、`mode_kind="mode_a"`），
因此下游 `WaymoFlowCacheDataset` / `FlowFeatureAssembler` / `SceneFlowMatching`
**零改动**消费。

与训练 Mode A 的区别：

| 维度 | 训练 Mode A | Validation |
|---|---|---|
| 编辑配置 | `final_info.json` + `build_edit_metadata.py` | `data/final_info_validation.json`（33 条） |
| 删除/资产 | 同一 Waymo 目标：删除后用其自身资产重渲染 | **解耦**：删 3 个源目标，插 3 个外部/复用资产 |
| 3D 框来源 | `instances_info.json` | `data/validation_info/all_object_info*` 的 tar（**不读 tfrecord**） |
| 每条产物 | 1 个 `.pt` | **5 个 `.pt`**：`combined` + `delete`/`add`/`replace`/`move` |

每条 validation entry 的 4 个编辑：

* **删除(delete)**：从高斯中删 `origin_object_dict.deletion`、`.replacement`、`.repositioning` 三个源目标。
* **添加(add)**：在 `all_object_info_insertion`（`insertion_0`）的逐帧框处插入 `insertion_candidates` 资产。
* **替换(replace)**：删 `replacement` 源目标，在其（精修后）框处插入 `replacement_candidates` 资产。
* **移动(move)**：删 `repositioning` 源目标，用其**自身重建资产**在“原框沿物体局部轴按
  `action_for_reposition` 平移 3.0m”处插入（对齐 `pointcloud_validation` 的
  `shift_object_tfm_by_action`，REPOSITION_DISTANCE_M=3.0）。

`combined` = 同时做以上 4 个编辑；`delete/add/replace/move` = 单一编辑类型，便于分类评测。

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
  `data/validation_info/all_object_info/{段名}.tar`、`data/validation_info/all_object_info_insertion/{段名}.tar`。
* 资产 `.ply`：单一目录 `--asset_root`（默认
  `/data/disk2/lyy_dataset/test_transfer/objects_ply_transformed`，文件名 `{id}.ply`）。
  **缺失资产** → 该 entry 跳过并记入 `_errors.jsonl`（把缺失 id 复制进 `--asset_root` 后重跑即可）。

## 1. 生成 validation 缓存

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u tools/precompute_flow_features_validation.py \
    --ckpt_path /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
    --asset_root /data/disk2/lyy_dataset/test_transfer/objects_ply_transformed \
    --out_root  /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_validation
```

输出（**扁平布局**，`index = entry_index*5 + variant_ord`，
`variant_ord` = combined0/delete1/add2/replace3/move4，使未改动的 Mode A 工具链
[verify_flow_cache_wysiwyg / WaymoFlowCacheDataset cache_root 模式] 直接可用）：

```
{out_root}/validation/{index:06d}.pt          # 例：entry 12 → 000060..000064
{out_root}/validation/_errors.jsonl           # {index, clip_name, reason, missing_asset_ids:[...]}
```

每条 entry **只跑一次 VGGT Pass-1 + 一次解耦定位（含位姿精修）**，再按 5 个 variant
切分已定位对象、分别跑 asset pass + phase1 打包 + splat/blend，组装与
`precompute_one_clip` 完全一致的 payload。

常用参数（`--help` 查看全部）：

* `--variants combined,delete` —— 只生成指定 variant（调试 / 省时）。
* `--start N --end M` —— 只处理 entry 索引 `[N, M)`（0..32）。
* `--force_overwrite` —— 覆盖已存在的 `.pt`（默认跳过已存在的）。
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
 "cache_path": ".../flow_cache_validation/validation/000060.pt"}
```

`index = entry_index*5 + variant_ord`（`combined/delete/add/replace/move` = 0..4），保证唯一。
也会写 `*.summary.json`（各 variant 计数）。

## 3. 校验

```bash
# 3.1 Smoke：选一条资产齐全的 entry，生成 5 个 .pt
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -u tools/precompute_flow_features_validation.py \
    --ckpt_path /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
    --asset_root /data/disk2/lyy_dataset/test_transfer/objects_ply_transformed \
    --out_root /tmp/valcache_smoke --start 12 --end 13 --sync_save
ls /tmp/valcache_smoke/validation/   # 000060.pt 000061.pt 000062.pt 000063.pt 000064.pt

# 3.2 Schema 与 Mode A 逐字段对齐
PYTHONPATH=. python -c "
from dggt.utils.flow_cache_io import load_flow_cache
a=load_flow_cache('/data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_a/training/000190.pt')
b=load_flow_cache('/tmp/valcache_smoke/validation/000060.pt')   # entry 12 combined
assert set(a)==set(b), (set(a)^set(b))
assert a['schema_version']==b['schema_version']==6 and b['mode_kind']=='mode_a'
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

# 3.4 WYSIWYG（整 29 帧，Mode A 同工具，扁平布局直接可用）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run -n dggt --no-capture-output \
    python -u tools/verify_flow_cache_wysiwyg.py \
    --cache_path /tmp/valcache_smoke/validation/000060.pt \
    --output_dir runs/val_wysiwyg_combined --skip_live_compare --splat_pca --nrow 6

# 3.5 对齐 sanity：tar Waymo 3D 框投影 vs DGGT 精修框（确认同一世界系）
#     需要 .tfrecord 复算时用：conda activate waymo160
```

通过标准：

* entry 12 → `/tmp/valcache_smoke/validation/000060..000064.pt` 5 个齐全（资产缺失时 `_errors.jsonl` 有对应行）。
* 3.2 打印 `schema OK`（顶层 key 全等；`pass1`/`phase1_localized`/`pass2_splatted_tok_low`
  绑定块与 Mode A 子 key 全等；`meta` 仅多出 `variant`/`validation_edit` 等附加键）。
* 3.3 `WaymoFlowCacheDataset` 5 行全部取样不抛错（过 `_validate_v6_payload` + `_subset_phase1_localized`
  + `_build_asset_pass` + `_subset_pass2_splatted_tok_low`）。
* 3.4 WYSIWYG 在 `runs/val_wysiwyg_combined/` 下生成 mask/coverage/scaffold/depth 可视化网格。

> 已在 entry 12（资产齐全）实测：5 个 variant 全产出，schema 与下游加载、语义计数
> （combined: 2 资产/135022 删除；delete: 0/135022；add: 1/0；replace: 1/107782；
> move: 0/0——该 clip 帧窗内无 reposition 目标，合法 no-op）均通过。

## 3.6 完整可视化（RGB + flow_features，WYSIWYG）

`verify_flow_cache_wysiwyg.py` 只产出 `flow_features/{masks,coverage,scaffold,depth}`，**没有 RGB
渲染结果**。用 `tools/visualize_flow_cache.py` 产出与 `docs/flow_cache_cmd.md` 的 Mode A
`inference_scene_editor.py --dump_features` **完全一致**的可视化集合，且全部从 `.pt` 反读
（走训练同款 `WaymoFlowCacheDataset → build_clean_scene_state → FlowFeatureAssembler` 路径，
保证所见即所得）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run -n dggt --no-capture-output \
    python -u tools/visualize_flow_cache.py \
    --cache_path /tmp/valcache_smoke/validation/000060.pt \
    --output_dir runs/flow_cache_vis --splat_pca --nrow 6
# 显存充足时加 --ckpt_path <dggt.pt> 用真实 scene_tokenizer 存真 flow_features.pt
```

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
与 `verify_flow_cache_wysiwyg` / 训练消费的 bundle 逐值对齐。该工具对任意 v6 缓存
（含训练 Mode A / Mode B）通用，不限 validation。

## 4. 实现要点（与 Mode A 的复用关系）

* `datasets/waymo_validation_edit_dataset.py:WaymoValidationEditDataset`
  —— entry → 6 槽位 sample（0–2 删除源 / 3–5 资产目标）。删除槽
  `object_valid_mask=True`；资产槽 `object_valid_mask=False`（使
  `_collect_protected_boxes` 跳过，避免 slot4=slot1 框反而“保护”了要删的替换源）。
* `dggt/utils/validation_edit_localize.py:localize_validation_objects`
  —— 删除槽走 `editor.localize(..., load_asset=False)`；资产槽自定义放置
  （insertion=`_transform_track_box`+可选 depth-snap；replacement/reposition 复用删除
  槽精修框，reposition 再做 `_apply_reposition_shift`）。
* `tools/precompute_flow_features_validation.py` —— 复用
  `_pack_pass1_tokens / _build_training_predictions_from_cache_payload /
  _pack_mode_a_asset_pass_result / _pack_phase1_localized /
  _compute_and_pack_pass2_splatted_tok_low / _build_object_meta /
  AsyncFlowCacheWriter`（均未改动）。
* `tools/build_flow_validation_manifest.py` —— 扫描产物写 JSONL（含 `variant` 字段）。

测试环境：`conda activate dggt`；如需读 `.tfrecord` 做对齐复核：`conda activate waymo160`。
