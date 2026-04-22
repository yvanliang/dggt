# Waymo 全量数据处理命令（`waymo_processed_dggt` 版本）

## 注意

- 仓库里的 `data/waymo_train_list.txt` 只有 49 行，不是全量 training split。要处理全部 Waymo scene，不能直接用它。
- `build_asset_cache.py` 会写自己的 `asset_ready_index.json`，所以 training 和 validation 必须用不同的 `asset_root`。

## 全量运行命令

### 1. 准备目录和空 scene list

```bash
mkdir -p /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/{lists,annotations,metadata,assets,manifests}
: > /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/lists/all_scenes.txt
```

### 2. 预处理全部 training scenes

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 python datasets/preprocess_waymo.py \
  --data_root /data/disk2/lyy_dataset/waymo_flow \
  --target_dir /data/disk2/lyy_dataset/waymo_processed_dggt \
  --dataset waymo \
  --split training \
  --scene_list_file /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/lists/all_scenes.txt \
  --num_workers 10 \
  --process_keys images lidar calib pose ground dynamic_masks objects \
  --json_folder_to_save /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/annotations
```

### 3. 预处理全部 validation scenes

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 python datasets/preprocess_waymo.py \
  --data_root /data/disk2/lyy_dataset/waymo_flow \
  --target_dir /data/disk2/lyy_dataset/waymo_processed_dggt \
  --dataset waymo \
  --split validation \
  --scene_list_file /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/lists/all_scenes.txt \
  --num_workers 10 \
  --process_keys images lidar calib pose ground dynamic_masks objects \
  --json_folder_to_save /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/annotations
```

### 4. 提取 sky / fine dynamic masks

需要先进入装好 mmseg 和 SegFormer 的环境。  
`sky masks` 是 required；`fine dynamic masks` 是 optional，但这里也一并生成（通过 `--process_dynamic_mask`）。

```bash
conda activate segformer
segformer_path=/home/dancer/code/tools/SegFormer/
checkpoint=/home/dancer/code/tools/SegFormer/segformer.b5.1024x1024.city.160k.pth
```

```bash
python datasets/tools/extract_masks.py \
  --data_root /data/disk2/lyy_dataset/waymo_processed_dggt/training \
  --start_idx 0 \
  --num_scenes 798 \
  --process_dynamic_mask \
  --segformer_path $segformer_path \
  --config $segformer_path/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py \
  --checkpoint $checkpoint \
  --device cuda:0
```

```bash
python datasets/tools/extract_masks.py \
  --data_root /data/disk2/lyy_dataset/waymo_processed_dggt/validation \
  --start_idx 0 \
  --num_scenes 202 \
  --process_dynamic_mask \
  --segformer_path $segformer_path \
  --config $segformer_path/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py \
  --checkpoint $checkpoint \
  --device cuda:0
```

可选：如果你想按文件或索引范围处理，也可以用下面写法（`--process_dynamic_mask` 同样保留）：

```bash
python datasets/tools/extract_masks.py \
  --data_root /data/disk2/lyy_dataset/waymo_processed_dggt/validation \
  --segformer_path $segformer_path \
  --checkpoint $checkpoint \
  --split_file data/waymo_example_scenes.txt \
  --process_dynamic_mask
```

`--split_file data/waymo_example_scenes.txt` 可替换为 `--start_idx 0 --num_scenes 200`，或省略以使用默认设置。

### 5. 构建 metadata + manifest

- 输入标注固定为仓库内的 `data/final_info.json`
- 每条 `final_info.json` 记录直接对应一个 29 帧 clip
- `object_list` 里的 id 既是 scene 中的目标 `raw_object_id`，也是要加载的 asset id
- asset 直接读取已经存在的 `/data/disk2/lyy_dataset/test_transfer/objects_ply_transformed/*.ply`
- `final_info.json` 里的 2D bbox 坐标是相对 DGGT transfer 图像 `704x1280`，不是 Waymo 原图；因此尺寸过滤阈值固定为长边 `1280 / 10 = 128 px`
- `build_edit_metadata.py` 会先按 frame/view 做两个过滤：
  - 小目标过滤：若 bbox 宽或高小于 `128 px`，该 frame/view 不编辑该目标
  - 遮挡过滤：若同一 frame/view 中两个目标 bbox 的交集面积分别除以任一目标框面积后超过 `0.8`，则读取 Waymo 3D track，比较目标中心到相机的深度，删除更远的那个
- `build_edit_metadata.py` 现在会同时写：
  - `metadata/<split>/mode_a_candidates.jsonl`
  - `manifests/<split>/<split>_mode_a_views1.jsonl`
  - `manifests/<split>/<split>_mode_a_views3.jsonl`
- 不再写旧的 `manifests/<split>/<split>_mode_a.jsonl` 别名；切到这套 schema 后必须重建 cache，不能继续复用旧 manifest
- 候选 metadata 统一只保留新命名字段：
  - object 级：`bbox_present_by_view`、`bbox_editable_by_view`
  - clip 级：`frame_has_front_present_object`、`frame_has_front3_present_object`
  - clip 级：`frame_has_front_editable_object`、`frame_has_front3_editable_object`
  - clip 级：`editable_object_slots_by_frame_front`、`editable_object_slots_by_frame_front3`
- `views=1` 版本会删除整个 29 帧 clip 中 `front` 始终没有可编辑目标的样本
- `views=3` 版本会删除整个 29 帧 clip 中 `front/front_left/front_right` 都始终没有可编辑目标的样本

```bash
python datasets/tools/build_edit_metadata.py \
  --processed_root /data/disk2/lyy_dataset/waymo_processed_dggt \
  --output_root /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache \
  --split training \
  --final_info_path /home/dancer/code/dm/dggt/data/final_info.json \
  --asset_root /data/disk2/lyy_dataset/test_transfer/objects_ply_transformed
```

如果后续你有单独的 validation 标注 json，再把 `--split validation` 和对应的 `--final_info_path` 换进去；当前仓库里只有 training 用的 `data/final_info.json`。

### 6. 跑 Hunyuan3D-2 + Mesh2Splat

相关代码为 `/home/dancer/code/3d/hunyuan3d-2.1/generate_textured_mesh_and_render.py`。

## 建议先验收的文件

- processed:
  - `/data/disk2/lyy_dataset/waymo_processed_dggt/training/000/instances/object_id_map.json`
  - `/data/disk2/lyy_dataset/waymo_processed_dggt/training/000/appearances_vehicle.json`
- metadata:
  - `/data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/metadata/training/mode_a_candidates.jsonl`
  - `/data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/metadata/training/metadata_summary.json`
- assets:
  - `/data/disk2/lyy_dataset/test_transfer/objects_ply_transformed/0QXtLAoMcF26x6k0m-7gVQ.ply`
- manifest:
  - `/data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_mode_a_views1.jsonl`
  - `/data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_mode_a_views3.jsonl`
