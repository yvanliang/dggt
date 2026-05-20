# Flow 缓存流水线（Mode A + Mode B）

本文档介绍 FlowDGGT 的完整离线缓存 → 扩散训练数据流水线。运行完这些步骤后，你将得到：

* `/data/flow_cache_mode_a/{split}/{scene}/{clip:04d}.pt` — Mode A 片段（确定性删除 GT Waymo 目标，并带有成对的资产图像渲染）。
* `/data/flow_cache_mode_b/{split}/{scene}/{clip:04d}.pt` — Mode B 片段（规划器想象的目标位置 + 场景 Gaussian 伪删除）。
* `/data/flow_cache/{split}_manifest.jsonl` — 扩散数据加载器使用的合并训练 manifest（每个片段一行，两种模式混合在一起）。

两种缓存共享相同的磁盘 schema。它们只在 payload 顶层的 `mode_kind` 字段以及下面两个 sibling 中哪一个被填充上有所不同：

| 字段 | Mode A | Mode B |
|--------------|---------------------------------------|-------------------------------------|
| `mode_kind` | `"mode_a"` | `"mode_b"` |
| `asset_pass` | `{int → {I_asset, A_asset, F_g_lut_asset_int8, ptr_*, G_asset_*}}` | `{}` |
| `mode_b` | `None` | `{imagined_objects, delete_mask, delete_mask_per_frame, …}` |
| `phase1_localized` | `{slot_idx, frame_idx, source_front_index, delete_mask, shell_mask, …}` | `None` |
| `pass2_splatted_tok_low` | tokenizer 前的 splat→blend 特征（int8） | tokenizer 前的 splat→blend 特征（int8） |
| `pass1` | 相同（gs_map / depth / dyn / int8 LUTs / cameras_dggt） | 相同 |
| `raw` | 相同（images_u8, sky_mask, dynamic_mask） | 相同 |
| `meta`, `object_meta` | 相同 | 相同 |

下游模块（`WaymoFlowCacheDataset` + `FlowFeatureAssembler`）会检查 `mode_kind`，并路由到对应的在线代码路径。

## Cache 语义：full-source-Gaussian splat

v7 的 `pass2_splatted_tok_low` 缓存的是 tokenizer 之前的 splat→blend 特征。它的 source Gaussian 集合是完整 clip 的所有帧；训练时随机选连续 4-8 帧窗口时，dataset 只在 target frame 维度做 `index_select`。v7 还把 `pass1.gs_conf` 从旧版 fp16 改为 finite fp32，避免大置信度值溢出为 `inf`。

因此 `cache.index_select(subset)` 不应该和“先把 `gs_map` 裁成 subset，再 live 重跑 `FeatureSplatter`”逐 token 相等。后者的 source Gaussian 少了其他帧，遮挡和补洞都会变。验证时应检查 full clip cache 与 full clip live recompute 一致；对子集只检查缓存切片结构正常，以及 mask、`z_clean`、asset tokens、scaffold 等非 pass2 字段一致。这个语义更接近实际推理：推理通常对完整 clip 做 live splat。


## 0. 前置条件

```bash
conda activate dggt
```

首先构建 Mode A 和 Mode B 的候选项 / manifest JSONL（每个 split 只需执行一次）。这些步骤不会使用 GPU —— 它们只读取 Waymo 标注和 Phase 1 元数据：

```bash
# Mode A 候选项 / manifest（现有工具）
python datasets/tools/build_edit_metadata.py --split training
# → /processed_root/waymo_edit_cache/metadata/training/mode_a_candidates.jsonl
# → /processed_root/waymo_edit_cache/manifests/training/training_mode_a_views1.jsonl

# Mode B 候选项 / manifest
python datasets/tools/build_mode_b_manifest.py --split training
# → /processed_root/waymo_edit_cache/metadata/training/mode_b_candidates.jsonl
# → /processed_root/waymo_edit_cache/manifests/training/training_mode_b_views1.jsonl
```

Mode B 构建器会排除已经被 Mode A 使用的片段，因此两个 split 不会重叠（合并后的 manifest 中每个 scene/clip 也不会有重复项）。


## 1. Mode A 预计算（目标删除 + 资产图像特征）

生成 Mode A 缓存：VGGT Pass 1 + Phase 1 对齐 + Phase 4（`AssetAggregatorPass`）逐目标资产 Gaussian 特征和渲染。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/precompute_flow_features.py \
    --edit_mode mode_a \
    --ckpt_path /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
    --split training \
    --out_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_a \
    --manifest_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_mode_a_views1.jsonl \
    --candidate_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/metadata/training/mode_a_candidates.jsonl \
    --views 1
```

如果已有 v6 cache，只想升级旧文件并保留已生成的 v7 文件，可在 Mode A / Mode B 预计算命令里加 `--overwrite_v6`。它会读取已存在 `.pt` 的 `schema_version`：v7 直接跳过，非 v7 或无法读取的文件会重新生成并覆盖。`--force_overwrite` 仍表示无条件重算覆盖。

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python inference_scene_editor.py \
    --output_dir runs/mode_a_all_vis \
    --ckpt_path /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
    --dump_features

# 输出可视化结果：runs/mode_a_smoke/{deleted_render_grid, asset_image_grid,
#          flow_features/{flow_features.pt, masks/, coverage/, scaffold/, depth/}}
```

`--dump_features` 会直接调用训练同用的 `FlowFeatureAssembler`，因此
`flow_features/masks/{M_preserve,M_source,M_dest}_grid.jpg` 可视化的就是训练实际消费的 mask。


## 2. Mode B 预计算（想象目标放置 + 伪删除）

生成 Mode B 缓存：VGGT Pass 1 + `ModeBPlanner.plan(...)` + `apply_mode_b(...)`。不包含 asset pass —— 扩散模型会幻化新内容。

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python tools/precompute_flow_features.py \
    --edit_mode mode_b \
    --ckpt_path /home/dancer/liangyiyuan/dggt/model_latest_waymo.pt \
    --split training \
    --out_root /data/intelssd/liangyiyuan/waymo_processed_dggt/flow_cache_mode_b \
    --views 1 \
    --planner_seed 0 \
    --allow_empty_plan --start 500
```

`--allow_empty_plan` 会为规划器无法满足条件的片段保留缓存文件（这样 manifest 条目可以与 manifest split 保持 1:1）。如果不使用它，脚本会报错并跳过该片段。

规划器阈值（默认值与 `inference_mode_b.py` 一致）可以通过 `--min_visible_frames`、`--max_semantic_overlap_px`、`--max_trials_per_object` 参数进行调整。完整参数列表请查看 `tools/precompute_flow_features.py --help`。

如果要对一个 Mode B 样本做临时可视化（想象框叠加 + 伪删除点云渲染 + D_map）：
```bash
CUDA_VISIBLE_DEVICES=3 python inference_mode_b.py --output_dir runs/mode_b_all_vis \
    --ckpt_path /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
    --planner_seed 0 \
    --dump_features
# 输出：runs/mode_b_all_vis/{imagined_boxes_overlay, deleted_render_grid,
#          d_map_grid, mode_b_summary.json,
#          flow_features/{masks/, coverage/, scaffold/, depth/}}
```

Mode B 的 `--dump_features` 同样走训练用 `FlowFeatureAssembler(mode_kind="mode_b")`，
不会另写一套 mask 计算逻辑；导出的 mask 是 bundle 中的
`M_preserve/M_source/M_dest` 的 JPG 可视化。注意：非空 imagined 区域会触发一次
训练同路径的 feature splat，因此比只画框和 `D_map` 慢。


## 3. 构建合并训练 manifest

扩散数据加载器会读取一个覆盖两种模式的 manifest。使用：

```bash
python tools/build_flow_train_manifest.py \
    --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_a:mode_a \
    --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_b:mode_b \
    --split training \
    --out_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl
```

`--cache_root` 可以重复指定。可选的 `:mode_a` / `:mode_b` 后缀会固定模式，不需要打开任何 `.pt`（低成本路径）；`:auto` 或不加后缀会强制脚本查看每个文件的 `mode_kind`（更慢，在 NVMe 上约为 1 s/clip）。

输出：

```
/data/flow_cache/training_manifest.jsonl          # 每个片段一行
/data/flow_cache/training_manifest.jsonl.summary.json
```

每一行 manifest：
```json
{"mode_kind":"mode_a","split":"training","scene_name":"003",
 "clip_name":"003_0","clip_start":0,"num_frames":29,"num_objects":2,
 "cache_path":"/data/flow_cache_mode_a/training/003/0000.pt"}
```

对 `--split validation` 重复上述步骤，即可构建 held-out manifest。


## 4. 训练扩散模型

```bash
torchrun --nproc_per_node=8 train_scene_flow.py \
    --ckpt_path /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
    --manifest_path /data/flow_cache/training_manifest.jsonl \
    --log_dir runs/flow_t1 \
    --batch_size 1 --grad_accum_steps 2 \
    --min_frames 4 --max_frames 8 \
    --max_steps 40000 --vis_every 1000
```

常用选项：

* `--mode_filter mode_a` — 仅限制为 Mode A（调试）。同一个参数也接受 `--mode_filter mode_a,mode_b`（默认）。
* `--cache_root /data/flow_cache_mode_a` — 绕过 manifest，直接遍历一个缓存目录（单模式运行）。

在每个 `--vis_every` step，rank-0 会将一组可视化结果保存到 `runs/flow_t1/vis/step_<N>/flow_features/` 下：

* 两种模式都有：`flow_features.pt`、`masks/`、`coverage/`、`scaffold/`、`depth/`。
* 仅 Mode B：额外包含 `mode_b/{clean_grid, I_map_grid, imagined_boxes_overlay}.jpg`。

仅 Mode A：`coverage/` 中已经包含每个资产目标对应的 `I_per_obj_slot{XX}_grid.jpg`。


## 5. 验证

```bash
# Sanity：缓存 schema + 规划器集成
pytest tests/test_offline_cache.py tests/test_mode_b_planner.py \
       tests/test_scene_pointers.py tests/test_per_token_noise.py -q

# WYSIWYG / pass2 校验说明：
# - 校验对象是完整 29 帧 cache，不是训练时随机采样出的连续 4-8 帧窗口。
# - 默认会从 .pt 反读完整 29 帧，输出 cache-derived 可视化，并用完整 29 帧
#   live assembler 重算 pass2_splatted_tok_low，和 .pt 中保存的 int8/scale 做逐值校验。
# - 默认使用 zero tokenizer stub；这是因为 cache 实际保存的是 tokenizer 前的
#   pass2_splatted_tok_low。需要额外验证/保存 29 帧 latent 时再加 --with_tokenizer --ckpt_path。
# - --chunk_channels 默认是 64，和当前 verifier 默认值一致；显存紧张时可手动降到 32。

# Mode A：完整 29 帧 WYSIWYG 校验。将 000000.pt 换成实际存在的 Mode-A cache。
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. conda run -n dggt --no-capture-output \
    python -u tools/verify_flow_cache_wysiwyg.py \
    --cache_path /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_a/training/000000.pt \
    --output_dir runs/flow_cache_wysiwyg_mode_a \
    --nrow 4

# Mode B：完整 29 帧 WYSIWYG 校验。将 002025.pt 换成实际存在的 Mode-B cache。
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. conda run -n dggt --no-capture-output \
    python -u tools/verify_flow_cache_wysiwyg.py \
    --cache_path /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_b/training/002025.pt \
    --output_dir runs/flow_cache_wysiwyg_mode_b \
    --nrow 4

# 快速只看 cache-derived 可视化，不重算 live pass2：
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. conda run -n dggt --no-capture-output \
    python -u tools/verify_flow_cache_wysiwyg.py \
    --cache_path /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_b/training/002025.pt \
    --output_dir runs/flow_cache_wysiwyg_mode_b_quick \
    --skip_live_compare \
    --splat_pca \
    --nrow 4

# 显存充足时，额外使用真实 scene_tokenizer 生成并保存 flow_features.pt。
# 这一步会显著增加 29 帧验证的显存占用。
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. conda run -n dggt --no-capture-output \
    python -u tools/verify_flow_cache_wysiwyg.py \
    --cache_path /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_b/training/002025.pt \
    --ckpt_path /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
    --output_dir runs/flow_cache_wysiwyg_mode_b_with_tokenizer \
    --with_tokenizer \
    --nrow 4

# Smoke：一个 Mode A 片段 + 一个 Mode B 片段端到端测试
CUDA_VISIBLE_DEVICES=3 python tools/precompute_flow_features.py \
    --edit_mode mode_a --ckpt_path .../model_latest_waymo.pt \
    --out_root /data/flow_cache_mode_a --start_clip_idx 0 --end_clip_idx 1
CUDA_VISIBLE_DEVICES=3 python tools/precompute_flow_features.py \
    --edit_mode mode_b --ckpt_path .../model_latest_waymo.pt \
    --out_root /data/flow_cache_mode_b --start_clip_idx 0 --end_clip_idx 1 \
    --allow_empty_plan

python tools/build_flow_train_manifest.py \
    --cache_root /data/flow_cache_mode_a:mode_a \
    --cache_root /data/flow_cache_mode_b:mode_b \
    --split training --out_path /data/flow_cache/training_manifest_smoke.jsonl

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 train_scene_flow.py \
    --ckpt_path .../model_latest_waymo.pt \
    --manifest_path /data/flow_cache/training_manifest_smoke.jsonl \
    --log_dir runs/flow_smoke --batch_size 1 --max_steps 5 \
    --min_frames 4 --max_frames 4 --vis_every 2
```

通过标准：
* 两种模式下都存在 `runs/flow_smoke/vis/step_*/flow_features/flow_features.pt`。
* Mode B step 还会额外输出 `mode_b/imagined_boxes_overlay.jpg`。
* 5 个 step 内 loss 为有限值。


## 6. Schema 参考（供下游消费者使用）

`FlowFeatureAssembler.forward(...)` 返回的 `FlowFeatureBundle` 在两种模式下具有相同字段：

| 字段 | Mode A | Mode B |
|------------------------|---------------------------------|----------------------------------|
| `M_preserve, M_source, M_dest` | K/(K+D+I+ε), D/…, I/…；D+I≈0 的未编辑 token 强制 preserve | K/(K+I+ε), 0, I/…；I≈0 的未编辑 token 强制 preserve |
| `K_map, D_map, I_map` | scene-kept α, scene-deleted α, asset α | scene-kept α, **0**, imagined-Gaussian α |
| `splatted_tok_low` | scene LUT splat 到 kept + asset Gaussians 上 | scene LUT 只 splat 到 kept Gaussians 上 |
| `F_asset_tokens` | `[B, K·S·P, 3072]`，来自缓存的 asset LUTs | `[B, 0, 3072]`（空） |
| `extras["mode_kind"]` | `"mode_a"` | `"mode_b"` |
| `extras` extras | `localized_objects` | `imagined_objects, num_imagined_objects, rejection_reason, delete_mask` |

`SceneFlowMatching`（Phase 6）会统一消费该 bundle：一个 Mode B 样本可以视为一个 Mode A 样本，其中 `F_asset_tokens` cross-attention 是 no-op（K/V 序列为空），`M_source` 处处为零，`M_dest` mask 是想象目标的 footprint。
