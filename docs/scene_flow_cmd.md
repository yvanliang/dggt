# SceneFlow 训练命令

本文档记录 SceneFlow 的两阶段训练命令：

1. **Pretrain**：不用 flow cache，直接从 raw DGGT Waymo 数据在线提取 tokenizer latent，训练 `WanSceneFlow` 的无条件/伪编辑生成能力。
2. **正式训练**：使用 `docs/flow_cache_cmd.md` 生成的 Mode A + Mode B flow cache manifest，在正式编辑任务上训练。

## 0. 当前路径约定

```bash
export WAYMO_DGGT_ROOT=/data/disk2/lyy_dataset/waymo_processed_dggt/training
export WAYMO_DGGT_VAL_ROOT=/data/disk2/lyy_dataset/waymo_processed_dggt/validation
export DGGT_CKPT=/data/lyy_dataset/model/dggt/model_latest_waymo.pt
export TOKENIZER_CKPT=/home/dancer/code/dm/dggt/logs/tokenizer_t0_waymo_views1/ckpt/scene_tokenizer_step_014000.pt
```

注意：

* `pretrained/model__latest_waymo.pt` 当前是断链，不要用它；使用上面的 `$DGGT_CKPT`。
* 这份 DGGT Waymo 数据的 tokenizer patch grid 是 `25x37`，pretrain 必须传 `--patch_grid_h 25 --patch_grid_w 37`。
* `logs/tokenizer_t0_waymo_views1/feature_stats.pt` 是 tokenizer 训练用的 `4x3072` aggregator stats，不能直接给 SceneFlow pretrain；SceneFlow 需要下面重新计算的 `768` 维 latent stats。

## 1. Pretrain 正式参数

先计算正式 latent stats。`--max_batches 800` 是校准预算，可根据时间增减。

```bash
CUDA_VISIBLE_DEVICES=2 python -u tools/compute_pretrain_feature_stats.py \
    --image_dir $WAYMO_DGGT_ROOT \
    --dggt_ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --output_path logs/scene_flow_pretrain/feature_stats_pretrain.pt \
    --scene_start 0 --scene_end 800 \
    --sequence_length 4 \
    --batch_size 1 \
    --num_workers 2 \
    --max_batches 800 \
    --log_every 20
```

单卡 pretrain：

```bash
CUDA_VISIBLE_DEVICES=2 python -u train_scene_flow_pretrain.py \
    --image_dir $WAYMO_DGGT_ROOT \
    --val_image_dir $WAYMO_DGGT_VAL_ROOT \
    --dggt_ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --feature_stats_path logs/scene_flow_pretrain/feature_stats_pretrain.pt \
    --log_dir logs/scene_flow_pretrain \
    --scene_start 0 --scene_end 800 \
    --sequence_length 8 \
    --pretrain_task full_scene \
    --patch_grid_h 25 --patch_grid_w 37 \
    --batch_size 4 \
    --grad_accum_steps 2 \
    --num_workers 8 \
    --max_steps 100000 \
    --warmup_steps 5000 \
    --save_every 5000 \
    --val_scene_start 0 --val_scene_end 198 \
    --val_every 1000 \
    --wandb_log_every 50 \
    --val_batches 1 \
    --val_log_images 8 \
    --val_sample_steps 30 \
    --seed 0 \
    --precision bf16 \
    --wandb \
    --wandb_project dggt-flow \
    --wandb_name scene_flow_pretrain_waymo_s1
```

多卡 pretrain（按实际可用 GPU 数调整 `CUDA_VISIBLE_DEVICES` 和 `--nproc_per_node`）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n dggt --no-capture-output \
    torchrun --nproc_per_node=4 train_scene_flow_pretrain.py \
    --image_dir $WAYMO_DGGT_ROOT \
    --val_image_dir $WAYMO_DGGT_VAL_ROOT \
    --dggt_ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --feature_stats_path logs/scene_flow_pretrain/feature_stats_pretrain.pt \
    --log_dir logs/scene_flow_pretrain \
    --scene_start 0 --scene_end 800 \
    --sequence_length 4 \
    --pretrain_task full_scene \
    --patch_grid_h 25 --patch_grid_w 37 \
    --batch_size 1 \
    --grad_accum_steps 4 \
    --num_workers 2 \
    --max_steps 100000 \
    --warmup_steps 5000 \
    --save_every 5000 \
    --val_scene_start 0 --val_scene_end 50 \
    --val_every 1000 \
    --val_batches 8 \
    --val_log_images 4 \
    --val_sample_steps 30 \
    --seed 0 \
    --precision bf16 \
    --wandb \
    --wandb_project dggt-flow \
    --wandb_name scene_flow_pretrain_waymo_s1_ddp
```

新增运行行为：

* pretrain 训练使用 tqdm 进度条；如果日志系统不适合交互式进度条，可加 `--no_tqdm`。
* tqdm 会每个 optimizer step 实时显示当前 loss 和 lr；train 标量也会每个 optimizer step 写入 wandb。
* `--seed` 会设置 Python/NumPy/PyTorch/CUDA 随机种子；DDP 下每个 rank 使用 `seed + rank`。
* `--val_image_dir` 指定 validation split 根目录；`--val_scene_start/--val_scene_end` 是在该 validation split 内部选 scene 范围，不要用 training split 的 800-850 做验证。
* `--val_every 1000` 表示每 1000 个 optimizer step 跑一次 validation；不是每 1000 个 batch，也不是每 1000 个 epoch。
* `--val_batches 8` 表示每次 validation 只遍历 8 个 validation batch 估计 loss，用来控制验证耗时；它不会限制训练数据量。
* `--pretrain_task full_scene` 是默认/推荐语义：`M_preserve=0, M_dest=1, F_asset_tokens=empty`，训练目标是从纯噪声重新生成完整 scene latent，而不是局部编辑填补。`pseudo_edit` 只作为兼容旧实验的可选模式保留。
* validation 图像会保存到 `logs/scene_flow_pretrain/validation/step_xxxxxx/`；默认包含 `generated_raw_3dgs_rgb.jpg`（纯噪声经 SceneFlow 采样、tokenizer decode，并用 decoded token 派生的 DGGT `camera_head/depth_head/instance_head/gs_head` 输出渲染；不使用 GT sky mask、GT sky background、GT depth 或 GT dynamic）、`dggt_clean_3dgs_rgb.jpg`（原始 DGGT clean 3DGS render）、`input_rgb_gt.jpg`、latent PCA、误差图和 mask 图。
* 当前 tokenizer latent 只覆盖 patch tokens，不覆盖 camera/register special tokens；因此 `generated_raw_3dgs_rgb.jpg` 的相机 special token 仍来自 validation batch，用于定义要渲染的参考视角。它不是 GT 图像内容、GT depth 或 GT dynamic。
* `--val_sample_steps` 只控制 validation 图像采样步数，不影响训练本身。`15` 偏少，适合 smoke test；正式看图建议先用 `30`，需要更稳定的样本再用 `50`。Wan/FlowMatch 的生成采样也不是训练时的 1000 timestep 全跑，而是在 scheduler timestep 上做几十步推理。
* 如需只记录 latent/mask 诊断图并跳过较慢的 3DGS RGB 渲染，可额外加 `--no_val_render_rgb`。
* 若当前机器未登录 wandb，可先执行 `wandb login`，或临时去掉 `--wandb` 相关参数。

最终 warm-start 权重路径：

```bash
logs/scene_flow_pretrain/ckpt/pretrain_step100000_weights_only.pt
```

## 2. 正式训练前置：flow cache manifest

正式训练不直接读 raw Waymo，而是读 flow cache manifest。先按 `docs/flow_cache_cmd.md` 完成：

```bash
python tools/build_flow_train_manifest.py \
    --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_a:mode_a \
    --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_b:mode_b \
    --split training \
    --out_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl
```

## 3. 正式训练参数

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n dggt --no-capture-output \
    torchrun --nproc_per_node=4 train_scene_flow.py \
    --ckpt_path $DGGT_CKPT \
    --manifest_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl \
    --log_dir logs/scene_flow_t1 \
    --batch_size 1 \
    --grad_accum_steps 2 \
    --num_workers 2 \
    --min_frames 4 \
    --max_frames 8 \
    --max_steps 40000 \
    --save_every 2000 \
    --vis_every 1000 \
    --log_every 20 \
    --lr 2e-4 \
    --weight_decay 0.05 \
    --warmup_steps 2000 \
    --grad_clip_norm 1.0 \
    --precision bf16
```

调试只跑 Mode A：

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n dggt --no-capture-output \
    python -u train_scene_flow.py \
    --ckpt_path $DGGT_CKPT \
    --manifest_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl \
    --mode_filter mode_a \
    --log_dir logs/scene_flow_t1_mode_a_smoke \
    --batch_size 1 \
    --grad_accum_steps 1 \
    --num_workers 0 \
    --min_frames 4 \
    --max_frames 4 \
    --max_steps 5 \
    --vis_every 2 \
    --log_every 1
```

备注：当前 `train_scene_flow.py` 的 CLI 尚未接入 `--scene_flow_pretrain_path` 这类 warm-start 参数；pretrain 产物路径已经固定记录在上面，等正式训练入口支持 warm-start 后应加载 `pretrain_step*_weights_only.pt` 中的 `scene_flow` state dict。
