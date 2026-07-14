# SceneFlow 训练命令

本文档记录 SceneFlow 的两阶段训练命令：

1. **Pretrain**：不用 flow cache，直接从 raw DGGT Waymo 数据在线提取 tokenizer latent，只训练 full-scene latent prior。
2. **正式训练**：使用 `docs/flow_cache_cmd.md` 生成的 Mode A + Mode B flow cache manifest，在 masked local edit 任务上训练。

下面所有命令都假设已经手动切换到正确的 Python/conda 环境；训练命令不再包额外的 conda wrapper。

## 0. 当前路径约定

```bash
export WAYMO_DGGT_ROOT=/data/disk2/lyy_dataset/waymo_processed_dggt/training
export WAYMO_DGGT_VAL_ROOT=/data/disk2/lyy_dataset/waymo_processed_dggt/validation
export DGGT_CKPT=/data/lyy_dataset/model/dggt/model_latest_waymo.pt
export TOKENIZER_CKPT=/home/dancer/code/dm/dggt/logs/tokenizer_t0_stageB/ckpt/scene_tokenizer_step_040000.pt
export FEATURE_STATS=logs/scene_flow_pretrain_1024/feature_stats_pretrain_v3_798.pt
export SCENE_FLOW_PRETRAIN_CKPT=logs/scene_flow_pretrain_1024/ckpt/pretrain_step100000.pt
export SCENE_CAPTION_ROOT=/data/disk2/lyy_dataset/waymo_processed_dggt/training_captions
export SCENE_CAPTION_VAL_ROOT=/data/disk2/lyy_dataset/waymo_processed_dggt/validation_captions
export SCENE_FLOW_TRAIN_MANIFEST=/data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl
export SCENE_FLOW_VAL_MANIFEST=/data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/validation/validation_manifest.jsonl
export QWEN_TEXT_ENCODER=/home/dancer/model/Qwen/Qwen3-0.6B/
```

注意：

* `pretrained/model__latest_waymo.pt` 当前是断链，不要用它；使用上面的 `$DGGT_CKPT`。
* 这份 DGGT Waymo 数据的 tokenizer patch grid 是 `25x37`，pretrain 必须传 `--patch_grid_h 25 --patch_grid_w 37`。
* `logs/tokenizer_t0_waymo_views1/feature_stats.pt` 是 tokenizer 训练用的 `4x3072` aggregator stats，不能直接给 SceneFlow pretrain；SceneFlow 需要下面重新计算的 latent stats（维度 = tokenizer latent dim）。
* **删除 ae29c424 及所有 v2 camera stats/checkpoint，重新计算 stats 并从头 pretrain。** 当前代码不提供旧 v2 resume、推理或迁移兼容。
* camera stats 固定来自 `$DGGT_CKPT` 的冻结 CameraHead；即使使用 `--latent_stats_path` 复用 latent stats，仍必须传 `--dggt_ckpt_path`。stats/pretrain/T1/offline inference 会核对同一 DGGT SHA256。
* 当前 `feature_stats_pretrain.pt` 是缺少 v3 camera provenance 的旧文件，会被代码拒绝；正式命令必须使用上面的 `feature_stats_pretrain_v3_798.pt` 或重新运行 stats 命令生成同等 v3 文件。
* **正式 pretrain 和正式训练统一使用 1024-dim tokenizer latent**：`$TOKENIZER_CKPT`、`$FEATURE_STATS`、`--latent_dim 1024`、`$SCENE_FLOW_PRETRAIN_CKPT` 必须来自同一套 1024 tokenizer / SceneFlow pretrain。三者维度不一致会在 `load_into_buffers`/`set_latent_stats` 或 warm-start 时直接报错。
* 实现设计说明见 `docs/scene_flow_model_design.md`。本文档只维护运行命令和参数。
* `--shift 10.0`、`--weighting_scheme waver --mode_scale 1.29`、`--lambda_repa 0.5`、EMA 验证默认开启；pretrain 和正式训练保持一致。
* Gaussian 时间轴固定为 `clip_local_frame_id / 4`，新 cache 必须携带 `meta.gaussian_time_representation=clip_local_frame_id_div4_v1`。缺少该标记的 cache 会被训练/offline 拒绝；修改代码前已经启动的 precompute 进程不会热加载新实现，必须重启。chunked cache summary 同样携带该标记，因此 `--overwrite_v7` 能识别并重建旧时间轴 cache。
* mRoPE 坐标已固定为 A1 设计，不再提供 `--mrope_temporal_margin`：text/timestep 使用 RAE-style zero RoPE；video、asset、edit-control 与 camera 共享视频时间轴；camera 空间位置固定在 patch grid 中心；pretrain sky atlas 使用独立 temporal offset `128`。checkpoint 会记录 `rope_layout_version=a1_camera_center_sky128`，旧全局 margin 版本不要直接续训或推理。

## 1. Pretrain 正式参数

先用同一次冻结 DGGT forward 同时计算 tokenizer latent stats 与 DGGT CameraHead-derived
camera anchor/delta 11D stats。该路径不读取 Waymo extrinsics/intrinsics。输出文件必须包含
`mu_z/sigma_z`、`camera_anchor_mean/std`、`camera_delta_mean/std`、target space/source/version、
DGGT checkpoint SHA256 与 anchor/delta count；
`--max_batches 800` 是校准预算，可根据时间增减。

```bash
CUDA_VISIBLE_DEVICES=2 python -u tools/compute_pretrain_feature_stats.py \
    --image_dir $WAYMO_DGGT_ROOT \
    --dggt_ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --output_path $FEATURE_STATS \
    --scene_start 0 --scene_end 800 \
    --sequence_length 10 \
    --batch_size 1 \
    --num_workers 2 \
    --max_batches 800 \
    --log_every 20
```

### 2×80GB A100 正式 pretrain

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_scene_flow_pretrain.py \
    --image_dir $WAYMO_DGGT_ROOT \
    --val_image_dir $WAYMO_DGGT_VAL_ROOT \
    --dggt_ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --feature_stats_path $FEATURE_STATS \
    --log_dir logs/scene_flow_pretrain_1024 \
    --caption_root $SCENE_CAPTION_ROOT \
    --val_caption_root $SCENE_CAPTION_VAL_ROOT \
    --text_encoder_path $QWEN_TEXT_ENCODER \
    --scene_start 0 --scene_end 800 \
    --sequence_length 10 \
    --patch_grid_h 25 --patch_grid_w 37 \
    --latent_dim 1024 \
    --batch_size 8 \
    --grad_accum_steps 1 \
    --num_workers 16 \
    --lr 2e-4 \
    --final_lr 2e-5 \
    --scheduler_type linear \
    --decay_end_steps 150000 \
    --weight_decay 0.0 \
    --ema_decay 0.9995 \
    --warmup_steps 3000 \
    --max_steps 150000 \
    --save_every 5000 \
    --shift 10.0 \
    --weighting_scheme waver \
    --mode_scale 1.29 \
    --loss_weighting_scheme none \
    --prediction_type x \
    --lambda_repa 0.5 \
    --base_model_coeff 0.25 \
    --lambda_boundary 0.25 \
    --lambda_camera_flow 0.1 \
    --lambda_camera_pose 1.0 \
    --lambda_sky_flow 0.1 \
    --sky_unobserved_loss_weight 0.05 \
    --lambda_rgb_render 0.01 \
    --rgb_render_every 4 \
    --rgb_render_start_step 5000 \
    --rgb_render_warmup_steps 5000 \
    --rgb_render_max_samples 1 \
    --rgb_render_max_frames 4 \
    --rgb_render_stride 2 \
    --rgb_render_camera_grad_scale 0.0 \
    --rgb_render_sky_mask_grad_scale 0.0 \
    --rgb_render_lpips_weight 0.01 \
    --uncond_drop_prob 0.1 \
    --text_uncond_drop_prob 0.1 \
    --asset_uncond_drop_prob 0.2 \
    --camera_uncond_drop_prob 0.2 \
    --all_cond_drop_prob 0.05 \
    --guidance_scale 1.0 \
    --asset_control_guidance_scale 1.0 \
    --camera_guidance_scale 1.0 \
    --camera_text_guidance_scale 1.0 \
    --camera_absolute_weight 1.0 \
    --camera_relative_weight 1.0 \
    --camera_smoothness_weight 0.25 \
    --val_guidance_scales "1.0,2.0,4.0" \
    --val_scene_start 0 --val_scene_end 100 \
    --val_every 1000 \
    --val_batches 1 \
    --val_log_images 10 \
    --val_sample_steps 35 \
    --seed 0 \
    --precision bf16 \
    --ddp_timeout_minutes 60 \
    --wandb \
    --wandb_project dggt-flow \
    --wandb_name scene_flow_pretrain_waymo_2a100
```

有效 batch / 学习率 / schedule 取值依据：

| 参数 | 取值 | 依据 |
|---|---|---|
| 有效 batch | `2 GPU × batch_size 8 × grad_accum 1 = 16` clip/step | 每 clip = 10 帧 × 925 patch ≈ 9.25K token-row → 每 step ≈ **148K token-row**，和 RAE DiT-XL 同量级，足够稳定 |
| `--lr 2e-4` | 默认 GMuon；如需 AdamW 显式加 `--optimizer_type adamw`，`--weight_decay 0.0` | RAE/DiT-XL 从头训冻结-encoder latent 的标配区间 1e-4–2e-4；不稳降 1e-4，开 REPA 后若仍停滞可升 3e-4 |
| `--warmup_steps 3000` | ≈ 2% of max_steps | 大 batch 从头训练使用较短 warmup |
| `--max_steps/--decay_end_steps 150000` | linear 衰减终点 | 与代码默认 `scheduler_type=linear` 对齐；LR 从 `2e-4` 衰减到 `final_lr=2e-5`，不会衰到 0 |
| `--ema_decay 0.9995` | half-life ≈ 1.4K step | RAE 取值；EMA 验证默认开启，见文末 |

显存兜底（80GB 仍 OOM 时按序降级，保持有效 batch≈16）：

* `--batch_size 1 --grad_accum_steps 8`（有效 batch 不变）
* 再不够：`--sequence_length 6`
* 仍不够：`--val_batches 4 --val_log_images 2 --no_val_render_rgb`（只降验证开销，不动训练）

> 旧的 4 卡 `--sequence_length 4 --batch_size 1` 配置已弃用；当前命令统一使用 `--sequence_length 10 --latent_dim 1024`。

新增运行行为：

* pretrain 训练使用 tqdm 进度条；如果日志系统不适合交互式进度条，可加 `--no_tqdm`。
* tqdm 会每个 optimizer step 实时显示当前 loss 和 lr；train 标量也会每个 optimizer step 写入 wandb。
* RGB loss 的前向渲染始终使用 SceneFlow 生成的 DGGT camera、generated depth/GS 和 predicted sky mask；`camera/sky_mask_grad_scale=0` 只切断对应 RGB 梯度，不会回退 teacher geometry。建议主 flow 收敛后再把 camera gradient 小幅升到 `0.05～0.1`。
* `--seed` 会设置 Python/NumPy/PyTorch/CUDA 随机种子；DDP 下每个 rank 使用 `seed + rank`。
* `--val_image_dir` 指定 validation split 根目录；`--val_scene_start/--val_scene_end` 是在该 validation split 内部选 scene 范围，不要用 training split 的 800-850 做验证。
* `--val_every 1000` 表示每 1000 个 optimizer step 跑一次 validation；不是每 1000 个 batch，也不是每 1000 个 epoch。
* `--val_batches 8` 表示每次 validation 只遍历 8 个 validation batch 估计 loss，用来控制验证耗时；它不会限制训练数据量。
* pretrain 现在固定为 full_scene；旧的 `pseudo_edit/random_inpaint/mixed` CLI 参数已经删除。
* `--uncond_drop_prob` 保留为旧参数 fallback；正式控制建议显式传 `--text_uncond_drop_prob --asset_uncond_drop_prob --camera_uncond_drop_prob --all_cond_drop_prob`。
* 默认训练 sky generation；如需关闭，加 `--no_sky_generation`。
* sky mask 只用于构造 sky RGB target 和 `sky_gen_loss_weight`；SceneFlow forward 不接收 GT 派生的 sky attention mask。`--sky_unobserved_loss_weight 0.05` 给未观测 atlas cell 弱监督，保持开放推理时完整 sky atlas 可生成。
* validation 图像会保存到 `logs/scene_flow_pretrain_1024/validation/step_xxxxxx/`；默认包含 `generated_raw_3dgs_rgb__cfg*.jpg`、`generated_sky_rgb__cfg*.jpg`、`generated_pred_sky_mask__cfg*.jpg`、latent PCA 和误差图。额外 CFG scale 会追加 `*_cfg{scale}` 后缀。
* `--val_sample_steps` 只控制 validation 图像采样步数，不影响训练本身。`15` 偏少，适合 smoke test；正式看图建议先用 `30`，需要更稳定的样本再用 `50`。FlowMatch/RAE 的生成采样也不是训练时的 1000 timestep 全跑，而是在 scheduler timestep 上做几十步推理。
* 训练内 validation 默认 `--val_sliding_window 10 --val_sliding_stride 7`，即相邻窗口重叠 3 帧。长序列必须满足 `1 <= stride < window`；`stride>=window` 直接报错。采样维护 full video/camera/sky 状态，对 video/camera/mask logits 用 cosine coverage 逐帧归一化；scene-global sky 使用 `sum(w/C)` 窗口权重，使每个全局帧贡献相等。
* 如需只记录 latent/mask 诊断图并跳过较慢的 3DGS RGB 渲染，可额外加 `--no_val_render_rgb`。
* 若当前机器未登录 wandb，可先执行 `wandb login`，或临时去掉 `--wandb` 相关参数。

## 1.5 关键参数说明

* `--no_val_ema`：默认 validation 使用 EMA 权重；加这个参数才会用实时裸权重验证。
* mRoPE A1 固定坐标：video/asset/edit-control 使用真实 `(t,y,x)`；camera condition 和 camera generation 使用同一帧 `t`、空间中心 `(H//2,W//2)`，避免把全局相机条件绑到左上角 patch；sky generation 使用独立 sky atlas grid 和 temporal offset `128`，避免 scene-level 天空 token 与视频 patch 共享位置。这个设置是模型结构的一部分，不通过命令行覆盖，并通过 `rope_layout_version=a1_camera_center_sky128` 写入 checkpoint config。
* `--weighting_scheme waver --mode_scale 1.29 --shift 10.0 --prediction_type x`：pretrain、正式训练、训练内 validation、离线 inference 需要保持一致。`logit_mean/logit_std` 只在显式切回 `--weighting_scheme logit_normal` 做 ablation 时使用。`x` 是 RAEv2 T2I 的 clean-latent 输出参数化；代码仍会把它转换成 velocity 做 flow matching loss 和 ODE 采样。
* `--lambda_repa 0.5`：推荐命令显式打开；CLI 默认仍是 `0.0`。
* `--guidance_scale --asset_control_guidance_scale --camera_guidance_scale --camera_text_guidance_scale`：前三者分别控制全局 text、asset/control residual、camera condition residual；最后一个只控制 camera 输出的 text residual，默认 `1.0`。因此 `--val_guidance_scales` / offline `--cfg` 扫描全局 text scale 时 camera trajectory 保持不变。
* 扩散/flow 的训练 loss 只能作为粗略健康检查；样本质量以 EMA validation 图像为准。

正式配置已切到 **1024-dim 6 万 iter tokenizer**。切 tokenizer 时务必重算 `$FEATURE_STATS`，并保持 pretrain / T1 的 `--latent_dim 1024` 与 warm-start checkpoint 维度一致。tokenizer 续训时建议跑满 RAE 式 decoder noise augmentation（denoise-recon 阶段），让 decoder 对 SceneFlow latent 误差鲁棒，这直接削 grid。

最终 warm-start 权重路径（完整 checkpoint；warm-start / 推理默认取其中的 `ema_scene_flow`，不要用裸权重 `_weights_only.pt`）：

```bash
$SCENE_FLOW_PRETRAIN_CKPT
```

新训练保存时也会额外导出 `pretrain_step100000_ema_weights_only.pt` 作为便捷 EMA-only state dict；旧产物若只有 `_weights_only.pt`，那只是裸权重，不能代表 EMA 出图质量。

## 2. 正式训练前置：flow cache manifest

正式训练不直接读 raw Waymo，而是读 flow cache manifest。先按 `docs/flow_cache_cmd.md` 完成：

```bash
python tools/build_flow_train_manifest.py \
    --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_a:mode_a \
    --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_b:mode_b \
    --split training \
    --out_path $SCENE_FLOW_TRAIN_MANIFEST
```

内部 holdout 仍可用：不传 `--val_manifest_path/--val_cache_root` 时，`train_scene_flow.py` 会把 training manifest 按 `--val_fraction 0.1` 做确定性切分。但内部 holdout 只能使用 `--caption_root`，不要传不同的 `--val_caption_root`。

如果要使用 validation captions，必须先构建独立 validation flow cache/manifest，再传 `--val_manifest_path ... --val_caption_root $SCENE_CAPTION_VAL_ROOT`：

```bash
python tools/build_flow_validation_manifest.py \
    --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_validation \
    --split validation \
    --out_path $SCENE_FLOW_VAL_MANIFEST
```

## 3. 正式训练参数

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_scene_flow.py \
    --ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --feature_stats_path $FEATURE_STATS \
    --latent_dim 1024 \
    --scene_flow_pretrain_path $SCENE_FLOW_PRETRAIN_CKPT \
    --scene_flow_pretrain_ema \
    --caption_root $SCENE_CAPTION_ROOT \
    --val_manifest_path $SCENE_FLOW_VAL_MANIFEST \
    --val_caption_root $SCENE_CAPTION_VAL_ROOT \
    --text_encoder_path $QWEN_TEXT_ENCODER \
    --manifest_path $SCENE_FLOW_TRAIN_MANIFEST \
    --log_dir logs/scene_flow_t1_1024 \
    --sequence_length 10 \
    --batch_size 2 \
    --grad_accum_steps 4 \
    --num_workers 4 \
    --prefetch_factor 1 \
    --lr 2e-4 \
    --weight_decay 0.0 \
    --ema_decay 0.9995 \
    --final_lr 2e-5 \
    --scheduler_type linear \
    --decay_end_steps 150000 \
    --warmup_steps 3000 \
    --max_steps 150000 \
    --save_every 5000 \
    --vis_every 1000 \
    --log_every 20 \
    --wandb_log_every 50 \
    --shift 10.0 \
    --weighting_scheme waver \
    --mode_scale 1.29 \
    --loss_weighting_scheme none \
    --prediction_type x \
    --lambda_repa 0.5 \
    --base_model_coeff 0.1 \
    --lambda_boundary 0.25 \
    --lambda_identity 1.0 \
    --edit_domain_threshold 1e-4 \
    --edit_domain_dilation 1 \
    --lambda_rgb_render 0.01 \
    --rgb_render_every 4 \
    --rgb_render_start_step 5000 \
    --rgb_render_warmup_steps 5000 \
    --rgb_render_max_samples 1 \
    --rgb_render_max_frames 4 \
    --rgb_render_stride 2 \
    --rgb_render_lpips_weight 0.01 \
    --uncond_drop_prob 0.1 \
    --guidance_scale 1.0 \
    --asset_control_guidance_scale 1.0 \
    --val_guidance_scales "1.0,2.0,4.0" \
    --val_every 1000 \
    --val_batches 1 \
    --val_log_images 10 \
    --val_sample_steps 50 \
    --grad_clip_norm 1.0 \
    --seed 0 \
    --precision bf16 \
    --wandb \
    --wandb_project dggt-flow \
    --wandb_name scene_flow_t1_waymo_2a100
```

正式训练与 2×80GB A100 正式 pretrain 的对齐项：

| 项 | 正式训练取值 | 对齐方式 |
|---|---|---|
| tokenizer latent | `--latent_dim 1024` + `$TOKENIZER_CKPT` + `$FEATURE_STATS` | 与 pretrain 使用同一个 1024 tokenizer / latent stats |
| warm-start | `$SCENE_FLOW_PRETRAIN_CKPT` + `--scene_flow_pretrain_ema` | 从正式 pretrain 的 EMA SceneFlow 权重继续训练 |
| 帧数 | `--sequence_length 10` | 固定 10 帧，和 pretrain `--sequence_length 10` 一致 |
| 有效 batch | `2 GPU × batch_size 2 × grad_accum 4 = 16` clip/optimizer update | 与 pretrain 完全一致；DataLoader 现在返回完整 micro-batch list，不再丢弃 `batch[1:]` |
| micro-batch 执行 | 默认将 `batch_size>1` 的 bundle 合并后一次 forward/backward | 如需回退旧路径可加 `--no_batch_scene_flow` |
| asset/cache 输入 | cache 读取所有可用 asset LUT levels，并使用 cached `pass2_splatted_tok_low` | 不再 live splat/blend；cache 字段缺失会在 DataLoader/assembler 阶段报错 |
| camera 输入/渲染 | SceneFlow 条件输入使用 Waymo `camera_to_world_corrected + intrinsics` 摘要；RGB render 固定使用输入图像经 DGGT 预测的 DGGT camera | Waymo camera 只作为条件，不直接给 renderer；正式训练/离线推理不启用 camera generation token，也不让 edited latent 重新定义相机 |
| RGB 几何链 | generated video latent 经 frozen tokenizer decoder + DGGT depth/GS/instance heads 后可微渲染；GT sky mask 同时用于渲染和 masked LPIPS | teacher depth 不进入主 RGB renderer；正式阶段固定 input-DGGT camera，pretrain 才生成 camera/sky |
| sky handling | T1 不启用 pretrain sky generation 参数 | 正式训练 RGB validation 仍使用 cache/GT sky 相关字段 |
| DataLoader | `--num_workers 4 --prefetch_factor 1`，默认不启用 `pin_memory` | 每个 cache 文件平均约 651MB，低 prefetch 避免 8 workers × 2 prefetch × batch_size 2 造成几十个大文件并发读；GB 级 batch 走 pin-memory 线程容易触发 `received 0 items of ancdata` |
| worker tensor sharing | 默认 `--mp_sharing_strategy file_system` | 减少 multiprocessing 通过大量 fd 传递超大 tensor 时的稳定性问题；若系统 `/dev/shm`/临时目录策略特殊，可显式改回 `file_descriptor` |
| cache 读取 | 默认读取 chunked zstd `.pt`；每个样本只解压 10 帧窗口需要的 chunk | 逻辑 `schema_version` 仍为 v8。旧 monolithic cache 可用 `tools/convert_flow_cache_to_chunked.py` 转换并逐 tensor 校验 |
| sigma / target | `--shift 10.0 --weighting_scheme waver --mode_scale 1.29 --loss_weighting_scheme none --prediction_type x` | 与 pretrain 保持一致 |
| REPA | `--lambda_repa 0.5` | 与 pretrain 保持一致 |
| EMA | `--ema_decay 0.9995`，validation 默认用 EMA | checkpoint 同时保存 raw / full / EMA-only 权重 |
| validation | `--val_manifest_path $SCENE_FLOW_VAL_MANIFEST --val_caption_root $SCENE_CAPTION_VAL_ROOT --val_every 1000 --val_batches 8 --val_log_images 10 --val_sample_steps 50 --guidance_scale 1.0 --asset_control_guidance_scale 1.0 --val_guidance_scales "1.0,2.0,4.0"`，长 clip 训练内采样才加 `--val_sliding_window 10 --val_sliding_stride 7` | 独立 validation manifest/cache 可使用 validation captions；内部 holdout 仍可用，但不能混用 validation caption root。采样噪声用 `seed + step`，和 pretrain validation 一样可复现。滑窗采样按 full latent 状态做 per-step velocity blending，并传入 clip-local `frame_ids` |
| schedule | `--lr 2e-4 --final_lr 2e-5 --scheduler_type linear --decay_end_steps 150000 --weight_decay 0.0 --warmup_steps 3000 --max_steps 150000` | 与正式 pretrain 的 optimizer / warmup / linear horizon 一致 |

注意：`train_scene_flow.py` 的 `global_step` 现在和 pretrain 一样是 optimizer update 口径；`--max_steps/--save_every/--vis_every/--val_every` 都按 optimizer update 触发。

注意：正式训练 validation 会保存 loss 标量、latent PCA / mask / CFG 采样诊断图到 `logs/scene_flow_t1_1024/validation/step_xxxxxx/` 并写入 wandb。训练内 validation 保持轻量，避免每 1000 step 做 3DGS 渲染拖慢训练。

注意：正式 offline 入口是 `inference_scene_flow.py`，不是不存在的 `inference_scene_flow_validation.py`。29 帧输入会自动解析为 `window=10, stride=7`（重叠 3 帧）；它维护 full 29 帧 latent，在每个采样步对窗口 velocity 做 cosine coverage 归一化后统一更新。正式阶段始终使用 input DGGT camera 与 GT sky/mask，不生成 11D camera/sky。

## 2.1 Camera cache 修复与旧 checkpoint 迁移

cache 缺少相机 GT 或原图尺寸时，普通 `.pt` 与 SQLite chunk cache 使用同一命令修复：

```bash
python tools/backfill_flow_cache_camera_gt.py \
    --cache_root /path/to/cache_or_root \
    --processed_root /data/disk2/lyy_dataset/waymo_processed_dggt \
    --split training --force
```

旧 48K pretrain 的 `dggt_hidden_v1/2048D` camera head 不能 resume 到 v3。先用上面的
stats 工具生成 v3 stats，然后把显式允许的 legacy shared-trunk checkpoint（265bd939）传给 `--warm_start_path`：程序只迁移 EMA
materialized shared trunk 中同名同 shape 参数，跳过 camera generation projection/decoder、
role embedding 与 stats，重置 EMA step/optimizer/scheduler，并从 pretrain step 0 开始。
`--resume_path` 只接受 representation、dimension、stats version 完全一致的新完整 checkpoint。

## 4. 四类 validation / offline inference

训练内 pretrain validation 使用 `train_scene_flow_pretrain.py` 的
`--val_sliding_window 10 --val_sliding_stride 7 --val_sample_steps 35`；正式训练 validation
使用 `train_scene_flow.py` 的同名滑窗参数。正式 validation 不 pack、不加噪、不预测 camera/sky
generation state。

Pretrain offline 单窗：

```bash
python inference_scene_flow_pretrain.py \
  --weights $SCENE_FLOW_PRETRAIN_CKPT --dggt_ckpt_path $DGGT_CKPT \
  --tokenizer_ckpt_path $TOKENIZER_CKPT --feature_stats_path $FEATURE_STATS \
  --val_image_dir $WAYMO_DGGT_VAL_ROOT --val_caption_root $SCENE_CAPTION_VAL_ROOT \
  --num_frames 10 --val_sliding_window 10 --val_sliding_stride 7 --cfg 1 2 4
```

Pretrain offline 只需指定 `--num_frames 29`；当请求帧数超过单次推理上限 10 时，入口会自动启用
`window=10, stride=7`（重叠 3 帧）的滑窗。`--val_sliding_window <= 0` 表示自动选择，不再表示禁用滑窗；显式传入
大于 10 的窗口也会被截断为 10。默认 `--camera_text_guidance_scale 1`，所以全局 CFG sweep 不改变
camera trajectory。

正式 offline 单窗/短 clip 与重叠长 clip 统一使用真实入口：

```bash
python inference_scene_flow.py \
  --scene_flow_ckpt_path /path/to/flow_stepXXXXXX.pt --ckpt_path $DGGT_CKPT \
  --tokenizer_ckpt_path $TOKENIZER_CKPT --manifest_path $SCENE_FLOW_VAL_MANIFEST \
  --output_dir runs/scene_flow_offline --window 10 --window_stride 7 \
  --edit_domain_threshold 1e-4 --edit_domain_dilation 1 \
  --guidance_scales 1,2,4
```

长度不超过 10 时公共 scheduler 自动返回单窗；长 clip 无论是否显式传 `--window` 都自动走最大 10 帧的
overlap schedule。`--window <= 0` 表示自动选择，`--window > 10` 会被截断为 10。所有入口都禁止
实际启用滑窗时 `stride>=window`。

旧 monolithic cache 转换为当前 chunked cache：

```bash
python tools/convert_flow_cache_to_chunked.py \
    --manifest_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl \
    --mode_a_source_dir /data/disk3/lyy_dataset/waymo_processed_dggt/flow_cache_mode_a/training \
    --mode_a_output_dir /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_a/training \
    --workers 2 \
    --zstd_level 1 \
    --verify --verify_items 8
```

转换脚本默认原地覆盖 `.pt`；这批 Mode-A 迁移使用上面两个参数从 disk3 读取旧 cache，并写到 disk2。manifest 不需要改变；`--verify` 会在覆盖/落盘前比较原始文件和临时 chunked 文件。后续重新 precompute cache 时直接使用默认 `--save_compression chunked_zstd --gzip_level 1`。

注意：正式训练 wandb 会在 rank-0 按 `--wandb_log_every` 记录 averaged `train/loss`、`train/loss_flow`、`train/loss_repa`、`train/sigma_mean`、`train/lr` 等标量；如果机器未登录 wandb，可先执行 `wandb login`，或临时去掉 `--wandb` 相关参数。

调试只跑 Mode A：

```bash
CUDA_VISIBLE_DEVICES=2 python -u train_scene_flow.py \
    --ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --feature_stats_path $FEATURE_STATS \
    --latent_dim 1024 \
    --scene_flow_pretrain_path $SCENE_FLOW_PRETRAIN_CKPT \
    --manifest_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl \
    --mode_filter mode_a \
    --log_dir logs/scene_flow_t1_mode_a_smoke \
    --sequence_length 10 \
    --batch_size 1 \
    --grad_accum_steps 1 \
    --num_workers 0 \
    --max_steps 5 \
    --vis_every 2 \
    --log_every 1
```

备注：`train_scene_flow.py --scene_flow_pretrain_path` 默认从完整 `pretrain_step{N}.pt` 读取 `ema_scene_flow` 做 warm-start；如传入没有 EMA 的旧 `_weights_only.pt` 会直接报错。只有明确要复现实验里的裸权重效果时，才加 `--no_scene_flow_pretrain_ema`。
