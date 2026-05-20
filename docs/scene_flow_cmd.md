# SceneFlow 训练命令

本文档记录 SceneFlow 的两阶段训练命令：

1. **Pretrain**：不用 flow cache，直接从 raw DGGT Waymo 数据在线提取 tokenizer latent，训练 `WanSceneFlow` 的无条件/伪编辑生成能力。
2. **正式训练**：使用 `docs/flow_cache_cmd.md` 生成的 Mode A + Mode B flow cache manifest，在正式编辑任务上训练。

## 0. 当前路径约定

```bash
export WAYMO_DGGT_ROOT=/data/disk2/lyy_dataset/waymo_processed_dggt/training
export WAYMO_DGGT_VAL_ROOT=/data/disk2/lyy_dataset/waymo_processed_dggt/validation
export DGGT_CKPT=/data/lyy_dataset/model/dggt/model_latest_waymo.pt
export TOKENIZER_CKPT=/home/dancer/code/dm/dggt/logs/tokenizer_t0_stageA/ckpt/scene_tokenizer_step_060000.pt
export FEATURE_STATS=logs/scene_flow_pretrain_1024/feature_stats_pretrain.pt
export SCENE_FLOW_PRETRAIN_CKPT=logs/scene_flow_pretrain_1024/ckpt/pretrain_step100000.pt
```

注意：

* `pretrained/model__latest_waymo.pt` 当前是断链，不要用它；使用上面的 `$DGGT_CKPT`。
* 这份 DGGT Waymo 数据的 tokenizer patch grid 是 `25x37`，pretrain 必须传 `--patch_grid_h 25 --patch_grid_w 37`。
* `logs/tokenizer_t0_waymo_views1/feature_stats.pt` 是 tokenizer 训练用的 `4x3072` aggregator stats，不能直接给 SceneFlow pretrain；SceneFlow 需要下面重新计算的 latent stats（维度 = tokenizer latent dim）。
* **正式 pretrain 和正式训练统一使用 1024-dim tokenizer latent**：`$TOKENIZER_CKPT`、`$FEATURE_STATS`、`--latent_dim 1024`、`$SCENE_FLOW_PRETRAIN_CKPT` 必须来自同一套 1024 tokenizer / SceneFlow pretrain。三者维度不一致会在 `load_into_buffers`/`set_latent_stats` 或 warm-start 时直接报错。
* `--shift 3.0`、`--lambda_repa 0.5`、EMA 验证默认开启 —— pretrain 和正式训练保持一致，详见文末「优化说明」。

## 1. Pretrain 正式参数

先计算正式 latent stats。`--max_batches 800` 是校准预算，可根据时间增减。

```bash
CUDA_VISIBLE_DEVICES=2 python -u tools/compute_pretrain_feature_stats.py \
    --image_dir $WAYMO_DGGT_ROOT \
    --dggt_ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --output_path $FEATURE_STATS \
    --scene_start 0 --scene_end 800 \
    --sequence_length 8 \
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
    --feature_stats_path $FEATURE_STATS \
    --log_dir logs/scene_flow_pretrain_1024 \
    --scene_start 0 --scene_end 800 \
    --sequence_length 8 \
    --pretrain_task full_scene \
    --patch_grid_h 25 --patch_grid_w 37 \
    --latent_dim 1024 \
    --batch_size 2 \
    --grad_accum_steps 4 \
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
    --val_guidance_scales "1.0,2.0,4.0" \
    --seed 0 \
    --precision bf16 \
    --wandb \
    --wandb_project dggt-flow \
    --wandb_name scene_flow_pretrain_waymo_s1
```

### 2×80GB A100 正式 pretrain

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run -n dggt --no-capture-output \
    torchrun --nproc_per_node=2 train_scene_flow_pretrain.py \
    --image_dir $WAYMO_DGGT_ROOT \
    --val_image_dir $WAYMO_DGGT_VAL_ROOT \
    --dggt_ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --feature_stats_path $FEATURE_STATS \
    --log_dir logs/scene_flow_pretrain_1024 \
    --scene_start 0 --scene_end 800 \
    --sequence_length 8 \
    --pretrain_task full_scene \
    --patch_grid_h 25 --patch_grid_w 37 \
    --latent_dim 1024 \
    --batch_size 2 \
    --grad_accum_steps 4 \
    --num_workers 8 \
    --lr 2e-4 \
    --weight_decay 0.0 \
    --ema_decay 0.9995 \
    --warmup_steps 3000 \
    --max_steps 150000 \
    --save_every 5000 \
    --shift 3.0 \
    --weighting_scheme logit_normal \
    --logit_mean 0.0 --logit_std 1.0 \
    --loss_weighting_scheme none \
    --lambda_repa 0.5 \
    --uncond_drop_prob 0.1 \
    --guidance_scale 1.0 \
    --val_guidance_scales "1.0,2.0,4.0" \
    --val_scene_start 0 --val_scene_end 50 \
    --val_every 1000 \
    --val_batches 8 \
    --val_log_images 4 \
    --val_sample_steps 50 \
    --seed 0 \
    --precision bf16 \
    --wandb \
    --wandb_project dggt-flow \
    --wandb_name scene_flow_pretrain_waymo_2a100
```

有效 batch / 学习率 / schedule 取值依据：

| 参数 | 取值 | 依据 |
|---|---|---|
| 有效 batch | `2 GPU × batch_size 2 × grad_accum 4 = 16` clip/step | 每 clip = 8 帧 × 925 patch ≈ 7.4K token-row → 每 step ≈ **118K token-row**，和 RAE DiT-XL（256 img × 256 tok ≈ 65K）同量级，足够稳定 |
| `--lr 2e-4` | AdamW, β=(0.9,0.95), `--weight_decay 0.0` | RAE/DiT-XL 从头训冻结-encoder latent 的标配区间 1e-4–2e-4；不稳降 1e-4，开 REPA 后若仍停滞可升 3e-4 |
| `--warmup_steps 3000` | ≈ 2% of max_steps | 大 batch + 恒等初始化（patch_embedding/proj_out/DDT 全 zero-init）需要较短 warmup 即可 |
| `--max_steps 150000` | cosine 衰减锚点 | **务必设成"现实总预算的上限而非下限"**：cosine 在 `max_steps` 处衰到 0，若你只训得到 40–60K，lr 仍在高位（好）；若把它设成 30000 而实际想训更久，lr 会过早衰到 ~0（坏） |
| `--ema_decay 0.9995` | half-life ≈ 1.4K step | RAE 取值；EMA 验证默认开启，见文末 |

显存兜底（80GB 仍 OOM 时按序降级，保持有效 batch≈16）：

* `--batch_size 1 --grad_accum_steps 8`（有效 batch 不变）
* 再不够：`--sequence_length 6`
* 仍不够：`--val_batches 4 --val_log_images 2 --no_val_render_rgb`（只降验证开销，不动训练）

> 旧的 4 卡 `--sequence_length 4 --batch_size 1` 配置已弃用：S=4 削弱了跨帧 attention 上下文，且未传 `--latent_dim` 会用默认 1024 与 768 tokenizer/stats 维度不符直接报错。

新增运行行为：

* pretrain 训练使用 tqdm 进度条；如果日志系统不适合交互式进度条，可加 `--no_tqdm`。
* tqdm 会每个 optimizer step 实时显示当前 loss 和 lr；train 标量也会每个 optimizer step 写入 wandb。
* `--seed` 会设置 Python/NumPy/PyTorch/CUDA 随机种子；DDP 下每个 rank 使用 `seed + rank`。
* `--val_image_dir` 指定 validation split 根目录；`--val_scene_start/--val_scene_end` 是在该 validation split 内部选 scene 范围，不要用 training split 的 800-850 做验证。
* `--val_every 1000` 表示每 1000 个 optimizer step 跑一次 validation；不是每 1000 个 batch，也不是每 1000 个 epoch。
* `--val_batches 8` 表示每次 validation 只遍历 8 个 validation batch 估计 loss，用来控制验证耗时；它不会限制训练数据量。
* `--pretrain_task full_scene` 是默认/推荐语义：`M_preserve=0, M_dest=1`，训练目标是从纯噪声重新生成完整 scene latent，而不是局部编辑填补。当前实现会保留由 dynamic mask 提取的 pseudo-asset `F_asset_tokens` 作为弱 cross-attn 条件，并通过 `--uncond_drop_prob` 训练 null-KV 路径；`pseudo_edit` 只作为兼容旧实验的可选模式保留。
* validation 图像会保存到 `logs/scene_flow_pretrain_1024/validation/step_xxxxxx/`；默认包含 `generated_raw_3dgs_rgb__cfg*.jpg`（纯噪声经 SceneFlow 采样、tokenizer decode，并用 decoded token 派生的 DGGT `camera_head/depth_head/instance_head/semantic_head/gs_head` 输出；sky/non-sky mask 来自 generated semantic logits，不使用 GT sky mask、GT sky background、GT depth 或 GT dynamic）、`generated_pred_sky_mask__cfg*.jpg`（generated semantic sky mask 诊断图）、`dggt_clean_3dgs_rgb.jpg`（原始 DGGT clean 3DGS render）、`input_rgb_gt.jpg`、latent PCA、误差图和 mask 图。
* 当前 tokenizer latent 只覆盖 patch tokens，不覆盖 camera/register special tokens；因此 `generated_raw_3dgs_rgb.jpg` 的相机 special token 仍来自 validation batch，用于定义要渲染的参考视角。它不是 GT 图像内容、GT depth 或 GT dynamic。
* `--val_sample_steps` 只控制 validation 图像采样步数，不影响训练本身。`15` 偏少，适合 smoke test；正式看图建议先用 `30`，需要更稳定的样本再用 `50`。Wan/FlowMatch 的生成采样也不是训练时的 1000 timestep 全跑，而是在 scheduler timestep 上做几十步推理。
* 如需只记录 latent/mask 诊断图并跳过较慢的 3DGS RGB 渲染，可额外加 `--no_val_render_rgb`。
* 若当前机器未登录 wandb，可先执行 `wandb login`，或临时去掉 `--wandb` 相关参数。

## 1.5 本轮优化说明（shift / REPA / EMA 验证）

为什么旧版 loss 降了但 RGB 还是糊、且 CFG=2 比 CFG=1 好：

1. **EMA 验证（默认开启，最大影响）**：旧版 `run_validation` 用实时裸权重采样。扩散模型训练中途裸权重单步去噪误差会在 30–50 步推理里累积成糊状；DiT/SD3/Wan/RAE 一律用 EMA 权重出图。现在验证（loss + CFG 采样 + RGB 渲染）默认在 EMA 权重下跑，所有 rank 同步交换参数，DDP 不会失同步。要对比可加 `--no_val_ema`。
2. **`--shift 3.0`（旧 10–16）**：`logit_normal` 采样下 shift=13 把约 90% 训练样本压在 clean-progress σ<0.15 的纯噪声端，低噪声「清理/细节」regime 几乎没训过 → 出图糊、无细节。RAE 的 shift≈13 是 1.4M step 的渐进最优；预算受限下 shift=3 让结构形成（σ→0）和细节清理（σ→1）都训得到。建议在 EMA 验证上做 `{1, 3}` 两点扫描定档。
3. **`--lambda_repa 0.5`（旧硬编码 0，现已接 CLI）**：REPA（Yu et al. 2024）把 trunk 中段特征对齐到干净 latent `z_clean_n`，是从头训 DiT 公认最强加速器（~2–17×）。代码早已实现，只是被关掉。同时它让 `repa_proj` 参与反传，修掉 DDP `find_unused_parameters=False` 在 2 卡上的崩溃。
4. **CFG=2>CFG=1 不是 bug**：`full_scene` 预训练本质是**无条件**生成（z_splat/scaffold 全 0、mask 常量、仅有从目标场景动态 patch 抠出的弱 KV）。欠训期条件输出是弱糊均值，CFG 外推把它推向更锐区域所以更好看；收敛后应回到 s≈1 最优。**预训练阶段不要拿生成结果和某个特定 GT 场景比对来判质量**——它本就不知道该生成哪个场景。看 EMA 样本是否落在 latent 流形（latent PCA 像真 latent、decode 出合理但不同的场景）+ loss 趋势即可，真实编辑质量留到 T1 条件训练再评。

判质量的正确方式：扩散/flow 的 MSE loss 有不可约方差下界（`v_gt=z_clean−eps`，eps 每步重采），从头训的 DiT/ADM/SD/Wan **全都**在 ~1–2K step 后 loss 基本压平、而样本质量再升 100K+ step。**loss 绝对值对扩散模型几乎无诊断价值，只看 EMA 样本。**

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
    --out_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl
```

正式训练不需要单独的 validation manifest。`train_scene_flow.py` 会把 training manifest 按 `--val_fraction 0.1` 做确定性 9:1 entries 切分：90% 用于训练，10% holdout 用于轻量 validation；两边都使用同样的 8 帧窗口、sigma、REPA/EMA 配置。

## 3. 正式训练参数

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_scene_flow.py \
    --ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --feature_stats_path $FEATURE_STATS \
    --latent_dim 1024 \
    --scene_flow_pretrain_path $SCENE_FLOW_PRETRAIN_CKPT \
    --scene_flow_pretrain_ema \
    --manifest_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl \
    --log_dir logs/scene_flow_t1_1024 \
    --sequence_length 8 \
    --batch_size 2 \
    --grad_accum_steps 4 \
    --num_workers 4 \
    --prefetch_factor 1 \
    --lr 2e-4 \
    --weight_decay 0.0 \
    --ema_decay 0.9995 \
    --warmup_steps 3000 \
    --max_steps 150000 \
    --save_every 5000 \
    --vis_every 1000 \
    --log_every 20 \
    --wandb_log_every 50 \
    --shift 3.0 \
    --weighting_scheme logit_normal \
    --logit_mean 0.0 --logit_std 1.0 \
    --loss_weighting_scheme none \
    --lambda_repa 0.5 \
    --uncond_drop_prob 0.1 \
    --guidance_scale 1.0 \
    --val_guidance_scales "1.0,2.0,4.0" \
    --val_every 1000 \
    --val_fraction 0.1 \
    --val_batches 8 \
    --val_log_images 4 \
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
| 帧数 | `--sequence_length 8` | 固定 8 帧，和 pretrain `--sequence_length 8` 一致 |
| 有效 batch | `2 GPU × batch_size 2 × grad_accum 4 = 16` clip/optimizer update | 与 pretrain 完全一致；DataLoader 现在返回完整 micro-batch list，不再丢弃 `batch[1:]` |
| DataLoader | `--num_workers 4 --prefetch_factor 1`，默认不启用 `pin_memory` | 每个 cache 文件平均约 651MB，低 prefetch 避免 8 workers × 2 prefetch × batch_size 2 造成几十个大文件并发读；GB 级 batch 走 pin-memory 线程容易触发 `received 0 items of ancdata` |
| worker tensor sharing | 默认 `--mp_sharing_strategy file_system` | 减少 multiprocessing 通过大量 fd 传递超大 tensor 时的稳定性问题；若系统 `/dev/shm`/临时目录策略特殊，可显式改回 `file_descriptor` |
| cache 读取 | zstd / gzip / plain 均自动识别；plain torch cache 默认 `mmap=True` | 支持 schema v6/v7；v6 读取时会把 fp16 溢出的 `pass1.gs_conf` 修成 finite fp32，v7 直接读取 finite fp32。空间不足时优先把现有 gzip cache 原地转 zstd：同样保留 `.pt` 路径，实测更小且解压显著更快；plain+mmap 最快但空间约翻倍 |
| sigma / target | `--shift 3.0 --weighting_scheme logit_normal --logit_mean 0.0 --logit_std 1.0 --loss_weighting_scheme none` | 正式训练代码已改为和 pretrain 一样调用 `build_rectified_flow_target`，用 clean-progress sigma 训练 `v=z_clean-eps` |
| REPA | `--lambda_repa 0.5` | 正式训练已启用 `return_mid` + `compute_total_loss(... lambda_repa=0.5)` |
| EMA | `--ema_decay 0.9995`，validation 默认用 EMA | checkpoint 同时保存 raw / full / EMA-only 权重 |
| validation | `--val_fraction 0.1 --val_every 1000 --val_batches 8 --val_log_images 4 --val_sample_steps 50 --guidance_scale 1.0 --val_guidance_scales "1.0,2.0,4.0"` | 从 training manifest 内部 9:1 holdout，cadence / CFG scales / sampling steps 与正式 pretrain 一致 |
| schedule | `--lr 2e-4 --weight_decay 0.0 --warmup_steps 3000 --max_steps 150000` | 与正式 pretrain 的 optimizer / warmup / cosine horizon 一致 |

注意：`train_scene_flow.py` 的 `global_step` 现在和 pretrain 一样是 optimizer update 口径；`--max_steps/--save_every/--vis_every/--val_every` 都按 optimizer update 触发。

注意：正式训练 validation 会保存 loss 标量、latent PCA / mask / CFG 采样诊断图到 `logs/scene_flow_t1_1024/validation/step_xxxxxx/` 并写入 wandb。T1 的完整 3DGS RGB validation 仍建议用 `inference_scene_flow_validation.py` 在 checkpoint 上离线跑；训练内 validation 保持轻量，避免每 1000 step 做 3DGS 渲染拖慢训练。

空间不足时的 cache 提速建议：

```bash
conda run -n dggt --no-capture-output python tools/recompress_flow_cache.py \
    --manifest_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl \
    --compression zstd \
    --gzip_level 1 \
    --workers 1
```

该命令逐文件原地替换，manifest 不需要改；峰值额外空间约为当前正在转换的一个 zstd 临时文件。后续重新 precompute cache 时可直接使用 `--save_compression zstd --gzip_level 1`。

注意：正式训练 wandb 会在 rank-0 按 `--wandb_log_every` 记录 averaged `train/loss`、`train/loss_flow`、`train/loss_repa`、`train/sigma_mean`、`train/lr` 等标量；如果机器未登录 wandb，可先执行 `wandb login`，或临时去掉 `--wandb` 相关参数。

调试只跑 Mode A：

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n dggt --no-capture-output \
    python -u train_scene_flow.py \
    --ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --feature_stats_path $FEATURE_STATS \
    --latent_dim 1024 \
    --scene_flow_pretrain_path $SCENE_FLOW_PRETRAIN_CKPT \
    --manifest_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl \
    --mode_filter mode_a \
    --log_dir logs/scene_flow_t1_mode_a_smoke \
    --sequence_length 8 \
    --batch_size 1 \
    --grad_accum_steps 1 \
    --num_workers 0 \
    --max_steps 5 \
    --vis_every 2 \
    --log_every 1
```

备注：`train_scene_flow.py --scene_flow_pretrain_path` 默认从完整 `pretrain_step{N}.pt` 读取 `ema_scene_flow` 做 warm-start；如传入没有 EMA 的旧 `_weights_only.pt` 会直接报错。只有明确要复现实验里的裸权重效果时，才加 `--no_scene_flow_pretrain_ema`。
