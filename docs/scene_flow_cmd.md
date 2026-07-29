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
export FEATURE_STATS=logs/scene_flow_pretrain_1024/feature_stats_pretrain_v3.pt
export SCENE_FLOW_PRETRAIN_CKPT=logs/scene_flow_pretrain_1024_v3/ckpt/pretrain_step100000.pt
export SCENE_CAPTION_ROOT=/data/disk2/lyy_dataset/waymo_processed_dggt/training_captions
export SCENE_CAPTION_VAL_ROOT=/data/disk2/lyy_dataset/waymo_processed_dggt/validation_captions
export SCENE_FLOW_TRAIN_MANIFEST=/data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl
export SCENE_FLOW_VAL_MANIFEST=/data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/validation/validation_manifest.jsonl
export QWEN_TEXT_ENCODER=/home/dancer/model/Qwen/Qwen3-0.6B/
```

注意：

* **实际 pretrain 一律通过仓库根目录的 `pretrain_*.sh` 启动器运行**，不再手写 `torchrun`。启动器内部
  已经固化了下面 §1 的全部超参；本文档的裸命令只作为参数说明和单卡 smoke test 用。
  各启动器的 `PROJECT_ROOT` 默认是集群路径 `/home/wuzn/liangyy/dggt`，本机调试需要覆盖
  `LIANGYY_ROOT` / `PROJECT_ROOT` / `DATASET_ROOT`。
* `pretrained/model_latest_waymo.pt` 在**本机**是断链（软链指向 `.../model__latest_waymo.pt`，
  双下划线，实际文件是单下划线的 `model_latest_waymo.pt`）。本机调试请直接用上面的绝对路径
  `$DGGT_CKPT`；集群上 `${PROJECT_ROOT}/pretrained/model_latest_waymo.pt` 是有效文件，启动器用的就是它。
* 这份 DGGT Waymo 数据的 tokenizer patch grid 是 `25x37`，pretrain 必须传 `--patch_grid_h 25 --patch_grid_w 37`。
* `logs/tokenizer_t0_waymo_views1/feature_stats.pt` 是 tokenizer 训练用的 `4x3072` aggregator stats，不能直接给 SceneFlow pretrain；SceneFlow 需要下面重新计算的 latent stats（维度 = tokenizer latent dim）。
* **当前统一的统计文件是 `feature_stats_pretrain_v3.pt`**，与所有 `pretrain_*.sh` 一致。
  它比 `_v2.pt` 多了 `placement_mean/placement_std/placement_count` 三个键，
  这是 factorized asset condition 的归一化参数。`asset_condition_protocol='factorized_v1'`（当前
  pretrain 的固定配置）在缺少这两个键时会直接抛
  `KeyError: Factorized SceneFlow pretraining requires placement_mean/placement_std ...`，
  所以 **`_v2.pt` 已经不能用于当前代码**，不是风格差异而是硬失败。
  文件内部 camera state 仍是 DGGT CameraHead v3 表示，camera stats 契约为 `v4_global_context`。
* camera stats 固定来自 `$DGGT_CKPT` 的冻结 CameraHead；即使使用 `--latent_stats_path` 复用 latent stats，仍必须传 `--dggt_ckpt_path`。stats/pretrain/T1/offline inference 会核对同一 DGGT SHA256。
* 旧的 `feature_stats_pretrain.pt`、`feature_stats_pretrain_v2.pt` 和 `feature_stats_pretrain_v3_798.pt`
  都不再用于训练；正式命令统一使用上面的 10-frame `feature_stats_pretrain_v3.pt`。训练入口会核对 `source.sequence_length`。
* **正式 pretrain 和正式训练统一使用 1024-dim tokenizer latent**：`$TOKENIZER_CKPT`、`$FEATURE_STATS`、`--latent_dim 1024`、`$SCENE_FLOW_PRETRAIN_CKPT` 必须来自同一套 1024 tokenizer / SceneFlow pretrain。正式训练、resume 和 offline inference 会在 checkpoint 加载后逐项核对 latent/camera stats；数值不一致会直接报错，不允许用新 stats 静默改变 checkpoint 的坐标系。
* `--tokenizer_ckpt_path` 只有在基础 DGGT checkpoint 本身完整包含 `scene_tokenizer.*` 时才可省略。旧 DGGT checkpoint 通常不含 tokenizer；此时 pretrain、正式训练和 offline inference 都会直接报错，禁止保留 `VGGT()` 随机初始化的 tokenizer。
* 实现设计说明见 `docs/scene_flow_model_design.md`。本文档只维护运行命令和参数。
* `--shift 10.0`、`--weighting_scheme waver --mode_scale 1.29`、`--lambda_repa 0.5`、EMA 验证默认开启；pretrain 和正式训练保持一致。
* 新 SceneFlow checkpoint 的 full/raw/EMA-only 三种导出都携带 `flow_schedule_config`：版本、flow path、训练 timestep 分布、`shift`、有效的 `mode_scale` 或 `logit_mean/logit_std`、loss weighting、`prediction_type`、`t_eps`、clean reconstruction 语义、推理时间网格和 Euler solver。resume 与 pretrain→T1 warm-start 会逐项拒绝不一致；offline inference 以 checkpoint 为准，命令行 `--shift` 仅作为可选一致性断言。`sample_steps` 是可调的 solver 精度/成本参数，但必须满足 `sample_steps <= shift/t_eps-shift+1`，避免进入 RAE pseudo-velocity 的非精确 clamp 区间。旧 v1 x-prediction schedule 可无损迁移；旧 v-prediction schedule 无法证明使用了正确 clean inverse，会被拒绝。旧 weights-only 无法证明 schedule，也会被拒绝。
* Gaussian 时间轴固定为 `clip_local_frame_id / 4`，新 cache 必须携带 `meta.gaussian_time_representation=clip_local_frame_id_div4_v1`。缺少该标记的 cache 会被训练/offline 拒绝；修改代码前已经启动的 precompute 进程不会热加载新实现，必须重启。chunked cache summary 同样携带该标记，因此 `--overwrite_v7` 能识别并重建旧时间轴 cache。
* mRoPE 坐标使用 A3 设计：text/timestep 使用 zero RoPE；video、asset、edit-control 与 camera 共享 `[0,15000)` 视频时间轴；camera 位于 patch grid 中心；sky 使用以 `15000` 为中心的上半球 Cartesian 坐标，使经度首尾天然相邻。这里不使用 seam loss。`rope_max_position=16384` 会真实检查越界，checkpoint 记录 `rope_layout_version=a3_camera_center_spherical_sky15000`。

## 1. Pretrain 正式参数

### 1.0 feature stats（v3）

先对完整 29 帧 caption clip 做一次冻结 DGGT forward，同时计算 tokenizer latent stats、
DGGT CameraHead-derived camera anchor/delta 11D stats 与 factorized asset 的 12D placement stats；
10 帧 latent 窗口只在完整 DGGT 上下文输出之后切片。该路径不读取 Waymo extrinsics/intrinsics。
输出文件必须包含 `mu_z/sigma_z`、`camera_anchor_mean/std`、`camera_delta_mean/std`、
**`placement_mean/placement_std/placement_count`**、target space/source/version、
DGGT checkpoint SHA256 与 anchor/delta count。

当前在用的 `feature_stats_pretrain_v3.pt` 是**不带 `--max_batches` 的完整一遍统计**
（`count=7381500`），与启动器一致。缩短标定时间可以加 `--max_batches`，但那样得到的
`placement` 分布覆盖的目标更少。

```bash
CUDA_VISIBLE_DEVICES=2 python -u tools/compute_pretrain_feature_stats.py \
    --image_dir $WAYMO_DGGT_ROOT \
    --dggt_ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --output_path $FEATURE_STATS \
    --scene_start 0 --scene_end 800 \
    --sequence_length 10 \
    --latent_dim 1024 \
    --camera_anchor_window_probability 0.5 \
    --batch_size 1 \
    --num_workers 4 \
    --log_every 20
```

### 1.1 正式 pretrain 启动器（当前实际使用）

正式 pretrain 通过仓库根目录的启动器运行，全部启动器共用同一份 `build_train_args`：

| 启动器 | 拓扑 | `BATCH_SIZE_PER_GPU` × `GRAD_ACCUM_STEPS` | 全局 batch |
|---|---|---|---|
| `pretrain_single_node.sh` | 1 节点 × 8 GPU | 1 × 8 | 64 |
| `pretrain_single_node30.sh` | 1 节点 × 8 GPU（10.199.7.30） | 1 × 8 | 64 |
| `pretrain_two_nodes26.sh` | 2 节点 × 8 GPU（26 + 25） | 1 × 4 | 64 |
| `pretrain_two_nodes31.sh` | 2 节点 × 8 GPU（31 + …） | 1 × 4 | 64 |
| `pretrain_three_nodes.sh` | 3 节点 × 8 GPU | 1 × 3 | 72 |
| `pretrain_four_nodes.sh` | 4 节点 × 8 GPU | 1 × 2 | 64 |
| `pretrain_half_node_p6000.sh` | 1 节点 × 4 GPU | 1 × 8 | 32 |
| `pretrain_ppu.sh` | PPU，1 节点 × 2 卡（可用环境变量覆盖） | 1 × 4 | 8 |

```bash
# 在主节点执行；脚本会自检文件/Python/GPU/网卡/IB，SSH 拉起副节点，再启动本节点。
bash pretrain_two_nodes26.sh                 # 启动
bash pretrain_two_nodes26.sh --stop-master   # 停止本节点
bash pretrain_two_nodes26.sh --stop-worker   # 停止副节点
```

共用的训练参数（等价的裸命令，便于单卡 smoke test；集群上请用上面的启动器）：

```bash
torchrun --nproc_per_node=8 train_scene_flow_pretrain.py \
    --image_dir $WAYMO_DGGT_ROOT \
    --val_image_dir $WAYMO_DGGT_VAL_ROOT \
    --dggt_ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --feature_stats_path $FEATURE_STATS \
    --log_dir logs/scene_flow_pretrain_1024_v3 \
    --caption_root $SCENE_CAPTION_ROOT \
    --val_caption_root $SCENE_CAPTION_VAL_ROOT \
    --text_encoder_path $QWEN_TEXT_ENCODER \
    --scene_start 0 --scene_end 800 \
    --sequence_length 10 \
    --val_sliding_window 10 \
    --val_sliding_stride 7 \
    --camera_anchor_context_dropout 0.25 \
    --patch_grid_h 25 --patch_grid_w 37 \
    --latent_dim 1024 \
    --batch_size 1 \
    --grad_accum_steps 8 \
    --num_workers 8 \
    --prefetch_factor 2 \
    --lr 1e-4 \
    --final_lr 1e-5 \
    --weight_decay 0.0 \
    --optimizer_type gmuon \
    --ema_decay 0.9995 \
    --warmup_steps 4000 \
    --max_steps 200000 \
    --save_every 2000 \
    --shift 10.0 \
    --weighting_scheme waver \
    --mode_scale 1.29 \
    --loss_weighting_scheme none \
    --prediction_type x \
    --lambda_repa 0.5 \
    --base_model_coeff 0.25 \
    --lambda_boundary 0.25 \
    --lambda_camera_flow 0.1 \
    --lambda_camera_pose 0.5 \
    --lambda_sky_flow 0.1 \
    --rgb_render_every 2 \
    --uncond_drop_prob 0.1 \
    --text_uncond_drop_prob 0.1 \
    --asset_uncond_drop_prob 0.2 \
    --camera_uncond_drop_prob 0.2 \
    --all_cond_drop_prob 0.05 \
    --guidance_scale 1.0 \
    --asset_control_guidance_scale 1.0 \
    --camera_guidance_scale 1.0 \
    --val_guidance_scales "1.0,2.0,4.0" \
    --val_scene_start 0 --val_scene_end 100 \
    --val_every 1000 \
    --val_batches 1 \
    --val_log_images 10 \
    --val_sample_steps 35 \
    --grad_clip_norm 1.0 \
    --seed 0 \
    --precision bf16 \
    --ddp_timeout_minutes 60 \
    --wandb --wandb_project dggt-flow \
    --wandb_name scene_flow_pretrain_waymo_gb64_lr1e4_v3
```

启动器额外做的事：设置 `WANDB_API_KEY`、清理残留 conda 状态、固定
`PYTHONNOUSERSITE=1 / OMP_NUM_THREADS=4 / HF_HUB_OFFLINE=1`、按 `NETWORK_MODE`
在双 200G HDR IB（`mlx5_4/5`）→ `mlx5_bond_0` RDMA → `bond0` socket 之间自动降级，
并把日志/PID 写到 `logs/distributed_launch/`。

**启动器里没有显式传、但依赖默认值的参数**（默认值恰好等于本文件旧版显式写出的取值，
所以省略不改变行为）：

```text
--camera_anchor_window_probability 0.5   --scheduler_type linear
--decay_end_steps 0  → 回退成 max_steps=200000
--lambda_rgb_render 0.1   --lambda_level_consistency 0.1   --lambda_head_consistency 0.1
--rgb_render_start_step 5000   --rgb_render_warmup_steps 5000   --rgb_render_sigma_power 2.0
--rgb_render_max_samples 1  --rgb_render_max_frames 0  --rgb_render_stride 1
--rgb_render_camera_grad_scale 0.0   --rgb_render_sky_mask_grad_scale 0.05
--rgb_render_lpips_weight 0.01   --rgb_render_sky_weight 1.0
--lambda_sky_mask 0.05   --lambda_sky_mask_refine 0.1
--sky_mask_endpoint_start_step 5000   --sky_mask_endpoint_every 4
--sky_unobserved_loss_weight 0.05   --lambda_sky_view_reconstruction 0.1
--camera_{translation,rotation,fov,absolute,relative}_weight 1.0   --camera_smoothness_weight 0.25
--camera_text_guidance_scale 1.0
```

### 1.2 启动器参数复核结论

以下是对 `pretrain_*.sh` 现有取值的逐项复核。**打勾的不需要改；标 ⚠️ 的是运行隐患，
本文档只记录，未改动脚本。**

| 项 | 取值 | 结论 |
|---|---|---|
| `--feature_stats_path` | `feature_stats_pretrain_v3.pt` | ✅ 必须是 v3。v2 缺 `placement_mean/std`，`factorized_v1` 会 `KeyError` |
| `--latent_dim 1024` / `--patch_grid_h 25 --patch_grid_w 37` | — | ✅ 与 tokenizer / 数据一致 |
| `--shift 10 --weighting_scheme waver --mode_scale 1.29 --prediction_type x` | — | ✅ 与 RAEv2 数值对拍一致；`--val_sample_steps 35` 满足 `steps ≤ shift/t_eps-shift+1 = 191` |
| 全局 batch | 64（3 节点为 72） | ✅ 但 3 节点的 72 与其它拓扑不可严格互比，`WANDB_NAME` 已按 `gb72` 区分 |
| `--lr 1e-4 --final_lr 1e-5` | — | ✅ 可用。RAEv2 t2i 在 gmuon + 全局 batch 1024 下用 2e-4；这里 batch 64 用 1e-4 偏保守，若前 2 万步 loss 下降过慢可以试 2e-4 |
| `--warmup_steps 4000` | — | ✅ 未传 `--warmup_from_zero`，所以这是 4000 步的**初始 LR 平台**，不是从 0 升温 |
| `--decay_end_steps`（未传） | 回退 `max_steps` | ✅ 线性从 1e-4 衰减到 1e-5，终点 200000 |
| `--save_every 2000` | — | ✅ 与 argparse 默认一致 |
| `--camera_anchor_context_dropout 0.25` | — | ⚠️ argparse 默认与 `docs/scene_flow_model_design.md` §1 都是 `0.0`，理由是"delta-only 窗口本身已覆盖后续滑窗不含 anchor 的输入分布，该参数仅作消融开关"。0.25 会额外让约 12.5% 的 micro-batch（P(含 anchor)=0.5 × P(drop)=0.25）被 `select_rgb_render_rows` 整体排除出 RGB render loss。**要么把脚本改回 0.0，要么更新设计文档说明为什么改用 0.25** |
| `--lambda_camera_pose 0.5` | — | ⚠️ argparse 默认与本文档旧版都是 `1.0`。0.5 是有意降权还是遗留？建议在设计文档里记一笔 |
| `--rgb_render_every 2` | — | ✅ 与默认一致（每 2 个 optimizer step 算一次 RGB/consistency） |
| `--val_batches 1` | — | ⚠️ 只用 1 个 batch 估计 validation loss，噪声很大，曲线不可读。采样出图才是大头，建议至少 4 |
| `--num_workers 8 --prefetch_factor 2` | — | ⚠️ 见下方"数据吞吐"，当前是主要瓶颈 |
| `pin_memory`（未传） | `False` | ⚠️ 单样本 tensor 不小（29 帧 DGGT context + 5 个 canonical reference），建议开 `--pin_memory` |
| `--resume_path .../pretrain_step018000.pt` | 写死在 `two_nodes26/four_nodes`；`two_nodes31/three_nodes` 写死 `step016000` | ⚠️ **代码没有 auto-resume-latest**。每次重跑启动器都会退回到那个固定 step，之后的进度全部丢失。启动前必须手动改成最新 ckpt，或改成从环境变量读取 |
| `--wandb_run_id 0bmu7zky --wandb_resume must` | 4 个多机启动器全部写死同一个 id | ⚠️ 不同集群的运行会写进同一个 wandb run；`must` 在该 run 不存在时直接失败。建议改成 `${WANDB_RUN_ID:-}` 并按 LOG_DIR 区分 |
| `LOG_DIR=logs/scene_flow_pretrain_1024_v3` | 8 个启动器全部相同 | ⚠️ 单机/2 机/3 机/4 机共用一个 ckpt 目录。同时跑两个拓扑会互相覆盖 `ckpt/` 与 `validation/`。建议按拓扑加后缀 |
| `--optimizer_type gmuon` | — | ✅ 与 RAEv2 t2i 一致；`gmuon` 不可用时代码会显式报错而不是静默回退 AdamW |

**数据吞吐（实测）**：`WaymoOpenDataset.__getitem__` 在本机实测约 **33 s/样本**，其中
`_project_pretrain_object_slots` 占 23 s（同一帧相机逆矩阵被重复求解上万次、每个候选目标×每个
参考帧都重新解码一张 dynamic mask PNG、大量 `torch.tensor(np_array.tolist())`）。
按 `--num_workers 8`、`--grad_accum_steps 8` 估算，每个 rank 每个 optimizer step 需要 8 个样本，
数据侧约 33 s，而模型 step 约 1–2 s。**当前 pretrain 是彻底的数据瓶颈**，加大 `num_workers`
受限于每节点 8 GPU × N workers 的 CPU 核数，只能靠修数据集代码解决。

### 1.3 有效 batch / 学习率 / schedule 依据

| 参数 | 当前取值 | 依据 |
|---|---|---|
| 有效 batch | `nodes × 8 GPU × batch_size 1 × grad_accum` = **64**（3 节点为 72） | 每 clip 的 video span 是 10 帧 × 925 patch，再加 text / 5×10×33 asset / camera / 512 sky token，full-attention 序列约 1.1 万 token。`batch_size=1` + grad_accum 是为了把 `B×N²×heads` 的 attention 峰值压住；改拓扑时用 `grad_accum` 补回全局 batch，启动时脚本会打印 `GLOBAL_BATCH_SIZE` |
| `--lr 1e-4 --final_lr 1e-5` | GMuon | RAEv2 t2i 在全局 batch 1024 下用 `2e-4`；这里全局 batch 64，`1e-4` 偏保守但安全。若前 2 万步 loss 明显停滞可提到 `2e-4`。换 AdamW 需显式 `--optimizer_type adamw` |
| `--warmup_steps 4000` | ≈ 2% of max_steps | 未传 `--warmup_from_zero`，所以这是 4000 步的**初始 LR 平台**，之后才开始 linear decay，不是从 0 升到 `1e-4` |
| `--max_steps 200000`（`--decay_end_steps` 未传） | linear 衰减终点 = max_steps | `scheduler_type=linear` 默认；LR 从 `1e-4` 线性降到 `final_lr=1e-5`，不会衰到 0 |
| `--ema_decay 0.9995` | half-life ≈ 1.4K step | RAE 取值；EMA 验证默认开启 |
| `--save_every 2000` | — | 与 argparse 默认一致；200K step 会产生约 100 个 full checkpoint，注意磁盘 |

显存兜底（80GB 仍 OOM 时按序降级，保持有效 batch 64）：

* 增大 `GRAD_ACCUM_STEPS`、保持 `BATCH_SIZE_PER_GPU=1`（有效 batch 不变）
* 再不够：`--sequence_length 6`（同时按 `docs/scene_flow_model_design.md` 检查 stride 是否仍合法）
* 仍不够：`--val_batches 1 --val_log_images 2 --no_val_render_rgb`（只降验证开销，不动训练）

> 旧的 4 卡 `--sequence_length 4 --batch_size 1` 与 `2 GPU × batch_size 8` 有效 batch 16 的配置都已弃用；
> 当前统一使用 `--sequence_length 10 --latent_dim 1024`、全局 batch 64。

新增运行行为：

* pretrain 训练使用 tqdm 进度条；如果日志系统不适合交互式进度条，可加 `--no_tqdm`。
* tqdm 会每个 optimizer step 实时显示当前 loss 和 lr；train 标量也会每个 optimizer step 写入 wandb。
* RGB loss 的前向渲染始终使用 SceneFlow 生成的 DGGT camera、generated depth/GS 和 predicted sky mask；`camera/sky_mask_grad_scale=0` 只切断对应 RGB 梯度，不会回退 teacher geometry。建议主 flow 收敛后再把 camera gradient 小幅升到 `0.05～0.1`。
* `--seed` 会设置 Python/NumPy/PyTorch/CUDA 随机种子；DDP 下每个 rank 使用 `seed + rank`。
* `--val_image_dir` 指定 validation split 根目录；`--val_scene_start/--val_scene_end` 是在该 validation split 内部选 scene 范围，不要用 training split 的 800-850 做验证。
* `--val_every 1000` 表示每 1000 个 optimizer step 跑一次 validation；不是每 1000 个 batch，也不是每 1000 个 epoch。
* `--val_batches` 表示每次 validation 只遍历几个 validation batch 估计 loss，用来控制验证耗时；它不会限制训练数据量。启动器当前用 `1`，loss 曲线噪声较大，只能看趋势；想读数值建议至少 `4`。
* pretrain validation 的局部 loss 会按与长窗相同的 clip-global 起点轮转；10 帧、stride 7 时为 `0/7/14/19`，因此同时覆盖含唯一 anchor 的首窗和三个 delta-only 后窗。采样可视化固定使用完整 29 帧 clip，并以训练 `sequence_length` 作为窗口做滑窗 rollout；若配置 stride 不适用于更短的训练窗口（例如 `sequence_length=6, stride=7`），会自动改用该窗口的三帧重叠默认值。
* pretrain 现在固定为 full_scene；旧的 `pseudo_edit/random_inpaint/mixed` CLI 参数已经删除。
* `--uncond_drop_prob` 保留为旧参数 fallback；正式控制建议显式传 `--text_uncond_drop_prob --asset_uncond_drop_prob --camera_uncond_drop_prob --all_cond_drop_prob`。
* 默认训练 sky generation；如需关闭，加 `--no_sky_generation`。
* sky target 是 `32×64` 上半球 RGB atlas，每个方向选置信度最高的可见帧；未观测区域 observation weight 为零，不再填全局均色。通过固定 `2×2` pixel-unshuffle 打包为 `16×32×12`，SceneFlow 仍只处理 512 个 sky token，不需要独立 sky tokenizer 或额外 checkpoint。
* validation 图像会保存到 `${LOG_DIR}/validation/step_xxxxxx/`（启动器为 `logs/scene_flow_pretrain_1024_v3`）；默认包含 `generated_raw_3dgs_rgb__cfg*.jpg`、`generated_sky_rgb__cfg*.jpg`、`generated_pred_sky_mask__cfg*.jpg`、latent PCA 和误差图。额外 CFG scale 会追加 `*_cfg{scale}` 后缀。
* pretrain offline inference 在 checkpoint 加载后只接受与 checkpoint 内 `mu_z/sigma_z` 和四组 camera anchor/delta buffers **逐元素完全一致**的 stats 文件；不一致会报错，不会再用外部文件覆盖 checkpoint 坐标系。
* `--val_sample_steps` 只控制 validation 图像采样步数，不影响训练本身。启动器用 `35`。`15` 偏少，只适合 smoke test；需要更稳定的样本可用 `50`。FlowMatch/RAE 的生成采样也不是训练时的 1000 timestep 全跑，而是在 scheduler timestep 上做几十步推理。RAE 的 target 使用 `max(sigma,t_eps)`，因此最后一个非零采样点不能低于 `t_eps`；代码会拒绝越界配置。默认 `shift=10,t_eps=0.05` 时最多 191 步。
* 训练内 validation 默认 `--val_sliding_window 10 --val_sliding_stride 7`，即相邻窗口重叠 3 帧。长序列必须满足 `1 <= stride < window`；`stride>=window` 直接报错。采样维护 full video/camera/sky 状态，对 video/camera/mask logits 用 cosine coverage 逐帧归一化；scene-global sky 使用 `sum(w/C)` 窗口权重，使每个全局帧贡献相等。
* 如需只记录 latent/mask 诊断图并跳过较慢的 3DGS RGB 渲染，可额外加 `--no_val_render_rgb`。
* 若当前机器未登录 wandb，可先执行 `wandb login`，或临时去掉 `--wandb` 相关参数。

## 1.5 关键参数说明

* `--no_val_ema`：默认 validation 使用 EMA 权重；加这个参数才会用实时裸权重验证。
* mRoPE A3 固定坐标：video/asset/edit-control 使用真实 `(t,y,x)`；camera condition/generation 使用同帧 `t` 和 patch grid 空间中心；sky 使用以 `15000` 为中心的上半球 Cartesian 坐标。frame id ≥ `15000` 会 fail-fast。
* `--weighting_scheme waver --mode_scale 1.29 --shift 10.0 --prediction_type x`：pretrain、正式训练、训练内 validation、离线 inference 需要保持一致，并由 checkpoint 的 `flow_schedule_config` 强制执行。`logit_mean/logit_std` 只在显式切回 `--weighting_scheme logit_normal` 做 ablation 时使用。`x` 是 RAEv2 T2I 的 clean-latent 输出参数化；代码仍会把它转换成 velocity 做 flow matching loss 和 ODE 采样。offline 命令可省略 `--shift`；若显式传入，则必须与 checkpoint 相等。
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

> **阻断提示（2026-07-29）**：当前 pretrain 用
> `asset_condition_protocol="factorized_v1"` 构造模型（`train_scene_flow_pretrain.py:6373`），
> `train_scene_flow.py` 会用 checkpoint 里的 `scene_flow_config` 原样重建同一配置，
> 但它的 `train_step` 和三个采样入口都不传 `factorized_asset_condition`，
> forward 会直接抛
> `RuntimeError: This SceneFlow configuration requires FactorizedAssetCondition; the legacy target-token copy/oracle path is disabled.`
> 因此下面这条 `--scene_flow_pretrain_path $SCENE_FLOW_PRETRAIN_CKPT` 的命令
> **在新 pretrain checkpoint 上还不能启动**。需要先决定正式训练的 asset 协议：
> 要么把正式训练也改成从 flow cache 的 3D box + canonical reference 构造
> `FactorizedAssetCondition`，要么给 pretrain 保留一个兼容开关。
> 详见 `docs/review_scene_flow_2026-07-29.md` §A1。

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
    --lambda_rgb_render 0.1 \
    --lambda_level_consistency 0.1 \
    --lambda_head_consistency 0.1 \
    --rgb_render_every 1 \
    --rgb_render_start_step 5000 \
    --rgb_render_warmup_steps 5000 \
    --rgb_render_sigma_power 2.0 \
    --rgb_render_max_samples 1 \
    --rgb_render_max_frames 0 \
    --rgb_render_stride 1 \
    --rgb_render_lpips_weight 0.01 \
    --uncond_drop_prob 0.1 \
    --guidance_scale 1.0 \
    --asset_control_guidance_scale 1.0 \
    --val_guidance_scales "1.0,2.0,4.0" \
    --val_every 1000 \
    --val_batches 8 \
    --val_log_images 10 \
    --val_sample_steps 50 \
    --no_val_render_rgb \
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
| 有效 batch | `2 GPU × batch_size 2 × grad_accum 4 = 16` clip/optimizer update | ⚠️ 与当前 pretrain 的全局 batch **64** 不一致（pretrain 已由 `pretrain_*.sh` 统一到 64）。若要 T1 与 pretrain 严格可比，需要把 T1 拉到同样的全局 batch，或明确记录这是有意的小 batch 微调。DataLoader 现在返回完整 micro-batch list，不再丢弃 `batch[1:]` |
| micro-batch 执行 | 默认将 `batch_size>1` 的 bundle 合并后一次 forward/backward | 如需回退旧路径可加 `--no_batch_scene_flow` |
| asset/cache 输入 | cache 读取所有可用 asset LUT levels，并使用 cached `pass2_splatted_tok_low` | 不再 live splat/blend；cache 字段缺失会在 DataLoader/assembler 阶段报错 |
| camera 输入/渲染 | SceneFlow 条件输入使用 Waymo `camera_to_world_corrected + intrinsics` 摘要；RGB render 固定使用 cache 中完整 29 帧上下文预测的 DGGT camera，窗口渲染只切片对应 `pose_enc` | Waymo camera 只作为条件，不直接给 renderer；正式训练/离线推理不启用 camera generation token，也不让 edited latent 或局部窗口 CameraHead 重新定义相机 |
| Reconstruction feedback / RGB 几何链 | generated video latent 经 frozen tokenizer decoder + DGGT depth/GS/instance heads 后，计算四层 feature consistency、frozen-head consistency 并可微渲染；三者共享 `every/start/max_samples/max_frames/stride`、warmup 和连续 sigma 权重 | teacher 为 `stopgrad(D(z_clean)) → H`，不读取额外 cache head；正式阶段固定 input-DGGT camera，pretrain 才生成 camera/sky |
| sky handling | T1 不启用 pretrain sky generation 参数 | 正式训练、训练内 validation 和 offline inference 均执行 `GT_sky_mask * input_GT_RGB + (1-GT_sky_mask) * rendered_edit`，不调用 sky model 或做 min-max |
| DataLoader | `--num_workers 4 --prefetch_factor 1`，默认不启用 `pin_memory` | 每个 cache 文件平均约 651MB，低 prefetch 避免 8 workers × 2 prefetch × batch_size 2 造成几十个大文件并发读；GB 级 batch 走 pin-memory 线程容易触发 `received 0 items of ancdata` |
| worker tensor sharing | 默认 `--mp_sharing_strategy file_system` | 减少 multiprocessing 通过大量 fd 传递超大 tensor 时的稳定性问题；若系统 `/dev/shm`/临时目录策略特殊，可显式改回 `file_descriptor` |
| cache 读取 | 默认读取 chunked zstd `.pt`；每个样本只解压 10 帧窗口需要的 chunk | 当前逻辑 `schema_version=9`。Mode-A 的 `[S,P]` asset patch mask 存在轻量 per-asset meta chunk，fast loader 不读取 `A_asset`、RGB、Gaussian 或 pointer；缺少 `alpha_max_t005_v1` mask 的旧 Mode-A cache 必须重新生成。Mode-B 不含真实 asset condition，不依赖该字段。 |
| sigma / target | `--shift 10.0 --weighting_scheme waver --mode_scale 1.29 --loss_weighting_scheme none --prediction_type x` | 与 pretrain 保持一致 |
| REPA | `--lambda_repa 0.5` | 与 pretrain 保持一致 |
| EMA | `--ema_decay 0.9995`，validation 默认用 EMA | checkpoint 同时保存 raw / full / EMA-only 权重；三个正式训练导出都携带训练后的 `scaffold_packer`，EMA-only 是 EMA SceneFlow + 当前训练后 packer，与训练内 EMA validation 一致 |
| validation | `--val_manifest_path $SCENE_FLOW_VAL_MANIFEST --val_caption_root $SCENE_CAPTION_VAL_ROOT --val_every 1000 --val_batches 8 --val_log_images 10 --val_sample_steps 50 --guidance_scale 1.0 --asset_control_guidance_scale 1.0 --val_guidance_scales "1.0,2.0,4.0" --no_val_render_rgb` | 独立 validation manifest/cache 可使用 validation captions；内部 holdout 仍可用，但不能混用 validation caption root。默认 T1 validation dataset 仍是 10 帧窗口，不会仅因设置 sliding 参数自动扩展为 29 帧；完整 29 帧 T1 滑窗链路用 formal offline inference 验证。采样噪声使用 `seed + step`，可复现 |
| schedule | `--lr 2e-4 --final_lr 2e-5 --scheduler_type linear --decay_end_steps 150000 --weight_decay 0.0 --warmup_steps 3000 --max_steps 150000` | ⚠️ optimizer 与 `linear` 类型和 pretrain 一致，但**数值已经不一致**：当前 pretrain 启动器用 `--lr 1e-4 --final_lr 1e-5 --warmup_steps 4000 --max_steps 200000`。T1 是 fine-tune，LR 不高于 pretrain 更合理，建议改成 `--lr 1e-4 --final_lr 1e-5`（或更低）。默认 `warmup_from_zero=false`，前 N step 是初始 LR plateau，不是 from-zero warmup |
| RoPE 时间轴 | T1 调用 `forward(fps=None)`，pretrain 传 `fps=10` | ⚠️ **不一致**：pretrain 走 Cosmos fps 调制，帧 id 被乘 `24/10=2.4`（实测 `t=[0,2.4,4.8,...]` vs T1 `t=[0,1,2,...]`）。warm-start 后所有 video/asset/camera token 的时间位置被压缩 2.4 倍，必须统一。见 `docs/review_scene_flow_2026-07-29.md` §B3 |

注意：`train_scene_flow.py` 的 `global_step` 现在和 pretrain 一样是 optimizer update 口径；`--max_steps/--save_every/--vis_every/--val_every` 都按 optimizer update 触发。

注意：上面的正式命令显式使用 `--no_val_render_rgb`，因此训练内 validation 只保存 loss、latent PCA / mask / CFG 采样诊断图并写入 wandb，不运行 3DGS RGB 渲染。若删除该参数，代码默认会执行 3DGS RGB validation。训练内 validation 的标量也不是完整训练目标：它不包含按训练 schedule 触发的 RGB/endpoint objective。

注意：正式 offline 入口是 `inference_scene_flow.py`，不是不存在的 `inference_scene_flow_validation.py`。29 帧输入会自动解析为 `window=10, stride=7`（重叠 3 帧）；它维护 full 29 帧 latent，在每个采样步对窗口 velocity 做 cosine coverage 归一化后统一更新。`--val_log_images` 默认是 10，只控制最终导出/拼图/渲染的帧数；需要导出完整 29 帧时必须显式传 `--val_log_images 29`。正式阶段始终使用 cache 中 full-context input DGGT camera，并在 GT sky mask 内逐像素保留 input GT RGB，不生成 11D camera/sky；即使启用 `--render_per_window`，也只切片同一条 29 帧 DGGT `pose_enc`，不会逐窗口重跑 CameraHead。

## 2.1 Camera cache 修复与旧 checkpoint 迁移

cache 缺少相机 GT 或原图尺寸时，普通 `.pt` 与 SQLite chunk cache 使用同一命令修复：

```bash
python tools/backfill_flow_cache_camera_gt.py \
    --cache_root /path/to/cache_or_root \
    --processed_root /data/disk2/lyy_dataset/waymo_processed_dggt \
    --split training --force
```

旧 48K pretrain 的 `dggt_hidden_v1/2048D` camera head 不能 resume 到当前的
`dggt_relative_se3_rot6d_logfov_v3` camera representation。先用上面的 stats 工具生成
`dggt_camera_anchor_delta_per_channel_v4_global_context` stats，然后把显式允许的 legacy shared-trunk checkpoint（265bd939）传给 `--warm_start_path`：程序只迁移 EMA
materialized shared trunk 中同名同 shape 参数，跳过 camera generation projection/decoder、
role embedding 与 stats，重置 EMA step/optimizer/scheduler，并从 pretrain step 0 开始。
`--resume_path` 只接受 representation、dimension、stats version 完全一致的新完整 checkpoint。

## 4. 四类 validation / offline inference

训练内 pretrain validation 使用 `train_scene_flow_pretrain.py` 的
`--val_sliding_window 10 --val_sliding_stride 7 --val_sample_steps 35`，其采样可视化会使用完整
29 帧 clip。正式训练 validation 使用 `train_scene_flow.py`，但默认 validation dataset 仍输出
10 帧窗口；同名滑窗参数不会把样本自动扩展成 29 帧。完整 29 帧的 T1 滑窗链路应使用下面的
formal offline inference 验证。正式 validation 不 pack、不加噪、不预测 camera/sky generation state。

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
  --scene_flow_ckpt_path /path/to/flow_stepXXXXXX_ema_weights_only.pt --ckpt_path $DGGT_CKPT \
  --tokenizer_ckpt_path $TOKENIZER_CKPT --manifest_path $SCENE_FLOW_VAL_MANIFEST \
  --output_dir runs/scene_flow_offline --window 10 --window_stride 7 \
  --val_log_images 29 \
  --edit_domain_threshold 1e-4 --edit_domain_dilation 1 \
  --sample_steps 50 \
  --guidance_scales 1,2,4
```

正式训练的 training-time validation 与正式 offline inference 均使用 50 个 Euler sampling steps；
`inference_scene_flow.py` 的 `--sample_steps` 默认值也是 50。`15` 仅用于 smoke test；若明确接受较低
采样精度来节省开销，可手动降到 Cosmos3 视频生成常用的 35 steps。

正式推理推荐直接使用新导出的 `flow_stepXXXXXX_ema_weights_only.pt`。它包含 EMA SceneFlow
和同一步训练后的 `scaffold_packer`；旧版正式 `_weights_only.pt` / `_ema_weights_only.pt` 若缺少
`scaffold_packer` 会被推理入口拒绝，不能用随机初始化 packer 代替。完整 `flow_stepXXXXXX.pt`
同样包含二者，并会默认物化其中的 EMA SceneFlow 权重。

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
