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
export TOKENIZER_CKPT=/home/dancer/code/dm/dggt/logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt
export FEATURE_STATS=logs/scene_flow_pretrain_1024/feature_stats_pretrain_v5.pt
export SCENE_GAUGE_PATH=data/scene_gauge/training.json
export VAL_SCENE_GAUGE_PATH=data/scene_gauge/validation.json
export VAL_SCENE_GAUGE_SHA256="$(sha256sum "$VAL_SCENE_GAUGE_PATH" | awk '{print $1}')"
export PULLBACK_CALIBRATION_PATH=data/scene_gauge/pullback_d63b34f7.json
# Phase 2 尚未开始；v2 stats 已生成，pretrain checkpoint 仍必须由 v2 训练产生。
export SCENE_FLOW_PRETRAIN_CKPT=logs/scene_flow_pretrain_tokenizer_v2/ckpt/pretrain_step100000.pt
export SCENE_CAPTION_ROOT=/data/disk2/lyy_dataset/waymo_processed_dggt/training_captions
export SCENE_CAPTION_VAL_ROOT=/data/disk2/lyy_dataset/waymo_processed_dggt/validation_captions
export SCENE_FLOW_TRAIN_MANIFEST=/data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl
export SCENE_FLOW_VAL_MANIFEST=/data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/validation/validation_manifest.jsonl
export QWEN_TEXT_ENCODER=/home/dancer/model/Qwen/Qwen3-0.6B
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
* tokenizer 自身的 aggregator stats 不能直接给 SceneFlow pretrain；SceneFlow 需要重新计算
  latent/camera/gauge/placement stats。当前 `$FEATURE_STATS` 已用 tokenizer v2 对完整 training split
  从头计算：4787/4787 trunks、44,279,750/44,279,750 latent，`stats_status=complete`，
  `latent_stats_path=null`，没有复用 v1 latent moments。
  2026-08-02 的 `feature_stats_pretrain_v4.pt` 绑定 tokenizer v1 SHA `75e566ef...`，只保留为历史证据，
  v2-only loader 会拒绝它，不能复制或重命名旧文件。
* camera stats 来自 Waymo 米制 camera target，不再来自 DGGT CameraHead。`$DGGT_CKPT` 仍用于
  tokenizer latent 提取及 provenance，`$TOKENIZER_CKPT` 必须显式传入。
* 训练 split 的完整 `$SCENE_GAUGE_PATH` 必须先由 29 帧离线 gauge 工具生成，才能计算训练用 v4
  feature stats；`training_shard_*.json` 不等价于完整 `training.json`。`$VAL_SCENE_GAUGE_PATH`
  独立阻塞 validation/pretrain inference 与 validation cache provenance，不是训练 stats 的输入。
  启动器会用 `check_file` 直接阻止误启动，不会创建占位文件。
* `$PULLBACK_CALIBRATION_PATH` 是与 v2 tokenizer/DGGT checkpoint SHA 绑定的 Scheme-A artifact；
  v1 schema/runtime/artifact 一律拒绝加载。
* **正式 pretrain 和正式训练统一使用 1024-dim tokenizer latent**：`$TOKENIZER_CKPT`、`$FEATURE_STATS`、`--latent_dim 1024`、`$SCENE_FLOW_PRETRAIN_CKPT` 必须来自同一套 1024 tokenizer / SceneFlow pretrain。pretrain→formal warm-start 会校验源 checkpoint 的 `pretrain_feature_stats_contract`（stats SHA、sequence length 10、DGGT context 29、grid 25×37）；formal resume/inference 再由 formal metric contract 保护自己的坐标系。任何不一致都会直接报错，不允许用新 stats 静默改变 checkpoint 的坐标系。
* `--tokenizer_ckpt_path` 在 metric-gauge pretrain 中必须显式提供，以便校验 pullback artifact
  的 tokenizer SHA；即使基础 DGGT checkpoint 内嵌 tokenizer 也不能省略。
* 实现设计说明见 `docs/scene_flow_model_design.md`。本文档只维护运行命令和参数。
* `--shift 10.0`、`--weighting_scheme waver --mode_scale 1.29`、`--lambda_repa 0.5`、EMA 验证默认开启；pretrain 和正式训练保持一致。
* 新 SceneFlow checkpoint 的 full/raw/EMA-only 三种导出都携带 `flow_schedule_config`：版本、flow path、训练 timestep 分布、`shift`、有效的 `mode_scale` 或 `logit_mean/logit_std`、loss weighting、`prediction_type`、`t_eps`、clean reconstruction 语义、推理时间网格和 Euler solver。resume 与 pretrain→T1 warm-start 会逐项拒绝不一致；offline inference 以 checkpoint 为准，命令行 `--shift` 仅作为可选一致性断言。`sample_steps` 是可调的 solver 精度/成本参数，但必须满足 `sample_steps <= shift/t_eps-shift+1`，避免进入 RAE pseudo-velocity 的非精确 clamp 区间。旧 v1 x-prediction schedule 可无损迁移；旧 v-prediction schedule 无法证明使用了正确 clean inverse，会被拒绝。旧 weights-only 无法证明 schedule，也会被拒绝。
* Gaussian 时间轴固定为 `clip_local_frame_id / 4`，新 cache 必须携带 `meta.gaussian_time_representation=clip_local_frame_id_div4_v1`。缺少该标记的 cache 会被训练/offline 拒绝；修改代码前已经启动的 precompute 进程不会热加载新实现，必须重启。chunked cache summary 同样携带该标记，因此 `--overwrite_v7` 能识别并重建旧时间轴 cache。
* 正式 flow cache 的干净切断版本是 schema v10。它在 payload meta 和 chunked
  summary 中同时绑定 `metric_box_mapping_mode=metric_gauge_v4`、scene-gauge 表 SHA
  与 DGGT checkpoint SHA。schema v9 及更旧 cache 一律重新预计算，物理格式转换
  或 backfill 不能升级这三项证据。
* mRoPE 坐标使用 A3 设计：text/timestep 使用 zero RoPE；video、asset、edit-control 与 camera 共享 `[0,15000)` 视频时间轴；camera 位于 patch grid 中心；sky 使用以 `15000` 为中心的上半球 Cartesian 坐标，使经度首尾天然相邻；单个 scene-global gauge token 固定在 `(15100,15100,15100)`。这里不使用 seam loss。`rope_max_position=16384` 会真实检查越界，checkpoint 仍记录兼容字段 `rope_layout_version=a3_camera_center_spherical_sky15000`，同时由 metric-gauge provenance 锁定 gauge 布局。

## 1. Pretrain 正式参数

### 1.0 feature stats（metric-gauge v4）

先完成 training/validation split 的完整 29 帧 scene-gauge 表，再对训练 split 做冻结 DGGT
forward，统计 tokenizer latent、9D 米制 Waymo camera、3D scene gauge 与 factorized-v3 16D
placement。输出必须是完整一遍数据的正式 v4 stats；`--max_batches` 只用于 smoke test，不能把
缩短统计得到的文件放到 `$FEATURE_STATS` 正式路径。

```bash
CUDA_VISIBLE_DEVICES=0 python -u tools/compute_pretrain_feature_stats.py \
    --image_dir $WAYMO_DGGT_ROOT \
    --dggt_ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --scene_gauge_path $SCENE_GAUGE_PATH \
    --output_path $FEATURE_STATS \
    --scene_start 0 --scene_end 800 \
    --sequence_length 10 \
    --latent_dim 1024 \
    --camera_anchor_window_probability 0.5 \
    --batch_size 1 \
    --num_workers 4 \
    --device cuda:0 \
    --precision bf16 \
    --require_dynamic_mask \
    --log_every 20
```

### 1.0.1 2026-08-02 tokenizer-v1 历史 artifact 与线路验收

本小节只保留不可变历史证据；其中 stats、smoke checkpoint 和 SHA 均不属于当前 v2 运行输入。

- training gauge：`data/scene_gauge/training.json`，4787/4787 trunks、798 scenes、0 errors，
  metric-scale/FOVx/FOVy valid counts `[4216,4787,4787]`，actor coverage 1662；SHA-256
  `39e0a32372e616e9aac4aef6109c8329ebdf382c16a913bd9e4d025b984e00af`。
- validation gauge：`data/scene_gauge/validation.json`，1212/1212 trunks、202 scenes、0 errors；SHA-256
  `5014e5c0ba5bd570c1a3d13e3fd222d15e32fe10276046dda763b7e87d9559fa`。
- 修复后的 v4 stats：先输出为 `.pt.inprogress`，在 schema、exact scope、latent count、finite tensor 与 provenance
  全部通过后才移动到 `$FEATURE_STATS`。覆盖 4787/4787 trunks、44,279,750/44,279,750 latent，
  `stats_status=complete`、`max_batches=null`；gauge counts `[4216,4787,4787]`、placement count 172605。
  camera anchor/delta counts 为 `[2416,45454]`；
  修复后的 `log_z_depth mean/std=2.998025/1.519740`。stats SHA-256 为
  `f5177c9262c878c1595c0f0e41ebd9cf42680de3676f0fccb789ed3cbc7a9111`，相邻 sidecar SHA-256 为
  `e0767b8bb3b86116f3748144b7c306d73ba6229a5568d47eb18a62cdf5d40539`。
- stats source contract：sequence length 10、DGGT context 29、grid 25×37、batch 1、workers 4、bf16，
  并绑定 tokenizer SHA `75e566efa3b66baa43f82cb9999c2de60a9f3feeb0f714e1caf38d1f6e8137eb`、DGGT SHA
  `352652738a5480b8d3ee9dd521ce07b528e5a297bd3feca4d07427dac6d87def` 与上述 training-gauge SHA。
- 旧 `logs/metric_gauge_one_step_cuda0/` 及其 inference 输出绑定修复前 v4（SHA `6fbdd3c5...eb81a`）/v2 contract，已被
  `rgb_patch_teacher_anchor_v3` 与 `factorized_asset_v3` clean cut 拒绝。
- 当前 CUDA 0 smoke 位于 `logs/metric_gauge_postreview_one_step_cuda0/`，使用修复后 v4 stats 且启用 sky
  generation；一步 loss/flow/sky-flow/gauge-flow/gauge-direct 为
  `3.4283/1.4045/0.1020/0.6241/0.0219`，LiDAR diagnostic available=1。其 EMA-only checkpoint
  实际 mmap 加载确认携带 sky-v3、factorized-v3、修复后 v4 stats SHA 与 10/29/25×37 contract；checkpoint
  SHA-256 为 `4daae958f043721f47a8e89c94cb5fe3a3b3e7ea7cfaf8586915c6e5ee85d9a6`。

旧版一步随机模型的 metric-depth error 是 17.29%，并在其后续 inference 中把所有候选点判成 sky，使 cycle 返回
`insufficient_support`。这些结果只验证可运行性与 fail-closed 行为；`gauge_vs_prior_gain`、相机可控性和
真实车长/车道米制误差必须等完整 pretrain/formal retrain 后评估，不能据此声称科学指标通过。

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
    --scene_gauge_path $SCENE_GAUGE_PATH \
    --val_scene_gauge_path $VAL_SCENE_GAUGE_PATH \
    --pullback_calibration_path $PULLBACK_CALIBRATION_PATH \
    --log_dir logs/scene_flow_pretrain_tokenizer_v2 \
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
    --pin_memory \
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
    --lambda_camera_pose 1.0 \
    --lambda_sky_flow 0.1 \
    --text_uncond_drop_prob 0.1 \
    --joint_generation_prob 0.2 \
    --camera_controlled_prob 0.2 \
    --asset_camera_controlled_prob 0.6 \
    --guidance_scale 1.0 \
    --asset_control_guidance_scale 1.0 \
    --camera_guidance_scale 1.0 \
    --val_guidance_scales "1.0,2.0,4.0" \
    --val_scene_start 0 --val_scene_end 100 \
    --val_every 2000 \
    --val_batches 8 \
    --val_log_images 10 \
    --val_inference_scenes 10 \
    --val_sample_steps 50 \
    --grad_clip_norm 1.0 \
    --seed 0 \
    --precision bf16 \
    --ddp_timeout_minutes 60 \
    --wandb --wandb_project dggt-flow \
    --wandb_name scene_flow_pretrain_waymo_gb64_lr1e4_v4
```

启动器额外做的事：设置 `WANDB_API_KEY`、清理残留 conda 状态、固定
`PYTHONNOUSERSITE=1 / OMP_NUM_THREADS=4 / HF_HUB_OFFLINE=1`、按 `NETWORK_MODE`
在双 200G HDR IB（`mlx5_4/5`）→ `mlx5_bond_0` RDMA → `bond0` socket 之间自动降级，
并把日志/PID 写到 `logs/distributed_launch/`。

**启动器里没有显式传、但依赖 trainer 默认值的参数**。这些损失权重有意只在
`train_scene_flow_pretrain.py` 维护，避免重平衡后被某个启动器静默钉回旧值：

```text
--scheduler_type linear
--decay_end_steps 0  → 回退成 max_steps=200000
--lambda_rgb_render 1.0   --lambda_level_consistency 1.0   --lambda_head_consistency 1.0
--rgb_render_every 1   --rgb_render_start_step 5000   --rgb_render_warmup_steps 5000   --rgb_render_sigma_power 2.0
--rgb_render_max_samples 1  --rgb_render_max_frames 0  --rgb_render_stride 1
--rgb_render_sky_mask_grad_scale 0.05
--rgb_render_lpips_weight 0.01   --rgb_render_sky_weight 1.0
--lambda_sky_mask 0.05   --lambda_sky_mask_refine 0.1
--sky_mask_refine_boundary_weight 4.0   --sky_mask_refine_boundary_loss_weight 0.125
--sky_atlas_h 128   --sky_atlas_w 256   --sky_grid_h 16   --sky_grid_w 32
--sky_unobserved_loss_weight 0.005   --lambda_sky_view_reconstruction 1.0
--lambda_gauge_flow 0.1   --lambda_gauge_direct 1.0
--metric_depth_diagnostic_every 500   --metric_depth_diagnostic_start_step 0
--metric_depth_diagnostic_max_samples 1
```

head-consistency 内部对 dynamic-confidence 通道使用
`DYNAMIC_HEAD_LOSS_WEIGHT=4.0`；它是损失实现常量，不是启动器可覆盖的 CLI 参数。

### 1.2 启动器参数复核结论

以下是对 `pretrain_*.sh` 现有取值的逐项复核。**打勾的是当前正式配置；标 ⚠️ 的是仍需调用方留意的运行隐患。**

| 项 | 取值 | 结论 |
|---|---|---|
| metric-gauge 输入 | v5 stats artifact（schema v4）+ training/validation gauge table + pullback artifact | ✅ 四者都由启动器逐文件检查；任何一项缺失都会在 `torchrun` 前失败 |
| `--feature_stats_path` | `feature_stats_pretrain_v5.pt` | tokenizer v2 正式统计；必须包含 9D metric camera、3D gauge、factorized-v3 16D placement 统计和 v2 tokenizer provenance |
| `--latent_dim 1024` / `--patch_grid_h 25 --patch_grid_w 37` | — | ✅ 与 tokenizer / 数据一致 |
| gradient checkpointing | DLC 默认 three_quarter | ✅ 公共 DLC launcher 传 `--three_quarter_gradient_checkpointing`，交错 checkpoint 21/28 encoder blocks，DDT 为 0/2；`GRADIENT_CHECKPOINTING=1` 使用 full，`half` 使用 14/28 + 1/2，`0` 完全关闭 |
| `--shift 10 --weighting_scheme waver --mode_scale 1.29 --prediction_type x` | — | ✅ 与 RAEv2 数值对拍一致；冻结的 `--val_sample_steps 50` 满足 `steps ≤ shift/t_eps-shift+1 = 191` |
| 全局 batch | 64（3 节点为 72） | ✅ 但 3 节点的 72 与其它拓扑不可严格互比，`WANDB_NAME` 已按 `gb72` 区分 |
| `--lr 1e-4 --final_lr 1e-5` | — | ✅ 可用。RAEv2 t2i 在 gmuon + 全局 batch 1024 下用 2e-4；这里 batch 64 用 1e-4 偏保守，若前 2 万步 loss 下降过慢可以试 2e-4 |
| `--warmup_steps 4000` | — | ✅ 未传 `--warmup_from_zero`，所以这是 4000 步的**初始 LR 平台**，不是从 0 升温 |
| `--decay_end_steps`（未传） | 回退 `max_steps` | ✅ 线性从 1e-4 衰减到 1e-5，终点 200000 |
| `--save_every 2000` | — | ✅ 与 argparse 默认一致 |
| `--camera_anchor_context_dropout 0.25` | — | ✅ 启动器有意保留的生成分支鲁棒性训练：只对确实含全局 anchor 的窗口隐藏 anchor，并 mask 对应 camera anchor/absolute-pose 监督。D3 后 RGB render 使用 detached teacher pose，**不再**因该 dropout 排除样本；内部日志会报告实际 dropout fraction。argparse 的 `0.0` 仍用于无额外消融的裸命令。 |
| `--lambda_camera_pose 1.0` | — | ✅ 与 argparse 默认及 `lambda_gauge_direct=1.0` 对齐；各拓扑启动器均显式固定该值 |
| RGB render cadence/window flags | 启动器不显式覆盖 | ✅ 统一服从入口的正式默认值，避免不同拓扑静默分叉 |
| `--val_batches 8` | — | ✅ 与 argparse 默认一致；固定的黄金比例散布覆盖集提供 8 个 validation batch，不再连续停留在同一 scene/trunk |
| `--num_workers 8 --prefetch_factor 2` | — | ✅ 真实 raw-data benchmark 为 3.065 sample/s/rank；8 workers 明显优于 4 workers 的 1.464 sample/s/rank |
| `--pin_memory` | `True` | ✅ 所有正式 pretrain 启动器均显式开启；CUDA 0 实测将中位 H2D 从 10.35 ms 降到 2.43 ms |
| `--resume_path` | 启动器不传 | ✅ metric-gauge v4 按 clean cut 从头训练；旧 v3 checkpoint 会被入口拒绝 |
| wandb resume | 多机启动器不传旧 run id | ✅ clean-cut 训练会创建独立 v4 run；PPU DLC 显式使用 `resume=never` |
| `LOG_DIR` | `logs/scene_flow_pretrain_tokenizer_v2`（可由环境变量覆盖） | ✅ 与所有 v1-bound run 隔离；并行跑多个拓扑时仍应各自覆盖成不同目录 |
| `--optimizer_type gmuon` | — | ✅ 与 RAEv2 t2i 一致；`gmuon` 不可用时代码会显式报错而不是静默回退 AdamW |

**数据吞吐（2026-08-02，真实 training scenes 300–305，CUDA 0）**：旧版文档记录的
`33 s/样本` 与 `_project_pretrain_object_slots=23 s` 已经过时。当前单进程 `__getitem__`
约 **2.36 s/样本**，其中图像解码/缩放 1.44 s、对象投影 0.86 s；主要剩余成本是必须执行的
JPEG、sky/semantic 数据解码与 resize。持久 worker、worker 内有界 LRU、uint8 IPC、29 帧只解码
一次等优化已经生效。

同一批真实样本、`batch_size=1/prefetch_factor=2` 的 DataLoader 吞吐如下：

| workers | sample/s/rank | 结论 |
|---:|---:|---|
| 0 | 0.350 | 同步读取，不用于正式训练 |
| 2 | 0.872 | 默认值仅适合调试 |
| 4 | 1.464 | 可用于 CPU/内存较紧节点 |
| 8 | 3.065 | 正式启动器取值 |

8 workers 下连同真实递归 H2D 测量，开启 pinned memory 后组合吞吐由 **1.686 提升到
2.801 sample/s/rank**，H2D 中位数由 10.35 ms 降到 2.43 ms。构建 bundle 时训练窗口现在直接从
已经搬到 GPU 的 29 帧上下文 gather，不再重复搬运同一组图像。LiDAR metric-depth 诊断若设
`--metric_depth_diagnostic_every 0`，数据集完全不读取；正常 cadence 下，仅在诊断/RGB render
实际执行的 step 才搬到 GPU。

Pinned-memory 预算约为每 rank `8 workers × 2 prefetch × 50 MiB ≈ 0.8 GiB`，8-rank 节点约
6.4 GiB host memory；资源不足时优先把 `PREFETCH_FACTOR` 降为 1，再考虑减少 workers。

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
* `sequence_length` 不能降到 6：v4 pullback 与 tokenizer 正式契约固定为 10 帧，入口会拒绝其它值。
  仍 OOM 时应关闭训练期 RGB/LPIPS 辅助项或减少 validation 渲染帧数，再用更大的梯度累积保持有效 batch。
* 仍不够：可显式用 `--val_batches 1 --val_log_images 2 --no_val_render_rgb` 做应急降级；这会覆盖正式默认的 8 个 validation batch，只适合排障，不适合比较标量曲线。

> 旧的 4 卡 `--sequence_length 4 --batch_size 1` 与 `2 GPU × batch_size 8` 有效 batch 16 的配置都已弃用；
> 当前统一使用 `--sequence_length 10 --latent_dim 1024`、全局 batch 64。

新增运行行为：

* pretrain 训练使用 tqdm 进度条；如果日志系统不适合交互式进度条，可加 `--no_tqdm`。
* tqdm 会每个 optimizer step 实时显示当前 loss 和 lr；train 标量也会每个 optimizer step 写入 wandb。
* RGB loss 的几何来自 generated depth/GS，sky mask 来自预测；相机默认固定为 detached 的完整
  29 帧 teacher DGGT pose（D3 gate）。旧 `--rgb_render_camera_grad_scale` CLI 已删除；底层兼容参数
  只接受 `0.0`，非零会 fail-fast。`--render_use_predicted_gauge` 只用于显式消融：它替换尺度/FOV gauge，不替换
  teacher 的旋转与轨迹形状。
* `metric_depth_rel_err` 有独立 cadence（默认每 500 optimizer step、从 step 0 开始），不再等到
  RGB loss 的 5000-step warm-up；非诊断步保留 `available=0` sentinel。
* `--seed` 会设置 Python/NumPy/PyTorch/CUDA 随机种子；DDP 下每个 rank 使用 `seed + rank`。
* `--val_image_dir` 指定 validation split 根目录；`--val_scene_start/--val_scene_end` 是在该 validation split 内部选 scene 范围，不要用 training split 的 800-850 做验证。
* 正式启动器的 `--val_every 2000` 表示每 2000 个 optimizer step 跑一次 validation；不是每 2000 个 batch，也不是每 2000 个 epoch。
* `--val_batches` 表示每次 validation 遍历几个 validation batch 估计 loss，用来控制验证耗时；它不会限制训练数据量。argparse 和全部正式启动器当前都用 `8`，并由固定的黄金比例散布 sampler 覆盖不同 scene/trunk/window，而不是按已消费样本数沿 trunk-major 索引缓慢前进。
* `--val_inference_scenes 10` 为每次采样安排 10 个场景 × 3 个 CFG scale。前 5 个场景 pinned，且只有它们进入 `validation/sample_*` 均值；后 5 个轮换且只出图。world size 小于 30 时回退到 2 个场景（1 pinned + 1 rotating，共 6 个任务），任务按 rank round-robin 分配。
* pretrain validation 的固定 spread cover 会从与长窗相同的 clip-global 起点取样；10 帧、stride 7 时为 `0/7/14/19`，因此每次都同时覆盖含唯一 anchor 的首窗和三个 delta-only 后窗。采样可视化固定使用完整 29 帧 clip，并以训练 `sequence_length` 作为窗口做滑窗 rollout；若配置 stride 不适用于更短的训练窗口（例如 `sequence_length=6, stride=7`），会自动改用该窗口的三帧重叠默认值。
* pretrain 现在固定为 full_scene；旧的 `pseudo_edit/random_inpaint/mixed` CLI 参数已经删除。
* `--uncond_drop_prob` 仅作为 `--text_uncond_drop_prob` 的兼容别名。asset/camera 不再独立 dropout，而由 `--joint_generation_prob --camera_controlled_prob --asset_camera_controlled_prob` 三项结构化任务概率控制；三者必须和为 1，且不会产生 asset-without-camera。
* 默认训练 sky generation；如需关闭，加 `--no_sky_generation`。
* sky target 是 `128×256` 上半球 RGB atlas，每个方向选置信度最高的可见帧；未观测区域经球面邻域补全，并以 `--sky_unobserved_loss_weight 0.005` 参与训练监督。通过固定 `8×8` pixel-unshuffle 打包为 `16×32×192`，SceneFlow 仍只处理 512 个 sky token，不增加主干序列长度，也不需要独立 sky tokenizer 或额外 checkpoint。该布局属于 `rgb_patch_teacher_anchor_v4`，checkpoint 会严格校验。
* validation 图像会保存到 `${LOG_DIR}/validation/step_xxxxxx/`（多数启动器默认
  `${PROJECT_ROOT}/logs/scene_flow_pretrain_v6`）；默认包含生成渲染、sky、mask、latent PCA
  和误差图。额外 CFG scale 会追加 `*_cfg{scale}` 后缀。validation 的基准条件固定为
  完整 TCMGA；相邻 event 只轮换后半部分的可视化场景，不轮换结构条件任务。固定
  slot 跨 event 保持相同初始噪声，同一 slot 的全部 CFG scale 也共享该噪声，便于直接比较。
* pretrain offline inference 在 checkpoint 加载后只接受与 checkpoint 内 `mu_z/sigma_z` 和四组 camera anchor/delta buffers **逐元素完全一致**的 stats 文件；不一致会报错，不会再用外部文件覆盖 checkpoint 坐标系。
* `--val_sample_steps` 只控制 validation 图像采样步数，不影响训练本身。当前值冻结为 `50`，入口会拒绝其它值。FlowMatch/RAE 的生成采样不是训练时的 1000 timestep 全跑，而是在 scheduler timestep 上做几十步推理。RAE 的 target 使用 `max(sigma,t_eps)`，因此最后一个非零采样点不能低于 `t_eps`；默认 `shift=10,t_eps=0.05` 时最多 191 步。
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

正式训练不直接读 raw Waymo，而是读 flow cache manifest。先按
`docs/flow_cache_cmd.md` 用 training/validation 各自的完整 29 帧 gauge 表生成 schema v10
cache，再构建 manifest：

```bash
python tools/build_flow_train_manifest.py \
    --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_a:mode_a \
    --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_b:mode_b \
    --split training \
    --out_path $SCENE_FLOW_TRAIN_MANIFEST
```

内部 holdout 仍可用：不传 `--val_manifest_path/--val_cache_root` 时，`train_scene_flow.py` 会把 training manifest 按 `--val_fraction 0.1` 做确定性切分。但内部 holdout 只能使用 `--caption_root`，不要传不同的 `--val_caption_root`。

manifest 不是信任源。正式训练在第一个 optimizer step 之前会遍历并预检每个
cache：training 的 mapping mode / gauge SHA 来自已验证的 metric-gauge pretrain
checkpoint，validation 的 gauge SHA 来自下面显式的 `--val_scene_gauge_sha256`，
DGGT SHA 始终由当前进程对 `$DGGT_CKPT` 实际文件现场计算。三项中任意一项
缺失或不相等都会在启动阶段报错。

如果要使用 validation captions，必须先构建独立 validation flow cache/manifest，再传 `--val_manifest_path ... --val_caption_root $SCENE_CAPTION_VAL_ROOT`：

```bash
python tools/build_flow_validation_manifest.py \
    --cache_root /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_validation \
    --split validation \
    --out_path $SCENE_FLOW_VAL_MANIFEST
```

## 3. 正式训练参数

> **v4 状态（2026-08-01）**：正式训练的 factorized asset condition 已接通 Mode A/Mode B、
> cache batch 和 sliding sampler；29 帧 offline bundle 的 scene/asset tokenizer encode/decode 也统一
> 分成不超过 10 帧的窗口。入口会严格校验 9D metric camera、3D gauge、16D placement、feature
> stats、gauge table 和 pullback provenance，旧 checkpoint 直接拒绝。

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_scene_flow.py \
    --ckpt_path $DGGT_CKPT \
    --tokenizer_ckpt_path $TOKENIZER_CKPT \
    --feature_stats_path $FEATURE_STATS \
    --pullback_calibration_path $PULLBACK_CALIBRATION_PATH \
    --latent_dim 1024 \
    --scene_flow_pretrain_path $SCENE_FLOW_PRETRAIN_CKPT \
    --scene_flow_pretrain_ema \
    --caption_root $SCENE_CAPTION_ROOT \
    --val_manifest_path $SCENE_FLOW_VAL_MANIFEST \
    --val_caption_root $SCENE_CAPTION_VAL_ROOT \
    --val_scene_gauge_sha256 $VAL_SCENE_GAUGE_SHA256 \
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
| cache 读取 | 默认读取 chunked zstd `.pt`；每个样本只解压 10 帧窗口需要的 chunk | 当前逻辑 `schema_version=10`。Mode-A 的 `[S,P]` asset patch mask 存在轻量 per-asset meta chunk；帧级 effective/raw/valid/fallback gauge 证据及有效通道均值也会保留。正式 loader 还会对 payload meta 和 chunked summary 同时校验 mapping mode、gauge SHA 和实际 DGGT SHA；任意旧 schema 必须重建。 |
| sigma / target | `--shift 10.0 --weighting_scheme waver --mode_scale 1.29 --loss_weighting_scheme none --prediction_type x` | 与 pretrain 保持一致 |
| REPA | `--lambda_repa 0.5` | 与 pretrain 保持一致 |
| EMA | `--ema_decay 0.9995`，validation 默认用 EMA | checkpoint 同时保存 raw / full / EMA-only 权重；三个正式训练导出都携带训练后的 `scaffold_packer`，EMA-only 是 EMA SceneFlow + 当前训练后 packer，与训练内 EMA validation 一致 |
| validation | `--val_manifest_path $SCENE_FLOW_VAL_MANIFEST --val_caption_root $SCENE_CAPTION_VAL_ROOT --val_every 1000 --val_batches 8 --val_log_images 10 --val_sample_steps 50 --guidance_scale 1.0 --asset_control_guidance_scale 1.0 --val_guidance_scales "1.0,2.0,4.0" --no_val_render_rgb` | 独立 validation manifest/cache 可使用 validation captions；内部 holdout 仍可用，但不能混用 validation caption root。默认 T1 validation dataset 仍是 10 帧窗口，不会仅因设置 sliding 参数自动扩展为 29 帧；完整 29 帧 T1 滑窗链路用 formal offline inference 验证。采样噪声使用 `seed + step`，可复现 |
| schedule | `--lr 2e-4 --final_lr 2e-5 --scheduler_type linear --decay_end_steps 150000 --weight_decay 0.0 --warmup_steps 3000 --max_steps 150000` | ⚠️ optimizer 与 `linear` 类型和 pretrain 一致，但**数值已经不一致**：当前 pretrain 启动器用 `--lr 1e-4 --final_lr 1e-5 --warmup_steps 4000 --max_steps 200000`。T1 是 fine-tune，LR 不高于 pretrain 更合理，建议改成 `--lr 1e-4 --final_lr 1e-5`（或更低）。默认 `warmup_from_zero=false`，前 N step 是初始 LR plateau，不是 from-zero warmup |
| RoPE 时间轴 | T1 与 pretrain 都传 `fps=10`（T1 统一常量 `FORMAL_SCENE_FPS`） | ✅ video/asset/camera 的时间位置不再在 warm-start 前后缩放；scene-global gauge 使用固定位置，不依赖 fps。 |

注意：`train_scene_flow.py` 的 `global_step` 现在和 pretrain 一样是 optimizer update 口径；`--max_steps/--save_every/--vis_every/--val_every` 都按 optimizer update 触发。

注意：上面的正式命令显式使用 `--no_val_render_rgb`，因此训练内 validation 只保存 loss、latent PCA / mask / CFG 采样诊断图并写入 wandb，不运行 3DGS RGB 渲染。若删除该参数，代码默认会执行 3DGS RGB validation。训练内 validation 的标量也不是完整训练目标：它不包含按训练 schedule 触发的 RGB/endpoint objective。

注意：正式 offline 入口是 `inference_scene_flow.py`，不是不存在的 `inference_scene_flow_validation.py`。29 帧输入会自动解析为 `window=10, stride=7`（重叠 3 帧）；它维护 full 29 帧 latent，在每个采样步对窗口 velocity 做 cosine coverage 归一化后统一更新。`--val_log_images` 默认是 10，只控制最终导出/拼图/渲染的帧数；需要导出完整 29 帧时必须显式传 `--val_log_images 29`。正式编辑阶段始终使用 cache 中 full-context input DGGT camera，并在 GT sky mask 内逐像素保留 input GT RGB；它不启动 pretrain 的 9D metric-camera、3D gauge 或 sky 生成流。即使启用 `--render_per_window`，也只切片同一条 29 帧 DGGT `pose_enc`，不会逐窗口重跑 CameraHead。正式 decoder/render 仍强制加载与 checkpoint SHA 绑定的 pullback artifact，并显式走 render-identity boundary。

## 2.1 Camera cache 修复与旧 checkpoint 迁移

cache 缺少相机 GT 或原图尺寸时，普通 `.pt` 与 SQLite chunk cache 使用同一命令修复：

```bash
python tools/backfill_flow_cache_camera_gt.py \
    --cache_root /path/to/cache_or_root \
    --processed_root /data/disk2/lyy_dataset/waymo_processed_dggt \
    --split training --force
```

旧 48K pretrain 的 `dggt_hidden_v1/2048D`、以及后续 11D DGGT/FOV camera checkpoint 都不能
resume 或 warm-start 到当前 `waymo_metric_relative_se3_rot6d_v4` 9D camera representation。
本次是 clean cut：必须使用完整 v4 feature stats 和 gauge/pullback provenance 从头训练；
`--resume_path`、formal warm-start 与 inference 都只接受 representation、dimension、stats version、
gauge table/tokenizer/DGGT/pullback SHA 完全一致的新 checkpoint。

## 4. 四类 validation / offline inference

训练内 pretrain validation 使用 `train_scene_flow_pretrain.py` 的
`--val_sliding_window 10 --val_sliding_stride 7 --val_sample_steps 50`，其采样可视化会使用完整
29 帧 clip。正式训练 validation 使用 `train_scene_flow.py`，但默认 validation dataset 仍输出
10 帧窗口；同名滑窗参数不会把样本自动扩展成 29 帧。完整 29 帧的 T1 滑窗链路应使用下面的
formal offline inference 验证。正式 validation 不 pack、不加噪，也不启动 pretrain 的
camera/sky/gauge generation state。

Pretrain offline 单窗：

```bash
python inference_scene_flow_pretrain.py \
  --weights $SCENE_FLOW_PRETRAIN_CKPT --dggt_ckpt_path $DGGT_CKPT \
  --tokenizer_ckpt_path $TOKENIZER_CKPT --feature_stats_path $FEATURE_STATS \
  --pullback_calibration_path $PULLBACK_CALIBRATION_PATH \
  --val_scene_gauge_path $VAL_SCENE_GAUGE_PATH --export_units metric \
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
  --tokenizer_ckpt_path $TOKENIZER_CKPT --feature_stats_path $FEATURE_STATS \
  --pullback_calibration_path $PULLBACK_CALIBRATION_PATH \
  --manifest_path $SCENE_FLOW_VAL_MANIFEST \
  --split validation --cache_scene_gauge_sha256 $VAL_SCENE_GAUGE_SHA256 \
  --output_dir runs/scene_flow_offline --window 10 --window_stride 7 \
  --val_log_images 29 \
  --edit_domain_threshold 1e-4 --edit_domain_dilation 1 \
  --sample_steps 50 \
  --guidance_scales 1,2,4
```

offline 入口不会把 cache 自报的 SHA 当作信任源。training cache 的 gauge SHA 固定取自
SceneFlow checkpoint；validation 或其他独立 split 必须显式传入其 production 表的可信
`--cache_scene_gauge_sha256`。入口现场 hash `$DGGT_CKPT`，再用所选 split 的可信 gauge SHA
对 manifest 内每个 schema v10 cache 做全量预检。省略 validation SHA、误传 training SHA，
或传入与 cache 不同的任意 SHA 都会在正式采样前报错。
`--cache_scene_gauge_sha256` 只选择当前 cache split 的 provenance；`$FEATURE_STATS`
仍与 SceneFlow checkpoint 的 **training** gauge 表绑定，不会被 validation SHA 覆盖。

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

仅对已携带完整 schema v10 geometry provenance 的 monolithic cache，可转换为当前
chunked 物理格式：

```bash
python tools/convert_flow_cache_to_chunked.py \
    --manifest_path /data/disk2/lyy_dataset/waymo_processed_dggt/waymo_edit_cache/manifests/training/training_manifest.jsonl \
    --mode_a_source_dir /data/disk3/lyy_dataset/waymo_processed_dggt/flow_cache_mode_a/training \
    --mode_a_output_dir /data/disk2/lyy_dataset/waymo_processed_dggt/flow_cache_mode_a/training \
    --workers 2 \
    --zstd_level 1 \
    --verify --verify_items 8
```

转换脚本默认原地覆盖 `.pt`；这批 Mode-A 迁移使用上面两个参数从 disk3 读取已符合 v10 的 monolithic cache，并写到 disk2。manifest 不需要改变；`--verify` 会在覆盖/落盘前比较原始文件和临时 chunked 文件。**schema v9 及更旧 cache 不能用该转换命令升级**，必须按 `docs/flow_cache_cmd.md` 重新 precompute。后续重新 precompute cache 时直接使用默认 `--save_compression chunked_zstd --gzip_level 1`。

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
