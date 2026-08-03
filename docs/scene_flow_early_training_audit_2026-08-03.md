# SceneFlow gauge 版早期训练修复与归因手册（2026-08-03）

## 本次修复

### 3：米制 camera pose 辅助损失在训练初期过强

旧实现把 DGGT 单位中的平移损失直接迁移到米制目标，但沿用了原数值权重。训练集
`log_metric_scale` 约为 3–4，即一 DGGT 单位对应约 20–65 米，因此同一个物理误差在
clean cut 后会被无意放大。

现在的正式训练路径：

- absolute / relative / acceleration 三类平移残差分别除以 10 m / 1 m / 1 m，再进入
  Smooth-L1；旋转仍以弧度计算；
- 在缺少 step-2000/4000 多样本梯度归因的情况下采用保守参数：
  `lambda_camera_pose=0.25`，并在前 10000 optimizer steps 从 0 线性升到 0.25；
- camera flow loss 从第一步保持开启，9D 米制相机目标、模型结构和 checkpoint clean cut
  均不改变；
- 日志新增 `camera_pose_ramp`、`camera_pose_effective_weight`、
  `loss_camera_pose_weighted`，避免只看未加权 loss。

### 6：validation 随机污染和归因盲区

- eval 模式下 `camera_anchor_context_dropout` 固定关闭，而且不消耗 validation RNG；
- validation loss 调用显式接收当前 `global_step`，所以 camera pose warmup 与训练同一步一致；
- 最终 ODE 采样 gauge 现在单独记录 `sample_gauge_*` 指标，而不是只看训练前向里的
  x-pred gauge；
- inference 新增固定生成几何的 camera × gauge 四臂归因；
- BF16 下的相机积分、SE(3) 求逆和 gauge 换算统一用 FP32，避免 `torch.linalg.inv`
  的 BF16 运行时错误。

## 四臂怎么读

四臂共享同一份生成 video latent、sky、sky mask 和噪声，只替换 camera 与 gauge：

| arm | camera | gauge | 用途 |
|---|---|---|---|
| TT | teacher | teacher | 生成几何本身在正确相机/gauge 下的基线 |
| GT | generated | teacher | 相对 TT 的变化主要归因于生成相机 |
| TG | teacher | generated | 相对 TT 的变化主要归因于生成 gauge |
| GG | generated | generated | 实际端到端组合及非加性交互 |

优先看 L1；正的 effect 表示退化。若 TT 已明显差，而 GT/TG 相对 TT 变化很小，问题主要
不在 camera/gauge，而在生成 latent/几何或共享表示。PSNR 保留作辅助读数，不用于选择
Gaussian 尺寸类常数。

step-2500 建议一次跑 4 个 validation rows（CUDA 0）：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n dggt --no-capture-output \
python -u inference_scene_flow_pretrain.py \
  --weights /path/to/pretrain_step002500.pt \
  --cfg 1.0 --sample_steps 35 --num_frames 10 \
  --start 0 --end 4 \
  --camera_gauge_attribution \
  --output_dir runs/audits/four_arm_step2500
```

每个样本的 `camera_gauge_attribution__cfg1.json` 是主结果，四张 arm 图用于人工确认。

## step-2500 共享骨干梯度审计

必须使用完整 raw checkpoint（不是 EMA-only 或 weights-only），因为脚本需要当时的 raw
训练权重和保存的启动参数：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n dggt --no-capture-output \
python tools/audit_scene_flow_gradient_balance.py \
  --checkpoint /path/to/pretrain_step002500.pt \
  --dataset_index 0 \
  --output runs/audits/step2500_gradient_balance_idx0.json
```

建议把 `dataset_index` 换成 0、1、2、3 各跑一次。主要读：

- `gradient_norm_ratios_to_video.camera_aux`：camera flow + warmup 后 pose 相对视频主损失；
- `gradient_norm_ratios_to_video.gauge_aux`：gauge flow + direct 相对视频主损失；
- `gradient_cosines.*_vs_video_core`：负数表示方向冲突。

经验报警线是“多个样本上 ratio > 0.5”或“多个样本上 cosine < -0.1”；单样本不作为改模型
的充分依据。如果 camera 持续超线，下一步应先降低 camera flow 权重或延长整个 camera aux
的 warmup；如果 gauge 持续超线，则给 gauge aux 加同类 warmup；如果二者都不超线但 TT
仍差，应转查主视频流、task mixture 和共享表示，而不是继续改 gauge 定义。

## 本机验证

- 相关单元测试、validation RNG、sliding-window 与 checkpoint clean-cut 测试通过；
- `tools/audit_scene_flow_gradient_balance.py` 已用本机 step-1 完整权重完成真实 Waymo batch
  冒烟并写出 JSON；
- 四臂已用 step-1、2-step ODE 完成 DGGT/3DGS CUDA 0 端到端冒烟，四张图和 JSON 均落盘。

step-1 梯度比值不能用于判断模型设计：视频输出 head 近零初始化，此时共享骨干上的视频
梯度极小，会把任何辅助梯度比值人为放大。当前拿不到 step-2000/4000 权重完成可靠归因，
所以正式启动脚本统一采用上述保守参数。相较原配置，在 step 4000 时 camera pose 的有效权重
由 `1.0 * 4000/5000 = 0.8` 降为 `0.25 * 4000/10000 = 0.1`，降低 8 倍；归一化空间的
`lambda_camera_flow=0.1` 保持不变，避免相机生成分支在 warmup 期间失去直接监督。后续若取得
多样本梯度结果，再按 ratio/cosine 判据决定是否恢复更高峰值，而不是凭单次 validation 调参。

## PPU tokenizer/RoPE illegal-address 诊断

PPU 的一次梯度审计在 tokenizer 2D RoPE 的 `int(positions.max())` 报
`hggcErrorIllegalAddress`。对比确认：从可正常 PPU 训练的 `ebd3cb6a` 到本次修改前的
HEAD，`rope.py`、`attention.py`、`block.py`、`joint_scene_tokenizer.py`、
`tokenizer_window.py` 以及 DGGT/tokenizer 加载函数均没有变化。因此该堆栈不能证明
RoPE 是本次回归；CUDA/PPU illegal-address 也可能把更早的异步 kernel 错误报告在首次同步
的 `positions.max()` 上。

正式 RoPE 代码暂不依据单次错误改动。先在全新进程用 `CUDA_LAUNCH_BLOCKING=1` 重跑；
若仍失败，再运行隔离脚本。每个 case 使用新进程，某个 illegal address 不会污染后续 case：

```bash
python tools/diagnose_ppu_tokenizer_rope.py \
  --case all \
  --output runs/audits/ppu_tokenizer_rope.json
```

结果解释：`current_rope2d` 或 `large_grid_tiny_tokenizer` 稳定失败、对应 safe case 通过，
才足以采用 CPU position/static-capacity 的候选修复；`safe_rope2d` 通过但
`formal_shape_attention` 失败说明问题在 PPU attention；所有 probe 都通过但完整审计仍失败，
则应以 blocking 模式的完整审计堆栈定位真实上游算子。单次重跑通过更符合瞬时 runtime/
设备上下文问题，但仍建议至少重复两次。

PPU 实测结果：上述隔离 cases 和重新测试均成功，未复现 illegal address。因此本次错误归类为
一次性的 PPU runtime/设备上下文异常，不是可稳定复现的 RoPE、attention 或 tokenizer 算子
不支持。正式 RoPE 实现保持 `ebd3cb6a` 的已验证版本，不采用候选兼容改写。
