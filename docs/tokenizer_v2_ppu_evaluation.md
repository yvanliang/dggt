# Tokenizer v2：单机 16 PPU checkpoint 评测

## 启动

在 PAI-DLC 建立单节点、每节点 16 张真武 810E PPU 的任务，启动命令：

```bash
bash /mnt/workspace/dggt/evaluate_tokenizer_v2_ppu_dlc.sh
```

脚本固定使用 `conda dggt`。第一轮为：

```text
55000,60000,65000,70000,75000 × 10,12,14 帧 = 15 configs
```

15 个配置分别占用 rank 0–14，rank 15 负责共同的输入协议校验和最终汇总。默认每个配置评测 300 个互不重复 scene。

第二轮复用完全相同的 scene、clip、起始帧和全部帧号，只改 checkpoint steps：

```bash
STEPS=80000,85000,90000,95000,100000 \
OUTPUT_DIR=/mnt/workspace/dggt/runs/tokenizer_v2_ppu_eval_round2 \
bash /mnt/workspace/dggt/evaluate_tokenizer_v2_ppu_dlc.sh
```

不要修改或删除第一轮生成的：

```text
/mnt/workspace/dggt/runs/tokenizer_v2_fixed_selection_300.json
```

它是跨轮次固定样本集的正式 manifest。每条 case 保存 scene、clip、dataset index，以及 10/12/14 帧各自的 sampling seed、起始 local/global frame 和完整帧号。后续运行会逐项复算和断言；dataset 排序、场景或帧号有任何漂移都会失败，而不是静默换样本。

## 指标

生成质量以原始 RGB 视频为 GT：

- PSNR、11×11 window SSIM、AlexNet LPIPS；
- 相邻采样帧的 temporal-delta L1；
- 同时报告 direct DGGT render 对 GT 的 ceiling，以及 tokenizer render 对 direct render 的退化。

3D 空间指标以 `scene_flow_metric_gauge_retest_2026-07-31.md` 的 v1 问题为主：

- `depth_recon_over_direct`，理想值 1；v1 参考值约 1.0307；
- `gs_recon_over_direct`；
- paired same-pixel `paired_gs_over_depth`，理想值 1；v1 参考值约 0.7964；
- depth AbsRel/log-RMSE、camera/world rotation-invariant XYZ displacement；
- Gaussian 三轴 anisotropy drift；
- 在原始 `depth_flows_4` 非零 LiDAR cell 中直接采样 dense depth，报告 metric AbsRel、RMSE(m)、δ1。稀疏零值 depth 图不会被 resize。

所有 pixel/frame 先在 case 内做 frame-balanced robust 聚合，再以 300 个独立 scene 做 bootstrap 95% CI。重叠帧或同一 scene 不会膨胀统计样本数。

checkpoint 主排名按三个帧长的平均 3D recovery score：

```text
|log(depth ratio)| + |log(paired GS/depth)| + depth AbsRel + GS anisotropy
```

越低越好；RGB PSNR 只作为并列时的 tie-breaker。

## 输出

每轮输出目录包含：

- `summary.json`：完整 provenance、逐配置 scene-bootstrap 汇总和 checkpoint 排名；
- `per_case.jsonl`：每个 scene 的原始指标和帧号；
- `checkpoint_ranking.csv`；
- `REPORT.md`；
- `visuals/`：每个配置的少量 Stage-A 风格对照图；
- `launch.log`。

本机无 PPU/权重时，可运行不依赖模型、数据、CUDA、LPIPS 或 gsplat 的分布式 mock：

```bash
conda run --no-capture-output -n dggt \
  python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/evaluate_tokenizer_v2_ppu.py \
  --mock --mock-cases 3 --expected-world-size 2 \
  --bootstrap-samples 64 --output-dir /tmp/tokenizer_v2_mock
```

mock 结果带有 `scientific_result=false`，只能验证调度、指标、LiDAR cell sampling、bootstrap 和结果写出，不能用于选择 checkpoint。
