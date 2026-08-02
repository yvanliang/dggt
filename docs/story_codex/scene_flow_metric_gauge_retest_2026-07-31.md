# Scene Flow Pretrain：Waymo 米制空间与 DGGT gauge 独立复测

> 日期：2026-07-31
> 环境：`conda dggt`，`cuda:0`
> 样本：Waymo training scene 300–329，每个 scene 的 trunk 0/1/2；共 90 个 29-frame trunk
> 尺度定义：`s = DGGT unit / metre`，因此 `metric = DGGT / s`

## 结论先行

先前“Waymo 米制空间与 DGGT 空间无法换算”的判断需要撤回。独立复测强力支持：在约 2.5% 的跨 trunk ratio 离散度下，**相机中心平移与直接 DepthHead 的 z-depth 近似共享一个逐 trunk 标量 gauge**。以下先给完整 29 帧轨迹跨度至少 5 m 的较稳健子集；它恰有 66 个 trunk，但与 A.8 legacy 10 帧 gate 的 66 个只是数量相同，筛选口径和成员并不相同：

| 指标 | 独立复测 |
|---|---:|
| 有效 trunk / scene | 66 / 24 |
| `s_cam` 范围 | 0.01635–0.03945 DGGT unit/m |
| 1 DGGT unit | 25.35–61.16 m |
| `s_cam / s_depth` | 0.9941 ± 0.0253 |
| ratio median / p95 | 0.9899 / 1.0406 |
| Pearson / Spearman | 0.9913 / 0.9924 |

因此，在 camera scale 可辨识的移动 trunk 中，将 DGGT 相机平移或直接深度除以相应的 `s`，可以近似换算回米，并应同时报告尺度估计的不确定性。A.8 的核心新结论——“存在逐 29-frame teacher trunk 的尺度，且相机与深度近似共尺”——得到强复现；20 个 span <2 m 的 trunk 不能由相机尺验证这一点。

但“整个 DGGT 三维世界只是米制世界的纯相似缩放副本”“渲染一定不会失配”仍然过强。独立测试发现三个必须分开的层次：

1. **相机中心 + direct z-depth 的一维 gauge：在约 2.5% ratio 离散度下近似成立。**
2. **完整射线/欧氏三维：不是仅差一个标量。** DGGT 与 Waymo 的 FOV 明显不同，移动样本的稀疏 LiDAR cell 平均射线夹角约 3.2°；只修尺度仍留下米级横向/XYZ 误差。
3. **SceneFlow 实际 latent decode：不保 uniform similarity gauge。** 10 帧 JointSceneTokenizer round-trip 后，深度相对 direct head 整体约放大 3%，而 Gaussian 三轴几何平均尺度约缩到 0.83；同像素配对的 `GS/depth` 只有约 0.80，理想值应为 1。

对 Scene Flow pretrain 最稳妥的表述应是：**系统存在可标定的逐 trunk metric gauge，但 camera、depth、intrinsics、Gaussian covariance 和 tokenizer round-trip 必须分别审计；不能用两把尺的一致性代替完整 3D 自洽性。**

## 1. 为什么重写测试

本次没有运行或调用 `lyy_tools/verify_*.py`、dataset loader、已有相机转换函数、已有 geometry helper 或已有测试断言。新工具只复用实际待测的模型组件与 checkpoint，以下部分均独立实现：

- 直接从磁盘读取 RGB、pose、extrinsics、intrinsics 与稀疏 LiDAR depth；LiDAR validity 由有限且非零的 depth cell 独立构造，Gaussian sensitivity 另读原始 sky/dynamic masks；
- 独立处理 quaternion `xyzw`、`w2c → c2w`、anchor 归一化与 Waymo/OpenCV 轴约定；
- 独立复刻原图到 518×350 canvas 的 resize 几何；
- LiDAR 主估计直接在原 320×480 非零 cell 对应的 model-grid 位置采样 dense prediction，**不缩放含大量零值的稀疏 depth 图**；
- trunk-local 帧号显式转换为 `global = trunk * 29 + local`；
- 相机尺度只由相机中心的成对距离估计，rotation/FOV 不进入尺度拟合；
- direct 与 tokenizer-reconstruction 使用相同 LiDAR cell 做配对比值；
- 五个重叠 10 帧窗口仅作为 repeated measures，统计推断以 trunk、scene 平衡为主。

新工具：

- `tools/retest_scene_flow_metric_gauge.py`：相机、direct depth、FOV、近似 3D、tokenizer depth round-trip。
- `tools/retest_scene_flow_gaussian_gauge.py`：实际 tokenizer round-trip 前后，depth 与 Gaussian scale 的同像素配对 gauge。

## 2. 可复现绑定

| 项目 | 值 |
|---|---|
| git commit | `920b3049f3f589cc8562dcf398cccb6f8648bc23` |
| 主测试脚本 SHA-256 | `9e91dd09c7057d5cf2a04a6027e2bf8088aee6ce400c1121a71ff1c4ae15a3e1` |
| Gaussian 测试脚本 SHA-256 | `6c192559c8c6a68071149a37e7def90818f938c7ca8d74a98264012ac37d502f` |
| 主结果 JSON SHA-256 | `424451a754fbb728f0abb822aa3387d65f8f8c6ffa80f3119a0504c0bb29fd38` |
| Gaussian 结果 JSON SHA-256 | `2a07a367da7c9af918b3172c041bf1e2c6ec8b6c48e5866e677de0bd39bcfc40` |
| DGGT checkpoint | `/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt` |
| DGGT checkpoint SHA-256 | `352652738a5480b8d3ee9dd521ce07b528e5a297bd3feca4d07427dac6d87def` |
| tokenizer checkpoint | `/data/disk2/lyy_dataset/logs/tokenizer_t0_stageB/ckpt/scene_tokenizer_step_040000.pt` |
| tokenizer SHA-256 | `75e566efa3b66baa43f82cb9999c2de60a9f3feeb0f714e1caf38d1f6e8137eb` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition，CUDA 12.8 |
| PyTorch | 2.7.1+cu128 |

完整运行命令：

```bash
conda run --no-capture-output -n dggt python -u tools/retest_scene_flow_metric_gauge.py \
  --scenes 300-329 --trunks 0,1,2 --device cuda:0 --precision bf16 \
  --depth-chunk 4 --raw-lidar-audit \
  --tokenizer-checkpoint /data/disk2/lyy_dataset/logs/tokenizer_t0_stageB/ckpt/scene_tokenizer_step_040000.pt \
  --output-json runs/metric_gauge_retest/a8_300_329_trunks012_production_path.json

conda run --no-capture-output -n dggt python -u tools/retest_scene_flow_gaussian_gauge.py \
  --scenes 300-329 --trunks 0,1,2 --device cuda:0 --precision bf16 \
  --depth-chunk 4 \
  --result-json runs/metric_gauge_retest/a8_300_329_trunks012_production_path.json \
  --output runs/metric_gauge_retest/gaussian_gauge_300_329_trunks012.json
```

仓库中的 `pretrained/model_latest_waymo.pt` 当前是一个失效 symlink，目标文件名多了下划线；本次没有静默沿用它，而是绑定到上表真实存在的 checkpoint。

数值路径按生产训练行为拆开：aggregator/tokenizer 在 bf16 autocast 域运行（部分输出仍按模块实现回到 fp32）；CameraHead 主结果使用 SceneFlow pretrain 外层 bf16 autocast，另跑 fp32 sensitivity；DepthHead 与 tokenizer 后的 DepthHead/GaussianHead 都关闭 autocast、使用 fp32。CameraHead 的 `fp32 / bf16` 尺度比为 `0.99941 ± 0.00095`，精度模式不是尺度结论的来源。

### 2.1 先验失败注入

在真实数据前，工具必须通过以下合成断言：已知尺度 0.03 的恢复误差、整体刚体旋转不改变尺度、`w2c` inversion、稀疏零值污染、仿射 grid sampling、frame-balanced 与 pooled-pixel 统计差异、相机/世界坐标下精确 Sim(3)，以及 FOV 50° 对 40° 的边缘射线误差。所有断言通过。

此外，90 个 trunk 的首帧均用原始 LiDAR 点重新投影并与保存 depth 比较：mask IoU mean `0.999933`、median `1.0`、min `0.999577`；重叠像素 mean absolute error 的跨 trunk 均值仅 `6.73e-7 m`。88/90 个 case 的 valid-pixel 数完全相同；另两个各差一个 cell，重叠 cells 中的最大单 cell 误差分别为 3.46 cm 与 8.08 cm。因此它是高度一致的帧号/depth 通道/单位/投影检查，但不是 bit-exact。该 audit 只覆盖每个 trunk 首帧，即 90/2610 帧（3.45%），不能外推成全帧 raw-return 审计。

## 3. 相机与 direct depth：核心结论复现

相机尺度采用中心点成对距离的过原点最小二乘斜率，主估计只用 Waymo baseline ≥2 m 的 pairs；完整轨迹跨度 <2 m 时将 camera scale 明确记为无效，而不是让退化 Umeyama 返回一个看似正常的数。

| 轨迹 span 门槛 | trunk / scene | `s_cam/s_depth` mean ± std | median | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| ≥2 m | 70 / 26 | 0.9955 ± 0.0261 | 0.9910 | 0.9933 | 0.9936 |
| ≥5 m | 66 / 24 | 0.9941 ± 0.0253 | 0.9899 | 0.9913 | 0.9924 |
| ≥10 m | 60 / 22 | 0.9924 ± 0.0244 | 0.9887 | 0.9911 | 0.9931 |

对 ≥2 m 样本做 10,000 次 scene-cluster bootstrap：每个 scene 先取 trunk ratio 中位数，再 bootstrap 这些 scene scalars；总体 median 为 `0.99287`，95% CI `[0.98522, 0.99733]`。这说明 scene 平衡后的 ratio 中心仍接近 1，且不由少数 scene 支配；它本身不是 correlation CI。另以 scene 内 trunk mean 聚合 camera/depth scale，scene-level Pearson 为 `0.99554`，cluster-bootstrap 95% CI `[0.99036, 0.99824]`。

完整 29 帧 span <2 m 的 20/90 个 trunk 中，camera 尺按定义不可辨识；depth 尺仍然稳定。A.8 的 legacy `24 static / 66 moving` 使用的是 10 帧运动口径，不能与 29 帧的 `20 / 70` 混为同一个分母。

### 3.1 尺度确实跨 trunk 漂移

相邻且双方都能可靠估计 camera scale 的 44 对 trunk，其相对差 mean `8.21%`、median `5.49%`、p95 `21.89%`、max `30.82%`。例如：

| scene | trunk 0 / 1 / 2 `s_cam` | trunk 0 / 1 / 2 `s_depth` |
|---|---|---|
| 301 | 0.02740 / 0.03428 / 0.03945 | 0.02746 / 0.03475 / 0.04004 |
| 302 | 0.02381 / 0.02890 / 0.03362 | 0.02481 / 0.02861 / 0.03444 |
| 325 | 0.04223 / 0.03536 / 0.02592 | 0.04287 / 0.03645 / 0.02612 |

相机和 LiDAR-depth 的漂移同向，故这不是单一相机拟合器噪声。A.8 所说“尺度是 teacher trunk 的属性，不是场景常数”得到定性确认；独立估计的精确 drift 数字与 A.8 不同，不应混用。

更直接地看有符号漂移 `Δlog s = log(s_next/s_prev)`：camera 与 depth 在 41/44 对上方向一致，Pearson `0.9711`、Spearman `0.9593`，scene-cluster Pearson 95% CI `[0.9276, 0.9901]`。这排除了“二者只是拥有相近的无符号离散度”这一解释。

### 3.2 10 帧内估计只是带噪观测

从完整 29 帧输出抽取起点 0/5/10/14/19 的五个 10 帧窗口。若先选 full-trunk motion ≥5 m，再保留每个 case 的全部五窗，66 个 trunk 的 within-trunk scale CV mean/median/p95/max 为 `3.19% / 2.18% / 8.97% / 9.94%`。作为 sensitivity，若让每个 window 独立通过自己的 motion gate：window motion ≥0.5 m 时为 345 windows / 71 trunks / 26 scenes，CV mean/median/max `3.12% / 2.11% / 9.93%`；window motion ≥5 m 时为 260 / 55 / 21，CV `2.65% / 1.96% / 9.88%`。后一口径会选择性丢掉低运动子窗，不能代替训练时“同一 full trunk 的五窗共享一个 GT”的主口径。两种口径均说明 `log s` target 应由完整 trunk 的 LiDAR 尺产生，不能把 10 帧拟合抖动当成不同 GT。

## 4. 不能复现为“完整三维只差标量”的部分

### 4.1 旋转与轨迹形状

90 个 trunk 的 raw 坐标约定下，每个 trunk 的逐帧旋转误差均值之分布为 mean `0.371°`、median `0.184°`、p95 `1.254°`、最大 trunk-mean `3.228°`；所有帧的最大值为 `6.484°`。若先为每个 trunk 拟合一个常量旋转 `Q`，相应数字下降为 mean `0.201°`、median `0.124°`、p95 `0.630°`、最大 trunk-mean `1.793°`。

所以 A.8 的 `mean 0.168° / max 1.157°` 不能作为未注明配准方式的普遍数字。至少 scene 312、314、325 存在不可忽略的 outlier；报告时必须明确是 raw 坐标误差还是先拟合了常量 `Q` 后的残差。

在 span ≥2 m 的 70 个 trunk 中，先做 orientation-only constant-Q 后，shape RMSE/path 的 case mean/median/p95/max 为 `1.334% / 1.065% / 3.252% / 4.066%`；scene-balanced mean 为 `1.319%`，95% CI `[1.024%, 1.659%]`。A.8 的 `0.52% / max 2.38%` 未被这一独立估计器复现。它不推翻“形状高度相关”，但不支持把轨迹当成数值上几乎精确的一一映射。

### 4.2 FOV 不是小噪声

| FOV | Waymo | DGGT |
|---|---:|---:|
| x | 49.848° ± 0.262° | 38.325° ± 9.386°，范围 16.226°–66.248° |
| y | 34.426° ± 0.195° | 26.888° ± 6.626° |
| DGGT − Waymo | x: −11.523° ± 9.554° | y: −7.538° ± 6.751° |

这些值基本复现 A.8 的 FOV 统计。由于保存预处理只是 resize、没有 crop，原始 K 按 x/y resize factor 映到 model canvas 后，raw-FOV 与 model-FOV 在浮点误差内完全相同；这里不能把 resize 当成 FOV 大偏差的解释。

在 ≥2 m 的移动样本中，稀疏 LiDAR cell 的相机射线夹角按 scene 平衡后为 `3.286°`，95% CI `[2.483°, 4.197°]`。因此，即使 z-depth 与 camera translation 共用标量，像素对应的横向坐标也会因 K/FOV 不同而变化。

### 4.3 只除以 `s_cam` 后仍有米级 3D 残差

在保存预处理采用的 pinhole 约定下，以稀疏 depth cell 的中心作为像素位置，将 DGGT depth 除以独立 `s_cam` 后反投影。每个 trunk 先对有效 cells 求内部 mean，再在 scene 内聚合 trunk，并以 scene 为 bootstrap cluster：

| 指标 | scene-balanced mean | 95% CI |
|---|---:|---:|
| z absolute error | 1.738 m | [1.487, 2.020] m |
| lateral error | 1.354 m | [1.068, 1.692] m |
| camera XYZ error | 2.452 m | [2.062, 2.896] m |
| world XYZ error | 2.550 m | [2.167, 2.977] m |
| 仅由 FOV 产生的 XYZ error | 1.287 m | [1.003, 1.618] m |

这组数字是“保存 depth cell center + pinhole”下的近似误差，不是原始每个 LiDAR return 的逐点误差；数据中的 lens-distortion 参数并未被现有 RGB/depth 预处理使用，因此本测试也不声称验证了真实镜头畸变。它足以说明：**标量能近似标定 z/translation 的单位与主导 gauge，但不能单独恢复完整欧氏 3D。**

## 5. SceneFlow 实际 tokenizer 路径

主结论不能只看 `aggregator → direct DepthHead`。训练使用的是完整 29 帧 aggregator 特征，先 gather 一个 10 帧窗口，再经 JointSceneTokenizer encode/decode，最后送入冻结 heads。本次按这条实际路径评测 90 个 trunk × 5 个重叠窗口，共 450 次 round-trip。

同一批 LiDAR cells 上的 `depth_recon / depth_direct`，先在像素配对后聚合，再对每个 trunk 取五窗中位数：mean `1.0305`、median `1.0328`、std `2.35%`、p95 `1.0598`、max `1.1205`。再以 scene 内三个 trunk 的中位数聚合，scene-balanced mean/median 为 `1.0307 / 1.0344`，scene-median bootstrap 95% CI `[1.0208, 1.0421]`。即 tokenizer 稳定地把 depth gauge 放大约 3%。

对 window motion ≥5 m 的 260 个重叠窗口做 descriptive pooled 汇总，`s_cam/s_direct_depth` mean 约 `0.989`，而 `s_cam/s_recon_depth` mean 降至约 `0.948`；这些窗口不是独立样本。即便如此，case/scene-balanced 的 same-cell round-trip 结果也给出同向系统偏差。因此 A.8 的 direct-head camera/depth 一致性不能直接代表 SceneFlow latent decode 后的输出；若 `log s` 监督的是最终可执行状态，应针对 tokenizer 后的 head 输出定义或至少同时报告这一系统偏差。2026-08-01 的后续 LiDAR gate 已直接测试该偏差是否应在米制边界校正，见 §8。

### 5.1 Gaussian linear scale：uniform similarity 必要条件明确失败

GaussianHead 的 `gs_map[...,4:7]` 是 renderer 直接使用的三个线性 scale。若 tokenizer round-trip 只引入一个共同相似尺度 `a`，同一像素应满足：

```text
depth_recon / depth_direct = a
scale_recon[x,y,z] / scale_direct[x,y,z] = a
```

因此本测试先在每个像素计算

```text
e_p = mean_axis(log(clamp(scale_recon, 1e-5) / clamp(scale_direct, 1e-5)))
      - log(depth_recon / depth_direct)
```

再依次取 frame median、window median、trunk median、scene 内 trunk median，最后对 30 个 scene scalars 报 median/IQR。`exp(e)=1` 是共同相似尺度的必要条件。它使用 paired same-pixel 量，不是“GS 两组中位数 / depth 两组中位数”的非配对比值。

Primary support 要求 direct/recon depth 有限且为正、两边 opacity >0.05，并排除原数据的 sky 与 canonical fine-dynamic pixels。90 个 trunk、450 个窗口、最终 30 个 scene-balanced scalars 的结果为：

| scene-balanced 指标 | 结果 |
|---|---:|
| `depth_recon / depth_direct`（separate median-log） | 1.0399 |
| Gaussian geometric-axis scale ratio（separate median-log） | 0.8289 |
| **paired `GS/depth`，主判据** | **0.7964** |
| paired scene IQR | 0.7876–0.8120 |
| scene 一致性 | 30/30 均 <1，范围 0.6906–0.8444 |
| paired RMS-radius/depth sensitivity | 0.7996 |
| 三个 raw scale channels 的非均匀性 log-RMS median | 0.1368 |
| depth pixel-ratio log-MAD median | 0.0122 |

更换 mask 后失败方向不变；将 floor pixels 排除后也没有回到 1：

| sensitivity support | paired `GS/depth` |
|---|---:|
| all canvas | 0.8043 |
| LiDAR depth-cell centers | 0.8060 |
| 仅 opacity intersection >0.05 | 0.7987 |
| sky + legacy rough dynamic mask | 0.7960 |
| primary 且六个 direct/recon scale 均严格高于 floor | 0.7024（RMS-radius 0.7933） |

strict support 是按 direct/recon 六个 scale outcomes 选择的子集，典型只保留约 29.5% primary pixels，因此其 geometric-mean 幅度不能与 primary 作无条件横比；它能说明去掉 floor 后失败方向仍存在。更不敏感的 RMS-radius 判据从 `0.7996` 变为 `0.7933`，基本稳定。raw 三通道 anisotropy 也未审计 quaternion 对应或潜在轴置换，不能单独解释为完整 covariance 主轴逐一失配；主 geometric-axis/体积半径判据对轴置换不敏感，已经足以否定共同 uniform scale。

Gaussian 工具自己重算的 LiDAR-cell depth round-trip 与主工具 reference 的 450 个窗口最大 absolute log difference 仅 `1.21e-6`，排除了两份工具 depth 对齐方式不一致这一解释。

所以，实际 SceneFlow latent 路径直接否定了“仅凭 camera/depth 共尺即可证明内部完整 Sim(3) 自洽、进而保证渲染不失配”这一推理：**tokenizer 后 depth 变大，renderer 使用的 Gaussian linear radii/scale channels 却显著变小。** covariance 会按 linear scale 的平方进入几何，但本测试没有直接测最终 RGB 是否显著变差；它是 direct→reconstruction 的必要条件审计，既不证明 direct Gaussian 本身是否以 Waymo 米制标定，也不替代完整 renderer 的 opacity/quaternion/compositing 测试。

## 6. 对 A.8 与 Scene Flow pretrain 设计的修订建议

> **历史状态说明（2026-08-01）**：下面“可以保留 / 必须收窄 / 建议闭环”是 D3/D4 与 v1
> LiDAR metric-boundary gate 之前的审计建议，不再是执行清单。当前冻结实现以
> `docs/metric_scale_camera_redesign_plan.md`、A.8.9 及本文件后续第 8 节为准；尤其 camera 已改为
> 9D 米制、`log s` 只生成不条件、render boundary 为 identity、metric depth 使用 v1 loglinear。

### 可以保留（历史审计，已由新 plan 接管）

- Waymo 米制与 DGGT 不是“无法换算”；应显式使用逐 trunk 的 `s`。
- 离线 gauge GT 用完整 29 帧 LiDAR depth 最可靠；相机尺只在有足够运动时交叉验证。
- `log s` 不应做成逐帧值，且不能用 10 帧 camera fit 作为独立 target。
- teacher 的 gauge 跨 trunk 漂移，长序列只有一个生成尺度而没有唯一的逐-trunk teacher GT；这是实际 limitation。
- FOV target 若服务于现有 DGGT renderer，应沿用 DGGT 自己的 K；不能把 Waymo K 直接替换进去。

### 必须收窄

- 将“DGGT 是米制世界的内部自洽相似缩放副本”改成“在 camera scale 可辨识的移动样本中，camera centers 与 direct z-depth 在约 2.5% ratio 离散度下近似共享逐 29-frame teacher trunk 的 scalar gauge”。
- 将“相机/深度共尺，所以渲染不会失配”撤回。Gaussian covariance、tokenizer round-trip 和 K/FOV 都是独立自由度，必须分别测试。
- 将 A.8 的 `±4.4%` 从“最好情况下界”改为“一种 direct camera/depth consistency dispersion”。本次同口径约为 2.5%，而 tokenizer 又引入约 3% 系统偏移；它既不是信息论下界，也不是最终 `log s` 可达误差。
- “camera condition 无法决定尺度”应写成条件歧义：仅由米制轨迹的形状无法解析 teacher 自选 gauge，但图像/场景内容可能提供统计线索，不能称为数学上完全不可学习。
- actor box 是潜在的米制 ruler，但在新独立测试未验证之前，不能把它宣称为最佳 ruler；A.8 自身也显示 actor 覆盖有限且与 DGGT depth 并不统计独立。

### 建议的训练/评测闭环

1. `log s` GT：完整 29 帧、frame-balanced LiDAR depth ratio；报告有效像素与逐帧 MAD。
2. 训练时明确 target 对象：若最终生成的是 tokenizer latent，监督应对 tokenizer 后 depth/GS 的 metric gauge，而非只对 direct teacher head。
3. 把 gauge loss 拆成至少三项：camera translation、z-depth、Gaussian covariance；K/FOV 单独监督或固定为 DGGT gauge K。
4. 增加同像素约束：在 gauge 改变 `a` 时，depth、Gaussian 三轴 scale 与 camera translation 都应乘 `a`，rotation/color/opacity 不变。
5. 指标同时报告 direct 与 round-trip，并以 scene 为统计 cluster；重叠滑窗不能当独立样本。

## 7. 结果文件

- 主结果：`runs/metric_gauge_retest/a8_300_329_trunks012_production_path.json`，SHA-256 `424451a754fbb728f0abb822aa3387d65f8f8c6ffa80f3119a0504c0bb29fd38`
- Gaussian 结果：`runs/metric_gauge_retest/gaussian_gauge_300_329_trunks012.json`，`status=complete`、90 cases / 450 windows，SHA-256 `2a07a367da7c9af918b3172c041bf1e2c6ec8b6c48e5866e677de0bd39bcfc40`
- 单 case Gaussian smoke：`runs/metric_gauge_retest/gaussian_gauge_smoke_301_t0.json`

JSON 保留每个 trunk/frame/window 的原始中间统计、checkpoint/script hash、source stat manifest 和模型 load key counts，便于后续对单一离群点追溯。

本次边界也需要写清：没有复跑 A.8.5 的 leave-one-frame-out renderer PSNR，因此本文的 FOV/ray/近似 3D 描述统计不能冒充其 Branch-A 渲染判据；也没有重新实现 Actor ray-box ruler，故不对 A.8 的 Actor/LiDAR 覆盖与一致性数字作独立背书。source manifest 只绑定 `path,size,mtime_ns`，不是输入文件内容哈希；checkpoint、tokenizer 与两份测试脚本才是完整 SHA-256。GPU 结果也未宣称 bitwise determinism。

## 8. 后续验证：v1 tokenizer 的 LiDAR metric-boundary gate（2026-08-01）

本节是 07-31 独立复测的后续实验，不改写其原始 provenance。D4 已证明 `c_depth` 对
leave-one-out render PSNR 无增益，但 PSNR 不能判断米制深度是否正确。新脚本
`tools/calibrate_tokenizer_pullback.py` 因而固定 D4 只用 scenes 300–319 拟合的
loglinear profile：

```text
z0 = depth_recon / s_lidar
c(z0) = exp(-0.0405706428 + 0.0146570329*log(clamp(z0,0.5,80)/20))
```

在固定 selection scenes 320–329 的 30 个 trunk 上，identity 与 candidate 使用完全相同的
原始 LiDAR cells。误差逐 cell 计算，依次按 frame/window/trunk/scene 取 median；五个重叠
窗口不是独立样本，bootstrap 只重采样 scene。主口径按 Phase-1a 已冻结阈值保留 26 个 trunk：

| 口径 | identity AbsRel | loglinear AbsRel | 相对改善 | scene Δ bootstrap 95% CI | 方向 |
|---|---:|---:|---:|---:|---:|
| 26 个有效 trunk、全部 LiDAR cells | 7.567% | 6.901% | **8.81%** | **[0.052%,1.225%]** | 8/10 scene 改善 |
| 全部 30 trunk sensitivity | 7.599% | 6.913% | 9.02% | [0.124%,1.226%] | 8/10 |
| 26 trunk、static/non-sky sensitivity | 7.614% | 6.935% | 8.91% | [0.062%,1.248%] | 7/10 |

预注册 gate 是 scene-bootstrap CI 下界严格大于 0，因此 **v1 metric boundary 选择
loglinear**。不过精确 sign-flip sensitivity 的单侧/双侧 p 分别为 0.0352/0.0703，且
scene 328、329 变差，证据应描述为温和而非无条件规律。render 仍保持 identity，`c_gs=1.0`；
这项测试不修复 GS/depth=0.796 的相似性失败。

该结果绑定 tokenizer v1 SHA-256
`75e566efa3b66baa43f82cb9999c2de60a9f3feeb0f714e1caf38d1f6e8137eb`，输出明确标记
`diagnostic_only_v1` 与 `eligible_for_training=false`。因此它验证了协议和 v1 偏差，但不是
最终 `data/scene_gauge/pullback_*.json`；tokenizer v2 出来后必须在 300–319 重拟合、在
320–329 原样复测，结论允许改变。

结果文件：

- `runs/metric_gauge_retest/v1_tokenizer_lidar_metric_gate_320_329.json`
- script SHA-256：`4162bbc469ede617056333e1e57dde124f53456f215bfb4578ddd6eab0e05eae`
- result file SHA-256：`29f91842641c2bfc7565d4edbe0f711223ba6eb006de1ed2ded8d5d72a97ecb9`
- canonical payload SHA-256：`87fc01c4f7bbd5f62fe40711d8fb82125c0e758a8dadce9b3e2e243e00cd6de1`

CUDA 0 正式运行耗时 186.94 s，环境为 PyTorch 2.7.1+cu128 / CUDA 12.8 / RTX PRO 6000
Blackwell Server Edition。新增单测 14 passed，连同冻结 D4 回归共 33 passed；CPU synthetic
与独立 result-contract 校验也通过。

## 9. 2026-08-01 执行冻结：v1 现在是正式 production pullback

§8 的原始 JSON 继续保持 `diagnostic_only_v1` / `eligible_for_training=false`，这是正确的 provenance：
实验输出不应被 runtime 直接冒充配置。用户随后决定 **tokenizer v2 训练完成前，不阻塞 Scene Flow
Phase 1–8，而是按已通过 gate 的 v1 结果正式实现**。因此使用
`tools/freeze_tokenizer_pullback.py` 生成了严格 runtime artifact：

- 文件：`data/scene_gauge/pullback_75e566ef.json`
- 文件 SHA-256：`1bb159e374e2b1d00af5020f780ada9f74d84a1365a525bc484fccb6a4e34693`
- `artifact_role=production_pullback`，`eligible_for_training=true`
- tokenizer v1 SHA-256：`75e566efa3b66baa43f82cb9999c2de60a9f3feeb0f714e1caf38d1f6e8137eb`
- runtime contract：10 帧、25×37 patch grid；加载时还必须匹配 DGGT checkpoint、artifact hash
  与 gauge representation

冻结的作用域没有变化：

| 边界 | v1 正式策略 | 能说明什么 |
|---|---|---|
| render | depth identity；Gaussian scale identity | 保持 tokenizer 原生 recon 渲染链，不把病态 PSNR 扫描结果混进几何 |
| metric export / metric assertion | loglinear `c_depth(z0)`；同一 factor 乘 depth 与 GS scale；`c_gs=1.0` | 降低 v1 米制 depth 的系统偏差 |
| 完整相似性 | **未修复**；paired `GS/depth≈0.796` | 必须作为 tokenizer v1 limitation 明示，不能声称 Gaussian/depth 相似一致 |

这项冻结只把**已有实验证据**转成可加载、可追责的生产契约，不改变 §8 的数字，也不把 0.796
包装成已解决。tokenizer v2 是根治该缺陷的路径，但**不是当前 Phase 1–8 blocker**。

同时冻结两条 K 链，禁止跨链混用：DGGT depth/render/sky 使用 trunk-global gauge K；米制
bbox/`in_frustum`/`target_bbox_patch` 使用真实 Waymo K。

### tokenizer v2 出来后的重测 prompt

下面这段可原样交给实现代理；`<V2_CHECKPOINT>` 必须替换成真实文件，不能沿用 v1 artifact：

```text
仓库 /home/dancer/code/dm/dggt，分支 dev。先完整阅读：
1) docs/metric_scale_camera_redesign_plan.md（尤其 Phase 0、Phase 1b、Verification、附录 A）；
2) docs/story_codex/scene_flow_metric_gauge_retest_2026-07-31.md（尤其 §4–§9）；
3) docs/story_claude/scene_flow_iclr_story_2026-07-30_claude.md 的 A.8.9–A.8.11；
4) docs/scene_flow_cmd.md（正式 stats、训练、推理命令及 clean-cut provenance）。

tokenizer v2 checkpoint 是 <V2_CHECKPOINT>。使用 conda dggt、CUDA_VISIBLE_DEVICES=0，先计算并记录
checkpoint SHA-256。不要复用 data/scene_gauge/pullback_75e566ef.json 的系数；teacher-space gauge GT 表
不需重算。按原冻结 split 在 scenes 300–319 重拟合 tokenizer pullback，在 scenes 320–329 原样重跑
LiDAR metric-boundary gate，并用 tools/retest_scene_flow_gaussian_gauge.py 重测 paired GS/depth、
depth_recon/depth_direct、GS scale_recon/scale_direct。render 路径继续以 identity 为基线；不得用 PSNR
选择尺寸常数。若确需 renderer fallback，只能另做 LPIPS/SSIM 的有内点 gate。

选择规则：
- 若 v2 metric identity 已不劣于 candidate，metric c_depth 取 identity；否则只在预注册 LiDAR gate
  通过时冻结新的 checkpoint-bound profile。
- c_gs 默认仍为 1.0；若 paired GS/depth 已回到预注册容差内，确认根治；若未回到 1，明确 limitation，
  不得把 PSNR 单调上升当作物理校正证据。
- render 始终独立选择，不能因 metric gate 通过而自动施加 correction。

产出新的 data/scene_gauge/pullback_{v2_sha8}.json，严格绑定 tokenizer/DGGT/window_len/patch_grid，
先用 load_pullback_calibration 与单元测试验证，再决定 Scene Flow checkpoint/provenance 是否从 v1 切到 v2。
把完整命令、fit/selection 数字、CI、文件 SHA、环境和最终分支决策回写上述四个 md；不覆盖 v1 的
历史 provenance。最后在 CUDA 0 分别跑非滑窗与滑窗小脚本，证明两条路径的 gauge、metric export
和 artifact rejection 行为一致。
```

## 10. 2026-08-02 Phase 1–8 production acceptance（tokenizer v1）

本节记录 v2 尚未完成时，按 §9 冻结的 tokenizer-v1 production pullback 完成 Scene Flow Phase 1–8
后的可复现验收。它不改变 §4–§9 的历史实验数字，也不把 v1 的 paired `GS/depth≈0.796` 包装成已修复。

### 10.1 离线 gauge 与正式 feature stats

- `data/scene_gauge/training.json`：4787/4787 trunks、798 scenes、0 errors；metric-scale/FOVx/FOVy
  valid counts `[4216,4787,4787]`，actor coverage 1662，SHA-256
  `39e0a32372e616e9aac4aef6109c8329ebdf382c16a913bd9e4d025b984e00af`。invalid-reason 计数是
  531 个 frame-CV 与 53 个 ruler-ratio，两个原因允许重叠。
- `runs/metric_gauge_retest/scene_gauge_training_random20.json`：seed 20260801 的 20-trunk LiDAR gate，
  median/mean/p95/max AbsRel 为 1.9833%/3.0103%/7.6039%/9.8653%，median 通过 5% 阈值；结果
  SHA-256 `889545cbd9b2753c44a732869ba507a43706e6557e52e7631c3dbe9c5c874f5a`。独立固定 drift cohort
  为 44 pairs、mean 8.2020%、max 30.8056%，三项 drift checks 全部通过。
- `logs/scene_flow_pretrain_1024/feature_stats_pretrain_v4.pt`（修复版）：先生成 `.inprogress`，独立
  校验后再发布。覆盖 4787/4787 trunks、798 scenes、44,279,750/44,279,750 latent，
  `stats_status=complete`、`exact_scene_gauge_scope=true`、`max_batches=null`；camera anchor/delta counts
  `[2416,45454]`、gauge counts `[4216,4787,4787]`、placement count 172605；
  `log_z_depth mean/std=2.998025/1.519740`。stats SHA-256 为
  `f5177c9262c878c1595c0f0e41ebd9cf42680de3676f0fccb789ed3cbc7a9111`，sidecar SHA-256 为
  `e0767b8bb3b86116f3748144b7c306d73ba6229a5568d47eb18a62cdf5d40539`。source contract 精确绑定
  tokenizer SHA `75e566efa3b66baa43f82cb9999c2de60a9f3feeb0f714e1caf38d1f6e8137eb`、DGGT SHA
  `352652738a5480b8d3ee9dd521ce07b528e5a297bd3feca4d07427dac6d87def`、上述 training gauge SHA、
  sequence 10、DGGT context 29 与 grid 25×37。

### 10.2 CUDA 0 端到端线路 smoke

> **2026-08-02 post-review 状态**：本小节以下数字是修复前历史记录。其 checkpoint 使用 v4 stats、
> `factorized_asset_v2`、`rgb_patch_v2`，已被当前 clean cut 拒绝，不可作为当前 production smoke。

历史环境是 `conda dggt`、`CUDA_VISIBLE_DEVICES=0`。一步训练输出到
`logs/metric_gauge_one_step_cuda0/`，full/raw/EMA-only 三种 step-1 checkpoint 均可加载，且携带相同的
metric-gauge provenance 与 `pretrain_feature_stats_contract`。实测总 loss 3.4305、flow loss 1.4045、
gauge-flow/direct loss 0.6251/0.0219，全部为有限值；LiDAR 诊断实际执行，
`metric_depth_rel_err_available=1`，随机一步的误差为 17.29%。

同一 EMA-only checkpoint 随后运行：

- `logs/metric_gauge_inference_10f_cuda0/`：10 帧，`window=10`、`stride=7`、`sliding=false`；
- `logs/metric_gauge_inference_29f_cuda0/`：29 帧，`sliding=true`，窗口精确为
  `[0,10]/[7,17]/[14,24]/[19,29]`，overlap geometry verification 为 true。

两条路径都只有一个 `[1,1,3]` scene-global gauge，都输出完整 9D 米制相机 tensor，并在
`boundary=metric` 下写出逐帧 Gaussian/point PLY（分别 20/58 个）和
`generated_static_geometry_reprojection_cycle_v1`。随机一步模型尚没有场景质量：它把 10 帧的 28,600 个、
29 帧的 82,940 个候选点全部判成 sky，所以 cycle 的 `status=insufficient_support`、metric support=0；
这是 fail-closed 诊断的正确行为，不应改写成 0 error 或“质量通过”。

### 10.3 结论与仍待 v2/完整重训回答的问题

Phase 1–8 的 production 数据、哈希 clean cut、direct/sliding 双路径、米制导出和诊断 wiring 已闭环。
当前结果只证明**代码与产物契约可以从离线 GT 一直走到训练、推理和米制导出**。一步随机模型的
`gauge_vs_prior_gain=0`、17.29% metric-depth error 和空 PLY 不能用于判定方法有效性；完整 Scene Flow
重训后仍必须测 gauge 相对 marginal prior 的增益、相机条件可控性以及真实车长/车道宽度误差。
tokenizer v2 出来后必须继续使用 §9 prompt，重新拟合/选择 pullback 并生成新的 v2 stats；不得沿用
本节 v1 artifact 的系数或 SHA。

### 10.4 2026-08-02 post-review D1–D4 修复验收

审计提出的前三项均由代码与数据复核确认，第四项的四个失败测试也完整复现，处理如下：

1. sky atlas 不再把 Waymo ego `+z-up` c2w 只做尺度换算后误当 teacher 世界。训练写入与 render 读出
   共享 teacher-anchor `-y-up` pose；生成米制相机先按 metric trunk anchor 重基再换算。sky clean cut
   升为 `rgb_patch_teacher_anchor_v3`。
2. placement stats 只使用 `pretrain_camera_to_anchor`；全量修复版 v4 stats 的 ch3 为
   `log_z_depth=2.998025±1.519740`、count 172605，stats SHA 为
   `f5177c9262c878c1595c0f0e41ebd9cf42680de3676f0fccb789ed3cbc7a9111`。
3. ch14 改为 `tanh(speed/z_depth)`，极端近/后方目标也不会向 MLP 注入 1e4 量级；factorized clean cut
   升为 v3。
4. 三个不存在 API 的 flow assembler 旧断言按当前 CleanSceneState 语义重写，tokenizer Stage-B fixture
   补齐新增参数。定向回归 140 passed；CUDA 0 完整回归 732 passed、1 skipped。

`lyy_tools/verify_metric_gauge_postreview.py` 的结果文件为
`runs/metric_gauge_postreview/verification.json`，SHA-256
`8803d5941c91e057711ac3c9d197470dec70d2bf6d8ffd27bd675e4d6997da9e`。新的一步真实训练位于
`logs/metric_gauge_postreview_one_step_cuda0/`：启用 sky generation，loss/flow/sky-flow/gauge-flow/
gauge-direct=`3.4283/1.4045/0.1020/0.6241/0.0219`，LiDAR diagnostic available=1。EMA-only
checkpoint 实际 mmap 加载确认含 sky-v3、factorized-v3 与修复后 v4 stats SHA；checkpoint SHA-256 为
`4daae958f043721f47a8e89c94cb5fe3a3b3e7ea7cfaf8586915c6e5ee85d9a6`。
