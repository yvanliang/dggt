# 度量尺度显式生成 + 相机米制化改造

> **v5 — 2026-08-05**。v1 写于 Phase 0 之前，其 Phase 1–8 建立在两条尚未验证的假设上。
> Phase 0（D1/D2，见 A.8.5/A.8.6）与独立复测（`docs/story_codex/scene_flow_metric_gauge_retest_2026-07-31.md`）
> 已推翻其中一条、收窄另一条；D3/D4 与 Phase 1b-0 LiDAR gate 又冻结了 render/metric
> 两个作用域。v4 冻结过的 tokenizer v1 production contract 只保留为不可变历史证据；本版新增
> tokenizer v2 的哈希隔离正式复测，并按预注册 gate 落到 **Phase 1b 方案 A**：render 与 metric
> boundary 都是 identity，`c_gs=1.0`。最终 runtime 是 **v2-only clean cut**：代码、默认配置和 loader
> 均拒绝 tokenizer v1 schema/checkpoint/pullback；Phase 2 未由本次工作启动。

## 修订说明（v1 → v5）

| # | v1 的说法 | 实测 | v5 的处理 |
|---|---|---|---|
| 1 | FOV 用哪套 K 待定（分支 A/B） | D1 判定 **Branch A**：`K_pred` 优于 `K_Waymo` +0.472 dB（CI [+0.253,+0.741]） | DGGT depth/render/sky 使用 gauge K；米制 bbox/`in_frustum` 保持 Waymo K，两条链路不交叉 |
| 2 | 「DGGT 是米制世界的内部自洽相似缩放副本」 | **过强**。除以 `s_cam` 后仍余 camera-XYZ 2.452 m，其中 1.287 m 纯由 FOV 差异产生 | Context 收窄为「camera center + z-depth 共享一维 gauge」，不再宣称完整 Sim(3) |
| 3 | gauge GT = `log(1/s_depth)`，用 direct head 算，直接当作生成量的单位 | **不完整**。训练/推理的几何经过 tokenizer 往返：depth ×1.031、GS 三轴 scale ×0.829，配对 `GS/depth` = **0.796**（30/30 场景 < 1） | **Phase 1 拆成两件事**：teacher 空间 gauge 表（与 tokenizer 无关）+ checkpoint-bound pullback 审计/metric 标定 |
| 4 | 把约 3% depth 偏移直接称为 render bug，并计划在 render 中施加 `c_depth` | D4 render 增益 −0.000933 dB；但 v1 LiDAR gate 把 metric AbsRel `7.567%→6.901%` | **按作用域拆开**：render 恒 identity；loglinear 只用于 v1 metric boundary，v2 后重拟合 |
| 5 | 旋转 0.168°、轨迹形状 0.52%，故米制相机「几乎无代价」 | 独立估计器给 raw 0.371°（拟合常量 `Q` 后 0.201°）、形状 **1.334%**（max 4.066%） | 论据换成更强的那条（去掉不可观测噪声 vs 残留可测偏差），并新增 **D3** 实测该替换在渲染上的代价 |
| 6 | `s_cam/s_depth = 1.0073 ± 0.0442` | 两个独立估计器在 29 帧口径下均给 ~2.5–2.6% | 数字更新；`±4.4%` 不再当作「最好情况下界」 |
| 7 | 「米制 → DGGT 只差一个标量」，故 `target_bbox_patch` 应改用 gauge K、`in_frustum` 有视锥矛盾 | **两条都错**。映射是各向异性的（横向 ×0.748）；而 box 投影本就走全米制自洽链路，无矛盾 | **撤回该改动**：`target_bbox_patch`/`in_frustum` 保持 Waymo K；改为「两条链路各自一个 K，永不交叉」；编辑路径新增各向异性 `metric_box_to_dggt` |
| 8 | v1 metric boundary 需要 loglinear、paired `GS/depth=0.796` 留作 tokenizer v2 根治 | v2 paired point `0.99985`、scene-bootstrap 95% CI `[0.99015,1.00753]` 均落在预冻结 `[0.95,1.05]`；v2 loglinear 在 LiDAR selection 上使 AbsRel `7.762%→8.507%` | **Phase 1b 方案 A**：v2 render/metric 都用 identity、`c_gs=1.0`；v1 artifact 与结论只作历史保留 |

---

## Context

### 一、已由实测确定的事实

所有尺度统一定义 `s = DGGT unit / metre`，`log_metric_scale = log(1/s) = log(米 / DGGT 单位)`。
下表合并 A.8（`lyy_tools/verify_*.py`）与独立复测（`tools/retest_*.py`）两套**互相不共享代码**的估计器，
均为 Waymo training scene 300–329 × trunk 0/1/2 = 90 个 29 帧 trunk。

| 分量 | 由 Waymo 参数确定？ | A.8 | 独立复测 | 采纳 |
|---|---|---|---|---|
| 世界原点 | **是** | `\|c2w[0]−I\|` max 7.1e-4 | 一致 | 锚点按构造是单位阵 |
| 旋转 | **近似是** | mean 0.168°，max 1.157° | raw mean 0.371° / median 0.184° / max frame 6.484°；拟合常量 `Q` 后 mean 0.201° | **报告区间 0.17°–0.37°，并注明配准方式**；scene 312/314/325 有离群 |
| 轨迹形状 | **近似是** | 残差 0.52% 路长，max 2.38% | constant-Q 后 mean 1.334%、p95 3.252%、max 4.066% | **采纳更保守的 1.33%**；不宣称数值上一一对应 |
| 平移尺度 | **否** | lidar 尺 [0.01548, 0.05076]，median 0.02790 | `s_cam` [0.01635, 0.03945]，1 单位 = 25.4–61.2 m | 1 DGGT 单位 ≈ **25–64 米，逐 trunk 变化** |
| FOV | **否** | DGGT 38.13°±9.51°（Waymo 49.85°±0.26°） | 38.325°±9.386°（y: 26.888°±6.626°） | 完全复现 |
| camera 尺 vs depth 尺 | — | 29f: `0.99995 ± 0.02639`，corr 0.99296（n=70） | `0.9955 ± 0.0261`，corr 0.9933（n=70/26 场景） | **共尺，离散度 ~2.5%** |

**尺度是 29 帧 teacher trunk 的属性，不是场景常数。** 相邻 trunk 相对漂移 mean 8.2% / p95 21.9% / max 30.8%（44 对）；
camera 与 lidar 的有符号漂移 `Δlog s` 在 41/44 对上同向、Pearson 0.9711 —— 排除「只是拟合噪声」的解释，
**是冻结 VGGT 自己在不同片段选了不同尺度**（`train.py:82-93`：`camera_head` 全程 `requires_grad=False`，
沿用 VGGT 的逐场景尺度归一化，米制信息从未进入这个 head）。

### 二、必须收窄的三条（否则论文经不起审）

1. **不是完整 Sim(3)：米制 → DGGT 的映射是各向异性的。** 把 DGGT depth 除以 `s_cam` 后反投影，
   仍余 z 1.738 m / 横向 1.354 m / camera-XYZ 2.452 m；其中**仅由 FOV 差异产生的就有 1.287 m**，
   稀疏 LiDAR cell 的射线夹角 scene-balanced 3.286°（CI [2.483°, 4.197°]）。

   机制是确定的、可写成闭式的。DGGT 的几何由 `K_dggt` 反投影像素得到，对同一像素 `u`、米制深度 `Z`：

   ```
   Waymo:  X      = (u−cx)·Z / fx_w
   DGGT:   X_dggt = (u−cx)·(Z/s) / fx_d = (X/s) · (fx_w/fx_d)

   k_x = fx_w/fx_d = tan(FOV_dggt_x/2)/tan(FOV_waymo_x/2) = tan(19.16°)/tan(24.92°) = 0.748
   k_y                                                    = tan(13.44°)/tan(17.21°) = 0.772
   ```

   即在**相机坐标系**下 `p_dggt = diag(k_x, k_y, 1) · p_metric / s` —— **径向 ×(1/s)，横向 ×(0.748/s)，
   横向被压缩约 25%**。自洽性检验：25 m 处 3.286° 射线夹角 = 1.44 m，与 FOV-only 残差 1.287 m 同量级 ✓。

   正确表述：**camera center 与 z-depth 在 ~2.5% 离散度下共享一维 gauge；横向不共享，完整欧氏 3D
   不是只差一个标量。**

   > **对 3D box 的直接后果**（这是最容易踩的坑）：
   > | 量 | 单标量够吗 | 实际 |
   > |---|---|---|
   > | 中心位置 | ✗ | 30 m 处、横向偏离 5 m 的目标误差 5×0.252 ≈ **1.26 m**（≈ 一个车宽） |
   > | 尺寸 `lwh` | ✗ | **朝向相关**：4.5 m 车沿光轴 → `4.5/s`，横置 → `4.5·0.748/s`。前视相机 + 路面车流下长度几乎不变而**宽度压缩 25%** |
   > | yaw | ✗ | 45° → `atan(0.748·tan45°)` = 36.8° |
   >
   > **但适用范围很窄。** `datasets/dataset.py:2054-2071` 的 box 投影走的是**全米制链路**
   > （米制 `object_to_anchor` + 米制 `camera_to_anchor` + 真实 Waymo K），自洽且给出真实目标在
   > 真实图像里的正确像素足迹；渲染侧也从不需要换算（DGGT 高斯本就是用 `K_dggt` 反投影这批像素得到的）。
   > **两套约定在像素链路里从不交叉。** 只有把 Waymo box 当作 DGGT 空间里的**显式 3D 体积**使用时
   > （编辑路径：圈选/删除/搬移高斯、按米制坐标插入 asset）才必须用上面的各向异性形式。
2. **不是「共尺 ⇒ 渲染不会失配」。** 见下面第三节。
3. **不是「相机条件数学上不可能决定尺度」。** 正确表述：**仅由米制轨迹形状无法解析 teacher 自选的 gauge**；
   但图像内容（车辆尺寸、车道宽度、建筑）可能提供统计线索。这一点很重要 —— 它正是 Phase 2 让 gauge token
   与全部 video token 做全注意力的**机制假设**，且是可证伪的（见 Phase 4 的 marginal-prior baseline）。

### 三、Phase 0 与复测暴露的**五个**代码硬伤

1. **相机生成目标带不可观测噪声。** 目标 = DGGT 预测位姿（`camera_state_from_dggt_pose_enc`），平移通道混了一个
   逐 trunk、CV 23.5% 的标量；20D Waymo 条件却是米制的。模型被要求学「输入乘一个看不见的随机数」——
   **相机可控性在数学上不成立。**
2. **米制条件与生成几何量纲不符。** `build_placement_state` 返回原始米制 `center_anchor`/`velocity`/`log box_size`，
   而生成几何在 DGGT 单位里。`placement_mean/std` 是全局统计，`s=25` 与 `s=64` 的场景归一化后输入完全相同、
   却对应不同物理位置。
3. **FOV 目标是坏值。** 生成 `log tan(FOV/2)` 去拟合一个 σ=9.4° 的量，同时把正确的 Waymo FOV 当条件喂进去。
   D1 进一步说明：这个「坏值」其实是 gauge 的一部分（见下），当前写法是把它放错了地方。
4. **sky atlas 与 render 的内参来自两个不同来源。** atlas 用 `camera_pose_gt_dggt` 的 **teacher 逐帧 K**
   （`train_scene_flow_pretrain.py:5630-5633`），render 用 `render_pose_enc_dggt` 的**生成 K**
   （`rgb_render_loss.py:275`）。二者不相等，差值就是当前 FOV 生成误差（σ=9.4°）。
   > **注意（v2 更正）**：这里**不**是「atlas 用错了 Waymo K」。写入与读出用同一个 K 时，
   > 任何 K 都只是环境图的一个固定重参数化，会自行抵消；真正的 bug 是两侧 K 不同源、且逐帧抖动。
   > 同理，`in_frustum` / `project_anchor_boxes_to_patch_bboxes` 在 `datasets/dataset.py:2054-2071`
   > 里走的是**全米制链路**（米制 box + 米制相机 + 真实 Waymo K），自洽且正确，**不存在**
   > 「标为在视锥内但渲染不到」的矛盾 —— 渲染覆盖的正是被反投影的那 518×350 个像素。
5. **【新】render loss 里存在相机/几何 3% 尺度失配，且高斯半径系统性偏小 20%。**
   `rgb_render_loss.py:206,212` 的 `depth_head`/`gs_head` 作用在 **tokenizer 重建 token** 上；
   `train_scene_flow_pretrain.py:4812-4818` 的渲染相机来自**生成的相机**（teacher 空间）。实测往返后：

   | 量（配对同像素，scene-balanced） | 值 |
   |---|---:|
   | `depth_recon / depth_direct` | **1.0307**，CI [1.0208, 1.0421] |
   | Gaussian 三轴几何平均 `scale_recon / scale_direct` | **0.8289** |
   | **配对 `GS/depth`（相似性的必要条件，理想 = 1）** | **0.7964**，IQR 0.7876–0.8120，30/30 场景 < 1 |
   | RMS-radius 判据（对轴置换不敏感） | 0.7996 |
   | 帧内 depth ratio 的 log-MAD | 0.0122（**空间上高度均匀**） |

   两条独立后果：
   - **视差偏小 3%。** 几何被放大 1.031 而相机基线没有。10 帧窗口、ego ~10 m/s → 基线 ~10 m；
     `f ≈ 745 px`（518 宽、FOVx 38.3°），20 m 处视差 ≈ 376 px，3% 即 **≈ 11 px** 的系统性错位。
   - **高斯偏小 20%。** 相对其所在深度，splat 覆盖不足 → 空洞、锯齿、PSNR 损失。
   这是**既有 bug**，v1 没有意识到它，修它属于「提升模型效果」而非新增功能。

### 四、目标结果

- 相机生成目标改为**真实 Waymo 米制位姿**（确定性 GT、零尺度噪声），相机可控性变成恒等映射。
- 新增 **scene-global gauge token**，显式生成 `[log_metric_scale, log tan(FOVx/2), log tan(FOVy/2)]`，
  作为 DGGT 几何 ↔ 米制的唯一桥梁。
- **冻结解码器的 pullback 单独标定并按作用域冻结**：v1 历史 contract 曾在 metric boundary 使用
  checkpoint-bound loglinear `c_depth`，render 使用 identity，`c_gs=1.0`；v2 正式 gate 选择
  **方案 A**，两个 boundary 均为 identity、`c_gs=1.0`。runtime 只接受 v2 SHA/schema；v1 的
  `GS/depth=0.796` limitation 不被改写，但其 artifact 不再可加载，v2 已从源头通过相似等价 gate。
- 内参分成**两条永不交叉的链路**：DGGT 链路（depth 反投影 / render / sky atlas）统一到 gauge K；
  米制链路（box 投影 / `in_frustum` / `target_bbox_patch`）保持真实 Waymo K。
- asset 条件重参数化为「无量纲方向 + log 幅值 + 尺度不变比值」；16 维中 5 个通道标准化、
  11 个 passthrough。
- 推理可导出**米制**点云与米制高斯。

### 五、已冻结的决策

| 项 | 决定 | 依据 |
|---|---|---|
| 相机单位 | 米制（目标换成真实 Waymo 位姿） | 用户决定 + 硬伤 1 |
| `log s` 可控性 | 只生成，不做条件路径 | 用户决定 |
| FOV 归属 | **Branch A** —— gauge 的两个通道，GT = DGGT 29 帧 `mean(log tan(FOV/2))` | D1（A.8.5） |
| 内参消费者 | **两条 K 链**：DGGT depth/render/sky 用 gauge K；米制 bbox/`in_frustum`/`target_bbox_patch` 用真实 Waymo K | D1 + 各向异性复测；禁止跨链混用 |
| gauge 主尺 | **完整 29 帧 LiDAR 深度尺**（90/90 有效；相机尺在 20/90 静止 trunk 上按定义失效） | D2（A.8.6） |
| actor 尺 | 仅诊断字段，不作 GT（29/90 可用，3 个灾难性离群 0.94/0.54/0.54） | D2（A.8.6） |
| gauge GT 空间 | **teacher 空间**（与 tokenizer 无关）；tokenizer 偏差由独立、按作用域的标定/审计产物处理 | 模块化：换 tokenizer 只需重跑 pullback audit/gate，不用重算全表 |
| tokenizer v1 过渡策略（历史） | `data/scene_gauge/pullback_75e566ef.json` 及数字只作审计记录；runtime/default/config/loader **必须拒绝** v1 schema/checkpoint/pullback | v1 LiDAR gate 当时通过；结论保留但不再形成兼容分支 |
| tokenizer v2（当前） | **正式启用** `data/scene_gauge/pullback_d63b34f7.json`：render identity、metric identity、`c_gs=1.0`（方案 A） | 30-scene paired GS gate 通过；10-scene LiDAR gate 选择 identity；CUDA render smoke 通过 |
| 兼容性 | **v2-only 干净切断**：升版本号，v1 tokenizer/pullback/checkpoint 与旧 Scene Flow checkpoint 全部拒绝加载 | 用户决定 |

---

## Phase 0 — 诊断

### 已完成：D1 / D2

- **D1（`lyy_tools/verify_fov_consistency.py`，90/90 成功）→ Branch A。**
  决定性判据是静态区域的 **primitive-level leave-one-frame-out** 渲染（目标帧产生的 Gaussian means 完全排除，
  同一候选 `K` 同时用于源帧 unprojection 与目标帧 rasterization）：

  | 比较（dB，正值 = 左侧更好） | mean | scene-bootstrap 95% CI |
  |---|---:|---:|
  | trunk-mean `K_pred` − `K_Waymo`，fixed static mask | **+0.472** | [+0.253, +0.741] |
  | trunk-mean `K_pred` − `K_Waymo`，shared-alpha support | **+0.539** | [+0.298, +0.829] |
  | native per-frame `K_pred` − `K_Waymo` | +0.580 / +0.677 | 同向 |

  **可生成的 trunk-mean K 相对 native per-frame K 只掉 −0.108/−0.150 dB**，通过预注册的 −0.2 dB
  non-inferiority margin —— 即「把 FOV 压成 trunk 常量再生成」几乎不损失。
  FOV 的 29 帧**内** std 仅 0.262°/0.178°（x/y），而 trunk **间** std 达 9.376°/6.624°：
  **FOV 确实是 per-trunk 规范量，不是逐帧量。**
  `point_head` 在三种候选 K 下均判为 `coordinate_incompatible`（`used_for_branch_decision=false`）；
  **这不影响本计划** —— `rgb_render_loss.py:429` 用 `torch_unproject_depth(depth, K)` 生成 `point_map`，
  渲染路径根本不经过 `point_head`。

- **D2（`lyy_tools/verify_gauge_gt.py`，90/90 成功）→ LiDAR 主尺。**
  移动组 29f `s_cam/s_lidar = 0.99995 ± 0.02639`（corr 0.99296，92.86% 落在 5% 内）；
  LiDAR 尺 90/90 有效，逐帧 robust CV mean 0.688% / median 0.263% / max 5.777%；
  静止组 20/20 上相机尺按定义无效而 LiDAR 尺稳定 —— 这是选 LiDAR 的关键理由。

### 已完成：D3 / D4（2026-08-01，结果见 A.8.9）

**D3 —— 米制相机换算的渲染代价 → 判定 `teacher`**

同一批 world-space Gaussian、同一个 trunk-constant gauge K、同一 support 与属性，
只把 target raster view 从 native teacher pose 换成 `metric_c2w_to_dggt(Waymo c2w, s_lidar)`：

| mask | scene-balanced mean loss | 95% CI | 决策 |
|---|---:|---:|---|
| fixed static | **1.4143 dB** | [1.0248, 1.8377] | teacher |
| shared-alpha | **1.4357 dB** | [1.0583, 1.8437] | teacher |

这里的 **1.41–1.44 dB 不是推理画质损失**：推理时没有同一真实图像上的
teacher/metric 两臂可比，生成相机和生成几何仍可共同形成自洽渲染。它是 D3 的**完整 29 帧、
指定 source/target 协议下的控制保真度上界诊断**，衡量“交付的 teacher-space 轨迹”相对
“请求的 Waymo 米制轨迹”存在多少可测残差；引用时必须同时带上该窗口与协议限定。

**motion split 是关键诊断**：moving trunk 1.7642 dB，stationary trunk 仅 **0.1896 dB**。
说明这个代价**几乎全部来自随运动累积的轨迹形状残差**，而不是逐帧的静态错位——
静止时两套位姿都接近单位阵，自然重合。三个预先关注的离群 scene 排名靠前
（314: 4.55 dB、312: 3.73 dB、325: 2.45 dB），与 A.8.4.1 的旋转离群同一批。

> **时间偏移诊断（不冒充窗口长度实验）**：D3 本身已经使用完整 29 帧 trunk。按 target
> local index 0/7/14/21/28 后验分层，fixed loss 依次为
> `0.029/0.996/1.593/2.036/2.417 dB`，说明代价在同一 29 帧片段内随离锚点时间增加；
> 但没有匹配地重跑不同 source-window 长度，因此不能写成“29 帧会比 D3 更大”。

**D4 renderer gate + Phase 1b-0 LiDAR gate —— 必须按作用域拆开**

| 轴 | 标定集拟合形式 | render 作用域 | metric-boundary 作用域 |
|---|---|---|---|
| `c_depth` | **loglinear**，`a=−0.04057, b=+0.01466, c(20m)=0.9602`；slope CI [0.00926, 0.01902] **不含 0**，LOSO RMSE 改善 8.80% | **identity**；PSNR 增益 **−0.000933 dB** | v1 LiDAR gate **通过 loglinear**：AbsRel `7.567%→6.901%`，相对改善 `8.81%`，scene-bootstrap Δ CI `[0.052%,1.225%]` |
| `c_gs` | constant `1.2576`（bin Spearman = 0，loglinear 未过 2% gate） | **identity**；PSNR 单调涨到语义上界，`renderer_pathology_rejected` | **identity**；无独立物理尺，等待 tokenizer v2 从源头修复 |

深度分层的 `c_depth` 有真实趋势：0.898(2.5m) / 0.947(7m) / 0.968(14m) / 0.962(28m) / 0.970(57m)
——**近场畸变 11%，远场 3%**（近场 bin 只有 5 个标定 scene，该值较弱）。

`c_gs` 的拒绝是正确的科学判断：`c_gs=2.5` 会把配对 GS/depth 从 0.7964 推到约 **1.99× teacher**，
早已失去「pullback 逆变换」的物理含义。**持续上涨的 PSNR 是在奖励 splat 覆盖与模糊，
不是在识别一个几何尺度。**

> **方法论教训（记录下来，避免重试）：render PSNR 对「尺寸类常数」是单调的，不能用作它的判据。**
> 我在上一版把「用留出集 PSNR 扫描」当成绕开 population 选择问题的办法——**那是错的**，
> 因为这个目标函数根本没有内点极值。
> 若日后必须重扫，改用 **LPIPS 或 SSIM**（惩罚模糊，可能存在内点极值），不要再用 PSNR。

---

## Phase 1 — 离线 gauge GT 表 + 冻结解码器 pullback 标定

**为什么离线、为什么 29 帧**：10 帧窗口估尺度的 within-trunk CV 为 mean 3.19% / median 2.18% / max 9.94%
（独立复测；A.8 给 3.1%/1.9%/8.8%，一致），静止 trunk 直接发散到 23×–30×。
而 aggregator 训练时本来就对整个 29 帧跑一次（`train_scene_flow_pretrain.py:5532`），
真尺度按构造对该 trunk 的每个窗口相同。离线预计算一次即彻底消除窗口依赖。

### 1a. gauge GT 表（**teacher 空间，不依赖 tokenizer**）

新文件 `tools/compute_dggt_scene_gauge.py` → `data/scene_gauge/{split}.json`：

```json
"301/1": {
  "log_metric_scale": 3.372,          // log(1/s_lidar)，teacher 空间；实测范围约 [2.98, 4.17]
  "log_tan_half_fov": [-0.5741, -0.8339],   // [x, y]，= mean_29f(log tan(FOV/2))，Branch A
  "s_lidar": 0.03429, "s_cam": 0.03431, "ruler_ratio": 1.0006,
  "s_actor": 0.0340,                  // 诊断字段，永不作为 GT
  "n_valid_px": 412391, "frame_cv": 0.0031, "ego_motion_m": 3.87,
  "rot_residual_deg": 0.19, "shape_residual_pct": 0.94,   // 米制相机替换的逐 trunk 代价
  "fov_std_deg": [0.26, 0.18],
  "valid": [true, true, true]          // 三个 gauge 通道各自的有效性
}
```

- `log_metric_scale = log(1 / s_lidar)`；`s_lidar = median(dggt_depth / lidar_depth)`，有效像素 `1 < lidar < 80` m；
  **逐帧取 median，再跨 29 帧 MAD 去异常后取 median**（复刻 D2 已验证的协议）。
- `log_tan_half_fov` **严格按 `mean(log tan(FOV/2))` 构造，不用角度算术平均**（照 A.8.2 第 4 条）。
- `valid=false` 的情形：lidar 文件缺失（`{scene}/depth_flows_4/{f:03d}_0.npy`）、有效像素 < 5000、
  逐帧 robust CV > 3%（D2 实测 max 5.777%，取 3% 可剔除长尾）、或移动 trunk 上 `ruler_ratio` 偏离 1 超过 10%。
  `valid=false` 的样本**仍参与其余全部训练**，只是 gauge 的直接监督被逐通道 mask。
- `rot_residual_deg` / `shape_residual_pct` 是新增字段：把「米制相机替换的代价」变成**逐 trunk 可查的数**，
  而不是一个全局平均。**不用它做样本降权**——米制 Waymo 位姿是真值，divergence 是 teacher 的误差
  而非目标的误差。它的用途是**分层报告**：把相机误差按 trunk divergence 分组看
  （D3 已确认 314/312/325 是最高的三个）。
- 成本：每 trunk 一次 29 帧 aggregator + depth/camera head 前向，约几秒；**不需要 tokenizer 前向**。
  全训练集数 GPU-小时，可按 scene 分片并行。
- **复现坑（已踩过）**：`dggt_window_indices` 是 **trunk 局部**索引（0–28），lidar 文件必须用
  `global = trunk * 29 + local`。第一版脚本用局部索引取雷达，trunk 1/2 的尺度全错。

**当前正式脚本的 lean/golden 契约**（`tools/compute_dggt_scene_gauge.py`）：

- 只实例化 checkpoint-compatible 的 `Aggregator + CameraHead + DepthHead`；不构建 tokenizer、PointHead、
  GaussianHead 或其余 dense heads。CUDA 上 aggregator 用 bf16 autocast，CameraHead/DepthHead 明确回到
  fp32，分别复现 D1/D3 与 D2 的数值路径。
- 只发现**连续且完整的 29 帧 trunk**；输出使用严格 schema/protocol hash、原子 checkpoint、可恢复
  `--resume` 和拒绝重复 key 的 shard merge。全表未完成时不得把 shard 或 `status!=complete` 的文件当训练表。
- `phase0_golden_comparison` 是 fail-closed golden：同一 trunk 的 `s_lidar` 必须复现 D2，FOV 的
  29-frame std 必须复现 D1 的 fp32 路径；单测覆盖正确与故意错配两臂。validation 正式表已完成：
  `1212/1212` trunks、202 scenes、0 errors，metric-scale 有效 1082，失效原因为 115 个 frame-CV
  与 15 个 ruler-ratio，actor 有效 478；文件 SHA-256 为
  `5014e5c0ba5bd570c1a3d13e3fd222d15e32fe10276046dda763b7e87d9559fa`。training 正式表也已完成并
  发布：`4787/4787` trunks、798 scenes、0 errors，metric-scale 有效 4216，两个 FOV 通道各有效 4787，
  actor 有效 1662；文件 SHA-256 为
  `39e0a32372e616e9aac4aef6109c8329ebdf382c16a913bd9e4d025b984e00af`。invalid-reason 计数为
  531 个 frame-CV 与 53 个 ruler-ratio（原因可重叠）。独立 random20 LiDAR gate 的 median AbsRel
  为 1.9833%（mean 3.0103%、p95 7.6039%、max 9.8653%），通过 5% 预注册阈值；固定 drift cohort
  为 44 pairs、mean 8.2020%、max 30.8056%。

### 1b. 冻结解码器 pullback 标定（**v1 历史保留；2026-08-05 由 v2 方案 A 收口**）

> **D4 之后本节必须按作用域理解。** render 路径的两个常数都为 identity；
> 米制边界另有独立 LiDAR 尺，可以且已经对 v1 的 `c_depth` 作出选择。

#### 1b-0. LiDAR metric-boundary gate（v1 已实跑）

D4 把 **render PSNR gate 同时**用在 `c_depth` 与 `c_gs` 上。对 `c_gs` 这是对的
（没有独立的物理参照，只能问「渲染是否更好」，而结果证明这个问题本身病态）。
**对 `c_depth` 这是错的** —— 它有一个现成的、独立的物理参照：**LiDAR**。

`c_depth` 承担两件事，D4 只测了第一件：

| 作用 | 判据 | D4 的结论 | 有效吗 |
|---|---|---|---|
| (a) 修 render 的视差失配 | render PSNR | 增益 −0.0009 dB → 不需要 | ✓ 有效，接受 |
| (b) 让**米制导出**正确 | **与 LiDAR 的一致性** | D4 未测；Phase 1b-0 已补测 | ✓ 必须使用独立物理尺 |

(a) 的零增益是可以理解的：tokenizer 往返的**逐像素**深度误差远大于 3% 的系统偏移，
去掉一个被噪声淹没的系统项，PSNR 自然不动。
但 (b) 完全不同 —— 系统偏移正是米制精度唯一在乎的东西。

**后果是具体的**：`c_depth = 1.0` 时，
`exp(log_metric_scale) × depth_recon` 会**系统性高估距离 3%（远场）到 11%（近场）**，
因为 gauge 定在 teacher 空间而几何在 recon 空间。而目标通常落在 5–40 m，正是偏差较大的区间。

固定 D4 在 calibration scenes 300–319 拟合的
`a=-0.0405706428, b=0.0146570329`，在 selection scenes 320–329、trunks 0/1/2 上
比较 identity 与 loglinear；不在 selection 上重拟合。真实 CUDA 0 结果：

| 口径 | trunk / scene | identity AbsRel | loglinear AbsRel | 相对改善 | scene Δ 95% CI | 方向 |
|---|---:|---:|---:|---:|---:|---:|
| 主口径：Phase-1a 阈值有效 + 全 LiDAR cells | 26 / 10 | 7.567% | 6.901% | **8.81%** | **[0.052%, 1.225%]** | 8/10 scene 改善 |
| sensitivity：全部 trunk | 30 / 10 | 7.599% | 6.913% | 9.02% | [0.124%, 1.226%] | 8/10 |
| sensitivity：有效 trunk + static/non-sky | 26 / 10 | 7.614% | 6.935% | 8.91% | [0.062%, 1.248%] | 7/10 |

聚合严格为 pixel median → frame → window → trunk → scene，bootstrap 只重采样 10 个 scene；
五个重叠 10 帧窗口不是独立样本。主口径 CI 下界严格大于 0，故**按预注册 gate，v1 在
metric boundary 选择 loglinear**。精确符号翻转 sensitivity 为单侧 `p=0.0352`、双侧 `p=0.0703`，
且 scene 328/329 变差，因此证据应称为“通过预注册 gate 但幅度温和”，不能夸成普适定律。

该结果只在米制导出、模块 C 的米制断言、`metric_depth_rel_err` 使用；**render 路径恒为
identity**。原始 gate 结果继续保留 `artifact_role=diagnostic_only_v1`、
`eligible_for_training=false`，因为实验记录本身不能被 runtime 当作契约。用户随后冻结了 v1
过渡策略：`tools/freeze_tokenizer_pullback.py` 将同一证据写成严格、哈希绑定的 production artifact
`data/scene_gauge/pullback_75e566ef.json`；**这是 v4 当时的历史加载契约，不是 v5 runtime contract**。
v5 clean cut 保留该文件作为审计记录，但生产代码和配置拒绝加载它。
v4 对后续 v2 的要求是：在 300–319 重拟合并原样重跑 320–329，结论和系数允许改变；
该要求现已按 1b-4 完成，且 v2 selection 改选 identity。

#### 1b-1. 本节剩下的三个作用

1. **审计记录 + 哈希绑定**：`pullback_*.json` 仍然要生成，记录配对比值、深度剖面、
   tokenizer SHA-256、`window_len`。v1 记录只读归档；runtime loader 只接受唯一 v2 SHA/schema，
   任何 v1 tokenizer 或 pullback 都 fail-closed。
2. **`apply_pullback_calibration` 仍然要实现**，并显式传作用域；当前 v2 的 render/metric 都是
   identity、`c_gs=1.0`。保留函数是为了维持唯一施加点和 no-op/hash contract，而不是保留 v1
   loglinear 运行时分支。
3. **验证 tokenizer v2 是否成功的尺子**：见下。

#### 1b-2. `c_gs` 被拒：v1 limitation 与 v2 根治路径（历史结论）

对 tokenizer v1，`c_gs = 1.0` 意味着 **0.796 这个缺陷没有被 pullback 修复**：
高斯相对其深度仍然偏小约 20%。这是 v1 production artifact 必须保留的 limitation；v2 的最终
根治结果见 1b-4，不反向改写这段历史。

根因已定位、修复已进代码（见附录 A）。当时让配对比值从源头回到 1 的剩余路径是 tokenizer v2
重训；用户决定在 v2 完成前按 v1 结果实现 Scene Flow，所以 v4 将 v2 记为非阻塞并部署 v1。
v1 production contract 保留这个缺陷并显式报告：

| | 状态 |
|---|---|
| v4 的 v2 成功条件 | `gs_scale_sim_ratio → 1.0`，缺陷从源头消除，`c_gs` 永远不需要 |
| v1 历史 production | metric-depth loglinear；`c_gs=1.0`，0.796 缺陷保留并写进 limitation，且**不得声称 v1 渲染几何是相似一致的** |
| v2 失败分支（未触发） | 先按同一 LiDAR/paired-ratio gate 判定；只有仍需 renderer fallback 时，才用 **LPIPS/SSIM**（不是 PSNR）重扫 `c_gs` |
| v2 正式结果 | paired gate 通过；LiDAR 选择 identity；采用方案 A，见 1b-4 |

#### 1b-3. 标定文件格式（依赖 tokenizer，独立文件）

> 本小节完整保留 v1 artifact 的历史语义，便于复核旧结果；它不再是 v5 的实现需求。
> v2-only loader 必须拒绝该 schema/generation/path，即使文件内历史字段仍写着
> `eligible_for_training=true`。

`tools/calibrate_tokenizer_pullback.py` 的 v1 原始诊断写入
`runs/metric_gauge_retest/v1_tokenizer_lidar_metric_gate_320_329.json`；
`tools/freeze_tokenizer_pullback.py` 已将冻结决策生成正式文件
`data/scene_gauge/pullback_75e566ef.json`。其文件 SHA-256 为
`1bb159e374e2b1d00af5020f780ada9f74d84a1365a525bc484fccb6a4e34693`，严格绑定 tokenizer v1
SHA-256 `75e566efa3b66baa43f82cb9999c2de60a9f3feeb0f714e1caf38d1f6e8137eb`、DGGT checkpoint、
10 帧窗口与 25×37 patch grid。精简后的语义结构为：

```json
{"artifact_role": "production_pullback", "eligible_for_training": true,
 "tokenizer_generation": "t0_v1",
 "tokenizer": {"sha256": "75e566ef..."},
 "runtime_contract": {"window_len": 10, "patch_grid_hw": [25, 37]},
 "boundaries": {
   "render": {"depth": {"form": "identity"},
              "gaussian_scale": {"form": "identity", "c_gs": 1.0}},
   "metric": {"depth": {"form": "loglinear", "a": -0.0405706428,
                           "b": 0.0146570329, "reference_depth_m": 20.0,
                           "runtime_depth_clamp_m": [0.5, 80.0]},
              "gaussian_scale": {"c_gs": 1.0}}},
 "limitations": {"paired_gaussian_scale_over_depth_ratio": 0.7963,
                 "similarity_consistent": false}}
```

语义（全部相对 **direct teacher 空间**，即 gauge 表所在的空间）：

```
z0_metric            = depth_recon * exp(log_metric_scale)
c_depth_metric       = exp(a + b*log(clamp(z0_metric, 0.5m, 80m)/20m))
metric_means         = unproject(depth_recon*c_depth_metric, K_gauge) * exp(log_metric_scale)
metric_scales        = scale_recon * c_depth_metric * exp(log_metric_scale)  # c_gs=1
render_depth         = depth_recon
render_scales        = scale_recon                                           # identity pullback
```

**全局不变式**：render 保持 tokenizer 原生 recon 几何，不施加 pullback；只有跨入米制边界时，
先用同一 `log_metric_scale` 得到未校正的 `z0_metric`，再计算 checkpoint-bound `c_depth(z0)`。
生成的米制相机在 render 前先经 `metric_c2w_to_teacher_anchor_dggt` 相对 trunk 首相机重基，
再换成 DGGT translation；禁止把 `+z-up` ego-world rotation 直接当作 teacher-atlas 世界。

三个必须写清的性质：

1. **`c_depth`/`c_gs` 无量纲，标定 provenance 与 teacher gauge 表正交，但 runtime 求值并非函数上
   与 gauge 无关。** 系数是 recon/direct 的比值之比；换 tokenizer 不影响 gauge 表，也不允许重算
   teacher gauge。可是 production loglinear profile 明确在
   `z0_metric=depth_recon*exp(log_metric_scale)` 上求值，所以换 gauge 会改变当前像素的
   `c_depth(z0_metric)`。在未触发 clamp 的区间，把 `log_metric_scale` 加 `δ` 后，最终几何尺度响应为
   `exp((1+b)δ)`；v1 的 `b=0.014657...`，例如 `δ=log 2` 时约为 `2.0204×`。这是冻结 LiDAR 公式的
   预期行为，不应改成 identity 来迁就相似不变式。
2. **修正对象是「未标定的那个」。** depth 由 lidar 标定，scale 没有独立米制参照，所以往 direct 方向修 scale，
   而不是反过来动 depth 去迁就 scale。
3. **审计范围**（照抄复测的自我限定）：这只证明 `direct → recon` 不是相似变换，**不证明 direct 的 Gaussian
   本身以 Waymo 米制标定**，也不替代完整 renderer 的 opacity/quaternion/compositing 测试。

**标定实现只有一个，调用作用域必须显式**：`dggt/utils/scene_gauge.py` 提供共享 helper；
米制导出、模块 C 与 metric 诊断调用 `boundary="metric"`，训练/推理 render 调用
`boundary="render"`，v2 两者都断言返回 identity。v1 schema/profile 在 loader 层直接拒绝，
更不得接进 `rgb_render_loss.py::_decode_geometry`。

#### 1b-4. tokenizer v2 正式复测与最终分支（2026-08-05，方案 A）

本节**只新增 v2 结论**；上面的 v1 数字、原始结果和
`data/scene_gauge/pullback_75e566ef.json` 均不覆盖，但全部只读归档、不可被 runtime 加载。v2 checkpoint 是
`logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt`，完整 SHA-256 为
`d63b34f7b1193ed7da399f953db504cfadb4f98dce2519854227a0f44714c8e8`。预检确认
`global_step=max_steps=100000`，当前 `JointSceneTokenizer` 严格加载 460 个 state key、
741,022,336 个参数，模型和 optimizer tensor 全部 finite；DGGT checkpoint SHA-256 为
`352652738a5480b8d3ee9dd521ce07b528e5a297bd3feca4d07427dac6d87def`。

> **v2-only 收口说明（2026-08-05）**：下面的正式数值和 provenance 来自已经完成的
> Gaussian schema 2.3.0 与 LiDAR schema 2.1.0 运行。随后只删除了 Gaussian 工具中未启用的
> render/PSNR/`c_gs` 扫描路径，并把工具合同升级为 2.4.0；按用户决定没有再次执行 90-trunk audit
> 或 LiDAR selection。新代码只做单元测试、CPU synthetic 和单 trunk CUDA 路径检查，不能把既有
> JSON 说成由 2.4.0 脚本生成，也没有生成新的正式 result/script SHA。

checkpoint 的训练参数保存了 `min_frames=10`、`max_frames=14`、`sample_window=20` 以及
`lambda_head_anchor=0.6`、`lambda_gs_scale_sim=0.3`、`lambda_depth_log_bias=0.2`，但**没有显式保存
objective/version 字段，也没有保存固定的 production tokenizer `window_len`**；因此不能把这两项描述成
checkpoint 自证。正式结果和 production artifact 另行 fail-closed 绑定 10 帧窗口。更重要的 limitation 是：
v2 训练日志/目录完全不在本机，无法复核 `gs_scale_sim_ratio`、depth log bias、有效 support
count/fraction、三项新增 loss 的训练轨迹，也无法从日志排除 NaN、空 support 被记成 ratio=1 或
raw/cache 混写；这里的接受依据是 strict-load/finite 检查和独立正式复测，**不是对缺失训练日志或
trainer 日志实现的事后背书**。

正式协议固定为 scenes 300–329 × trunks 0/1/2 = 90 个完整 29 帧 trunk；每个 trunk 只用起点
`[0,5,10,14,19]` 的五个 10 帧 tokenizer 窗口，Aggregator 看完整 29 帧，DepthHead/GaussianHead
关闭 autocast 后以 FP32 运行。calibration scenes 300–319 只拟合 identity/constant/loglinear，
selection scenes 320–329 只在 identity 与冻结候选之间二选一；selection 不得回写 `a/b`、clamp、
support、窗口或统计方式。GS practical-equivalence margin 在查看 v2 结果前冻结为 `[0.95,1.05]`，
PSNR 未运行、也未用于 `c_gs`。

主 support `primary_static_nonsky_opacity_0p05` 的 scene-balanced 结果如下。ratio 的 point 是
30 个 scene 的 median-log；括号是 scene IQR。三轴行按同一 window→trunk→scene 层级分别聚合
`axis_log_ratio_median_of_frame_medians`：

| 量 | tokenizer v1（历史 D4） | tokenizer v2 | v2 判定 |
|---|---:|---:|---|
| `depth_recon/depth_direct` | 1.03989 | **1.04189**（IQR 1.00495–1.05669） | 仍有描述性偏移，不能据此越过 LiDAR gate |
| Gaussian scale x/y/z | 0.83805 / 1.00000 / 0.76931 | **1.06666 / 1.00000 / 1.03631** | 三轴已不再呈 v1 的系统性缩小 |
| Gaussian 三轴几何平均 `scale_recon/scale_direct` | 0.82894 | **1.03311**（IQR 1.01923–1.04466） | 描述性统计 |
| 配对 `GS/depth` | **0.79640** | **0.99985**（IQR 0.98205–1.01013） | point 在 margin 内 |

配对 ratio 的 scene-only bootstrap 95% CI 为 **`[0.99015,1.00753]`**；point 与整段 CI 都位于
预冻结 `[0.95,1.05]`，故 GS/depth practical-equivalence gate **通过**。这是与 v1 最关键的逐项
变化：v1 的 0.796 缺陷已由 v2 从源头修复，到 1 的误差减少约 99.93%；不需要、也不允许用
PSNR 拟合 `c_gs`。30 个 per-scene ratio 的范围是 `[0.95311,1.10088]`，只有 scene 306 高于
1.05、没有 scene 低于 0.95；等价结论来自预注册的跨 scene point + bootstrap CI gate，不表示每个
scene 都逐一落入 margin。
另一个 estimator 的 90-trunk same-cell case-balanced `depth_recon/depth_direct` mean/p50 为
`1.03358/1.03685`（v1 mean 为 `1.03048`），说明 depth 描述性偏移没有改善，也未凭配对 gate
自动消失，仍必须交给独立 LiDAR gate。

calibration split 的 depth profile 在 identity/constant/loglinear 中冻结为 loglinear：
`a=-0.028007614619865877`、`b=-0.04383812022323029`、`c(20m)=0.9723809624616624`，自变量仍是
**未校正 reconstructed metric depth**，runtime clamp `[0.5m,80m]`。calibration depth 分层为：

| 未校正 reconstructed metric depth | correction | `recon/direct` | scene count |
|---|---:|---:|---:|
| 0–5 m | 1.11662 | 0.89556 | 6 |
| 5–10 m | 0.96736 | 1.03374 | 20 |
| 10–20 m | 0.97372 | 1.02699 | 20 |
| 20–40 m | 0.95624 | 1.04576 | 20 |
| 40–80 m | 0.94300 | 1.06045 | 20 |

近场 0–5 m 只有 6 个 scene 且方向反转，是 profile 的明确 limitation。`c_gs` 的 calibration
constant/loglinear 分别为 `a=-0.0147907,b=0` 与 `a=-0.00793792,b=0.0549248`，但冻结候选相对
identity 没有达到 2% LOSO 改善，因此 `c_gs` profile 自身也选择 identity。

selection split 的正式 LiDAR 主口径（Phase-1a valid，all LiDAR）给出：

| 口径 | case / scene | identity AbsRel | frozen loglinear AbsRel | scene mean `identity-candidate` | scene-bootstrap 95% CI | improved scenes |
|---|---:|---:|---:|---:|---:|---:|
| 主口径 | 26 / 10 | **7.762%** | 8.507% | **−0.745%** | **[−1.553%, +0.113%]** | 3/10 |
| 全 30 trunk sensitivity | 30 / 10 | **7.705%** | 8.428% | −0.723% | [−1.521%, +0.132%] | 3/10 |
| valid static/non-sky | 26 / 10 | **7.772%** | 8.573% | −0.802% | [−1.620%, +0.073%] | 3/10 |

主口径 exact sign-flip sensitivity 为单侧 `p=0.93457`、双侧 `p=0.13281`。预注册规则要求
point delta > 0 且 scene-bootstrap CI 下界严格 > 0；两者均不满足，故选择 **identity**。注意这不是
把 calibration fit 改回 identity：冻结 profile 仍作为候选证据保留，只是 selection gate 拒绝部署它。
与 v1 的方向相反：v1 主口径 delta 为 `+0.0066645`、CI `[0.0005170,0.0122477]`、8/10 scenes
改善并选择 loglinear；其 `b=+0.0146570`，而 v2 是 `b=-0.0438381`。这既说明 v2 不需要 depth
correction，也直接证明不得复用 v1 的 `a/b`。
LiDAR result 本身保持 `artifact_role=candidate_v2`、`eligible_for_training=false`；只有下面经过全部 gate
与 smoke 验证的 production freeze 才能把独立文件标成可训练加载。

10 帧 CUDA 0 encode/decode + FP32 DepthHead/GaussianHead/render smoke 也通过：
`depth_recon/direct=0.99928`、`GS_recon/direct=1.00299`、paired `1.00317`、
render-vs-direct PSNR/SSIM/LPIPS 为 `39.296 dB / 0.97892 / 0.01618`，没有明显 reconstruction/render
退化。三项 gate 因而落到**方案 A**：

- render pullback = identity；
- metric-boundary `c_depth` = identity；
- `c_gs=1.0`；
- shared helper 仍要求显式 `boundary="render"|"metric"`，并保留 tokenizer SHA/window/grid/artifact
  hash 强制校验；identity 路径必须是精确 no-op，禁止接入 `rgb_render_loss.py::_decode_geometry`；
- 唯一 v2 production artifact 是 `data/scene_gauge/pullback_d63b34f7.json`，
  `artifact_role=production_pullback`、`eligible_for_training=true`、schema `2.0.0`；clean-cut 只把
  旧的 PSNR 否定字段迁移成显式 `c_gs_recommendation={form: identity, value: 1}`，不改 gate 数值；
- runtime、默认配置和 formal checkpoint provenance 只接受上述 v2 artifact/checkpoint；v1 schema、
  checkpoint、pullback 与兼容 fallback 全部删除或显式拒绝。

**正式证据与代码哈希**（大体积 `runs/*.json` 只保留为本地证据，不加入 git 暂存）：

| 对象 | SHA-256 |
|---|---|
| tokenizer v2 checkpoint | `d63b34f7b1193ed7da399f953db504cfadb4f98dce2519854227a0f44714c8e8` |
| DGGT checkpoint | `352652738a5480b8d3ee9dd521ce07b528e5a297bd3feca4d07427dac6d87def` |
| metric reference JSON | `2416f97b4afed0d9bf33556841cd419574b70dde1598474c5e4cd03899cf112b` |
| `tools/retest_scene_flow_metric_gauge.py` | `9e91dd09c7057d5cf2a04a6027e2bf8088aee6ce400c1121a71ff1c4ae15a3e1` |
| Gaussian audit JSON | `8676b7767f3ddda6097331466dc0db30f0fc8d35ce7e09ecb82a8550b27b95d6` |
| `tools/retest_scene_flow_gaussian_gauge.py` | `ebe3e49867fde426908a393fe3774b6e36fa6a6ff5ec35e7876dfac91984d10d` |
| LiDAR selection JSON | `ab82b2884afd2d40aa6e02a78abcae27185942251288a497ca5bcf281615c2b8` |
| LiDAR JSON canonical self-hash（排除 self 字段） | `46a7066881c07f2ecf6dda942bf3001966f537076c2d1cc2b88dde1040ea1046` |
| `tools/calibrate_tokenizer_pullback.py` | `9a206db00f58cdce870f1c86c85bbe56560bd409cbfc2f8e37e1cdd33a33c0b4` |
| CUDA render smoke JSON | `2519d4e3e21ed5f353fe4951d268f22f0c7d4eeae705b5cb6e29503e01690c89` |
| fixed selection manifest | `ce8290c997c2f9e5c9fd600ebd4178e86d40797077ae2874fb2774a7c1ca8cc6` |
| smoke visual | `c7db9b6a38821e7f7b0d9b1b5f1a83b30f27a07b047a15fdef0e103b13a857a2` |
| `tools/freeze_tokenizer_pullback.py` | `3e96b000245fa74b81e1fa2794ab620d08e1c6e9305c8756430a804a3acce46f` |
| v2 production artifact（clean-cut 前） | `d24e23f77bcd7b51cb022a591ac9cdee3a7108d233f00b2ae9a9ae8ea7d550fb` |

reference JSON 对 DGGT/tokenizer checkpoint 作完整 content SHA-256；对 90 个原始数据输入只保存每 case
的 `sha256(canonical JSON lines of path,size,mtime_ns)` stat manifest，**不是 raw file content hash**。
因此既有正式结果对 checkpoint、当时的脚本和上游结果有完整 content hash，但不能声称原始
Waymo/DGGT array 的每个字节都已 content-hash；这是当前可复现性 limitation。render smoke 还没有
记录独立 script SHA-256 或 elapsed，只对固定 selection、结果 JSON 与 visual 作了 content hash。
clean-cut 后的 production artifact 只做合同字段迁移，本轮按用户要求没有重新计算或登记文件 SHA；
Phase 2 首次消费前必须重新绑定它的实际 artifact SHA。

本次只验收并实施 Phase 1b；Phase 1a 的 teacher gauge 表保持已完成且无需重算。Phase 1c 及其后续
阶段仍按各自 gate 推进，**不得据本节单独宣称整个 Phase 1 已完成**，也未进入 Phase 2。

### 1c. 数据集接线

`WaymoOpenDataset.__init__` 新增 `scene_gauge_path`；`__getitem__` 按 `(scene, start_idx // 29)` 查表，
输出 `input_dict["scene_gauge"]`（3 维）与 `input_dict["scene_gauge_valid"]`（3 维 bool）。
查表键与 `datasets/dataset.py:2176-2202` 已有的 `context_base = (start_idx // trunk_frames) * trunk_frames`
完全一致 —— 不引入第二套 trunk 定义。

---

## Phase 2 — 模型：scene-global gauge 流

模板完全照抄 sky（唯一现存的非逐帧生成流）。

**常量**（`dggt/utils/scene_gauge.py`，新文件）：

```python
SCENE_GAUGE_DIM = 3            # [log_metric_scale, log_tan_half_fovx, log_tan_half_fovy]
SCENE_GAUGE_REPRESENTATION = "dggt_teacher_log_metric_scale_logfov_v1"
SCENE_GAUGE_STATS_VERSION = "scene_gauge_per_channel_v1"
GAUGE_MROPE_TEMPORAL_OFFSET = 15100
```

配套 `normalize_scene_gauge` / `denormalize_scene_gauge`（照 `camera_generation.py:342-371`）、
`metric_c2w_to_dggt(c2w, log_scale)` / `dggt_c2w_to_metric`、
以及 `gauge_to_pose_enc_fov(gauge)`（把两个 `log tan` 通道还原成 `rgb_render_loss.py:626` 断言的
`[B,S,9]` DGGT `pose_enc` 的 FOV 段，注意 DGGT 的通道顺序是 `[..., FOVy, FOVx]`）。

**`dggt/models/scene_flow.py`**：

- 模块（照 `:1092-1099` sky 的写法，但用 `ChannelScale` 而非 `RMSNorm` —— 低维物理量的范数本身有意义）：
  `gauge_gen_norm = ChannelScale(3)`、`gauge_gen_proj = nn.Linear(3, hidden)`、
  `gauge_gen_decoder = (RMSNorm, Linear, SiLU, Linear→3)`、`gauge_gen_modality_embed`；
  decoder 末层零初始化（照 `:1224-1227`）。
- `_build_gauge_generation(z_t, gauge_gen_tokens)` —— **只取 `b`，不取 `s`**，照 `_build_sky_generation`（`:2917-2949`）。
- `_gauge_position_ids`：单 token，`pos = (15100, 15100, 15100)`。避开视频带 `[0,15000)`、sky 球面 `15000±8`，
  且 < `rope_max_position` 16384。
- 序列改为 `video | camera | sky | gauge`，`:3237-3250` 拼接、`:3298-3309` 的累积偏移各加一段。
- 解码：`gauge_out = self.gauge_gen_decoder(F.silu(gauge_hidden + t_base)).reshape(b, 1, 3)` → `result["gauge"]`。
- 新增 buffer `gauge_mean` / `gauge_std` / `gauge_stats_valid`，配 `require_gauge_stats()` /
  `normalize_gauge()` / `denormalize_gauge()`（照 `:1410-1440`）。

**关键耦合（回答「用最合适的方式和其他变量交互」）** —— 照 `sky_context` 加进 `sky_mask_cond` 的现成写法（`:3341-3348`）：

```python
gauge_context = gauge_hidden if gauge_gen_len > 0 else enc_video.new_zeros((b, 1, hidden))
cond = self.s_projector(F.silu(enc_video + t_base + gauge_context))   # :3310
```

这条线做两件事：**生成的几何显式以生成的尺度为条件**；**video flow loss 的梯度反向流进 gauge token**。
加上 gauge token 通过 encoder 全注意力看到全部 video token，形成双向耦合 ——
这是让 `log s` 准的主要机制，而不是靠一个旁挂的回归头。

**机制假设与它的证伪条件（写进论文，也写进训练日志）**：
仅由米制轨迹形状无法解析 teacher 的 gauge（这是 Context 第二节第 3 条收窄后的正确表述），
所以 gauge 必须从**图像内容**里拿线索（车辆尺寸、车道宽度）。
若真如此，gauge 预测应显著优于「输出训练集 `log s` 均值」这个 marginal-prior baseline。
Phase 4 强制每次训练都打印这两个数的对比 —— **假设错了要看得见。**

**不把 gauge 加进相机解码条件**：相机现在是米制的，本就不该依赖它，保持解耦。

**为什么 gauge 是 3 维而不是 4 维（不把 tokenizer 偏差放进来）**：
`c_depth`/`c_gs` 是冻结 tokenizer 的确定性属性，不是逐场景未知量，生成它等于让模型去学一个常数。
另外 **FOV 通道天然免疫往返偏差** —— 它由 gauge token 直接生成，不经过 tokenizer decode。
而 K 在均匀深度缩放下保持有效（缩放只沿射线移动点，不改射线方向），
实测帧内 depth ratio 的 log-MAD 仅 0.0122，均匀性足够支撑这一点。

---

## Phase 3 — 相机目标米制化（v4）

**`dggt/utils/camera_generation.py`**：

- `CAMERA_GENERATION_DIM: 11 → 9`，`CAMERA_GENERATION_REPRESENTATION = "waymo_metric_relative_se3_rot6d_v4"`，
  `CAMERA_TARGET_SPACE = "waymo_metric_camera_to_world"`，`CAMERA_TARGET_SOURCE = "waymo_gt_extrinsics"`，
  `CAMERA_STATS_VERSION` 升版。
- **删除** `camera_state_from_dggt_pose_enc`，新增 `camera_state_from_waymo_c2w(c2w, anchor_to_world)`：
  `rel[t] = inv(anchor) @ c2w[t]`，`state[0] = rel[0]`（按构造是单位阵）、`state[t] = inv(rel[t-1]) @ rel[t]`，
  输出 `[平移(3, 米), rot6d(6)]`。同时**推翻并重写**现有 docstring 里
  「Waymo 参数绝不可进入目标空间」的不变量 —— 那是在不知道两者关系时定的。
- `decode_camera_trajectory`：去掉 FOV 段；`pose_encoding` 的 FOV 由调用方从 gauge 提供
  （`gauge_to_pose_enc_fov`）。complete / delta-only 的判定逻辑（`:185-216`）**完全不动**。
- `camera_geometry_loss`：去掉 `camera_log_fov` / `camera_acceleration_fov` 两项与 `fov_weight`。其余 6 项不变。

**为什么这是对的（论据换成更强的那条）**：不是「残差可忽略」——
实测残差是 0.20°–0.37° 旋转、1.33% 形状，并不可忽略。真正的论据是**残差的性质变了**：

| | v1 目标（DGGT 位姿） | v2 目标（Waymo 米制） |
|---|---|---|
| 与条件的关系 | `target = s · condition`，`s` 逐 trunk 随机且**不可观测** | 恒等映射 |
| 误差性质 | **不可约的随机乘性噪声**，CV 23.5% | **固定、可测、有界**的重建差异，且逐 trunk 已记入 gauge 表 |
| 可控性实验 | 数学上做不了 | 可以做 |

**`train_scene_flow_pretrain.py:5548-5570`**：目标不再来自 `camera_head`。改为从 batch 已有的
`camera_to_world_corrected` + `camera_trajectory_anchor_to_world_corrected` 构造；
`camera_previous_c2w` 改用 batch 里已有的 `camera_previous_to_world_corrected`
（`datasets/dataset.py:2264-2275`，**不需要改数据集**）。`camera_head` 只在 Phase 1 离线脚本里用于诊断，
训练主循环可以完全不跑它。

**条件与目标对齐（相机可控性的关键）**：共享 role-aware helper 先构造与生成目标完全相同的
9D 米制 camera state，再按 anchor/delta 角色使用同一套 `camera_anchor_mean/std` 与
`camera_delta_mean/std` 归一化，并替换 20D condition 的 `[..., 9:18]`。这里不存在旧的
`delta_t/10` 或 `translation_scale=10` 路径；完整 anchor 窗与 delta-only 窗都沿用全局 frame role，
后者显式使用 preceding metric pose。于是「复现给定轨迹」在参数化层面就是恒等映射，
不是一个要学的仿射变换。

**渲染换算（含 D3 gate）**：`compute_rgb_render_loss` 的 `render_pose_enc_dggt` 要求 `[B,S,9]` DGGT pose_enc
（`rgb_render_loss.py:626` 有硬断言）。装配方式：

```
c2w_dggt = metric_c2w_to_dggt(decode_camera_trajectory(...), log_metric_scale)   # 只除平移
pose_enc = assemble(c2w_dggt, fov_from_gauge)                                    # 9D
```

- 训练时 `log_metric_scale` 与 FOV **一律取离线表的 GT**（精确，且避免早期噪声 gauge 污染 render loss）；
  新增 `--render_use_predicted_gauge`（默认 false）留作后续实验。
- **D3 已判定（1.4143/1.4357 dB ≫ 0.3 dB）：`render_camera_space = teacher`，写死为默认。**
  `render_pose_enc_dggt` 用 teacher 空间位姿，米制相机只作为生成目标。
  这是**正确选择而非妥协**：latent 的 flow 目标就是 teacher 空间的 `Encoder(direct tokens)`，
  teacher 位姿才是与它配套的相机；用换算后的米制相机会反过来把几何往 Waymo 世界拉，主动制造偏差。

**D3 的两个必须处理的连带后果（v2 新增）**：

1. **render loss 不再向相机分支回传梯度。** `render_pose_enc_dggt` 变成 detached 的 teacher GT，
   `compute_rgb_render_loss(..., camera_grad_scale=args.rgb_render_camera_grad_scale)` 这条路彻底失效。
   **必须显式处理**：删除该参数，或断言其为 0 —— 留着一个静默无效的旋钮是最坏的情况。
   相机的精度不受影响（它现在有确定性 GT + 几何损失），但——
2. **生成相机与生成几何之间不再有任何光度耦合。** 训练时二者各自对齐各自的 GT；
   推理时二者都来自噪声，**没有任何损失强迫它们一致**。
   这条直接抬高了故事文档 E.4 **模块 C** 的地位：它从「一个加固项」变成
   **唯一负责 camera/gauge ↔ 几何互相一致的机制**。
   在 C 落地前，必须有一个可观测诊断顶上：正式实现是 **generated static-geometry
   reprojection/cycle diagnostic**（生成 depth 前向重投影、目标 depth 采样、反向 cycle），并记入
   run summary。系统没有独立 optical-flow/scene-flow correspondence head，故不得把它命名成
   “相机光流 vs 几何光流”。

**推理侧的正确表述（不要把 1.41 dB 说成质量损失）**：
推理时没有真实图像可比，几何与相机都是生成的，渲染仍然是自洽的。
1.41 dB 衡量的是**冻结 teacher 的世界与米制世界之间的差距**，它在 D3 的完整 29 帧、
指定 source/target 协议下给出**控制保真度上界诊断**（交付轨迹相对请求轨迹差 ~0.2° / ~1.3% 路径长），
而不是画质下降。且由 motion split 可知它随运动累积——静止片段仅 0.19 dB。

---

## Phase 4 — 损失与 trainer 接线

`train_scene_flow_pretrain.py`：

- `build_gauge_rectified_flow_target`：照 `build_camera_rectified_flow_target`（`:667-695`），**共用 video 的 sigma**。
- `--lambda_gauge_flow` 默认 0.1（对齐 `lambda_camera_flow` / `lambda_sky_flow`）：归一化空间的 masked MSE。
- `--lambda_gauge_direct` 默认 1.0（对齐 `lambda_camera_pose`）：对 **denormalize 后的 x-prediction**
  在物理 log 单位上做 smooth_l1，按 `scene_gauge_valid` 逐通道 mask。
  低维物理量上直接监督比纯 flow 损失条件数好得多 —— 这正是相机同时有 flow 和 geometry 两个损失的原因。
- bundle 新增 `scene_gauge_clean` / `scene_gauge_valid`（照 `sky_gen_clean` 在 `:4135-4137`、`:4166-4170` 的位置）。
- 相机损失：`camera_fov_weight` 及其校验（`:6383-6387`、`:6699-6712`）删除；
  `camera_target_state_dggt` 更名 `camera_target_state_metric`，`camera_loss_gt_space` 断言换成新常量。

**必须打印的四个诊断（无梯度）** —— 前两个是 v2 新增，直接决定「这次改造有没有用」：

| 日志名 | 定义 | 为什么需要 |
|---|---|---|
| `gauge_vs_prior_gain` | `\|log s_prior − log s_gt\| − \|log s_pred − log s_gt\|`，`prior` = 训练集均值 | **Phase 2 机制假设的证伪器**。≤ 0 说明模型没从图像里学到尺度，只是记住了先验 |
| `metric_depth_rel_err` | `median(\|exp(log s_pred)·c_depth·depth_recon − lidar\| / lidar)` | 唯一端到端、有物理意义的验收数；把 gauge + `c_depth` + 解码器一起测了 |
| `gauge_log_scale_error` / `gauge_fov_error_deg` | 逐通道绝对误差 | 常规 |
| `gauge_valid_frac` | 有效通道占比 | 监控 mask 覆盖 |

**统计**：`tools/compute_pretrain_feature_stats.py` 增加 gauge 的 per-channel 累积
（照 `:231-336` 的相机 anchor/delta 写法），`dggt/utils/feature_stats.py` 的
`load_all_stats_into_buffers` 增加 `set_gauge_stats`。相机统计因改米制、降到 9 维，**必须重算**。

---

## Phase 5 — 滑动窗兼容性审计

用户明确要求确认不与滑动窗冲突。逐条：

| 改动 | 与滑动窗的关系 | 结论 |
|---|---|---|
| gauge GT 表 | 离线按 `(scene, trunk)` 查表，与窗口起点无关 | ✓ 按构造窗口不变 |
| gauge 采样融合 | 复用 `scene_global_window_weight`（`dggt/utils/sliding_window.py:138-145`），`tests/test_sliding_window_v2.py:26-36` 已证明每帧贡献均等 | ✓ |
| `c_depth(z0)` / `c_gs` | 无量纲 checkpoint-bound profile/常数，逐像素施加，与窗口调度无关 | ✓ |
| 米制相机 anchor | 仍是 clip-global `frame_ids.eq(0)`；`:5681-5685` 的断言与 `decode_camera_trajectory` 的 complete/delta-only 检查不动 | ✓ 语义不变 |
| delta-only 窗口 | `camera_previous_c2w` 来源从 DGGT 改为 Waymo（batch 里已有），结构不变 | ✓ |
| 渲染换算 `exp(log_scale)` | per-row 标量，广播到整个窗口 | ✓ 窗口不变 |
| gauge K 用于 atlas / render | 单一 scene-global 量，整段共用一个 K | ✓ 比现状（teacher 逐帧 K）**更**窗口无关 |
| 米制链路的 Waymo K（box/`in_frustum`） | 不变，本就与窗口无关 | ✓ 不受本次改造影响 |
| placement v2 | 只改通道维度，`_slice_factorized_asset_condition` / `factorized_asset_conditions_by_window` 的窗口语义不动 | ✓ |
| **`c_depth`/`c_gs` 的标定窗口长度** | 它们是 tokenizer 在**给定窗口长度**上的属性；用 10 帧标定、却在 29 帧单窗推理下使用，会有偏差 | ⚠ **`pullback_*.json` 记录 `window_len`，加载时断言等于当前窗口长度**；`OFFLINE_MAX_SINGLE_WINDOW = 10` 已保证离线推理不超过 10 帧，训练窗口也是 10 —— 一致，但必须由断言而非约定保证 |
| 长于 29 帧的重建验证 | GT 尺度本身跨 trunk 漂移 mean 8.2% / p95 21.9% / max 30.8%，且 lidar 与 camera 同向漂移（41/44，Pearson 0.9711）—— 是 frozen VGGT 而非估计量的问题 | ⚠ gauge 指标限制在 ≤29 帧验证内报告，长序列作为已知 limitation 主动写出 |

**长序列 limitation 的正确表述**（写进论文，不要等审稿人问）：
训练 target 是逐 29 帧 trunk 的，而 teacher 跨 trunk 漂移 8–31%。因此 ≤29 帧生成干净（一 trunk 一尺度）；
**长序列滑动窗生成会得到一个全局尺度，它实际上比 teacher 更自洽，但不存在一致的 GT 尺度可供评测。**

**必须两处都改**：`_cfg_sample_pretrain_latents_sliding`（`:3358-3764`）与非滑窗主体（`:3796+`）
是同一契约的两份平行实现，只改一处会让长短 clip 推理静默分叉。gauge 需要加的五处，逐一对照 sky：
`gauge_z` 初始化（对照 `:3403-3408`）→ 不切片传入 `sf(...)`（对照 `:3526`）→ `_combine_cfg("gauge")`（对照 `:3609`）
→ `scene_global_window_weight` 累积（对照 `:3622-3626`）→ 归一化 + 单次全局 Euler（对照 `:3633-3635`）
→ 返回 `SimpleNamespace`（`:3752-3764`）。

---

## Phase 6 — 推理与米制导出

`inference_scene_flow_pretrain.py`：

- `cfg_sample_pretrain_latents(..., return_gauge=True)`；gauge 为渲染必需
  （用户选了「只生成」，所以没有外部传入路径，也不设开关）。
- `decode_pose_from_camera_features`（`train_scene_flow_pretrain.py:758-779`）现在产出**米制**位姿；
  FOV 从 gauge 取，用 `gauge_to_pose_enc_fov` 重组 9 维 `pose_enc` 供 `_predict_camera_mats` 使用；
  渲染前用 `metric_c2w_to_teacher_anchor_dggt` 重基并换算。
- 解码几何本身保持原样；跨入米制导出时才调用 `scene_gauge.py` 的唯一 shared helper，
  v2 方案 A 的 checkpoint-bound `c_depth=identity`、`c_gs=1.0`。render 调同一 helper 的 identity
  分支并断言；两个 boundary 都是精确 no-op，但仍保留显式 scope 与 provenance 门。
- > **v1 米制边界结论**：loglinear 将 scene-balanced LiDAR AbsRel 从 7.567% 降到 6.901%；
  > 它只记录 v1 当时的审计结论，不修复 GS/depth=0.796，也不是 v2 的最终常数；v2-only runtime
  > 必须拒绝该 profile，不能保留兼容开关。
- **`--export_units {dggt,metric}`（默认 metric）**：
  PLY 导出时 `means` 与高斯 `scales` **同乘** `exp(log_metric_scale)`；
  rotation / color / opacity **不变**。`export_generated_pointclouds`（`:660-754`）里那句写死的
  `"DGGT generated-camera world coordinates"`（`:744`）改为记录实际单位、gauge 值与两个标定常数。
  **这是本次改造对外最直接的能力：第一次能导出米制场景。**
- run summary（`:1373` 附近）记录 `log_metric_scale`、`metres_per_unit`、`fov_deg`、
  `tokenizer_sha256`、`gauge_table_sha256`、`c_depth`、`c_gs`。
- `sync_args_from_model`（`:590-601`）与配置回填（`:513-520`）加入 gauge 字段。

`inference_scene_flow.py`（edit 路径）：不生成相机/sky/gauge，但仍需同步
`SCENE_FLOW_CONFIG_COMPAT_FIELDS`（`:750-772`）并强制加载 hash 绑定的 pullback artifact。
它的 tokenizer decode / render 必须显式走 v2 `boundary="render"`，米制导出或米制断言走 v2
`boundary="metric"`；方案 A 下两者都是 identity。tokenizer v1 checkpoint/pullback/provenance 在
入口拒绝，不存在 legacy loglinear 或兼容 fallback。

---

## Phase 7 — s2.5.1 目标条件 v2

`dggt/utils/factorized_asset_condition.py:671-696`。原则：
**每个米制量拆成「无量纲方向」+「单个 log 幅值」，并显式补上尺度不变的比值。**
这样未知尺度只影响一个加性常数，而不是混在三个 xyz 通道里。

`PLACEMENT_STATE_DIM: 12 → 16`：

| ch | 内容 | 性质 |
|---|---|---|
| 0:3 | `unit_direction_anchor` | 无量纲，passthrough |
| 3 | **`log_z_depth`**（沿相机光轴的深度，**不是**欧氏 range） | 米制，标准化 |
| 4:7 | `log_box_lwh` | 米制，标准化 |
| 7 | `log(box_diag / z_depth)` | **尺度不变的角尺寸**，passthrough |
| 8:10 | `sin/cos yaw` | 无量纲，passthrough |
| 10:13 | `unit_velocity_dir`（零速时置零） | 无量纲，passthrough |
| 13 | `log_speed`（`clamp_min(1e-3)`） | 米制，标准化 |
| 14 | `tanh(speed / z_depth)` | 尺度不变、有界，passthrough |
| 15 | `in_frustum` | 无量纲，passthrough |

**签名变更**：`build_placement_state` 需新增 `camera_to_anchor` 入参才能算 z-depth。
已确认它在调用点 `factorized_asset_condition.py:762` 的作用域内可得
（紧邻的 `project_anchor_boxes_to_patch_bboxes(..., camera_to_anchor, ...)` 于 `:753-761` 就在用它），
**不需要任何数据集或上游改动**。

标准化通道只剩 `{3, 4:7, 13}`；`tools/compute_pretrain_feature_stats.py:243-266`、`:337-354` 里
强制 `mean=0,var=1` 的 passthrough 索引集合相应更新。`asset_placement_mlp` 输入 12→16。
`build_placement_state` 现有的与 `object_to_anchor` 交叉校验（`:707-742`）保留并适配新布局。

**为什么这解决了「条件与生成不同尺度」**：尺度不变的 passthrough 通道
（`0:3`、`7`、`8:10`、`10:13`、`14`、`15`）共 **11 个**，
**不需要 gauge 就可用**，模型立刻拿到无歧义的方向/形状/相对速度；
剩下需要统计标准化的米制通道（`3`、`4:7`、`13`）共 **5 个**，未知尺度只表现为
这些 log 幅值上的**同一个加性常数**，
而这个常数正是同一次前向里生成的 `log_metric_scale`（gauge token 与 asset token 在 encoder 里互相可见）。
v1 里 `center_anchor` 的三个 xyz 通道各自被全局统计归一化，`s=25` 与 `s=64` 的场景归一化后完全相同 —— 这是根因。

**`target_bbox_patch` 保持真实 Waymo K，不动**（v2 更正 —— 本文档上一版曾要求改成 gauge K，那是错的）。
`datasets/dataset.py:2054-2071` 用米制 `object_to_anchor` + 米制 `camera_to_anchor` + 真实 Waymo K 投影，
是一条**自洽的全米制链路**，产出的是真实目标在真实图像里的正确像素足迹。
把它改成 gauge K 等于用 DGGT 内参去投影米制 box —— 正好是混用两套约定，会把一个目前正确的量改坏。
`in_frustum` 同理保持不变。

**为什么距离通道用 `log_z_depth` 而不是欧氏 `log_range`**（这是各向异性的直接推论）：
在各向异性映射 `p_dggt = diag(k_x,k_y,1)·p_metric/s` 下，**只有 z 分量是纯标量**；
欧氏 range 是 `sqrt(k²X²+k²Y²+Z²)/s`，随离轴角变化，**不是**纯标量。
换成 z-depth 后，这个通道在两套约定之间只差 `1/s` 一个因子，与 gauge 的语义完全对齐。
第 7、14 两个比值通道随之统一到同一个距离量，整个向量里只有一种距离概念。

**于是 `target_bbox_patch` 与 `log_z_depth` 构成一对「约定无关」的完备参数化**，值得显式说清：
- 像素坐标同时被两套约定接受（Waymo 侧由 `K_waymo` 投影米制 box 得到；DGGT 侧由 `K_gauge` 反投影得到同一像素）——
  **横向位置由像素钉死，各向异性没有作用余地**；
- `log_z_depth` 给出沿光轴的距离，经 `exp(log_metric_scale)` 纯标量换算 —— **纵向位置由 gauge 钉死**。

两者合起来唯一确定目标在 DGGT 空间的位置，**全程不需要各向异性换算**。
这也是为什么位置相关的监督（`bbox_patch_mask`、RGB render、距离误差）不受本节发现影响。
`unit_direction_anchor`（米制空间方向，受 `k_x/k_y` 影响）与它们**互补而非冗余**，一并保留。

> **评测口径的直接后果**：报 **2D IoU（图像）+ 米制距离误差**，而不是裸的 3D IoU。
> DGGT 空间里的车天生横向窄 25%，裸 3D IoU 会把一个正确的结果判成差 —— 那是约定差异，不是模型误差。
> 若必须报 3D IoU，先套 `metric_box_to_dggt` 的闭式各向异性映射再算。

---

## Phase 8 — sky atlas 与视锥一致性

**要修的是什么（v2 更正）**：不是「atlas 用错了 Waymo K」。atlas 是纯方向采样
（`_build_directional_sky_tokens_from_images`，`:2007-2013`：world dir → cam dir 仅旋转 → 像素），
写入与读出用同一个 `K` 时，任何 `K` 都只是环境图的一个固定重参数化，会自行抵消。
真正的 bug 是**两侧 `K` 不同源**：

| | 修复前 | Phase 8 / post-review 后 |
|---|---|---|
| atlas 构建（`train_scene_flow_pretrain.py`） | `camera_pose_gt_dggt` 的 **teacher 逐帧 K** | **teacher c2w** + **trunk 常量 gauge K**（训练取表 GT） |
| render 读出（`rgb_render_loss.py:275,461-475`） | `render_pose_enc_dggt` 的 **生成 K**（σ=9.4°） | 同一个 **gauge K**（推理取生成值） |

同时 teacher 逐帧 K 在 trunk 内还有 0.26° 抖动 —— 同一世界方向在不同帧落进不同 atlas cell，
**target 自身就不自洽**。改成 trunk 常量 gauge K 后这一项也一并消除。平移在此无关，尺度问题不出现。

**2026-08-02 post-review 坐标系更正**：上一版表格把 atlas 的旋转写成 Waymo rotation 是错的。
`_sky_direction_grid` 明确定义在 teacher/OpenCV camera-anchor 世界（identity camera 下 image-up 为
world `-y`），Waymo `camera_to_world_corrected` 则位于 clip-start ego 世界（`+z` up）。二者不能只靠
平移尺度换算。训练 atlas 现在与 D3 render 一样复用 teacher c2w，只把 FOV 替换成 trunk-constant
gauge K；开放生成时，9D 米制相机先相对 `camera_trajectory_anchor_to_world_metric` 重基到首相机世界，
再把平移换成 DGGT units。这样 atlas 写入、训练 render 读出、validation/offline 生成读出共享同一
`-y-up` teacher-atlas 世界，首帧严格为 identity。

- **两条链路，各自一个 K，永不交叉**（取代上一版的「全局单一 K」原则，那条是错的）：

  | 链路 | K | 成员 |
  |---|---|---|
  | 米制链路 | **真实 Waymo K** | `project_anchor_boxes_to_patch_bboxes`、`in_frustum`、`target_bbox_patch` |
  | DGGT 链路 | **gauge K** | depth 反投影、RGB render、sky atlas 构建与读出 |

  加 lint 级检查：禁止任何路径把米制量喂给 gauge K，或把 DGGT 量喂给 Waymo K。
- **新增 `metric_box_to_dggt`**（放进 `dggt/utils/scene_gauge.py`，与 `metric_c2w_to_dggt` 并列）：
  编辑路径需要把 Waymo box 当作 DGGT 空间的显式 3D 体积（圈选/删除/搬移高斯、按米制坐标插入 asset）时，
  **必须用各向异性形式** `p_dggt = diag(k_x, k_y, 1) · p_metric / s`，
  `k_x = tan(FOV_dggt_x/2)/tan(FOV_waymo_x/2)`（由 gauge FOV 与 Waymo 内参算出，实测 ≈ 0.748/0.772），
  **不是** `p_metric / s`。且必须在**相机坐标系**下施加（`diag` 只对某一帧的光轴定义），
  跨帧的 box 需逐帧换算后再合并 —— 函数签名强制传入 `camera_to_anchor` 以防误用。

---

## 附录 A — tokenizer 侧根治（**v2 已训练并通过正式复测**；本节 2026-08-05 更新）

> 2026-08-01 版记录为「根因已定位、修复已进代码，只差重训」。该历史判断保留；2026-08-05
> 已取得 step-100000 v2 checkpoint，并按 1b-4 完成哈希隔离复测。由于训练日志完全缺失，不能补写
> loss/support 的训练轨迹；最终接受以 checkpoint finite/strict-load、CUDA smoke 与正式 gate 为准。

**根因（都在 `train_tokenizer.py`，不是架构问题）**：

1. `gs_anchor` 用**一个** per-sample std 归一化全部 11 个 `gs_map` 通道。该 std 由 rgb（~0.29）
   与 quat（~0.50）决定，而三个**线性** Gaussian scale 只有 ~1e-4 —— scale 误差被除以一个大约
   3700× 过大的数，一个 20% 的 scale 误差对 loss 的贡献小了约 **1.2e7 倍**。
   **这三个通道实际上从未被监督过。**
2. `gs_anchor` 与 `geom_anchor` 相互独立，**没有任何项约束二者的比值** —— 而光栅化器在几何上
   唯一在乎的就是这个比值。`render_anchor` 也钉不住它：Stage-B 从 step 0 就开着 render、
   权重 0.5 跑了 40k 步，比值仍是 0.796，因为**一个偏小的 splat 配上抬高的 opacity 渲染结果几乎一样**。
3. `geom_anchor` 是绝对误差判据，容忍均匀乘性偏移 —— 于是收敛到一个带 3% 尺度偏移的解。

**对应三项新损失（已在 `train_tokenizer.py` 中实现）**：
`gs_channel_group_huber_loss`（rgb / opacity / scale / quat **各按自己的 std** 归一化）、
`--lambda_gs_scale_sim`（配对同像素项，直接惩罚
`E_p = mean_axis(log(scale_recon/scale_direct)) − log(depth_recon/depth_direct)`）、
`--lambda_depth_log_bias`（惩罚 log 域的**偏移**而非逐像素误差）。
架构不变，但旧 checkpoint 与新目标不可比 → **从 0 重训**
（`train_tokenizer_ppu_dlc.sh`，Stage-A 100k + Stage-B 40k）。
训练时应盯 `gs_scale_sim_ratio`，它是被审计量的 `exp()`，目标是收敛到 1.0。本机没有 v2 训练日志，
所以该轨迹不可审计；独立 30-scene 复测得到 paired point `0.99985`、scene-bootstrap 95% CI
`[0.99015,1.00753]`，通过预冻结 practical-equivalence gate。

**与本计划的关系（这正是 Phase 1a/1b 拆分的回报）**：

| | 受 tokenizer v2 影响吗 |
|---|---|
| gauge GT 表（Phase 1a，teacher 空间） | **完全不受影响，不需要重算** |
| `c_depth` / `c_gs`（Phase 1b） | v2 已重新标定并通过 selection：metric/render depth 均 identity，`c_gs=1.0`，代码路径退化成严格 no-op |
| 相机 / placement / atlas（Phase 3/7/8） | 不受影响 |

**因此 Phase 1b 的角色变了**：它从「修复手段」变成 **(a) v2 未完全收敛时的兜底，(b) 对任何
已部署 tokenizer 的常设审计**。标定机器仍然要建 —— 它正是**验证 v2 是否真的成功**的那把尺子。
计划中「加载时断言 tokenizer SHA-256 匹配」的要求现在是 clean-cut 门：只允许冻结的 v2 SHA，
而不是为 v1/v2 双版本共存提供兼容性。

**执行顺序更新**：v4 曾冻结「Phase 1–8 使用 v1 production artifact 连续完成、不等待 v2」作为过渡策略，
该历史决定仍有效且不改写。现在 v2 已作为新的哈希隔离对象完成 D4/Phase 1b LiDAR gate 与 paired
`GS/depth` 审计，并生成 `data/scene_gauge/pullback_d63b34f7.json`；新的 Scene Flow
训练/推理 provenance 只使用 v2 artifact。v1 文件不就地改写、只作审计归档，所有 v1 tokenizer、
pullback 和旧 Scene Flow checkpoint 都由生产入口 fail-closed 拒绝。

---

## 兼容性（干净切断）

采用 **v2-only clean cut**。升版本号后，现有 `validate_scene_flow_checkpoint_config`（`train_scene_flow_pretrain.py:492-560`）
会对旧 checkpoint 明确报错：`camera_generation_representation` 门（`:509`）、
`metric_gauge_provenance` 块、必需 state-dict key 集合（需含 gauge 的 key）。旧的
`camera_dggt_provenance` 不作兼容 fallback。

需要同步的清单：

- `SCENE_FLOW_CONFIG_COMPAT_FIELDS` 两处：`train_scene_flow_pretrain.py:384-413`、`inference_scene_flow.py:750-772`
- `save_checkpoint`（`:1297-1356`）的 provenance 块加 **gauge 表哈希 + tokenizer SHA-256 + `pullback_*.json` 哈希**
- **新增强制校验**：加载 `pullback_*.json` 时断言其 `tokenizer_sha256` 等于
  `--tokenizer_ckpt_path`（`:6169`、`:6800`，经 `load_scene_tokenizer_state_dict_strict` 于 `:289-297` 加载）
  的实际哈希，且 `window_len` 等于当前窗口长度；此外 schema/generation 必须是 v2，SHA 必须是
  `d63b34f7...`。v1 schema、`t0_v1`、`75e566ef...` 与旧 artifact path 无条件拒绝，不能 fallback。
- feature stats 文件必须重算（相机降维+米制、gauge 新增、placement 重参数化）。
  **`target_bbox_patch` 的分布不变**（它保持 Waymo K），但它所在的 placement 向量整体重排，所以仍在重算范围内

**必须重跑**：feature stats、全部 baseline 数字。

> **环境坑**：`pretrained/model_latest_waymo.pt` 是失效 symlink（目标文件名多了下划线）。
> 用 `/data/lyy_dataset/model/dggt/model_latest_waymo.pt`（或复测所用的
> `/data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt`），不要静默沿用 symlink。

---

## 需要修改的关键文件

| 文件 | 改动 |
|---|---|
| `dggt/utils/scene_gauge.py` | **新增** —— 常量、归一化、米制↔DGGT 换算、`gauge_to_pose_enc_fov`、**`apply_pullback_calibration`（唯一实现）**；runtime 只接受 v2 production schema，identity 为精确 no-op，v1 直接拒绝 |
| `tools/compute_dggt_scene_gauge.py` | **新增** —— teacher 空间离线 GT 表 |
| `tools/calibrate_tokenizer_pullback.py` | **新增** —— v2 支持 calibration split 选出的 identity/constant/loglinear；selection 只比较 identity 与冻结候选；输出 `candidate_v2` 且不可训练加载 |
| `tools/freeze_tokenizer_pullback.py` | **新增** —— 只将通过 gate 的 v2 证据冻结成严格 production artifact；v1 文件仅作外部历史证据，不保留 v1 production 生成/授权分支 |
| `lyy_tools/verify_fov_consistency.py` | ✅ **D3 已完成**（相机换算 arm） |
| `tools/retest_scene_flow_gaussian_gauge.py` | ✅ 正式数值来自已归档的 2.3.0 运行；当前 2.4.0 工具保留 scene-only practical-equivalence CI，并已删除 render/PSNR/`c_gs` 扫描实现 |
| `train_tokenizer.py` + `train_tokenizer_ppu_dlc.sh` | v2 step-100000 已用三项新损失训练并通过独立 gate；本轮不把额外 trainer/延训改动纳入 Phase 1b 交付，训练日志仍不可得 |
| `dggt/losses/rgb_render_loss.py` | `render_pose_enc_dggt` **改用 teacher 位姿**（D3）；`_decode_geometry` 的 pullback 在 render 路径**恒为 identity**；删除失效的 `rgb_render_camera_grad_scale` CLI，底层兼容参数只接受 `0.0`，非零 fail-fast |
| `dggt/utils/camera_generation.py` | 11→9 维、米制、删 FOV、推翻旧不变量 |
| `dggt/utils/camera_condition.py` | 去 `translation_scale=10`，与生成共用统计 |
| `dggt/models/scene_flow.py` | gauge 流 + RoPE + 序列偏移 + `cond` 耦合 + buffer |
| `train_scene_flow_pretrain.py` | 目标构造、gauge 损失、两个采样器、sky atlas 内参、provenance 校验 |
| `inference_scene_flow_pretrain.py` | gauge 解码、pullback、米制换算、`--export_units` |
| `inference_scene_flow.py` | compat 字段；formal train/inference 的 decoder/render 始终显式走 v2 `boundary="render"`，只有米制导出或米制断言走 v2 `boundary="metric"`；v1 provenance 拒绝 |
| `dggt/utils/factorized_asset_condition.py` | placement v3（沿用 v2 的 12→16 布局，修复 z-depth stats 坐标系并将 motion ratio 有界化）。**`target_bbox_patch` / `in_frustum` 的 K 不动**（保持 Waymo，见 Phase 7） |
| `dggt/utils/feature_stats.py` / `tools/compute_pretrain_feature_stats.py` | gauge 统计、相机统计重算、passthrough 索引 |
| `datasets/dataset.py` | `scene_gauge_path` 查表输出 |

### 2026-08-02 当前实现/验证状态

- **历史快照（已由 2026-08-05 clean cut 取代）**：当时 v1 production pullback、scope-aware shared helper、严格 checkpoint/hash/window/grid loader，以及
  dataset gauge 查表和 factorized-v3 placement 已进入当前实现。最新全套回归在 `conda dggt`、
  `CUDA_VISIBLE_DEVICES=0` 下为 **732 passed, 1 skipped**；另有 clean-cut 与定向 CUDA 审计通过。
- 相机 condition 的 9:18 通道现在由同一个 role-aware helper 在 raw pretrain、formal T1、formal offline
  inference 与 external manifest 四条路径构造；真实 cache 的 `[S,V]` front-view 选择、完整 anchor 窗和
  delta-only 窗均有非 identity stats 回归。pretrain→formal warm-start 还会对 checkpoint 自带的
  `pretrain_feature_stats_contract` 做 stats SHA / 10 帧 / 29 帧 context / 25×37 grid fail-closed。
- validation teacher gauge 表已完成：`data/scene_gauge/validation.json` 为 1212/1212、202 scenes、0 errors，
  文件 SHA-256 `5014e5c0ba5bd570c1a3d13e3fd222d15e32fe10276046dda763b7e87d9559fa`；
  metric-scale 有效 1082，失效原因 115 个 frame-CV 与 15 个 ruler-ratio。validation v10 cache 的独立
  coverage 为 33/33 clips，5 个 invalid raw scale 恰由 5 个 production-mean fallback channel 承接，
  结果 SHA-256 `f8b77cac188e626cd1c8f825a82ce0c6b97f9a33ee7ae331e4fa6889bad08099`。
- training gauge 三分片已经按同一冻结数值协议合并成 `data/scene_gauge/training.json`：4787/4787、
  798 scenes、0 errors，metric-scale/FOVx/FOVy valid counts 为 `[4216,4787,4787]`，actor coverage 1662，
  SHA-256 `39e0a32372e616e9aac4aef6109c8329ebdf382c16a913bd9e4d025b984e00af`。random20 结果位于
  `runs/metric_gauge_retest/scene_gauge_training_random20.json`，SHA-256
  `889545cbd9b2753c44a732869ba507a43706e6557e52e7631c3dbe9c5c874f5a`；median LiDAR AbsRel
  1.9833% < 5%，固定 44-pair drift cohort 的 mean/max 为 8.2020%/30.8056%。
- training full-pass 修复版 v4 stats 已以 `.inprogress` 产物通过独立校验后发布到
  `logs/scene_flow_pretrain_1024/feature_stats_pretrain_v4.pt`：4787/4787 trunks、798 scenes、
  44,279,750/44,279,750 latent，`stats_status=complete`、`exact_scene_gauge_scope=true`；camera
  anchor/delta counts `[2416,45454]`，gauge counts `[4216,4787,4787]`，placement count 172605；
  `log_z_depth mean/std=2.998025/1.519740`。stats SHA-256 为
  `f5177c9262c878c1595c0f0e41ebd9cf42680de3676f0fccb789ed3cbc7a9111`，sidecar SHA-256 为
  `e0767b8bb3b86116f3748144b7c306d73ba6229a5568d47eb18a62cdf5d40539`；source contract 明确绑定
  sequence 10、DGGT context 29、grid 25×37、tokenizer/DGGT/training-gauge 三个 SHA，且 `max_batches=null`。
- tokenizer v2 已在同一 4787-trunk scope 从头重算 v5 stats，发布为
  `logs/scene_flow_pretrain_1024/feature_stats_pretrain_v5.pt`：44,279,750/44,279,750 latent、
  `stats_status=complete`、`exact_scene_gauge_scope=true`、`latent_stats_path=null`；不复用上述
  tokenizer v1 latent moments。camera anchor/delta counts 为 `[2418,45452]`，gauge counts 为
  `[4216,4787,4787]`，placement count 为 172183。
- CUDA 0 米制导出 smoke 已重跑通过：camera round-trip max-abs=0、render pullback identity、
  depth factor 范围 `[0.694053,1.051988]`、means/scales 同比 2.0、静态 cycle EPE=0（support 0.8125）。
- 生成相机/静态几何的诊断正式名称为
  **generated static-geometry reprojection/cycle diagnostic**，schema
  `generated_static_geometry_reprojection_cycle_v1`。它已实现独立单测并接到 pretrain inference 输出；
  它不是“预测光流 vs 几何光流”的比较（系统没有独立光流/对应点 head），也不使用 GT 图像。
- 旧 `logs/metric_gauge_one_step_cuda0/` 及其 direct/sliding 输出绑定修复前 v4 stats（SHA `6fbdd3c5...eb81a`）、
  `factorized_asset_v2` 和错误的 `rgb_patch_v2` atlas 世界，现已由 clean cut 明确拒绝；它们只保留为
  历史证据，不再属于当前 production acceptance。
- 2026-08-02 post-review 审计确认并修复 D1–D3：sky atlas 改回 teacher-anchor `-y-up` 世界，开放生成
  相机先按 metric trunk anchor 重基再缩放；stats 工具只接受 `pretrain_camera_to_anchor`；placement ch14
  改为 `tanh(speed/z_depth)`。D4 的 4 个陈旧测试也已按当前真实 API 修复。独立 CUDA 0 脚本
  `lyy_tools/verify_metric_gauge_postreview.py` 通过，结果 SHA-256
  `8803d5941c91e057711ac3c9d197470dec70d2bf6d8ffd27bd675e4d6997da9e`。
- 新 smoke `logs/metric_gauge_postreview_one_step_cuda0/` 使用修复后 v4 stats、启用 sky generation，并成功完成
  一步真实训练：loss/flow/sky-flow/gauge-flow/gauge-direct 为
  `3.4283/1.4045/0.1020/0.6241/0.0219`，LiDAR diagnostic available=1。EMA-only checkpoint 实际
  mmap 加载确认携带 factorized-v3、sky-v3 与修复后 v4 stats SHA；其 SHA-256 为
  `4daae958f043721f47a8e89c94cb5fe3a3b3e7ea7cfaf8586915c6e5ee85d9a6`。完整回归为
  **732 passed, 1 skipped**。

### 2026-08-05 tokenizer v2 / Phase 1b 最终状态

- checkpoint 预检、CPU synthetic、CUDA 0 真实 10 帧 encode/decode + FP32 heads/render smoke、正式
  30-scene/90-trunk Gaussian audit 与 10-scene/30-trunk LiDAR selection 均已执行；v2 选择方案 A。
  生产默认已切换到 `data/scene_gauge/pullback_d63b34f7.json` 并实施 v2-only clean cut；v1 artifact
  和旧结论不覆盖但只作审计记录，不能进入任何训练/推理 runtime。
- 上述正式 audit/LiDAR 运行发生在 v2-only 工具清理之前；清理后按用户决定不重跑正式结果链。
  当前合并定向回归为 **218 passed**，另有 CPU synthetic 与 CUDA 单 trunk 路径检查通过；严格 loader
  会从 per-scene rows 重算 primary Gaussian/LiDAR bootstrap CI，并拒绝手工构造的 v1 calibration；这些验证
  不能登记为新的正式 audit provenance。当前 2.4 freeze 工具会拒绝归档的 2.3/2.1 输入，因此
  production artifact 是对既有正式证据的合同字段迁移，不是由当前工具链重新生成的产物。
- 环境固定为 repo `/home/dancer/code/dm/dggt`、branch `dev`、conda `dggt`、
  `CUDA_VISIBLE_DEVICES=0`，Python 内使用 `cuda:0`。下面是既有正式结果绑定的历史命令；其中
  `--skip-d4-render-scan` 只属于当时的 2.3.0 脚本，当前 2.4.0 parser 已删除该参数：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dggt python -u \
  tools/retest_scene_flow_metric_gauge.py \
  --scenes 300-329 --trunks 0,1,2 --device cuda:0 --precision bf16 --depth-chunk 4 \
  --checkpoint /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
  --tokenizer-checkpoint logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt \
  --roundtrip-window-starts 0,5,10,14,19 --roundtrip-window-length 10 \
  --bootstrap-repetitions 10000 \
  --output-json runs/metric_gauge_retest/v2_metric_reference_300_329_trunks012_d63b34f7.json

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dggt python -u \
  tools/retest_scene_flow_gaussian_gauge.py \
  --scenes 300-329 --trunks 0,1,2 --device cuda:0 --precision bf16 --depth-chunk 4 \
  --checkpoint /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
  --tokenizer-checkpoint logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt \
  --result-json runs/metric_gauge_retest/v2_metric_reference_300_329_trunks012_d63b34f7.json \
  --skip-d4-render-scan --d4-form-bootstrap-samples 10000 \
  --paired-equivalence-bootstrap-samples 10000 \
  --output runs/metric_gauge_retest/v2_gaussian_gauge_300_329_trunks012_d63b34f7.json

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dggt python -u \
  tools/calibrate_tokenizer_pullback.py \
  --device cuda:0 --precision bf16 --scenes 320-329 --trunks 0 1 2 \
  --bootstrap-samples 10000 \
  --checkpoint /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
  --tokenizer-checkpoint logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt \
  --reference-json runs/metric_gauge_retest/v2_metric_reference_300_329_trunks012_d63b34f7.json \
  --d4-json runs/metric_gauge_retest/v2_gaussian_gauge_300_329_trunks012_d63b34f7.json \
  --output runs/metric_gauge_retest/v2_tokenizer_lidar_metric_gate_320_329_d63b34f7.json

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n dggt python \
  tools/freeze_tokenizer_pullback.py --authorize-v2-production \
  --diagnostic-json runs/metric_gauge_retest/v2_tokenizer_lidar_metric_gate_320_329_d63b34f7.json \
  --gaussian-audit-json runs/metric_gauge_retest/v2_gaussian_gauge_300_329_trunks012_d63b34f7.json \
  --reconstruction-smoke-json runs/tokenizer_v2_cuda0_render_smoke_300_d63b34f7/smoke.json \
  --reconstruction-smoke-selection-manifest runs/tokenizer_v2_fixed_selection_300.json \
  --reconstruction-smoke-visual runs/tokenizer_v2_cuda0_render_smoke_300_d63b34f7/visuals/step_100000_frames_10/00_training_300.jpg \
  --reference-json runs/metric_gauge_retest/v2_metric_reference_300_329_trunks012_d63b34f7.json \
  --tokenizer-checkpoint logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt \
  --dggt-checkpoint /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
  --output data/scene_gauge/pullback_d63b34f7.json
```

render smoke 的实际启动环境是
`CUDA_VISIBLE_DEVICES=0 TORCH_HOME=/home/dancer/.cache/torch PYTHONPATH=/home/dancer/code/dm/dggt`
加 `conda run --no-capture-output -n dggt python -u /dev/stdin`；内联程序调用
`tools/evaluate_tokenizer_v2_ppu.py` 的同一 runtime，固定
`training_300` / dataset index 151 / frames `[93,95,96,97,98,99,101,103,106,107]`，并写出上面
已哈希的 selection manifest、smoke JSON 与 visual。它不是新增的可复用 CLI；正式输入选择和输出已由
production freeze 作 content-hash 校验。

---

## Verification

**Phase 0（D3/D4）之后**：写进 `docs/story_claude/scene_flow_iclr_story_2026-07-30_claude.md` 的 A.8.9 新小节。

**离线表**：随机抽 20 个 trunk，把 `exp(log_metric_scale)` 乘回 DGGT depth，与 lidar 逐像素比，
中位相对误差 < 5%。跨 trunk gate 与随机 20 样本相互独立：固定取 training scene `[300,330)`、
trunk `[0,3)` 的可靠 `s_cam` 相邻对（至少 20 对），复现 mean 8–11% / max 25–31%；`s_lidar`
以及全表两种估计器继续输出为诊断，但不把 sample-size-dependent 的全表最大值硬套进该 gate。

**标定文件（D4 + v1 LiDAR gate 实跑）**：
- `c_gs`：**已判定 identity**，不再有 render 判据可用（PSNR 单调，见 Phase 0）。
  v1 的替代验收被定义为 tokenizer v2 的 `gs_scale_sim_ratio`/独立 paired gate；训练日志缺失，
  但 v2 正式 paired gate 已用独立 CUDA 审计通过。
- `c_depth`：v1 在 selection split 的主口径为 `7.567%→6.901%`，scene Δ bootstrap
  CI `[0.052%,1.225%]`，故 metric boundary 选择 loglinear；render 仍 identity。
  v1 的**原始诊断 JSON**不可训练加载；冻结后的
  `data/scene_gauge/pullback_75e566ef.json` 曾是 v1 历史正式、`eligible_for_training=true` 的加载对象；
  v5 production loader 无条件拒绝它，该字段只作为历史 payload 保留。
- **v2 最终验收**：paired point/scene-bootstrap CI 为 `0.99985/[0.99015,1.00753]`，通过
  `[0.95,1.05]` 等价 gate；calibration loglinear `a=-0.0280076,b=-0.0438381` 在 LiDAR selection
  上得到 identity/candidate AbsRel `7.762%/8.507%`，delta CI `[−1.553%,+0.113%]`，故选择
  identity。production 加载对象切换为 `data/scene_gauge/pullback_d63b34f7.json`（方案 A）。

**单元测试**：

- `tests/test_camera_generation_v2.py` —— 全套测试按 9 维米制重写，重点保留
  `test_sliding_windows_only_slice_the_global_anchor_mask`、
  `test_delta_only_window_keeps_global_roles_and_decodes_from_previous_frame`、
  `test_role_normalization_matches_noise_scale`。
  新增：从真实 Waymo c2w 编码再解码应精确还原（确定性 GT 的往返测试）。
- `tests/test_scene_gauge.py`（新）—— 归一化往返、`metric_c2w_to_dggt` 往返、
  gauge token 形状与 `S` 无关、缺失 GT 时逐通道 mask 生效、
  `gauge_to_pose_enc_fov` 产出满足 `rgb_render_loss.py:626` 的 `[B,S,9]` 断言。
- **`tests/test_gauge_similarity_invariance.py`（新，最重要的一个）** —— 分两层冻结语义：
  (a) pullback 前或 identity profile（`a=b=0`）把 `log_metric_scale` 加 `δ`，`means` 与 `scales`
  必须精确同乘 `exp(δ)`，`rotation` / `color` / `opacity` 逐元素不变；这是相似变换定义，也是
  「导出米制时忘了缩放高斯 scale」的直接捕手；(b) production fixture 必须绑定 v2 identity artifact，
  并另测 v1 schema/profile 直接拒绝。旧 v1 loglinear 的数学响应只留在历史文档，不留 runtime 测试分支。
- **`tests/test_metric_box_anisotropy.py`（新）** —— (a) `metric_box_to_dggt` 在 `FOV_dggt == FOV_waymo`
  时退化成纯标量 `1/s`；(b) 用实测 FOV 时横向压缩比落在 `[0.74, 0.78]`；
  (c) **回归护栏**：断言 `datasets/dataset.py` 的 box 投影用的仍是 Waymo K 而非 gauge K
  —— 上一版计划曾要求改掉它，这个测试防止那个错误被重新引入。
- `tests/test_pullback_calibration.py` / `tests/test_freeze_tokenizer_pullback_v2.py` —— v2-only schema
  与 v1 rejection、tokenizer/result/hash/window/grid mismatch fail-closed、identity/constant/loglinear candidate、显式
  render/metric boundary、identity 精确 no-op，以及只有完整通过 Gaussian/LiDAR/render smoke gate 的
  v2 candidate 才可 promotion 为 `eligible_for_training=true`。
- `tests/test_calibrate_tokenizer_pullback.py` —— 公式方向、严格 LiDAR support、29 帧全局索引、
  分层聚合、scene bootstrap、冻结 calibration/selection split、identity/constant/loglinear、哈希/window/
  boundary/eligibility contract；candidate 结果禁止直接冒充 production artifact。
- `tests/test_tokenizer_v2_losses.py` —— 三项新增 loss 的 finite/gradient、有效与空 support、混合 batch、
  gradient-accum cache 聚合；空 support 必须报告 NaN + zero count，不能伪装成 ratio=1 或污染 raw 日志。
- `tests/test_sliding_window_v2.py` —— 扩展到 gauge：覆盖 29 帧、window=10、stride=7、`x/v`
  prediction、CFG 1/2.5 与完整 modality namespace，验证 direct 与滑窗两份平行生成实现遵守同一契约。
- `tests/test_inference_scene_flow_pretrain.py` —— 米制导出与单位标注测试。
- `tests/test_scene_gauge_flow.py` —— gauge generation token、统计往返、逐通道 direct-loss mask 与
  gauge→video 梯度耦合。
- `tests/test_tokenizer_window.py` / `tests/test_formal_tokenizer_window_integration.py` —— 10 帧单窗
  精确一次 encode/decode；29 帧 formal cache、train、validation、inference 的每次 tokenizer 调用都 `S≤10`。
- `tests/test_formal_metric_gauge_provenance.py` / `tests/test_metric_gauge_checkpoint_clean_cut.py` ——
  gauge 表、pullback、tokenizer、DGGT、stats 的 hash/window/grid clean-cut，旧 checkpoint fail-closed。
- `tests/test_metric_depth_diagnostic.py` —— Phase 4 的 `gauge_vs_prior_gain`、
  `metric_depth_rel_err`、log-scale/FOV error 与逐通道 valid mask 均有纯函数回归；LiDAR 不可用时
  明确记录 unavailable，而不是把 NaN 冒充零误差。
- camera-condition 回归覆盖 raw pretrain、formal T1、formal offline inference、external manifest
  四条路径，以及真实 `[S,V]` cache 的 front-view 选择、delta-only preceding pose 和非 identity stats；
  pretrain→formal warm-start 还必须通过 `pretrain_feature_stats_contract` 的 stats SHA、10 帧、29 帧
  context 与 25×37 grid clean-cut。

**端到端**（`conda activate dggt`）：

下面第 1、2、4、5 项是**完整重训后的科学 gate**。2026-08-02 的 step-1/单步采样只完成了线路验收，
不能用 `gauge_vs_prior_gain=0`、17.29% metric-depth error 或空 PLY 对这些 gate 作通过/失败判断。

1. 小规模训练若干步，确认 `gauge_log_scale_error` 下降、`metric_depth_rel_err` < 10%、
   且 **`gauge_vs_prior_gain` 显著 > 0**（否则 Phase 2 的机制假设不成立，必须写进 limitation）。
2. `camera_translation_error` 应显著低于当前基线 —— 米制目标去掉了 CV 23.5% 的不可约噪声。
3. `inference_scene_flow_pretrain.py` 跑 10 帧与 29 帧（触发滑动窗）各一次，确认两条路径都只维护
   一个 `[B,1,3]` scene-global gauge、从不按窗口切片，并在固定 seed/schedule 下可复现。10 帧与 29 帧
   看到的上下文不同，**不要求数值 bitwise 相等**；新 checkpoint 产出后应预注册同一前 10 帧条件下的
   drift 容差，再报告差值，不能事后调阈值。
4. `--export_units metric` 导出 PLY，量一条已知车道宽度或车长，对照 Waymo 真值。
5. **相机可控性验收（本次改造的头号指标）**：给定 Waymo 相机条件、`camera_guidance_scale` 拉高，
   生成轨迹与条件轨迹的平移误差应接近 0。**改造前因为差一个未知 `s`，这个实验根本做不了。**
6. **~~渲染质量验收：施加 `c_depth`/`c_gs` 前后的 PSNR 对比~~ —— 此项已作废**（D4 证明 PSNR
   对尺寸类常数单调、无内点极值，扫到语义上界 2.5 仍在涨）。替代验收见上方「标定文件」两条。
   保留这行只为记录：**不要再用 PSNR 去定尺寸类常数。**
7. **硬伤 5 后半段的 v1 结论保留，但 v2 根治 gate 已通过。** v1 paired ratio 仍是 0.796；
   v2 正式复测为 `0.99985`、scene-bootstrap 95% CI `[0.99015,1.00753]`，整段位于
   `[0.95,1.05]`。该结论只适用于预注册 support/estimator，不扩张成 Gaussian 绝对协方差标定。
8. **generated static-geometry reprojection/cycle diagnostic（D3 的连带项）**：从生成的
   `depth_t` 用生成相机运动重投影到 `t+1`，在落点采样生成的 `depth_{t+1}`，再反投影回 `t`，
   报 flow-cycle EPE 与 z-depth log residual；显式排除 sky、dynamic、出视锥与遮挡 support。
   **它不是独立 optical-flow accuracy，也不使用 GT image**；纯静态相机应报告 degenerate，不能把零 flow
   当成一致性证据。在模块 C 落地前，它是生成相机/静态几何分叉的可观测代理诊断。

---

## 讲故事：每个 Phase 对应论文里的哪一句

叙事主线是 **「让 gauge 显式，并审计冻结解码器的 pullback」**，不是「加了一个模块」。

| Phase | 论文里的一句话 | 支撑数字 |
|---|---|---|
| Context | 冻结 teacher 定义的 latent 世界带一个**逐 clip 不可观测的规范自由度**，它污染了所有混用米制与 latent 的下游量 | 1 单位 = 25–64 m，相邻 trunk 漂移 mean 8.2% / max 30.8%，camera 与 lidar 同向（41/44） |
| 1a | 该自由度可以用一把独立的物理尺子（LiDAR）离线标定到 29 帧片段级 | 90/90 有效；逐帧 robust CV 0.688%；相机尺交叉验证 `0.99995 ± 0.026`，corr 0.993 |
| 1b | **冻结解码器的 pullback 必须按作用域、checkpoint 与独立物理尺审计**：GS 的相似性先从 tokenizer 目标修复，depth 候选再由 LiDAR selection 决定是否部署 | v1 metric `7.567%→6.901%` 但 GS/depth **0.796**；v2 paired **0.99985**、CI `[0.99015,1.00753]`，而 loglinear 使 LiDAR `7.762%→8.507%`，故方案 A（render/metric identity、`c_gs=1`） |
| 2 | 把规范量做成**显式生成的 scene-global 变量**，与几何双向耦合；并检验它是否真能从图像内容推断 | `gauge_vs_prior_gain` 是可证伪的判据 |
| 2/8 | teacher 的内参**不是要修的 bug，而是 gauge 的一部分** —— 我们用 leave-one-frame-out 渲染证明 teacher 自己的 K 比真实标定更好地解释它的几何 | **+0.472 dB，CI [+0.253, +0.741]**；trunk-mean K 相对 native 仅 −0.108 dB，通过 −0.2 dB 非劣界 |
| 3 | gauge 一旦显式，其余每个量都能搬回它**天然的确定性空间**：相机 → 米制 Waymo | 目标从「乘一个 CV 23.5% 的随机数」变成恒等映射 |
| 7 | asset 条件重参数化为**尺度不变量 + log 幅值**，未知尺度退化成同一个加性常数 | 16 维中 **11 个 passthrough + 5 个标准化通道** |
| 5 | 主动写出 limitation：长序列生成的全局尺度比 teacher 更自洽，但**不存在一致的 GT 可供评测** | 跨 trunk 漂移 8–31%，且 lidar 同步漂移 |

这条线最耐审的地方在于：**每一步都由一个可复现的实测判据驱动，且每个判据都自带证伪条件**
（D1 的非劣界、D4 的 renderer 病态拒绝、Phase 1b 的 paired-equivalence + LiDAR 双 gate、
Phase 4 的 prior baseline）。
Reviewer 换掉任何 baseline，这些数字都还在。
