# ICLR 全场景生成：代码梳理、相关工作谱系、SceneDirector 叙事迁移与故事设计

> **版本：2026-08-03 v9.1 —— 按落地代码重写创新点与叙事，并把全文用语改成正常的论文语言。**
> 前身为 `scene_flow_iclr_story_2026-07-30_claude.md`（v8），保留不改，本文件是它的后继。
> v8 的 Part D/E/F 写在「场景尺度尚未显式生成」的代码上；`docs/metric_scale_camera_redesign_plan.md`
> v4 的 Phase 1–8 已全部落地（回归 **732 passed, 1 skipped**），因此故事的核心从
> 「在解码之后测量误差」变成 **「在一个没有内建尺度的表示里生成场景，同时把这一段场景的
> 米制尺度生成出来，再用这套单位在解码之后测量误差」**。Part A/B/C 与 A.8 证据链保留并按当前契约更新。
>
> **v9 → v9.1 的两类修改**（2026-08-03，对照 `docs/story_codex/scene_flow_metric_readout_story_2026-08-03.md`）：
> （一）**用语**。删掉全部自造抽象词（"米制承诺""表示保持未承诺""可审计""规范量"
> "readout""fidelity / coherence"），改用领域标准词或大白话——一句话不该需要回查定义才能读懂。
> （二）**内容**。补上漏掉的已落地特性（合法任务层级）；修正目标绑定损失里一个真实的几何错误
> （box 中心 ≠ 可见表面深度）；新增目标身份一致性项与整组反事实实验。
>
> **v9.1 → v9.2（2026-08-04）：¶2 的缺口整体换掉，故事骨架随之调整。**
> 起因是逐篇核实了 11 个驾驶前馈重建方法（D.1.5 新增全表）：**它们的米制全部来自输入侧
> ——输入位姿、已标定 rig 外参、或 LiDAR 深度监督——按构造是对的。** 因此 v9 的"他们的米制是
> 生成器碰不到的固定属性"这条缺口**不成立，已从 ¶2 全部删除**。
> 新的 ¶2 是**三个各有机制的缺点**（阶段无反馈 / 先验在图像上 / 格点粗且断链），
> 米制移到 ¶3 作为**我们这条路线的必然代价**（无位姿 → 绝对尺度不可确定）。
> 受影响并已同步更新：§0、C.3.6、D.1.5、E.1–E.3、F.0–F.7。
>
> **2026-08-04 追加：¶3 拆成前后两半，方法定名 ChoraGen。**
> 前半（第 1–3 句，问题）由用户定稿，第 3 句把"没有尺度"这个代价翻译成**两个**具体后果
> （条件施加不上 / 结果度量不了）；后半（第 4–9 句，方法介绍，本轮草稿）逐一回答这两个后果，
> 并按 E.4 的三个模块各给一句。**旧的 F.5「¶4 四句骨架」随之作废**，见该节方框。
>
> **v9.2 → v9.3（2026-08-05）：Phase 1b tokenizer v2 正式复测与方案 A 冻结。**
> A.8.9/A.8.10 的 v1 数字与结论保留为历史证据；新增 A.8.11 记录 v2 正式复测。
> 原 A.8.11/A.8.12 顺延为 A.8.12/A.8.13，正文不覆盖。
>
> **v9.3 → v9.4（2026-08-05）：v2-only clean cut。** runtime、默认配置与 loader 不再提供
> tokenizer v1 兼容路径；v1 artifact/结果只读保留。正式数值仍绑定 A.8.11 已记录的 Gaussian 2.3.0
> 与 LiDAR 2.1.0 运行；清理旧 render/PSNR 扫描代码后按用户决定不重跑正式结果链，也不登记新的
> script/result SHA。
>
> 实测脚本（`conda activate dggt`）：`lyy_tools/verify_camera_gauge.py`（A.8.1–A.8.6）、`lyy_tools/verify_window_scale.py`（A.8.7）、`tools/retest_scene_flow_gaussian_gauge.py`（A.8.9/A.8.11）、`tools/calibrate_tokenizer_pullback.py`（A.8.10/A.8.11）、`tools/freeze_tokenizer_pullback.py`（A.8.11）、`lyy_tools/verify_metric_gauge_postreview.py`（A.8.13）
> 阅读范围：`train_scene_flow_pretrain.py`、`inference_scene_flow_pretrain.py`、`dggt/models/{scene_flow,joint_scene_tokenizer,canonical_asset_encoder}.py`、`dggt/utils/{scene_gauge,camera_generation,factorized_asset_condition,camera_geometry_flow_consistency}.py`、`dggt/losses/{flow_losses,reconstruction_feedback_loss,rgb_render_loss}.py`、`train_tokenizer.py`、`pretrain_single_node.sh`；`docs/metric_scale_camera_redesign_plan.md` 全文；`docs/ICLR_scene_generation_story_codex.md` §1–2（仅代码与来源部分）；`docs/ICML_SceneDirector.pdf` 全文；`paper/read/` 下一手审计。
> 外部核验：LSD-3D、ScenDi、SEM-ROVER、WorldFlow3D、WorldSplat、CVD-STORM、Envision4D、GaussianCity、PrITTI、Urban Architect、InfiniVerse、AnyScene、GaussianDWM 均已用网络检索独立复核。文中标 ⚠ 的是未逐行核实、写作时需再确认的细节。

> **当前正式契约（A.1–A.7 已按此更新）**：9D Waymo 米制 camera state
> （`waymo_metric_relative_se3_rot6d_v4`）、16D placement（`factorized_asset_v3`）、
> 3D scene-global gauge（`dggt_teacher_log_metric_scale_logfov_v1`）、
> tokenizer-v2 production pullback（`data/scene_gauge/pullback_d63b34f7.json`，
> render/metric depth 均为 identity，`c_gs=1`）。v1 artifact 保留为历史输入，不覆盖。
> 实现与验证清单以 `docs/metric_scale_camera_redesign_plan.md` 为准。

---

## 目录

- [0. 结论先行](#0-结论先行)
- [Part A. Scene Flow Pretrain 代码梳理](#part-a-scene-flow-pretrain-代码梳理)（A.1–A.7 已按当前契约更新；A.8 为尺度实测证据链）
- [Part B. SceneDirector 的叙事为什么成立](#part-b-scenedirector-的叙事为什么成立)
- [Part C. 相关工作谱系（准入标准与逐篇判定）](#part-c-相关工作谱系准入标准与逐篇判定)
- [Part D. 四个核心问题的回答](#part-d-四个核心问题的回答2026-08-02-按落地代码重写)（**新增 D.1.5：表示的细节与米制尺度，通常只能二选一**）
- [Part E. 统一故事与模块地图](#part-e-统一故事与模块地图2026-08-02-重写)（**三个核心模块：JST / 统一场景状态生成 / HDS，只有一个是损失**）
- [Part F. Introduction 逐段写作](#part-f-introduction-逐段写作2026-08-02-按新故事重写)（**¶2 三个缺点、¶3 方法段、¶4 能力段均已定稿，方法名 ChoraGen；F.4 后半草稿已作废，见 F.4.5**）
- [Part G. 实验闭环](#part-g-实验闭环2026-08-02-更新)（**新增 Fig.1 尺度量程图、G.5 米制导出实验、G.6 反事实实验组**）
- [Part H. 表述边界与风险](#part-h-表述边界与风险2026-08-02-更新)（**新增 H.5 证据分级表**）

> **v8 → v9.1 的四处结构性变化，先看这里：**
>
> 1. ~~**缺口从一层变两层。** 除"共同解释场景的三维在生成之后由单独阶段产生"外，新增
>    "它们的米制尺度固定在生成器不训练的部件里，因而不能逐片段自适应"。~~
>    **⚠ 本条已于 v9.2 全部推翻并从 ¶2 删除**（理由见上方 v9.1 → v9.2 方框与 D.1.5）。
>    保留仅作修订记录，**任何位置都不得再使用这条主张**。
> 2. **场景尺度生成已全部落地**，从"待新增模块"变成方法的第一条贡献。
> 3. **方法侧三个核心模块，只有一个是损失**：**JointSceneTokenizer**（让重建状态可生成，架构）、
>    **统一的场景状态生成**（场景/相机/天空/尺度一次采样 + 尺度对齐的条件读取，架构）、
>    **HDS**（解码之后的分层监督，含 teacher 置信度加权，损失）。
>    **目标满足度全部作为评测指标，不作为损失**——与 SceneDirector 报 ATE/AOE 的做法一致。
> 4. **补上一个漏掉的已落地特性**：合法任务层级 `text ⊂ text+camera ⊂ text+camera+actor`
>    （`train_scene_flow_pretrain.py:1539-1629`），它是 D.2「条件怎么进来」这一环的另一条腿。

---

## 0. 结论先行

**一句话故事（不用 bridge 句式；2026-08-04 重写）：**

> 在现有的驾驶 Gaussian 生成里，最终的三维总是在生成目标之外产生的——要么由一个独立的重建阶段
> 产生（视频先生成，重建不受惩罚），要么把图像平面的 latent 抬起来（先验没走出图像），
> 要么藏在体素的离散解码后面（先验终于在三维上，但格点太粗，梯度也回不来）。
> **我们改在一个前馈重建模型的内部状态里生成**：它的先验落在三维上，又不引入格点。
> **然而**这类模型只接受无位姿图像，绝对尺度在数学上不可确定，它们因此把尺度归一化掉；
> 而驾驶的条件与标注全是米。所以我们把**这一段的米制尺度和视场角作为三个数与场景一同生成**，
> 并**重参数化条件，使未知尺度只表现为一个加性常数**；随后在解码之后、用米测量误差。

**第二核心句（¶3 的转折，可直接进 Abstract 或 contribution 第一条）：**

> *Such a model takes unposed images, where absolute scale cannot be determined, so we generate the
> metric scale of the clip together with the scene, and reparameterize the conditions so that the
> unknown scale appears only as a single additive constant.*

> ⚠ **2026-08-04 的方向性修正**：v9 曾主张"现有方法的米制是生成器碰不到的固定属性"。
> 逐篇核实 11 个驾驶前馈重建方法后，这条**不成立**——它们的米制来自输入侧（已标定 rig 外参、
> 给定位姿、LiDAR 深度监督），按构造是对的（见 D.1.5）。米制现在只作为**我们这条路线的代价**
> 出现在 ¶3，不再是对手的短处。

**用语纪律：不造抽象词。** SceneDirector 全文只造了一个模块名（MGRA），它的核心概念
*structural reliability* / *semantic completion* / *uncertainty-aware allocation* 全是普通词组。
造得越少，越不像在包装。本文只保留两个方法名，其余一律用大白话：

| 角色 | 表述方式 |
|---|---|
| 失败机制（一） | 不造词：*the 3D scene that would jointly explain scene semantics, actors and trajectory is produced only after generation, by a separate stage, and never enters the generative objective.* |
| 失败机制（二） | 不造词：*their metric scale is fixed inside a component the generator never trains, so it cannot adapt to the clip being generated.* |
| 正面主张 | 不造词：*we generate the scene's metric scale and field of view jointly with the scene, as three scene-level numbers.* |
| 方法名（一） | **Hierarchical Decoding Supervision (HDS)**，三层就叫**特征层 / 高斯层 / 图像层** |
| 模块名（其余两个） | **JointSceneTokenizer**、**统一的场景状态生成**——都是描述性的，不造词 |

> **已弃用的自造词，全文不再出现**：米制承诺（→ 场景的米制尺度）、表示保持未承诺（→ 没有内建尺度的
> 表示）、可审计 / 可被核对（→ 直接说报告什么误差）、规范量（→ 场景尺度与视场角）、readout
> （→ decoder / 冻结的解码头）、fidelity vs coherence（→ 与真值比对的损失 vs 生成量之间互相一致的
> 损失）、世界级监督（→ 解码之后的监督）。代码里的 `scene_gauge`、`gauge_*`、`pullback_*` 等标识符
> 保留原样，但只在提到代码时出现。

**四个环节，互为前提（这是"一个故事"而不是"四个卖点"的证据）：**

| 环节 | 问题 | 机制 | 状态 |
|---|---|---|---|
| **在哪个空间生成** | 那个状态怎么才能被生成？四层三流每 patch 12288 维，直接扩散不可能 | **模块一 JointSceneTokenizer**：按被压缩对象的结构设计的 12:1 压缩 | 已有（v2 训练中） |
| **用什么单位** | 那个 decoder 解出来的几何，一个单位等于多少米？ | **模块二**：尺度与视场角作为 3 个数被生成 + 9D 米制相机 | **已落地** |
| **条件怎么进来** | 条件约束的是不是同一个状态，且与它同单位？ | **模块二**：16D placement（11 尺度不变 + 5 log 幅值）+ typed mRoPE + 合法任务层级 + **尺度对齐的条件读取** | 已有，减 $\hat g$ 待补 |
| **误差在哪里测** | 训练误差是不是在解码之后、用这套单位测的？ | **模块三 HDS**（特征 / 高斯 / 图像三层 + 置信度加权） | 已有，加权待补 |

依赖关系是真实的，不是修辞：
**没有一个可微且冻结的 decoder，三层损失根本写不出来；
没有生成出来的尺度，四族条件里有两族的量纲是坏的、"米制误差"根本没有定义；
没有解码之后的监督，生成出来的尺度就没有任何东西约束它。**

**三个模块之外，还需要补的两个部件（都很小，且都只调节一份已存在的信息）：**

- **HDS 的置信度加权**：用冻结解码头自己输出的 `depth_conf` / `gs_conf` 给高斯层和图像层的损失
  加权——解码可信的地方反馈强，解码本身就不确定的地方（天空、远景、细结构）反馈弱。
  这回答"你的 teacher 自己有误差，它的误差就成了你的目标"这一必问质疑。实现上是一次逐元素相乘。
- **尺度对齐的条件读取**：条件进入注意力前先减掉当前步生成的 $\hat g_{\text{scale}}$。
  价值不在省一个损失，而在于它制造了一条梯度回路——**条件用得对不对，会反过来监督尺度分支**。

> **实现优先级**：置信度加权最便宜，先做；减 $\hat g$ 次之；满足度评测脚本最后。
> **最大的非模块依赖是 tokenizer v2 的训练**（见 E.4.1 的状态方框），
> 模块一的全部实验数字都等它。

> ⚠ **实测边界（见 A.8，尤其 A.8.9/A.8.10）**：Waymo 相机参数与 DGGT camera state **不是确定性关系**。
> 相机中心与直接深度近似共享一个逐 29 帧片段的标量尺度，但视场角差异造成横向各向异性
> （$k_x=0.748$、$k_y=0.772$）。
> **tokenizer 往返的畸变结论已作废**——那是 v1 checkpoint 上的历史测量，当前实现是 v2 且尚未训完，
> 任何往返缩放的数字都必须等 v2 训练结束后重测，见 E.4.1。
> D3 实测把渲染相机换成米制换算相机的代价是 1.41–1.44 dB，因此
> "完整三维只差一个标量"与"渲染一定不失配"两句都已撤回。这个数是 D3 在完整 29 帧、
> 指定 source/target 协议下的**控制保真度上界诊断**，不是推理时的画质损失。

**Intro 核心谱系（一句话名单）：**

> MagicDrive3D、InfiniCube、DriveGen3D、X-Scene、LSD-3D、WorldSplat、CVD-STORM、ScenDi（带 urban-road 限定）。其余全部进 Related Work。

---

# Part A. Scene Flow Pretrain 代码梳理

## A.1 任务定义

尽管文件与类名保留 `scene_flow`、`WanSceneFlow` 等历史命名，**这一阶段解决的不是两帧之间的 scene-flow estimation**。`build_full_scene_bundle`（`train_scene_flow_pretrain.py:4125`）设置

$$
M_{\text{preserve}}=0,\quad M_{\text{source}}=0,\quad M_{\text{dest}}=1,\quad z_{\text{splat}}=0,
$$

即整个目标窗口从纯噪声生成，既不是 inpainting，也不是局部编辑；代码中保留的 preserve/boundary loss 在该模式下数学上恒为零。

任务是：**在冻结 DGGT 特征空间中，从噪声与结构化条件出发，条件生成一段完整的驾驶场景状态
以及给它一个米制解释的那 3 个数（尺度与两个视场角通道），并由冻结 DGGT heads 解码为可渲染、
可导出为米制的三维场景。**

> 后半句是 2026-08-02 新增的。它不是措辞修饰：**冻结 DGGT 状态本身没有单位**
> （1 DGGT 单位在不同 29 帧片段等于 25–64 米，见 A.8），而全部结构化条件——ego 轨迹、
> 3D box 中心与尺寸、速度——都是米制的。不生成这 3 个数，条件与解码几何就不在同一套单位里，
> "米制误差"也没有定义。

## A.2 输入输出

**数据**：Waymo 前视，默认 10 帧 @ 10 FPS。冻结 DGGT aggregator 对完整 29 帧 clip **只跑一次**，再抽取 10 帧目标窗口作为 teacher target——同一 trunk 内不同窗口共享同一 clip-global 几何上下文，但生成器只生成当前窗口。

Patch grid 25×37 = 925，scene latent $z\in\mathbb{R}^{B\times 10\times 925\times 1024}$。

**条件（全部 soft condition，进入同一 full attention 序列）：**

| 条件 | 表示 | 关键设计 |
|---|---|---|
| 全局文本 | 冻结 Qwen3-0.6B 编码的 caption tokens | 直接进入 full attention |
| 物理相机 | 20 维 Waymo condition，其中 `[..., 9:18]` 由**与生成目标完全相同的 9 维米制 state** 构造 | 请求与目标共用 role-aware helper、同一套 `camera_anchor/delta` 统计 → **复现给定轨迹是恒等映射**，不再是要学的仿射变换 |
| 目标外观 | 每对象 ≤32 个 canonical appearance tokens | 从**目标窗口之外**一帧取 RGBA reference，冻结 DGGT + tokenizer 以 `S=1` 编码（`CanonicalAssetEncoder`）；API 结构上无法接收目标 clip/轨迹/latent/bbox/mask——有价值的防泄漏设计 |
| 目标位置与运动 | 每对象每帧 **16 维** placement state（`factorized_asset_v3`） | 见 A.5 的通道表：**11 个尺度不变 passthrough + 5 个标准化米制 log 幅值**，≤5 对象；另由 3D box + **真实 Waymo K** 产生 target projected bbox |

**随机变量（噪声初始化，联合生成）：**

- scene latent $z$
- **camera generation target**：**9 维米制 Waymo camera state**（translation 3 米 + rotation-6D），
  第 0 帧绝对、其余帧相邻 $SE(3)$ delta；target 来自 **Waymo GT 外参**
  （`camera_to_world_corrected` + `camera_trajectory_anchor_to_world_corrected`），
  **不再**来自冻结 DGGT CameraHead。表示常量 `waymo_metric_relative_se3_rot6d_v4`
- **scene gauge**：**3 维 scene-global** token
  $[\log(\text{米}/\text{DGGT 单位}),\ \log\tan(\text{FOV}_x/2),\ \log\tan(\text{FOV}_y/2)]$，
  与 $S$ 无关，只有一个 token；GT 来自离线 LiDAR 标定表（A.8.6 / Phase 1a）。
  表示常量 `dggt_teacher_log_metric_scale_logfov_v1`，mRoPE 坐标 `(15100,15100,15100)`
- **sky**：目标视图转成 32×64 上半球方向 atlas，2×2 RGB patch 打包成 12 维 token；未观测方向低权重监督。
  atlas 构建与 render 读出现在共用 **trunk 常量 gauge K**（此前是 teacher 逐帧 K vs 生成 K，两侧不同源）
- patch 级与 refined dense 级 sky mask（`SkyMaskRefineDecoder`）

**最终可执行输出：**

$$
\hat z_{1024}\xrightarrow{D_{\text{JST}}}4\times\hat F_{3072}
\xrightarrow{\text{frozen DGGT heads}}
\{\text{depth},\text{GS attrs},\text{dynamic conf}\}
\xrightarrow{\text{gsplat}}\text{RGB}/\text{PLY}
$$

**Gaussian mean 不是 flow 直接预测的参数**，而是由生成 depth 与生成 camera 反投影得到；GS head 给出 RGB、opacity、scale、quaternion 与 `gs_conf`，renderer 用 `gs_conf` 做 lifespan-like 时间衰减；instance head 给出 dynamic confidence。point head 与 semantic head 当前不参与生成解码。

## A.3 核心模块

### JointSceneTokenizer（冻结）

四层（4/11/17/23）× 三流（DINO / frame-attention / global-attention）× 1024 维 = 每层 3072 维，概念上 4×3072 = 12288 维原始通道。Encoder（`joint_scene_tokenizer.py:415`）：分流归一化投影 → `DetailConvBranch` 保留局部细节 → `LayerAttnStack` + `LearnedQueryPool` 在四层间做 layer attention 与 query pooling → `FrameGlobalBlockPair` 空间与时间 attention → 1024 维 patch-aligned latent。

约 12:1 通道压缩，仍远高于常见 video VAE 的 16 通道 latent。**关键事实**：tokenizer 独立预训练不只用 feature MSE，还包含 cosine、depth/GS/dynamic head anchor、动态 mask、render anchor、noisy-latent decoder 与 latent-stat losses——bottleneck 被显式训练成**保留 DGGT heads 所需的信息**。本阶段完全冻结、严格完整加载。

> 写作红线：应称 **compressed multi-level DGGT feature lattice** 或 **reconstruction-grounded scene state**；不可写"在原始 DGGT 四层特征上直接扩散"、"pure world-space geometry token"、"直接 Gaussian diffusion"，也不可说这是公开 DGGT 自带的 tokenizer——它是本仓库的扩展。

### RAEVideoSceneFlow（生成器）

28 个 1440 维 full-attention encoder blocks + 2 层 2048 维 DDT decoder head（`scene_flow.py:739`）。未加载 Wan / RAEv2 / Cosmos3 的生成权重。

`forward`（`scene_flow.py:3043`）把两组 span 拼成**一个序列做 full self-attention**：

```
gen  = [ scene tokens (S·P) | camera-gen tokens (S) | sky-gen tokens (K) | gauge-gen token (1) ]
cond = [ timestep | text | camera-cond | asset | edit-control ]
full = concat(gen, cond)     ← 一次 full attention
```

只有 scene span 经 DDT head 预测 1024 维 clean latent；camera、sky 与 gauge 用独立轻量 decoder。

**五类 span 共用三维 mRoPE 但位置定义不同**，这是 typed addressing 的实现：

| span | mRoPE 位置 |
|---|---|
| scene | 全局 frame / $y$ / $x$（`[0,15000)`） |
| camera | 每帧图像中心 |
| **asset** | **canonical UV 线性映射到该帧 target projected bbox**（`scene_flow.py:1965-1982`）；出画对象放到 reserved 保留坐标带 |
| sky | 球面方向坐标 + 独立时间偏移区间（`15000±8`） |
| **gauge** | 单 token，`(15100,15100,15100)`——刻意落在视频带与 sky 带之外，且 < `rope_max_position` |

asset token 值为 `appearance + placement_MLP(16维) + slot_embed + modality_embed`（`scene_flow.py:1939-1944`），另加 per-frame summary token。**即：外观是"什么"、placement 是"在三维哪里"、RoPE 是"在图像哪里"，三者分解后相加。**

### Scene gauge 生成流（本轮新增，`scene_flow.py`）

模块完全照抄 sky（唯一现存的非逐帧生成流），但用 `ChannelScale` 而非 `RMSNorm`
——低维物理量的范数本身有意义：`gauge_gen_norm` / `gauge_gen_proj` / `gauge_gen_decoder`
（末层零初始化）/ `gauge_gen_modality_embed`，配 `gauge_mean`/`gauge_std` buffer
与 `set_gauge_stats` / `require_gauge_stats`（`:1121-1128`、`:1206-1208`、`:1498-1530`）。

**关键耦合（这是让 `log s` 学得准的主要机制，不是旁挂一个回归头）**（`:3465-3472`）：

```python
gauge_context = gauge_hidden if gauge_gen_len > 0 else zeros(b, 1, hidden)
cond = self.s_projector(F.silu(enc_video + t_base + gauge_context))
```

这条线做两件事：**生成的几何显式以生成的尺度为条件**；**video flow loss 的梯度反向流进 gauge token**。
加上 gauge token 通过 encoder 全注意力看到全部 video token，形成**双向耦合**。

**为什么 gauge 是 3 维而不是 4 维（不把 tokenizer 偏差放进来）**：`c_depth`/`c_gs` 是冻结 tokenizer 的
确定性属性、不是逐场景未知量，生成它等于让模型去学一个常数。且 **FOV 通道天然免疫往返偏差**
——它由 gauge token 直接生成，不经过 tokenizer decode。

### Rectified flow

$$
z_\sigma=(1-\sigma)z_{\text{clean}}+\sigma\epsilon,\qquad
v^\star=\frac{z_\sigma-z_{\text{clean}}}{\max(\sigma,0.05)}
$$

$\sigma\ge0.05$ 时 $v^\star=\epsilon-z_{\text{clean}}$。默认 `prediction_type=x`：预测 clean endpoint $\hat z_0$ 再转 pseudo-velocity。Waver 时间采样、shift 10、`mode_scale` 1.29。

## A.4 三层世界反馈（方法最独特的部分）

`reconstruction_feedback_loss.py` + `rgb_render_loss.py`。

**梯度链已确认贯通**：`compute_rgb_render_loss` 接收 flow 自己预测的 `z_clean_pred_n`（$\hat z_0$），`decode_generated_dggt_geometry` 在 `autocast(enabled=False)` 下调用 tokenizer decoder 与三个 DGGT heads，注释明确写 "Frozen DGGT/tokenizer parameters keep `requires_grad=False` but this module must not run their decode/head calls under `torch.no_grad`"——**参数冻结，梯度穿过**。teacher 分支在 `torch.no_grad()` 内。

| 层 | 名称 | 实现 | teacher | 度量 |
|---|---|---|---|---|
| L1 | **特征层**（feature level） | `_level_consistency` | $D_{\text{JST}}(z_{\text{clean}})$ 四层 | layernorm-L1 + cosine，both-zero 保护 |
| L2 | **高斯层**（Gaussian level） | `_head_error_maps` | 同上经冻结 depth/GS/instance heads | log-depth smooth-L1、log1p(conf)、GS rgb/opacity/log-scale、quaternion $1-\lvert q_s\cdot q_t\rvert$、dynamic |
| L3 | **图像层**（image level） | `_render_one_sample` + gsplat | **真实图像** | Charbonnier + LPIPS(spatial, 0.01) |

L1/L2 的 teacher 是 $D(z_{\text{clean}})$——**相对于目标世界**；L3 对齐真实像素——**绝对**。

**渲染装配严格对齐部署路径**：means 来自 `torch_unproject_depth(生成 depth, 相机, gauge K)`；static/dynamic 按 `dynamic_conf<0.5` 划分；static opacity 乘 $(1-p_{\text{dyn}})$ 与 $(1-p_{\text{sky}})$ 并按 `gs_conf` 做 $\exp(\ln 0.1\cdot\Delta t^2/\text{conf}^2)$ 时间衰减；背景用生成 sky atlas 可微投影（`sky_tokens_to_background`，align_corners 约定与 validation 严格一致）。

**render 路径的 pullback 恒为 identity**：`scene_gauge.py::apply_pullback_calibration` 是唯一实现，
但调用方必须显式选 `boundary`；render 侧断言返回 identity，只有跨入米制边界（导出、目标满足度评测的米制断言、
`metric_depth_rel_err`）才施加 checkpoint-bound `c_depth` loglinear。

**调度（对论文诚实性很重要）**：`rgb_render_start_step=5000`、`rgb_render_every=2`、`warmup=5000` 线性 ramp、per-sample 权重 $(1-\sigma)^2$、`max_samples=1`。世界反馈是**低占空比、只在低噪声区生效**的信号——这也是它算力可负担的原因。默认 $\lambda_{\text{rgb}}=\lambda_{\text{level}}=\lambda_{\text{head}}=0.1$、`sky_mask_grad_scale=0.05`。

> ⚠ **D3 之后的关键契约变更（必须在论文里如实写）**：训练 render 使用 **detached teacher pose**
> （`train_scene_flow_pretrain.py:5561` 的 `render_pose_enc_dggt=render_pose_teacher`），
> 且 `camera_grad_scale` 在 `rgb_render_loss.py:663` 已升级为**硬断言 0.0**，非零直接报错。
> 理由是正面的而非妥协：latent 的 flow 目标本身就是 teacher 空间的 $\text{Encoder}(\text{direct tokens})$，
> 用换算后的米制相机渲染会主动把几何往 Waymo 世界拉（实测代价 1.4143/1.4357 dB，见 A.8.9）。
> **连带后果是生成相机与生成几何之间没有任何光度耦合**——这正是把相机–几何一致性作为评测指标报告的理由；
> 在它落地前只有 `generated_static_geometry_reprojection_cycle_v1` 这个诊断顶着。

## A.5 其余监督

- 主 scene rectified-flow loss（`masked_flow_edit_loss`，按 `M_edit` 掩蔽）
- 第 8 层 early/base head loss，`base_model_coeff=0.25`
- REPA（$\lambda=0.5$）：中间 trunk feature 投影后与 clean tokenizer latent 做 MSE——**仍在 latent 空间**
- camera flow(0.1) + 绝对/相对 $SE(3)$ + 二阶平滑，`lambda_camera_pose=0.5`。
  **FOV 两项（`camera_log_fov`、`camera_acceleration_fov`）已删除**——FOV 迁到 gauge
- **gauge flow(`--lambda_gauge_flow`=0.1) + gauge direct(`--lambda_gauge_direct`=1.0)**：
  前者与 video 共用 sigma 的归一化空间 masked MSE，后者对 **denormalize 后的 x-prediction**
  在物理 log 单位上做 smooth_l1，按 `scene_gauge_valid` **逐通道 mask**。
  低维物理量上直接监督比纯 flow 损失条件数好得多——这正是相机同时有 flow 与 geometry 两个损失的原因
- sky flow(0.1) + patch mask + refined mask BCE/Dice/boundary

### 合法任务层级：条件不是可任意开关的集合（**本轮补写，v9 漏掉了这块**）

训练时**不再**对 text / camera / actor 做独立随机 dropout。`sample_pretrain_condition_tasks`
（`train_scene_flow_pretrain.py:1539-1629`）只采样三种组合：

$$
\text{text-only}\;\subset\;\text{text}+\text{camera}\;\subset\;\text{text}+\text{camera}+\text{actor}
$$

对应 `PRETRAIN_CONDITION_TASKS = ("joint_generation", "camera_controlled", "asset_camera_controlled")`，
默认概率 **0.2 / 0.2 / 0.6**（`--joint_generation_prob` 等，`:8106-8123`）。
评测时固定取 `asset_camera_controlled`，不做随机丢弃。

**为什么 actor-only 被结构性禁止**（这是这套设计的关键，也是它区别于普通 condition dropout 的地方）：
目标条件里的 `target_bbox_patch`、`in_frustum` 和 canonical-UV mRoPE **都要先由相机投影才有定义**。
一个"给了目标但没给相机"的样本，它的目标条件在数学上是没有意义的。
代码里还有一条兜底：`camera_available_rows` 为假的行会被强制降到 `joint_generation`（`:1609-1616`），
防止一个可选相机的样本意外保留了相机投影出来的 placement。

**推理端的层级 CFG 与训练分布严格对应**（`combine_pretrain_cfg_prediction`，`:1886-1934`）：

$$
\Delta_{\text{camera}} = v_{\text{text}+\text{cam}} - v_{\text{text}},
\qquad
\Delta_{\text{actor}} = v_{\text{full}} - v_{\text{text}+\text{cam}}
$$

于是相机增量和目标增量可以分别调节，而**永远不会去评估"NULL 相机 + 相机投影的目标 placement"
这个非法组合**——函数 docstring 里写明了这一点。三条通路的入口在
`inference_scene_flow_pretrain.py:355-357`（`none` / `cam` / `asset_cam`）。

### 16 维 placement 的通道语义（`factorized_asset_v3`）

| ch | 内容 | 性质 |
|---|---|---|
| 0:3 | `unit_direction_anchor` | 无量纲，passthrough |
| **3** | **`log_z_depth`**（沿光轴，**不是**欧氏 range） | 米制，标准化 |
| **4:7** | `log_box_lwh` | 米制，标准化 |
| 7 | `log(box_diag / z_depth)` | **尺度不变的角尺寸**，passthrough |
| 8:10 | `sin/cos yaw` | 无量纲，passthrough |
| 10:13 | `unit_velocity_dir`（零速置零） | 无量纲，passthrough |
| **13** | `log_speed` | 米制，标准化 |
| 14 | `tanh(speed / z_depth)` | 尺度不变、有界，passthrough |
| 15 | `in_frustum` | 无量纲，passthrough |

标准化通道 `{3,4,5,6,13}` 共 **5** 个；passthrough `{0,1,2,7,8,9,10,11,12,14,15}` 共 **11** 个。
**未知尺度只表现为那 5 个 log 幅值上的同一个加性常数**，而这个常数正是同一次前向里生成的
`log_metric_scale`（gauge token 与 asset token 在 encoder 里互相可见）。

**距离通道必须是 z-depth 而不是欧氏 range**：在各向异性映射
$p_{\text{dggt}}=\mathrm{diag}(k_x,k_y,1)\cdot p_{\text{metric}}/s$ 下**只有 z 分量是纯标量**；
欧氏 range 是 $\sqrt{k^2X^2+k^2Y^2+Z^2}/s$，随离轴角变化。

### 两条永不交叉的内参链路

| 链路 | K | 成员 |
|---|---|---|
| **米制链路** | **真实 Waymo K** | `project_anchor_boxes_to_patch_bboxes`、`in_frustum`、`target_bbox_patch` |
| **DGGT 链路** | **gauge K** | depth 反投影、RGB render、sky atlas 构建与读出 |

米制链路是一条**全米制自洽链路**（米制 `object_to_anchor` + 米制 `camera_to_anchor` + 真实 Waymo K，
`datasets/dataset.py:2054-2071`），给出真实目标在真实图像里的正确像素足迹。
**把它改成 gauge K 是错的**（v1 计划曾这样要求，已撤回，并有回归护栏测试防止重新引入）。

### 相机：请求与目标现在是同一个量

改造前必须分清"20 维 Waymo 物理 condition（请求）"与"11 维 DGGT camera state（生成）"，
因为二者差一个不可观测的 $s$。**现在不需要分清了**：condition 的 `[..., 9:18]` 与生成目标
由**同一个 role-aware helper** 构造，用**同一套** `camera_anchor/delta` 统计归一化，
完整 anchor 窗与 delta-only 窗都沿用全局 frame role。
**于是"复现给定轨迹"在参数化层面就是恒等映射。**

仍然**不能**写成"直接用输入相机渲染"——训练 render 用的是 teacher pose（见 A.4 的 D3 契约），
米制相机是生成目标，推理时用生成的那条。

## A.6 推理流程与当前缺口

标准 raw-validation 路径：噪声 → scene/camera/sky/**gauge** 联合 ODE 采样（脚本内显式 Euler，$\sigma:1\to0$，35 步；`WanSceneFlow.sample()` 本身未实现）→ 补零 special tokens → 冻结 heads 以 `images=None` 解码 → gsplat 渲染 / 逐帧 PLY。长序列用 10 帧窗口、stride 7、cosine overlap 融合；**gauge 是 scene-global 的，用 `scene_global_window_weight` 融合后做一次全局 Euler，从不按窗口切片**。

**本轮新增的推理能力（这是改造对外最直接的产出）**：

- **`--export_units {dggt,metric}`（默认 metric）**：导出 PLY 时 `means` 与高斯 `scales` **同乘**
  $\exp(\widehat{\log s})$，rotation / color / opacity 不变。**这是第一次能导出米制场景。**
  已冻结的相似不变式测试 `tests/test_gauge_similarity_invariance.py` 保证
  "导出米制时忘了缩放高斯 scale"这类错误会被立即捕获。
- run summary 记录 `log_metric_scale`、`metres_per_unit`、`fov_deg`、`tokenizer_sha256`、
  `gauge_table_sha256`、`c_depth`、`c_gs`——**任何米制数字都必须带 provenance**。
- `generated_static_geometry_reprojection_cycle_v1` 诊断已接进推理输出
  （`inference_scene_flow_pretrain.py:1381`）：报 flow-cycle EPE 与 z-depth log residual，
  显式排除 sky / dynamic / 出视锥 / 遮挡 support，纯静态相机标为 degenerate。
  **它不读任何 GT 图像**，因此在推理期仍然成立。

**缺口（必须写进 limitation）**：外部 manifest 分支目前 `return_camera=False`，只保存 normalized latent/sky/mask `.pt`，未接通 render/export——"任意外部 appearance/location/camera → Gaussian"的接口意图清楚但**未端到端验证**。raw-validation 的 asset、track、camera 都来自同一 Waymo clip，因此跨场景外观重组、反事实位置、显著 OOD 相机轨迹、prompt 组合泛化**均未证明**。输出是逐帧 image-grid-aligned dense Gaussians，**没有持久 canonical 4D field、没有显式 deformation field、没有 instance identity**。

## A.7 各模块如何配合

```
29 帧 RGB ──(冻结 DGGT aggregator, 跑一次)──> 四层三流 feature lattice
                                                    │
        窗外 RGBA reference ─(CanonicalAssetEncoder)─┤
                                                    ↓ (E_JST, 冻结)
  text ─(Qwen)──────┐                     z_clean [B,10,925,1024]
 20D camera cond ───┤   (其 9:18 = 与目标同参数化的米制 state)
 16D placement ─────┼──> typed mRoPE ──> RAEVideoSceneFlow ──> ẑ₀ , ĉam(9D 米制) , ŝky , ĝauge(3D)
  sky spherical ────┘      (28×1440 + DDT)          │  ▲                              │
                                                    │  └──── gauge_context 进 cond ────┘
                                                    │        (双向耦合：几何以尺度为条件，
                                                    │         video flow 梯度回流进 gauge)
                                                    ↓ (D_JST, 冻结；梯度穿过)
                                      四层 3072 维 feature ──> 冻结 depth/GS/instance heads
                                                    │              │
                                            L1 level loss    L2 head loss
                                                    │              ↓ (gsplat, gauge K + teacher pose)
                                                    │        L3 render loss vs 真实像素
                                                    │
                              ĝauge ──(exp)──> 米制边界: c_depth loglinear
                                                    ↓
                                     米制点云 / 米制 PLY / 米制 actor 断言
```

**离线 GT 表（Phase 1a，teacher 空间，与 tokenizer 无关）**：
`data/scene_gauge/{training,validation}.json` 按 `(scene, trunk)` 查表。
training 4787/4787 trunks / 798 scenes、validation 1212/1212 / 202 scenes，0 errors；
`log_metric_scale = log(1/s_\text{lidar})`，`s_lidar = median(dggt_depth / lidar_depth)`，
逐帧 median → 跨 29 帧 MAD 去异常 → median；`log_tan_half_fov` 严格按 $\text{mean}(\log\tan(\text{FOV}/2))$ 构造。
无效通道**逐通道 mask**，样本仍参与其余全部训练。

优势不来自任何单一模块，而来自"**在哪个空间生成、用什么单位、条件怎么进来、误差在哪里测**"
四者相连：

$$
\text{driving specification (metres)}
\to \text{scene state without built-in scale}+\text{generated metric scale}
\to \text{decoded metric geometry}
\to \text{supervision}
$$

---

## A.8 Waymo 相机与 DGGT 相机的实测标定（2026-07-31 实跑）

> 本节数字来自六个可复现诊断：`lyy_tools/verify_camera_gauge.py`、
> `lyy_tools/verify_window_scale.py`、`lyy_tools/verify_fov_consistency.py`（D1）和
> `lyy_tools/verify_gauge_gt.py`（D2），以及在前述脚本上扩展的 D3 与
> `tools/retest_scene_flow_gaussian_gauge.py`（D4）。环境为 `conda activate dggt`，
> GPU 为 `CUDA_VISIBLE_DEVICES=0`。D1–D4 均覆盖场景 300–329、每场景 trunk 0/1/2，
> 共 **90 个完整 29 帧 trunk**；D3/D4 的精确环境、哈希与取值见 A.8.9。

### A.8.1 起因

A.5 只说了"两种相机量必须分清"，没有回答更基本的问题：**20 维 Waymo 物理条件与 11 维 DGGT camera state 之间是不是确定性关系？** 若是，相机生成分支不过是在做一次坐标变换；若不是，模型被要求学的目标里就含有条件解释不了的成分。

### A.8.2 验证方法

所有尺度统一定义为

$$
s=\frac{\text{DGGT unit}}{\text{metric metre}}.
$$

1. **基础相机标定**（`verify_camera_gauge.py`）：冻结 DGGT aggregator +
   CameraHead 跑完整 29 帧，Waymo 侧取
   `inv(anchor) @ camera_to_world_corrected`；逐帧比较旋转与轨迹形状。历史 10 帧
   尺度复现仍按原协议报告，用于与已验证数字逐项对齐。
2. **D1 FOV 自洽性**（`verify_fov_consistency.py`）：一次 29 帧 bf16 aggregator
   前向同时解码 camera/depth/point/GS head。决定性测试是静态区域的
   **primitive-level leave-one-frame-out** 渲染：目标帧产生的 Gaussian means 被完全
   排除，同一个候选 $K$ 同时用于源帧 depth unprojection 与目标帧 rasterization，
   从而避免同帧 $K$ 抵消和只替换 rasterizer 对 $K_{pred}$ 的机械偏置。固定 GT
   static mask 与 shared-alpha support 各算一遍 PSNR；默认 stride=2，目标帧为
   0/7/14/21/28。分支统计仅用 29 帧最大两两相机中心跨度 $\gt 2$ m 的 trunk，并先在
   scene 内平均 trunk，再做 10,000 次 scene-cluster bootstrap。
   aggregator 仍看过目标 RGB，因此这是 decoded geometry 的跨视角自洽性测试，
   不是 target-masked encoder 的 novel-view 泛化实验。
3. **D2 三把尺度尺**（`verify_gauge_gt.py`）：
   - Lidar 主尺：每帧取 `median(DGGT depth / lidar depth)`，只用 $1 \lt d \lt 80$ m
     像素，帧间 MAD 去异常后再取 29 帧中位数；
   - Camera 尺：仅用 29 个相机中心做 Umeyama Sim(3)，最大两两跨度 $\le2$ m 时
     显式标为无效；
   - Actor 尺：不读取 lidar，以 Waymo 米制 OBB 与语义 actor 像素（本次默认只取
     vehicle）的 ray-box
     entry/exit 给出尺度区间，经 object/frame 最大共识及帧间 MAD+median 聚合。
     Actor 与 Lidar 的**米制参考不同**，但两者共享 DGGT depth，故不具统计独立性，
     Actor 只作诊断而不作离线 GT。
4. **FOV**：Waymo 内参经 `resize_crop_intrinsics_to_model_canvas` 映到 518×350
   canvas；DGGT pose 的通道顺序为 `[..., FOVy, FOVx]`。Branch-A 可实现的 trunk
   常量严格按 `mean(log(tan(FOV/2)))` 构造，不使用角度的算术平均。

> **复现时的坑**：`dggt_window_indices` 是 **trunk 局部**索引（0–28），激光雷达文件必须用全局帧号 `trunk*29 + local`。第一版脚本用局部索引取雷达，trunk 1/2 的 $s_{\text{depth}}$ 全是错的。

### A.8.3 结果：哪些准，哪些不准

| 分量 | 由 Waymo 相机参数决定？ | 实测（90 窗口） |
|---|---|---|
| 世界系原点 | **是** | $\lvert c2w[0]-I\rvert$ 最大 **7.1e-4**；DGGT 世界系 = 第 0 帧相机，与 Waymo camera-to-anchor 同一约定 |
| **旋转 / 朝向** | **是** | 逐帧测地角误差 平均 **0.168°**，最大 1.157° |
| **轨迹形状** | **是** | 拟合相似变换后残差 = 路径长度的 **0.52%**，最大 2.38% |
| **平移尺度** | **否** | $s_{\text{cam}}\in[0.016,0.041]$，CV **23.5%**；1 DGGT 单位 = **24.8–64.2 米** |
| **FOV** | **否** | GT 49.85°±0.26°（Waymo 前视恒定），DGGT 预测 **38.13°±9.51°**，范围 16.3°–65.3°，平均偏 **−11.7°** |

于是两个空间的关系可近似写成一行，**平移的主导差异就是一个标量**（旋转与轨迹
形状仍保留上表量级的小残差）：

$$
R_{\text{DGGT}}[t]\approx R_{\text{Waymo}}[t],\qquad
t_{\text{DGGT}}[t]\approx s\cdot t_{\text{Waymo}}[t]
$$

**尺度不是每场景的常数，而是每 29 帧窗口的常数**：scene 301 三个 trunk 给出 0.0281 / 0.0343 / 0.0407（CV 15.0%），scene 302 给出 0.0234 / 0.0307 / 0.0325（CV 13.6%）。

**关键的正面结果——相机与深度共用同一把尺子：**

$$
s_{\text{cam}}/s_{\text{depth}}=1.0073\pm0.0442,\qquad
\mathrm{corr}(s_{\text{cam}},s_{\text{depth}})=0.980
$$

这是 legacy 10 帧口径；D2 的完整 29 帧移动组进一步给出
$s_{\text{cam}}/s_{\text{lidar}}=0.99995\pm0.02639$、corr 0.99296。它只支持
**camera center 与 direct z-depth 近似共用一维 gauge**，不能推出完整欧氏 3D 是纯
Sim(3)：FOV 差异带来约 25% 的横向压缩，而 tokenizer 往返后的 depth 与 Gaussian
scale 又有独立偏差。A.8.5 与 A.8.9 分别用 leave-one-out render 审计 K 与相机外参，
不能由两把尺度尺的相关系数替代。

### A.8.4 机制（代码层面已确认）

`train.py:82-93`：DGGT 训练**只**打开 `gs_head`、`instance_head`、`sky_model`，`camera_head` 全程 `requires_grad=False` 且不在 optimizer 中；`train.py:127-128` 的 `point_map` 经 numpy 往返，梯度链在那里已断。**所以 `camera_head` 是原封不动的 VGGT 权重**，沿用 VGGT 的逐场景尺度归一化，米制信息从未进入这个 head。`camera_generation.py:121-126` 的 docstring 也明确禁止把 Waymo 外参转入目标空间——这是有意设计，不是疏漏。

### A.8.5 D1：FOV 的决定性结果是 Branch A

D1 的 90/90 trunk 均成功。判据人口为 70 个有足够视差的 trunk、26 个 scene；
20 个低运动 trunk 仍参与 FOV、PSNR 与 support 的全体描述统计，但不进入 PSNR
分支判定。两种评测 mask 都给出同一个结论：

| 比较（dB，正值表示左侧更好） | mean | scene bootstrap 95% CI | 判定 |
|---|---:|---:|---|
| trunk-mean $K_{pred}$ − $K_{Waymo}$，fixed static mask | **+0.472** | **[+0.253, +0.741]** | A |
| trunk-mean $K_{pred}$ − $K_{Waymo}$，shared-alpha support | **+0.539** | **[+0.298, +0.829]** | A |
| native per-frame $K_{pred}$ − $K_{Waymo}$，fixed static mask | +0.580 | [+0.331, +0.876] | A |
| native per-frame $K_{pred}$ − $K_{Waymo}$，shared-alpha support | +0.677 | [+0.418, +0.978] | A |

可生成的 trunk-mean K 相对 native per-frame K 仅下降 −0.108 dB（fixed，CI
[-0.151, −0.070]）和 −0.150 dB（shared-alpha，CI [-0.195, −0.107]），均通过
预注册的 −0.2 dB non-inferiority margin；该项按 JSON 标记仅为诊断
（`used_for_branch_decision=false`），Branch A 要求 trunk-mean 与 native per-frame
两条相对 Waymo 的比较同向。全体 90 个 trunk 的 shared-alpha support 占 fixed
static support 的均值为 97.24%，各 trunk 跨目标帧最小 support fraction 的
均值/最小值为 90.25%/66.61%，结论不是由两套 K 的空洞区域差异制造的。
若把低运动 trunk 也纳入纯描述统计，fixed-mask leave-one-out PSNR 的全体均值为
20.512 dB（trunk-mean $K_{pred}$）对 20.139 dB（$K_{Waymo}$）。

FOV 在同一 trunk 内确实近似常量，而主要变化发生在 trunk 之间：

| FOV 方差 | FOVx | FOVy |
|---|---:|---:|
| 29 帧内 std：mean / median / max | 0.262° / 0.179° / 1.768° | 0.178° / 0.112° / 1.343° |
| trunk mean 的跨 trunk std | 9.376° | 6.624° |

因此采用 **Branch A**：gauge 的 FOV GT 是 DGGT 预测的 29 帧
`mean(log(tan(FOV/2)))`。**DGGT 链路**（depth unprojection、sky atlas 与 RGB render）
统一使用生成 gauge 解出的 K；**米制链路**（`in_frustum`、bbox 投影、
`target_bbox_patch`）保持真实 Waymo K，两条链路不交叉。90 个 trunk 的可实现 trunk FOV 均值为
38.324°/26.892°（x/y），跨 trunk std 为 9.376°/6.625°；Waymo 对照为
49.848°/34.426°。

PointHead 几何残差也被实跑：跨 90 个 trunk，$K_{pred}$ / $K_{Waymo}$ 的
trunk-median residual 再取均值分别为 relative-L2 2.978/3.009、角度
133.84°/129.81°、重投影 969/657 px。三种候选 K 在 90/90 trunk 上均未通过宽松的
坐标兼容门槛（median angular $\le30$° 且 median reprojection 不超过图像对角线的
10%），因而判为 `coordinate_incompatible`。这不等于 PointHead 不能以任何未知
约定重投影，而是说明当前脚本测试的三种 K/坐标解释都不成立。因此该 head 在当前
checkpoint 下不能作为 D1 证据，结果 JSON 明确标记
`used_for_branch_decision=false`；Branch A 只由与当前实际 depth→GS 装配路径一致的
leave-one-out 渲染决定。

### A.8.6 D2：三把尺子与主 GT 选择

D2 同样是 90/90 trunk 成功且 Lidar 主尺 90/90 有效。legacy 10 帧协议精确复现
已知结果：24 个静止、66 个移动，移动组

$$
s_{\text{cam}}/s_{\text{depth}}=1.00729\pm0.04422,\qquad
\mathrm{corr}=0.98013.
$$

改用完整 29 帧与最大两两中心跨度门控后，20 个 trunk 判为静止、70 个为移动；
移动组一致性进一步收紧为：

| 尺子比较 | n | ratio mean ± std | median | corr | 5% 内 |
|---|---:|---:|---:|---:|---:|
| 29f Camera / Lidar | 70 | **0.99995 ± 0.02639** | 0.99526 | **0.99296** | 92.86% |
| Actor / Lidar（移动组） | 26 | **0.98878 ± 0.02245** | 0.99320 | **0.99780** | 96.15% |
| Actor / Camera（移动组） | 26 | 0.99346 ± 0.03174 | 0.99748 | 0.99594 | 92.31% |

静止组给出选择主尺子的直接证据：Camera Umeyama 在 20/20 上按定义无效，而
Lidar 尺在 20/20 上有效。静止组 29 帧逐帧尺度 robust CV 的 mean/median/max 为
0.688%/0.263%/5.777%，frame max/min 的 mean/median/max 为
1.029/1.010/1.186。全体 Lidar 尺范围为 [0.01548, 0.05076]，median 0.02790。

Actor 尺的正面结果只在有运动且可见性充分时成立。全体 90 个 trunk 中仅 40 个有
可用 actor interval、29 个形成最终 actor point。3 个静止有效点的 Actor/Lidar
分别为 0.9425/0.5405/0.5434，其中后两个是灾难性离群；这只能归因于 OBB 表面代理、
语义像素与可见性假设在这些样本上的联合失效，不能归结为单一因素。它们使全体
Actor/Lidar 降为 0.956±0.115、corr 0.818。全体 robust interval 对 Lidar 的覆盖率为
68.97%，移动组为 76.92%。再加上 Actor 与 Lidar 共享 DGGT depth，它不能充当独立
主尺。

> **结论：离线 gauge GT 必须使用 29 帧 Lidar 深度尺。** Camera 是有运动样本上的
> 高精度交叉验证；Actor 是覆盖有限、对 OBB/可见性敏感的第三诊断。不能只用相机
> 轨迹，否则约四分之一的 legacy 窗口会静默发散；也不能把 actor box 提升为主 GT。

### A.8.7 窗口起点与滑动窗对尺度的影响（脚本 `lyy_tools/verify_window_scale.py`）

训练窗口不总是从 trunk 第 0 帧开始（为支持滑动窗推理，`camera_anchor_window_probability` 控制含 anchor 窗口的比例），推理更是显式滑动窗。因此必须确认：**尺度是窗口的属性还是 trunk 的属性？**

**(1) 同一 trunk 内，窗口起点不影响真实尺度——按构造如此。** aggregator + CameraHead 在完整 29 帧上**只跑一次**（`train_scene_flow_pretrain.py:5532-5551`），10 帧窗口由 `batched_gather_frames` 从中取出，所有窗口共享同一组 DGGT 位姿。

实测（28 个有位移的 trunk × 5 个窗口起点 0/5/10/14/19）：

| 指标 | 数值 |
|---|---|
| 窗内尺度**估计**的 CV | mean **3.1%**，median **1.9%**，max 8.8% |
| 窗内 max/min | mean 1.09，max 1.27 |
| 静止 trunk（303/311） | **完全发散**，max/min 达 23× 与 30× |

> **实现约束 1：`log_metric_scale = log(1/s)` 的 target 必须用整段 29 帧估计，不要用 10 帧窗口估计。** 窗内那 2–3% 是估计噪声（10 个相机中心 vs 29 个），用窗口算等于给同一个物理量喂进随机抖动。

**(2) 跨 trunk 的尺度是真的会变。** 同场景 trunk0 vs trunk1（均用 29 帧全量估计），相对差 mean **10.6%**、median 10.2%、**max 25.1%**（301: 0.0274→0.0343；302: 0.0238→0.0289；305: 0.0269→0.0325）。这不是估计器问题——激光雷达独立算的 $s_{\text{depth}}$ 漂移一致（301 三 trunk: 0.0278/0.0341/0.0379）。**是冻结 VGGT 本身在不同片段选了不同尺度。**

**(3) 推理滑动窗不会重新选尺度。** 采样循环（`train_scene_flow_pretrain.py:3455-3480`）的融合发生在 **ODE 循环内部**：

```
每个去噪步：
  for 每个窗口: 前向 → 该窗口的 velocity
  cosine 加权累加进全局 v_acc / camera_acc
  用全局 v_acc 对整条序列的 z 与 camera_z 走一步 Euler
```

整条序列**只有一个 `camera_z` 张量、一个 anchor**（`camera_gen_anchor_mask = frame_ids.eq(0)`），后续帧由它链式积分。**不是**"生成窗口 1 再把窗口 2 接上去"。

> **实现约束 2：`log_metric_scale = log(1/s)` 做成 scene-global token，与 sky 同类。** 机制现成——`sliding_window.py:137-146` 的 `scene_global_window_weight`，已在 sky 分支使用（`train_scene_flow_pretrain.py:3624`）。
> - 不能挂在 anchor 帧：训练存在 delta-only 窗口（不含全局第 0 帧），那些窗口无处安放；
> - 不能做成逐帧通道：会被窗口融合当作逐帧量处理；
> - 概念上尺度本就是整段场景的属性。

**(4) 一条必须写进 limitation 的训练/推理不匹配。** 训练 target 是逐 29 帧 trunk 的，而 teacher 跨 trunk 漂移 10–25%。因此：≤29 帧生成干净（一 trunk 一尺度）；**长序列滑动窗生成会得到一个全局尺度，它实际上比 teacher 更自洽，但不存在一致的 GT 尺度可供评测**。这条要主动写出，不要等审稿人问。

### A.8.8 Phase 0 冻结的实现决策与复现命令

D1–D4 把后续原本未定的取值冻结如下；D3/D4 的最终数值与完整命令见 A.8.9：

1. gauge 三维定义为
   `[log(metres/DGGT-unit), log_tan_half_fovx, log_tan_half_fovy]`；
   第一维 GT 只来自完整 29 帧 Lidar 尺，后两维来自 DGGT 29 帧预测的
   `mean(log(tan(FOV/2)))`。
2. 相机生成目标改为 9 维米制 Waymo 相对 SE(3)，FOV 不再留在相机流中；D3 只让
   **训练 RGB render pose** 回退 teacher 空间，不撤销米制相机生成目标。
3. 两条内参链路永不交叉：米制 box/`in_frustum`/`target_bbox_patch` 保持真实 Waymo K；
   DGGT depth unprojection、sky atlas 构建/读出和 RGB render 使用 gauge K（Branch A）。
4. Actor 尺只保留为离线表的诊断字段；移动 Camera 尺用于交叉验证。二者均不替代
   Lidar 主 GT。
5. 29 帧是 GT 计算单位；10 帧窗口只消费查表结果。滑动窗与非滑窗采样必须生成同一个
   scene-global gauge，并以测试证明两条路径输出一致。
6. D4 的 **renderer 作用域**没有批准任何 tokenizer pullback：`c_depth=1.0`、
   `c_gs=1.0`。这不能外推到有独立 LiDAR 物理尺的米制导出；后者由 A.8.10 的
   Phase 1b-0 gate 单独决定。

完整 D1 命令：

```bash
source /home/dancer/anaconda3/etc/profile.d/conda.sh
conda activate dggt
CUDA_VISIBLE_DEVICES=0 GAUGE_DEVICE=cuda:0 \
python lyy_tools/verify_fov_consistency.py \
  --scenes 300-329 --trunks 0,1,2 \
  --output /tmp/verify_fov_consistency_full.json --fail-fast
```

D2 的单条 90-trunk 命令也已完整实跑，产物为
`/tmp/verify_gauge_gt_full.json`（`status=complete`、90/90、0 error）。此外还按
300–309/310–319/320–329 三个互斥 scene 分片独立复跑，校验合并后恰为 90 个唯一
`(scene,trunk)`、无 missing/extra，且与单条命令逐值一致：

```bash
CUDA_VISIBLE_DEVICES=0 GAUGE_DEVICE=cuda:0 \
python lyy_tools/verify_gauge_gt.py \
  --scenes 300-329 --trunks 0,1,2 \
  --output-json /tmp/verify_gauge_gt_full.json --strict
```

对应 CPU 单元测试为 `tests/test_verify_fov_consistency.py` 与
`tests/test_verify_gauge_gt.py`。它们覆盖 matched-K leave-one-out、scene bootstrap
判据、log-tan trunk FOV、29 帧 motion gate、逐帧 Lidar MAD、centres-only Umeyama、
actor OBB ray-box 与静止 trunk 稳定性。

### A.8.9 D3/D4：训练 render 相机与冻结解码器 pullback gate（2026-08-01）

#### D3：只换外参的决定性渲染测试

D3 保持同一批 world-space Gaussian primitives、同一个 trunk-constant gauge K、同一
support 与 Gaussian 属性，只把 target raster view 从 native DGGT teacher pose 换成
`metric_c2w_to_dggt(Waymo c2w, s_lidar)`。因此它测试的是**相机外参替换**，不是再次比较
Waymo K 与 gauge K。主损失定义为

$$
\Delta_{\text{cam}}=\operatorname{PSNR}(\text{teacher pose})
-\operatorname{PSNR}(\text{metric-converted pose}).
$$

固定门槛为 0.3 dB：scene-bootstrap CI 上界低于门槛才选 metric；CI 下界不低于门槛则选
teacher；其余为 inconclusive。90/90 trunk、30/30 scene 均有效，静止 trunk 也进入主 gate：

| mask | scene-balanced mean loss | scene-bootstrap 95% CI | 决策 |
|---|---:|---:|---|
| fixed static | **1.4143 dB** | **[1.0248, 1.8377]** | teacher |
| shared-alpha | **1.4357 dB** | **[1.0583, 1.8437]** | teacher |

**作用域纪律**：这两个 dB 数来自完整 29 帧 trunk 的冻结 D3 协议，衡量 teacher-space
交付轨迹相对 Waymo 请求轨迹的可测残差；它是**控制保真度上界诊断，不是推理画质损失**。
推理时相机与几何都由模型生成，没有对应真实图像可构造上述两臂。后面的 local-index 分层只能说明
残差随离锚点时间累积，也不能冒充另一次“不同窗口长度”实验。

motion split 只作描述：70 个 moving trunk 的 fixed/shared mean loss 为
1.7642/1.7919 dB，20 个 stationary trunk 为 0.1896/0.1890 dB。预先关注的三个离群
scene 确实靠前：

| scene | fixed mean / max | shared mean / max | fixed 排名 |
|---|---:|---:|---:|
| 314 | 4.5484 / 5.7150 | 4.3180 / 5.4162 | 1 |
| 312 | 3.7275 / 5.1206 | 3.2577 / 3.4177 | 2 |
| 325 | 2.4501 / 3.0653 | 2.3662 / 3.0539 | 6 |

**冻结值：相机生成 target 仍是 9D 米制 Waymo pose；训练
`compute_rgb_render_loss` 的 `render_camera_space=teacher`。** 推理仍使用生成的米制相机，
teacher/metric 残差作为 limitation 报告。D3 不推翻 D1：render K 仍是 gauge K。

#### D4：29 帧主尺的深度分层与形式 gate

D4 对 90 trunk × 5 个重叠 10 帧窗口（450 windows）重跑 production
aggregator→tokenizer→frozen heads 路径。每个 trunk 的五窗都严格共用 reference case 的
完整 29 帧 `s_lidar`，不使用任何 10 帧尺度。标定/选择 split 固定为 scene 300–319 / 320–329；
后十个 scene 被用于取值，所以应称 **selection/validation set**，不是 untouched test。

五个米制深度层的 scene-balanced correction 中位数为：

| depth bin (m) | representative depth | `c_depth` | `c_gs` | calibration scenes |
|---|---:|---:|---:|---:|
| [0,5) | 2.50 | 0.8978 | 1.2312 | 5 |
| [5,10) | 7.07 | 0.9465 | 1.2949 | 20 |
| [10,20) | 14.14 | 0.9675 | 1.2475 | 20 |
| [20,40) | 28.28 | 0.9617 | 1.2374 | 20 |
| [40,80] | 56.57 | 0.9695 | 1.2352 | 20 |

标定集形式 gate 对 `c_depth` 选择 loglinear：
`a=-0.0405706, b=+0.0146570, c(20m)=0.960241`，slope scene-bootstrap CI
[0.009262, 0.019018]，LOSO RMSE 相对 constant 改善 8.80%。`c_gs` 选择 constant：
`c=1.257629`；其 bin Spearman 为 0，loglinear 的 LOSO 改善仅 1.27%，未过 2% gate。
这些只是候选形式，不能越过 render gate 直接写入 render 路径；`c_depth` 是否服务于
米制边界由 A.8.10 的独立 LiDAR gate 决定。

#### D4：真实 renderer gate 与边界处置

相机策略由 D3 固定为 teacher；每个 target 做 primitive-level leave-one-frame-out，所有
候选共用 GT static/non-sky mask、K、view、color、quaternion 与 opacity。先扫描
`c_depth=(1/1.0307)×(1±3%)`、相对步长 0.2%，另含 identity；冻结其结果后再扫
`c_gs`。聚合顺序为 target→trunk→十个 scene，增益必须严格大于 0.05 dB。

| 轴 | identity PSNR | raw best | raw best PSNR | gain | 最终 |
|---|---:|---:|---:|---:|---|
| `c_depth` | 17.032570 | 0.99932085（网格上界） | 17.031637 | **−0.000933** | 不引入，1.0 |
| `c_gs`，原预注册 [1,1.5] | 17.032570 | 1.50（上界） | 17.377403 | +0.344833 | inconclusive，先扩网格 |
| `c_gs`，显式 sensitivity [1,2.5] | 17.032570 | 2.50（语义上界） | 18.030502 | **+0.997932** | 拒绝 renderer 取值，1.0 |

扩展网格的 scene-balanced PSNR 在 `c_gs=1.0/1.5/2.0/2.5` 分别为
17.03257/17.37740/17.71965/18.03050 dB；8/10 scene 的区间内最佳值仍为 2.5，
scene 321/327 则分别在 2.28/1.62 达峰。`c_gs=2.5` 会把 primary paired
GS/depth ratio 0.7964 推到约 **1.99× teacher**，已经失去“tokenizer pullback 逆变换”
的物理解释；持续上涨的 PSNR 在奖励 splat 覆盖/模糊，而不是识别一个全局几何尺度。

因此边界协议保守实现为：默认 `boundary_policy=block`；只有显式重跑精确的 2.5
semantic cap 才允许 `reject`。若 best-score 集仍触边，边界值**永不冻结**，输出
`renderer_pathology_rejected=true`，应用 identity。最终 D4 **renderer 结论**为
`status=complete_c_gs_rejected_renderer_pathology`，正式 recommendation 是
`c_depth=1.0, c_gs=1.0, form=identity, slopes=0`。这不否定 0.7964 的 tokenizer audit；
它只说明当前 leave-one-out RGB PSNR 无法把该 audit 唯一转成可部署的 `c_gs`。

#### 对 Phase 1–8 的最终输入

| 项 | Phase 0 冻结值 |
|---|---|
| 相机生成目标 | 9D 米制 Waymo pose |
| 训练 RGB render pose | **teacher** |
| DGGT depth/render/sky K | trunk-global **gauge K** |
| 米制 bbox/`in_frustum` K | 真实 **Waymo K** |
| render pullback | `c_depth=1.0`、`c_gs=1.0`，均 identity |
| v1 metric-boundary pullback | 由 A.8.10 的 LiDAR gate 选择 loglinear；原始诊断不可直接加载，冻结后的 production artifact 正式供当前 Phase 1–8 使用；v2 后重测 |

#### 可复现命令与绑定

```bash
source /home/dancer/anaconda3/etc/profile.d/conda.sh
conda activate dggt

CUDA_VISIBLE_DEVICES=0 python lyy_tools/verify_fov_consistency.py \
  --device cuda:0 --scenes 300-329 --trunks 0,1,2 \
  --render-stride 2 --geometry-stride 2 --bootstrap-samples 10000 \
  --output runs/metric_gauge_retest/verify_fov_consistency_d3_300_329_trunks012.json

CUDA_VISIBLE_DEVICES=0 python -u tools/retest_scene_flow_gaussian_gauge.py \
  --scenes 300-329 --trunks 0,1,2 --device cuda:0 --precision bf16 --depth-chunk 4 \
  --checkpoint /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
  --tokenizer-checkpoint /data/disk2/lyy_dataset/logs/tokenizer_t0_stageB/ckpt/scene_tokenizer_step_040000.pt \
  --result-json runs/metric_gauge_retest/a8_300_329_trunks012_production_path.json \
  --render-camera-policy teacher --render-window-start 10 --render-targets 0,5,9 \
  --render-stride 2 --c-gs-maximum 2.5 --c-gs-boundary-policy reject \
  --output runs/metric_gauge_retest/d4_pullback_300_329_trunks012_cgs250.json
```

绑定信息：DGGT checkpoint SHA-256 `352652738a5480b8d3ee9dd521ce07b528e5a297bd3feca4d07427dac6d87def`；
tokenizer SHA-256 `75e566efa3b66baa43f82cb9999c2de60a9f3feeb0f714e1caf38d1f6e8137eb`；
reference JSON SHA-256 `424451a754fbb728f0abb822aa3387d65f8f8c6ffa80f3119a0504c0bb29fd38`。
D3 script/result SHA-256 分别为 `d8ff57946f66348b97b493c80681459b57ccc5541528e2e7a4b9c95559ec9cda` /
`1498b1511fd293d159fc26b3f02c1b86ec80f044d80021f8e737f31623e55467`（D3 JSON 未内嵌
这些 hash，此处为运行后外部计算）；D4 schema 2.1.0 的 script/result SHA-256 分别为
`35213b0dd4773b4f979bf78e5b562dc919a12b93085eacbddcaf1689b6ae6d74` /
`94a33623c2607d88710f9f21f21dad6260114ec29bb9ae500ffb7f58258bcbda`。
D4 运行环境为 PyTorch 2.7.1+cu128、CUDA 12.8、NVIDIA RTX PRO 6000 Blackwell Server Edition；
耗时 1841.08 s。联合 CPU 回归为 **63 passed**，真实 CUDA 路径由上述两次 90-trunk 全量运行覆盖。

### A.8.10 Phase 1b-0：v1 tokenizer 的 LiDAR metric-boundary gate（2026-08-01）

D4 的 PSNR 只回答“校正是否改善 renderer”，不能回答“导出的米制深度是否正确”。因此本节
冻结 D4 在 calibration scenes 300–319 拟合的 loglinear profile，不在 selection 上重拟合：

$$
z_0=\mathrm{depth}_{\text{recon}}/s_{\text{lidar}},\qquad
c(z_0)=\exp\left[
-0.0405706428+0.0146570329
\log\frac{\mathrm{clamp}(z_0,0.5,80)}{20}
\right].
$$

在 scenes 320–329、每场景 trunks 0/1/2 上，读取原始 `depth_flows_4[...,0]` 相机 z-depth；
dense recon depth 只采样到原始非零 LiDAR cell center，`align_corners=False`，不 resize 稀疏零值图。
每个 trunk 只使用完整 29 帧的 `s_lidar`，五个 10 帧窗口固定起点 0/5/10/14/19。
误差先逐 cell 计算，再按 frame→window→trunk→scene 取 median；bootstrap 只重采样 10 个 scene。

| 口径 | trunk / scene | identity AbsRel | loglinear AbsRel | 相对改善 | scene mean Δ bootstrap 95% CI | 改善方向 |
|---|---:|---:|---:|---:|---:|---:|
| 主口径：Phase-1a 阈值有效、全部 LiDAR cells | 26 / 10 | **7.567%** | **6.901%** | **8.81%** | **[0.052%, 1.225%]** | 8/10 scene |
| sensitivity：全部 30 trunk | 30 / 10 | 7.599% | 6.913% | 9.02% | [0.124%, 1.226%] | 8/10 |
| sensitivity：有效 trunk、static/non-sky | 26 / 10 | 7.614% | 6.935% | 8.91% | [0.062%, 1.248%] | 7/10 |

主口径排除的四个 trunk 为 324/2、326/0、329/0、329/1，原因均是逐帧 robust CV >3%；
每个 selection scene 仍至少保留一个 trunk。预注册规则要求
`identity AbsRel − loglinear AbsRel` 的 scene-bootstrap 95% CI 下界严格大于 0；主口径下界为
0.000517，因此 **v1 metric boundary 选择 loglinear**。scene median absolute improvement 为
0.7566 个百分点。作为保守敏感性，精确 sign-flip 单侧 `p=0.0352`、双侧 `p=0.0703`，且
scene 328/329 变差；所以应写成“按预注册 bootstrap gate 通过，但证据温和”，不能写成普适常数。

作用域必须分开：训练/推理 **render 仍用 identity**；loglinear 只服务米制导出、目标满足度评测的
米制断言与 `metric_depth_rel_err`。`c_gs=1.0` 不变，GS/depth=0.796 的缺陷没有被本测试修复。
本次 tokenizer SHA-256 为
`75e566efa3b66baa43f82cb9999c2de60a9f3feeb0f714e1caf38d1f6e8137eb`，产物明确记录
`artifact_role=diagnostic_only_v1`、`eligible_for_training=false`。tokenizer v2 会直接改变 depth
bias/profile，出来后必须在 300–319 重拟合、在 320–329 原样重跑。这里的“不可训练加载”只指
**原始实验 JSON**；用户随后冻结了单独的、严格哈希绑定的 v1 production artifact，见下。

复现命令：

```bash
source /home/dancer/anaconda3/etc/profile.d/conda.sh
conda activate dggt
CUDA_VISIBLE_DEVICES=0 python tools/calibrate_tokenizer_pullback.py \
  --device cuda:0 --precision bf16 --scenes 320-329 --trunks 0 1 2 \
  --bootstrap-samples 10000 \
  --output runs/metric_gauge_retest/v1_tokenizer_lidar_metric_gate_320_329.json
```

脚本/result 文件 SHA-256 分别为
`4162bbc469ede617056333e1e57dde124f53456f215bfb4578ddd6eab0e05eae` /
`29f91842641c2bfc7565d4edbe0f711223ba6eb006de1ed2ded8d5d72a97ecb9`；JSON 内部去 self-field 的
canonical payload SHA-256 为 `87fc01c4f7bbd5f62fe40711d8fb82125c0e758a8dadce9b3e2e243e00cd6de1`。
输入 DGGT/reference/D4 SHA 与 A.8.9 一致。环境为 PyTorch 2.7.1+cu128、CUDA 12.8、
NVIDIA RTX PRO 6000 Blackwell Server Edition，真实 GPU 耗时 186.94 s。新增单测 14 passed；
连同冻结 D4 回归为 **33 passed**，另有 CPU synthetic 与独立 result-contract 校验通过。

#### A.8.10.1 v1 production 冻结（当前 Phase 1–8 的正式输入）

tokenizer v2 尚在训练期间，当前 Scene Flow **不等待 v2**。通过
`tools/freeze_tokenizer_pullback.py` 将上面的冻结 evidence 转成
`data/scene_gauge/pullback_75e566ef.json`：

- `artifact_role=production_pullback`、`eligible_for_training=true`；
- artifact SHA-256
  `1bb159e374e2b1d00af5020f780ada9f74d84a1365a525bc484fccb6a4e34693`；
- 严格绑定 tokenizer v1、DGGT checkpoint、`window_len=10`、`patch_grid_hw=[25,37]` 与 gauge
  representation；任何不匹配都拒绝加载；
- metric boundary 使用上述 loglinear `c_depth(z0)`，并把同一 factor 乘到 depth 与 GS scale；
  render boundary 返回原生 depth/GS tensor（identity）；`c_gs=1.0`。

这不是把 §8/本节的诊断 provenance 改名，而是新增一层 runtime contract。它只正式修正 v1 的
米制 depth 偏差；paired `GS/depth≈0.796` 仍是显式 limitation，不能宣称当前渲染几何相似一致。
tokenizer v2 是该缺陷的根治路径，但不是当前 Phase 1–8 blocker；v2 出来后必须重新拟合、重测，
并依据相同 gate 决定保留 identity、冻结新 profile 或调整方案，不能沿用 v1 artifact。

### A.8.11 Phase 1b：tokenizer v2 正式复测与方案 A（2026-08-05）

本节不改写 A.8.9/A.8.10 的 v1 provenance。v2 checkpoint 为
`logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt`，完整 SHA-256
`d63b34f7b1193ed7da399f953db504cfadb4f98dce2519854227a0f44714c8e8`；DGGT checkpoint SHA-256
为 `352652738a5480b8d3ee9dd521ce07b528e5a297bd3feca4d07427dac6d87def`。checkpoint
`global_step=100000`，当前 `JointSceneTokenizer` 的 460-key state 严格加载通过，model/optimizer
tensors 均 finite；训练 args 记录 `min_frames=10,max_frames=14` 与三项 v2 loss 权重，但没有显式
`objective_version` 或 formal `window_len` 字段。训练日志不在本机，因此无法从日志核实
`gs_scale_sim_ratio`、depth log bias、support fraction、三项 loss 曲线、NaN 或历史 cache 污染。
本轮不修改或背书 tokenizer trainer 的历史日志口径；缺失日志里的空 support 与 cache 聚合状态
仍是明确 limitation，不能由 checkpoint strict-load 或下游 audit 倒推。

> 本节以下正式数字来自已完成的 Gaussian schema 2.3.0 与 LiDAR schema 2.1.0 结果。之后当前
> Gaussian 工具升级到 2.4.0，并删除未启用的 render/PSNR/`c_gs` 扫描实现；按用户决定没有再次执行
> 90-trunk audit 或 LiDAR selection。代码清理后的单元测试、CPU synthetic 和单 trunk CUDA 检查
> 不能替代或改写下面的正式 provenance。

正式 audit 使用完整 29 帧 Aggregator、五个 10 帧窗口（起点 0/5/10/14/19）、bf16
Aggregator/tokenizer 与 FP32 DepthHead/GaussianHead，覆盖 scenes 300–329、trunks 0/1/2，
共 30 scenes / 90 trunks / 450 windows。calibration scenes 300–319 只拟合；selection scenes
320–329 只选择，不反向修改 form、系数、clamp、support、窗口或统计。PSNR scan 默认关闭，
没有用 PSNR 选择 `c_gs`。

primary static/non-sky/opacity>0.05 support 的 v1/v2 对比为：

| scene-balanced 指标 | tokenizer v1 | tokenizer v2 |
|---|---:|---:|
| `depth_recon/depth_direct`（median-log） | 1.0399 | 1.0419 |
| Gaussian geometric-axis ratio（median-log） | 0.8289 | 1.0331 |
| paired `GS/depth` | **0.7964** | **0.99985** |
| paired scene IQR | 0.7876–0.8120 | 0.9821–1.0101 |
| v2 x/y/z ratio | — | 1.0667 / 1.0000 / 1.0363 |

v2 paired point estimate 为 `0.9998545510`。在查看正式结果前冻结的 practical-equivalence margin
为 `[0.95,1.05]`；scene-only bootstrap 10,000 次的 95% CI 为
`[0.9901538116,1.0075338585]`，点估计与整个 CI 都在 margin 内，故 **GS/depth gate 通过**。
same-cell、90-trunk case-balanced depth round-trip mean/p50/p95 为
`1.03358/1.03685/1.06966`；独立 depth bias 没有完全消失，所以仍需 LiDAR gate。

calibration split 选出的冻结 depth 候选是 loglinear：

```text
log c(z0) = -0.02800761462 - 0.04383812022 * log(clamp(z0,0.5,80)/20)
c(20m) = 0.97238096246
```

constant 候选为 `c=0.97771399915`，identity 为 1。LiDAR selection 严格读取原始
`depth_flows_4[...,0]` 相机 z-depth 与 `1m<z<80m` support；dense prediction 采样到原始
LiDAR cell center，`align_corners=False`，不 resize 稀疏零值图。每个 trunk 只使用完整 29 帧
`s_lidar`，聚合为 pixel median→frame→window→trunk→scene，bootstrap 只重采样 scene：

| 口径 | identity | frozen loglinear | mean Δ（95% CI） | 改善 scene | sign-flip 单侧 p |
|---|---:|---:|---:|---:|---:|
| Phase-1a valid 26 trunks，all LiDAR | **0.077615** | 0.085069 | -0.007453 `[-0.015526,0.001132]` | 3/10 | 0.9346 |
| 全 30 trunks sensitivity | **0.077050** | 0.084283 | -0.007233 `[-0.015215,0.001322]` | 3/10 | 0.9287 |
| valid 26，static/non-sky | **0.077716** | 0.085734 | -0.008019 `[-0.016200,0.000727]` | 3/10 | 0.9443 |

三种口径都没有满足 point Δ>0 且 CI 下界严格大于 0，候选点估计反而更差，因此选择 identity。
v1 主口径曾是 identity `0.075671`、loglinear `0.069006`、Δ `+0.006665`、
CI `[0.000517,0.012248]`、8/10 scenes，正确选择 v1 loglinear；v2 重新拟合后得到相反结论，
没有沿用 v1 的 `a/b`。

固定 CUDA render smoke 的 `render_direct` PSNR/SSIM/LPIPS 为
`39.296 dB/0.97892/0.01618`，depth/GS/paired ratio 为
`0.99928/1.00299/1.00317`，point XYZ relative error `0.00591`，通过预注册阈值，未见明显退化。
综合 GS gate、LiDAR gate 与 smoke，Phase 1b 选择 **方案 A**：render depth identity、metric depth
identity、`c_gs=1`；保留 boundary 显式区分、tokenizer/DGGT/artifact SHA、10-frame window、patch grid
与 gauge representation 的 fail-closed 校验，以及精确 identity no-op helper。

production artifact 为 `data/scene_gauge/pullback_d63b34f7.json`，
`artifact_role=production_pullback`、`eligible_for_training=true`；v1 artifact 不覆盖。原 freeze 输出的
SHA-256 `d24e23f77bcd7b51cb022a591ac9cdee3a7108d233f00b2ae9a9ae8ea7d550fb` 只对应 clean-cut 前文件；
本轮只把 PSNR 否定字段迁移成显式 identity `c_gs` recommendation，按用户要求未重新登记 artifact SHA。
完整 provenance：reference script/result
`9e91dd09c7057d5cf2a04a6027e2bf8088aee6ce400c1121a71ff1c4ae15a3e1` /
`2416f97b4afed0d9bf33556841cd419574b70dde1598474c5e4cd03899cf112b`；Gaussian script/result
`ebe3e49867fde426908a393fe3774b6e36fa6a6ff5ec35e7876dfac91984d10d` /
`8676b7767f3ddda6097331466dc0db30f0fc8d35ce7e09ecb82a8550b27b95d6`；LiDAR script/result
`9a206db00f58cdce870f1c86c85bbe56560bd409cbfc2f8e37e1cdd33a33c0b4` /
`ab82b2884afd2d40aa6e02a78abcae27185942251288a497ca5bcf281615c2b8`；freeze script
`3e96b000245fa74b81e1fa2794ab620d08e1c6e9305c8756430a804a3acce46f`；smoke/selection manifest
`2519d4e3e21ed5f353fe4951d268f22f0c7d4eeae705b5cb6e29503e01690c89` /
`ce8290c997c2f9e5c9fd600ebd4178e86d40797077ae2874fb2774a7c1ca8cc6`。

关键历史命令如下；环境均为 `CUDA_VISIBLE_DEVICES=0`、conda `dggt`、代码设备 `cuda:0`。
`--skip-d4-render-scan` 属于结果所绑定的 2.3.0 脚本，当前 2.4.0 parser 已删除该参数：

```bash
conda run --no-capture-output -n dggt python -u tools/retest_scene_flow_metric_gauge.py \
  --scenes 300-329 --trunks 0,1,2 --device cuda:0 --precision bf16 --depth-chunk 4 \
  --checkpoint /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
  --tokenizer-checkpoint logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt \
  --roundtrip-window-starts 0,5,10,14,19 --roundtrip-window-length 10 \
  --bootstrap-repetitions 10000 \
  --output-json runs/metric_gauge_retest/v2_metric_reference_300_329_trunks012_d63b34f7.json

conda run --no-capture-output -n dggt python -u tools/retest_scene_flow_gaussian_gauge.py \
  --scenes 300-329 --trunks 0,1,2 --device cuda:0 --precision bf16 --depth-chunk 4 \
  --checkpoint /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
  --tokenizer-checkpoint logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt \
  --result-json runs/metric_gauge_retest/v2_metric_reference_300_329_trunks012_d63b34f7.json \
  --skip-d4-render-scan --d4-form-bootstrap-samples 10000 \
  --paired-equivalence-bootstrap-samples 10000 \
  --output runs/metric_gauge_retest/v2_gaussian_gauge_300_329_trunks012_d63b34f7.json

conda run --no-capture-output -n dggt python -u tools/calibrate_tokenizer_pullback.py \
  --device cuda:0 --precision bf16 --scenes 320-329 --trunks 0 1 2 --bootstrap-samples 10000 \
  --checkpoint /data/disk2/lyy_dataset/model/dggt/model_latest_waymo.pt \
  --tokenizer-checkpoint logs/tokenizer_t0_v2_stageA/ckpt/scene_tokenizer_step_100000.pt \
  --reference-json runs/metric_gauge_retest/v2_metric_reference_300_329_trunks012_d63b34f7.json \
  --d4-json runs/metric_gauge_retest/v2_gaussian_gauge_300_329_trunks012_d63b34f7.json \
  --output runs/metric_gauge_retest/v2_tokenizer_lidar_metric_gate_320_329_d63b34f7.json
```

clean-cut 后覆盖 Gaussian/calibrator/freeze、production loader、formal provenance、metric/render/
inference、feature stats 与 window contract 的合并定向回归为 **218 passed**；严格 loader 会从
per-scene rows 重算 primary Gaussian/LiDAR bootstrap CI，并拒绝手工构造的 v1 calibration；另有 CPU
synthetic 与 CUDA 0 单 trunk 路径检查通过。正式 90-trunk audit 与 LiDAR gate 沿用上面已绑定的既有结果，
清理后没有重跑，也没有新的 JSON/hash contract 结论。当前 2.4 freeze 工具会拒绝归档的 2.3/2.1
输入，因此 production artifact 是合同字段迁移，不是由当前工具链重新生成的产物。

限制：practical equivalence 只覆盖预注册 estimator/support；selection scenes 不是 untouched test；
raw source manifest 只绑定 `path,size,mtime_ns`，不是全部输入内容 SHA；render smoke 未记录独立 script
SHA/elapsed，CUDA 不承诺 bitwise determinism；clean-cut 后 artifact 的实际 SHA 要在 Phase 2 首次消费前
重新绑定。Phase 1a 的既有 gauge table 验收不变，Phase 1b 至此
完成；本轮没有进入 Phase 2。

### A.8.12 Phase 1–8 当前实现核对点（不覆盖 Phase 0 provenance）

- `tools/compute_dggt_scene_gauge.py` 使用 lean teacher：只构建 Aggregator/CameraHead/DepthHead，
  不加载 tokenizer、PointHead 或 GaussianHead；CUDA aggregator 走 bf16，两个 head 走与 D1/D2 一致的
  fp32 路径。其 protocol/resume/shard merge 与 `phase0_golden_comparison` 已有定向单测。validation 正式表
  `data/scene_gauge/validation.json` 已完成：1212/1212 trunks、202 scenes、0 errors，metric-scale 有效
  1082，失效原因为 115 个 frame-CV 与 15 个 ruler-ratio，actor 有效 478；文件 SHA-256 为
  `5014e5c0ba5bd570c1a3d13e3fd222d15e32fe10276046dda763b7e87d9559fa`。training 三分片也已按同一
  protocol 全部完成并合并为 `data/scene_gauge/training.json`：4787/4787 trunks、798 scenes、0 errors，
  metric-scale/FOVx/FOVy valid counts `[4216,4787,4787]`，actor 有效 1662；SHA-256 为
  `39e0a32372e616e9aac4aef6109c8329ebdf382c16a913bd9e4d025b984e00af`。invalid-reason 计数为
  531 个 frame-CV 与 53 个 ruler-ratio（原因可重叠）。独立 random20 gate 的结果文件 SHA-256 为
  `889545cbd9b2753c44a732869ba507a43706e6557e52e7631c3dbe9c5c874f5a`；20 个 trunk 的 LiDAR AbsRel
  median/mean/p95/max 分别为 1.9833%/3.0103%/7.6039%/9.8653%，median 通过 5% 阈值；固定 drift
  cohort 为 44 pairs、mean 8.2020%、max 30.8056%。
- 两条 K 链已冻结：DGGT depth/render/sky 只用 gauge K；米制 bbox/`in_frustum`/
  `target_bbox_patch` 只用真实 Waymo K。
- placement v3（沿用 v2 布局）的 16 维中，标准化索引 `{3,4,5,6,13}` 是 **5 个**，passthrough 索引
  `{0,1,2,7,8,9,10,11,12,14,15}` 是 **11 个**；此前“4+12”的算术是错的。
- 推理侧的相机/几何代理正式命名为 **generated static-geometry reprojection/cycle diagnostic**
  （`generated_static_geometry_reprojection_cycle_v1`）。它使用生成 depth 与生成 camera 做前向重投影、
  目标 depth 采样和反向 cycle；不是独立 optical-flow 比较，也不读取 GT 图像。
- 相机 condition 的 `[...,9:18]` 由同一个 role-aware helper 在 raw pretrain、formal T1、formal offline
  inference 与 external manifest 四条路径构造；真实 cache 的 `[S,V]` front-view 选择、完整 anchor 窗、
  delta-only 窗及其 preceding metric pose 均有非 identity stats 回归。pretrain→formal warm-start 还会
  fail-closed 校验源 checkpoint 的 `pretrain_feature_stats_contract`：stats SHA、10 帧、29 帧 context
  与 25×37 grid。
- 2026-08-02 在 `conda dggt`、`CUDA_VISIBLE_DEVICES=0` 下，当前全套回归为 **732 passed, 1 skipped**；
  staged-only fresh-checkout import/collect 与 Phase 5–8 定向审计也通过。CUDA 0 米制导出 smoke 实测：
  camera round-trip max-abs=0、render pullback identity、depth factor 范围 `[0.694053,1.051988]`、
  means/scales 同比约 2.0、静态 cycle EPE=0（support 0.8125）。这些证明代码契约与米制边界 wiring，
  不能由 synthetic smoke 代替完整重训后的科学指标。
- 2026-08-02 training full-pass 修复版 v4 stats 已先写入 `.inprogress`，经独立 schema/coverage/finite/provenance
  检查后再原子发布为 `logs/scene_flow_pretrain_1024/feature_stats_pretrain_v4.pt`。
  它覆盖 4787/4787 trunks、798 scenes、44,279,750/44,279,750 latent，`stats_status=complete`、
  `exact_scene_gauge_scope=true`、`max_batches=null`；camera anchor/delta counts `[2416,45454]`、gauge
  counts `[4216,4787,4787]`、placement count 172605；修复后的
  `log_z_depth mean/std=2.998025/1.519740`。stats SHA-256 为
  `f5177c9262c878c1595c0f0e41ebd9cf42680de3676f0fccb789ed3cbc7a9111`，sidecar SHA-256 为
  `e0767b8bb3b86116f3748144b7c306d73ba6229a5568d47eb18a62cdf5d40539`；其 source contract 绑定
  tokenizer v1、DGGT、training gauge 三个 SHA，以及 10 帧/29 帧 context/25×37 grid。
- 旧 `logs/metric_gauge_one_step_cuda0/` 及其 10/29 帧输出绑定修复前 v4 stats（SHA `6fbdd3c5...eb81a`）、factorized-v2 和错误的
  `rgb_patch_v2` atlas 世界，现已 fail-closed，不能再作为当前线路验收。修复后 v4/v3 的新 smoke 单列在
  post-review 记录；完整重训科学 gate 仍不变。

### A.8.13 D1–D4 post-review 审计与修复（2026-08-02）

另一实现审计报告提出四项问题。逐项按真实代码和 CUDA 0 小脚本复核后，D1–D3 确认存在，D4 的四个
失败测试也全部复现；没有把任何一项当成纯文档意见跳过。

- **D1 sky atlas 世界系**：Waymo `camera_to_world_corrected` 是 clip-start ego `+z-up` 世界，不能只缩放
  平移后送进 teacher/OpenCV `-y-up` atlas。训练 atlas 与 RGB render 现在直接共享同一个 detached
  `camera_pose_sky_gauge`（teacher c2w + trunk 常量 gauge K）；validation 和 pretrain offline 的生成
  米制相机则先相对 metric trunk anchor 重基，再换算 DGGT translation。sky contract 升为
  `rgb_patch_teacher_anchor_v3`，旧 v2 checkpoint fail-closed。
- **D2 placement stats 坐标系**：统计工具原来误把 ego-world c2w 当 camera-anchor c2w。现在 placement
  分支强制读取 `pretrain_camera_to_anchor`，缺字段直接失败；全量重算的修复版 v4 stats 覆盖 4787/4787 trunks，
  ch3 从污染的 `-5.072679/3.431808` 修正为 `+2.998025/1.519740`，placement count 172605。stats SHA
  为 `f5177c9262c878c1595c0f0e41ebd9cf42680de3676f0fccb789ed3cbc7a9111`。
- **D3 无界 motion ratio**：保留投影 near-plane 语义不动，只把 passthrough ch14 从裸
  `speed/z_depth` 改为单调有界的 `tanh(speed/z_depth)`；factorized contract 升为 v3，旧 v2 stats/
  checkpoint 不会被静默加载。
- **D4 陈旧测试**：三个 flow assembler 测试改为当前的 CleanSceneState/coverage/concat 语义；tokenizer
  Stage-B fixture 补齐新损失参数。定向回归 140 passed，完整 `tests/` 回归 732 passed、1 skipped。

独立脚本 `lyy_tools/verify_metric_gauge_postreview.py` 实测 teacher anchor identity max-abs
`1.11e-16`、image-up/`-y` error `0`、2 m→1 DGGT unit error `0`；修复版 v4 ch3 和 ch14 范围检查通过。
结果位于 `runs/metric_gauge_postreview/verification.json`，SHA-256
`8803d5941c91e057711ac3c9d197470dec70d2bf6d8ffd27bd675e4d6997da9e`。

新端到端 smoke 使用 `logs/metric_gauge_postreview_one_step_cuda0/`、修复后 v4 stats、sky generation 和 CUDA 0。
一步 loss/flow/sky-flow/gauge-flow/gauge-direct 为 `3.4283/1.4045/0.1020/0.6241/0.0219`，LiDAR
diagnostic available=1；EMA-only checkpoint 实际 mmap 加载确认携带 sky-v3、factorized-v3 和修复后 v4 stats
SHA，文件 SHA-256 为 `4daae958f043721f47a8e89c94cb5fe3a3b3e7ea7cfaf8586915c6e5ee85d9a6`。

---

# Part B. SceneDirector 的叙事为什么成立

## B.1 Introduction 如何从应用需求导出核心矛盾

四步，每步只做一件事：

**¶1 只卖任务，不卖模块。** AD 验证需要多样场景 → 真实采集偏向 nominal、safety-critical 稀缺昂贵 → 编辑已有 log 可扩展 → **同时**控制局部目标与全局 ego 轨迹才能构成 reactive scenarios。整段没有一个方法名或模块名。读者先接受"unified editing 值得做"，后面的"统一"才不是功能堆砌。

**¶2 把功能组合提升为结构性矛盾，且用因果机制而非方法名单表述。** 原文："unifying these tasks necessitates reconciling two conflicting requirements: generative flexibility for object editing and physical precision for trajectory control." 然后**解释两种失败为什么发生**：

- 目标编辑需要生成先验补阴影/遮挡/光照；无严格空间约束 → generative freedom compromises 3D alignment → viewpoint shift 下 geometric drift。
- 跨轨迹 view-consistency 需要显式三维；缺生成能力 → rigid representation 无法补遮挡、无法匹配光照 → 目标编辑时出 artifact。

**关键一击**在段末：`While recent works incorporate diffusion into reconstruction, they restrict it to texture refinement rather than semantic creation.` 这不是"某方法缺某模块"，而是"当他们真的把两者结合时，生成部分只被允许**润色**，不被允许**创造**"。**换掉任何 baseline 这个缺口仍然存在**——这是它耐审查的根本原因。

**¶3 给原则性回答，并给每个模块唯一职责。** Unified Geometric Scaffold = 显式几何，且**一个表示同时服务两类编辑**（这才是 unified 的技术含义）；Static Texture Bank = 外观/语义来源；MGRA = 仲裁者，按 uncertainty 决定何处结构优先、何处注入纹理。

**¶4 范围 + 贡献。** 先说能力边界（asset-provenance-agnostic、free-form viewpoint），再三条 bullet，每条回指 ¶2 的一个判断。

## B.2 核心概念如何贯穿全文

同一组概念——**structural reliability / semantic completion / uncertainty-aware allocation**——出现在六个地方：

| 出现位置 | 具体形态 |
|---|---|
| 问题定义 | 两类编辑依赖的证据可靠性不同 |
| **训练数据构造** | $V_{\text{ref}}$ 纹理丰富但错位；$V_{\text{geo}}$ 对齐但稀疏有洞；$M\in\{0,1,2,3\}$ 标记来源可靠性 |
| 模块 | $G=\sigma(F([h_l;\Psi(M)]))$，$h_{l+1}=h_l+\lambda(G\odot\text{Attn})$，$\lambda$ 零初始化 |
| 指标 | FID/FVD/CLIP-I（视觉）**与** ATE/AOE/lane F1/X-err（几何）**并列** |
| Ablation | w/o Ref.Attn（生成先验是否必要）→ Standard Attn（**gate** 是否必要，而不只是 cross-attn）→ Alt.Geo（是否只是更好的 depth model） |
| Gate 可视化 | 高噪声时依赖 mask 先验，低噪声时变 content-adaptive |

## B.3 偏功能组合的方法如何形成方法创新

1. **训练构造与论文矛盾同构。** 真实数据没有"另一条轨迹 + 目标编辑"的 paired GT，他们**主动制造**冲突：affine warp 得错位参考、round-trip projection 得对齐但退化的 scaffold、mask 掉 $V_{\text{ref}}$ 中的目标区域防 shortcut。这比只加一个 loss 更有说服力。
2. **普通组件获得唯一且可检验的职责。** depth completion 是"可靠结构来源"；point rendering 是"统一两类编辑的坐标接口"；frozen reference latent 是"语义/纹理来源"；gate 是"决定两类证据责任的机制"。**每个模块关闭因果链上恰好一环。**
3. **指标互不掩盖。** 视觉与几何分开报。
4. **"统一"用干扰实验证明。** Obj. Only 的 ATE 0.90 → Obj.+Traj. 的 0.95，退化可忽略 → "scaffold effectively decouples tasks"。**声称的性质被专门实验测了。**

## B.4 合理抽象 vs 写作包装

**合理的研究抽象：** 点云/depth 可靠性确实有空间异质性；错位参考直接 cross-attention 确实会 ghosting（Fig. 7 上排有证据）；自监督 triplet 主动模拟 inference mismatch；视觉/几何指标分离能检验两个目标。

**属于包装的部分：**

- **"inherent dilemma" 不是被证明的定理**，更准确说是"多源条件的可靠性分配问题"。
- **Unified Geometric Scaffold** = DMD3C depth completion + 点云反投影/z-buffer 投影 + Trellis asset 合成，全是已有组件。
- **Static Texture Bank** 本质是冻结的源视频 latent。
- **MGRA** = cross-attention + MLP sigmoid gate + 零初始化 residual。
- **"guarantees structural fidelity" 过强**——Appendix G 承认 thin structure、雨天 LiDAR、不同 camera rig 都会失败。
- **"unified" 只指一次 diffusion pass**，预处理仍需 LiDAR、depth completion、asset library、SAM2 分割与 lifting。
- 评测集各 64 个 curated scenario；object baseline 是单视角单目标方法，Multi-Edit 需顺序执行，公平性不完全对称；lane detector / PGD / StreetGaussian 都是代理指标。

**结论：它最成功的地方不是提出复杂新算子，而是把六个普通组件组织进同一个研究问题。可迁移的是论证结构，不是 "Bridging X and Y" 句式。**

## B.5 六条可迁移原则

1. 不复制句式，复制论证结构：任务 → 机制级缺口 → 一条中心原则 → 每模块关闭一环 → 训练构造实例化缺口 → 实验逐环验证。
2. 先定义样本最终必须满足的**可验证责任**，再指出现有流程为何**系统性**失配。
3. 让表示空间、联合条件、损失层级分别回答同一中心问题的不同环节，而不是三个并列卖点。
4. 训练目标应直接实例化论文指出的失败机制。
5. 实验同时测中间表示、最终 Gaussian/解码几何、渲染结果、控制满足度——四类分开报。
6. 小模块的价值来自它是否关闭一条清楚的因果链，而不是参数量。

---

# Part C. 相关工作谱系（准入标准与逐篇判定）

## C.1 准入标准（固定，不再变动）

Introduction 核心谱系只保留**同时**满足以下三条的方法：

1. **生成任务面向驾驶或道路城市场景**；
2. **不依赖目标场景 RGB / 历史观测**，从噪声与结构条件生成新场景（source-free）；
3. **最终得到持久化、可自由视角渲染的 3DGS / 4DGS**。

## C.2 第一类：生成视图或几何中介后构建 Gaussian

| 方法 | 生成链路 | 放入原因 |
|---|---|---|
| [MagicDrive3D](https://arxiv.org/abs/2405.14475) | map/boxes/text → 多视图视频 → 单目深度初始化 → 逐场景优化 3DGS | 明确生成新 street scene；代表 video-first + optimization |
| [InfiniCube](https://arxiv.org/abs/2412.03934) | map/boxes/text → voxel diffusion → video diffusion → 前馈动态 3DGS | 明确驾驶生成；代表 voxel 与视频共同支撑 Gaussian |
| [DriveGen3D](https://arxiv.org/abs/2510.15264) | text/BEV → 驾驶视频 → FastRecon3D 动态 Gaussian | 明确驾驶生成；代表 video-first + feed-forward reconstruction |
| [X-Scene](https://proceedings.neurips.cc/paper_files/paper/2025/hash/969f0d274237b8fe99baa74c2fad5d93-Abstract-Conference.html) | text/layout → occupancy → 多视图图像/视频 → 3DGS lifting | 代表 occupancy-mediated 路线。**不要对其未公开的 GS 梯度链做强断言**——正文只给三个独立 diffusion denoising loss，未报告 GS reconstruction loss 或训练链 |

共同点：

> 生成过程先落在 RGB/video、voxel 或 occupancy 等**显式中间表示**上，再通过逐场景优化或一个独立的重建模块形成 Gaussian。

DreamDrive 不在此列：它从目标场景图像起步，不是从零生成（违反准入 2）。

## C.3 第二类：生成变量直接进入三维 / Gaussian 状态

### C.3.1 一个重要的轴向修正 ⚠ 必读

原表述"生成变量直接进入三维/Gaussian 状态"作为 2A 的定义**在两个成员上站不住**：

- **WorldSplat** 的 latent 是多视图、**pixel-aligned** 的 RGB+depth+semantic latent；
- **CVD-STORM** 的 diffusion 明确在**多视图视频 latent** $z_t\in\mathbb{R}^{T\times V\times C\times H\times W}$ 上去噪，只是它的 VAE encoder 被 4D 重建辅助任务微调过。

只有 ScenDi 的 latent 是真正三维索引的（voxel）。若把 2A 定义成"latent 是三维的"，审稿人可以直接反驳；**而且更危险的是，这条轴会把质疑引向"你的 latent 也是 view-indexed"——这恰好是本方法的真实 scope 限制（frame/patch aligned，见 A.6）。**

**建议把 2A 的判据从"latent 是否三维"改为"生成变量是否经一次前馈解码就得到 Gaussian 场景"：**

| | 判据 | 成员 |
|---|---|---|
| **第一类** | 生成变量先被解码为**显式观测（图像/视频）或几何支架（voxel/occupancy）**，之后才形成 Gaussian | MagicDrive3D、InfiniCube、DriveGen3D、X-Scene |
| **2A** | 生成变量**一次前馈解码直接得到 Gaussian 场景**，无中间显式观测、无逐场景优化 | WorldSplat、CVD-STORM、ScenDi |
| **2B** | 生成三维状态后**逐场景优化** Gaussian | LSD-3D |

这条轴的好处：它就是你的 2×2 缺口论证所依赖的那条轴（摊销 vs 逐场景 / 监督是否跨越阶段边界），而不是一条会反过来打到自己的轴。**在 2A 内部可以用半句话承认 latent 索引方式不同（view-aligned：WorldSplat、CVD-STORM；voxel：ScenDi）**，这正是 DGGT 空间论证的落点。

### C.3.2 2A：摊销式 latent-to-Gaussian

| 方法 | 生成链路 | 写作限定 |
|---|---|---|
| [WorldSplat](https://openreview.net/forum?id=KWeX6tYno6) | noise + road/boxes/ego/text → multimodal latent（RGB+metric depth+semantics，pixel-aligned）→ latent Gaussian decoder 前馈输出动态 4DGS → **video refiner 精修渲染出的 novel view** | 最直接驾驶对手。原文 Sec. 3.1 明确 "The three modules are trained independently" |
| [CVD-STORM](https://arxiv.org/abs/2510.07944) | conditions（+可选 0/1/3 reference frames）→ **多视图视频 latent** → 两个并行分支：VAE decoder 出视频、GS decoder 出动态 3DGS | 见下方专门审计——**它是"世界监督止于表示"的最锋利证据** |
| [ScenDi](https://arxiv.org/abs/2601.15221) | noise → colored-voxel VQ latent → coarse 3DGS → **2D video diffusion 精修** | 保留为 **urban-road boundary**；不能写成 autonomous-driving-specific（Waymo + KITTI-360 城市生成），但它也不是 WorldFlow3D 式通用 world 方法 |

#### CVD-STORM 专项审计（已逐项核实）

| 项 | 事实 |
|---|---|
| 主任务 | cross-view **video** generation；GS 是一个 reconstruction decoder 分支 |
| 生成变量 | 多视图视频 latent $z_t\in\mathbb{R}^{T\times V\times C\times H\times W}$ |
| source-free？ | **可以**——原文 "Without reference frames, the model conducts pure video generation, producing content based solely on conditional inputs"；ablation Table 2a 覆盖 0/1/3 reference frames。**但 Table 1 的 headline FID 3.8 / FVD 14.0 用 3 帧 reference**，即主结果是 prediction 设定。**这个限定必须跟着引用一起走。** |
| 条件 | text、3D boxes、HD map、可选 reference frames |
| GS decoder 与谁联合训练 | **与 VAE 在 stage 1 联合**（$\mathcal L_{\text{STORM}}$）。stage 2 训练 diffusion 时 **"we freeze the encoder of STORM-VAE"** |
| diffusion 损失 | 纯 rectified flow $\lVert\epsilon_\theta(z_t,t,c)-(z_0-\epsilon)\rVert^2$。**没有任何 render/depth/Gaussian loss 回传到 diffusion** |
| 输出 | 视频 **与** 动态 3DGS 两者；Gaussian 可按预测速度变换后自由渲染 |
| 数据 | OpenDV-Youtube（单视图）、nuScenes、Waymo、Argoverse2（多视图） |

> **为什么它是最好的反例素材**：它的摘要就写着 "the **jointly-trained** Gaussian Splatting Decoder"——读者极易以为世界监督进入了生成器。但那个 "jointly" 指的是 **VAE ↔ GS decoder**，而不是 **diffusion ↔ world**；训练 diffusion 时整个 VAE（含 GS decoder）冻结，目标只有 latent。**这一句写进 Related Work，比任何抽象论证都更有说服力。**

### C.3.3 2B：直接三维状态，但逐场景优化

| 方法 | 生成链路 | 作用 |
|---|---|---|
| [LSD-3D](https://princeton-computational-imaging.github.io/LSD-3D/) | map / proxy geometry（层次化 voxel diffusion → NKSR mesh）→ 表面对齐 splat 初始化 → 2D-prior score distillation | 代表 **endpoint-aware 但需逐场景数千步优化**的另一端。6000 步、约 2h/H100；动态 actor 事后插入 |

### C.3.4 这三组正好构成核心矛盾

- **WorldSplat、CVD-STORM、ScenDi**：快速、摊销，但 Gaussian/render supervision 主要止于表示或 decoder 训练，负责新场景采样的 diffusion 仍由 latent objective 定义；
- **LSD-3D**：endpoint feedback 直接优化最终场景，但不能快速摊销生成；
- **本方法**：让 diffusion 实际预测的 clean DGGT state 经过 decoder、Gaussian heads 与 renderer，并让这些误差直接更新**同一个摊销式生成器**。

### C.3.5 缺口的 2×2 形式

|  | 只有 latent 监督 | 有解码之后的监督 |
|---|---|---|
| **摊销式条件生成器** | WorldSplat、**CVD-STORM**、ScenDi、DriveGen3D、InfiniCube、X-Scene | **← 空缺，本方法所在** |
| **逐场景优化 / 非生成** | — | LSD-3D、MagicDrive3D(FTGS)、DreamDrive（逐场景）；Envision4D、DGGT（重建而非生成） |

### C.3.6 第二根轴：这些场景的米制尺度是从哪来的

> ⚠ **2026-08-04 改变用途。** 这一轴曾被用来支撑 Intro ¶2 的第二层缺口（"他们的米制是固定属性"），
> **那个用途已作废**——逐篇核实后确认他们的米制来自输入侧且按构造正确（见 D.1.5 的 11 篇全表）。
> 本表现在只是一份**事实索引**：用来解释为什么我们这条路线必须自己生成尺度，
> **不能用来指控任何人**。

| 尺度来源 | 成员 | 事实（不含价值判断） |
|---|---|---|
| **表示自带**：固定分辨率 / 固定体积的 voxel 或 occupancy 格点 | ScenDi（0.4 m、固定体积）、LSD-3D、InfiniCube、X-Scene | 尺度按构造正确。**可写的缺点只有分辨率与断链，与尺度无关**——外观细节受表示分辨率限制是 ScenDi 自陈动机 |
| **输入位姿 / 已标定 rig** | STORM 与 CVD-STORM（输入内外参）、DrivingForward（rig 外参）、DrivingScene | 尺度随位姿一起进来，按构造正确 |
| **LiDAR 深度监督** | DrivingRecon、WorldSplat（GS decoder 的 metric depth $\ell_1$） | 尺度由训练时的物理测量锚定，按构造正确 |
| **由输入观测锚定** | Envision4D、PhiGenesis、DreamDrive | 不需要生成尺度；**违反准入 2**，不进核心谱系 |
| **无位姿、未标定 → 没有尺度** | **FRUC**、**DGGT（本方法）**、VGGT / DUSt3R 一族 | 绝对尺度在数学上不可确定，因此被归一化掉 |

> **写作纪律（已收紧）**：**不要**声称别人的尺度"不准""不能自适应""是固定属性"——三句都不成立。
> 这一轴在论文里唯一的用途是 ¶3 的转折：**我们选的是最后一行，所以必须自己把尺度生成出来。**
> 详见 D.1.5。

## C.4 只放 Related Work（一）：驾驶领域，但依赖目标场景观测

| 方法 | 排除原因（违反准入 2） |
|---|---|
| [Envision4D](https://arxiv.org/abs/2606.10656) | 输入 context RGB，重建并外推同一场景的未来 4DGS |
| [PhiGenesis](https://arxiv.org/abs/2509.20251) | 依赖历史多视图图像与相机，沿目标轨迹生成未来 4DGS |
| [DreamDrive](https://arxiv.org/abs/2501.00601) | 从 street-view / control image 出发生成视觉参考再构建 Gaussian |
| [FreeGen](https://arxiv.org/abs/2512.04830) | 用目标场景多视图重建 3DGS，再用 diffusion 改善偏移视角 |
| [GaussianDWM](https://arxiv.org/abs/2512.23180)（CVPR 2026，建议补入） | 3D Gaussian 是**理解任务的输入**；生成变量是 RGB-D video latent；observation-conditioned 未来预测；各阶段分开训练 |
| [InfiniVerse](https://arxiv.org/abs/2606.31109)（建议补入） | 单帧输入 → 重建 occupancy → video diffusion → 视频反投影回来增强 occupancy；输出视频 |
| DriveDreamer4D、ReconDreamer、RGE-GS、DriveX ⚠未逐篇核实 | 从已有真实日志重建、扩车道或扩视角，不采样新场景分布 |

> **最重要的表述**：Envision4D **不是**因为"不属于驾驶领域"被移出，而是因为**任务起点不同**才不进入 Introduction 核心。它在 Related Work 与 Part D.3 的论证边界里仍不可缺——它是"有完整 render gradient 的前馈网络"的存在性证明。

## C.5 只放 Related Work（二）：从零驾驶生成，但最终不是 Gaussian 场景

| 方法 | 排除原因（违反准入 3） |
|---|---|
| [SEM-ROVER](https://arxiv.org/abs/2604.06113) | 明确是驾驶场景生成（Waymo 60 seq + PandaSet 60 seq，无逐场景优化），但生成的是 **Σ-Voxfield**（每 occupied voxel 存 n 个彩色表面采样点）；Gaussian 只在光栅化时由 PCA 法向临时构造，照片级图像来自一个改造过的 Stable Diffusion deferred renderer；明确不建模动态 |
| [UniScene](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_UniScene_Unified_Occupancy-centric_Driving_Scene_Generation_CVPR_2025_paper.pdf) ⚠ | 生成 occupancy、video 与 LiDAR；Gaussian splatting 只是产生视频条件图的工具（此细节未逐行核实，写作前请确认） |
| [DiST-4D](https://arxiv.org/abs/2503.15208) | 生成/预测 RGB 与 metric depth，不输出持久化 Gaussian 场景 |
| [AnyScene](https://arxiv.org/abs/2605.26113)（建议补入） | BEV layout → 语义 occupancy 序列 → 视频；无持久 Gaussian |

它们会限制"首次显式三维驾驶场景生成"这种宽主张，但不构成 Gaussian generator 的直接反例。

## C.6 只放 Related Work（三）：已有 Gaussian 场景上的编辑或仿真

HorizonForge、SIMSplat、RoVES、BehaviorGaussian、Real2Sim ⚠（后三者未逐篇核实）。先重建既有场景，再插入车辆、修改轨迹、编辑道路或模拟行为。可用于讨论联合控制与场景编辑，但**不能证明已有方法可以从噪声生成完整 Gaussian 场景**。

## C.7 只放通用 3D Generation Related Work

| 方法 | 排除原因（违反准入 1，或 1+3） |
|---|---|
| [GaussianCity](https://arxiv.org/abs/2406.06526) | unbounded **city** generation。主设定 GoogleEarth 是 Manhattan/Brooklyn 上空 112–884 m 的 400 条无人机环绕轨迹；KITTI-360 只是 street-view 子设定，**baseline 是 StyleGAN2 / GSN / GIRAFFE / UrbanGIRAFFE**（3D-aware GAN 谱系）。条件只有 BEV semantic/height/density map + 相机参数；无 text、box、ego 轨迹、动态、时序 |
| [WorldFlow3D](https://arxiv.org/abs/2603.29089)（ECCV 2026，与 LSD-3D 同组） | 任务是跨域 unbounded **world** generation，Waymo outdoor 与合成室内（⚠ 用户称 3D-FRONT，我未能确认数据集名）并列以证明 cross-domain generalizability；输出 **TUDF 体素距离场，不是 Gaussian**；条件为 vectorized layout（polylines + boxes）+ discrete scene attributes，无 text、无相机轨迹 |
| Prometheus、Director3D、L3DG、Lyra、ST-Gen4D ⚠部分未核实 | 通用 3D/4D Gaussian generation，非驾驶专用任务 |
| GLD、OneWorld、Gen3R、Geometry Forcing、PixWorld ⚠部分未核实 | 用于审计 DGGT feature generation、多层监督与 endpoint gradient 的**通用先例**；不参与驾驶 Introduction 的领域缺口，但**必须出现在 Related Work 与 Part H 的创新边界里** |

> **两条必须保留的中和句**（成本近乎零，漏引代价高）：
>
> *GaussianCity decodes 3D Gaussians from BEV semantic/height/density maps in a single pass for unbounded city generation, evaluated against 3D-aware GAN baselines; it models neither actors, ego trajectories, temporal evolution, nor instance-level specification.*
>
> *WorldFlow3D performs latent-free hierarchical flow matching over volumetric distance fields for unbounded world generation across driving and indoor domains; it generates geometry and colour without decoding to Gaussians, and is conditioned on vectorized layouts rather than object appearance or camera trajectories.*
>
> 若正文任何地方写"推理时无需第二个二维生成阶段"，**WorldFlow3D 是唯一反例**，靠这一句中和。

---

# Part D. 四个核心问题的回答（2026-08-02 按落地代码重写）

> **本 Part 的重写原则**：v8 的四个答案写在「尺度尚未被生成」的代码上，因此 D.1 只能把 DGGT 空间
> 说成"一个更好的 latent"，D.2 的"联合控制"有两族条件在量纲上是坏的，D.4 的 pullback metric
> 论证站在一个**未经审计**的解码器上。三处现在都被代码补齐了，答案随之升级。

## D.0 一句话把四个问题连起来

四个问题不是并列的四个卖点，它们是同一条因果链上的四环：

```
在哪个空间生成    →  这个空间决定了能不能把三维几何解出来                    (D.1)
解出来的几何用什么单位 →  没有单位，条件和几何无法比较，米制误差根本没有定义     (D.1.5)
条件怎么进来      →  单位一致之后，条件才真的约束几何，而不只是被拼进注意力     (D.2)
误差在哪里测      →  在解码之后、用这套单位测，生成分布才对最终三维结果负责     (D.3 / D.4)
```

第二环是本轮新增的，也是让另外三环从"设计选择"变成"必要条件"的那一环。

---

## D.1 为什么用 DGGT 特征空间

### 安全且准确的抽象

> **reconstruction-grounded multi-view scene state**：一个 pose-free feed-forward 4D 重建模型的中间
> 状态——跨视图对应、depth、相机关系、动态归属与 pixel-aligned Gaussian 属性已被**联合编码**，
> 并可由**已预训练的可信 heads 联合解码**。形式上 $z\in\mathbb R^{S\times P\times1024}$ 是四层三流 DGGT
> feature lattice 的 12:1 压缩；$D_{\text{JST}}\circ\text{heads}$ 是一条**已经训好的解码链**。

**不可写**：view-independent 3D space、canonical 3D grid、world-space geometry token、直接 Gaussian
diffusion。**必须承认**：latent 仍是 view-indexed（frame/patch aligned），作为 scope 而非隐瞒。

**一句重要的限定**：这个状态给出的几何**本身没有单位**。它给的是"世界的形状"，不是"世界的
尺寸"；尺寸由 D.1.5 那 3 个数单独给出。把这句话写清楚，D.1 就从"我们换了个 latent"变成
"我们把表示与它的度量分开了"。

### 与核心谱系的对照（新增一列：米制从哪来）

| | 生成变量 | 谁定义"世界" | **米制从哪里来** | 高保真外观最终落在哪 | 单场景代价 | 动态 |
|---|---|---|---|---|---|---|
| **LSD-3D** | voxel occupancy（latent diffusion）；Gaussian 参数是**逐场景优化变量** | NKSR proxy mesh + 深度条件 2D teacher | **表示自带**：体素网格本身定义在米制格点上 | 逐场景优化的 2D oriented planar splats | **6000 步 ~2h/H100** | 事后插入外部资产 |
| **ScenDi** | 单目深度融合得到的彩色 voxel grid 的量化 latent（**0.4 m**，固定体积） | 从头训 300k 步的 Voxel-to-3DGS VQ-VAE | **表示自带**：0.4 m 体素 + 固定体积 | **第二阶段 2D video diffusion**；远景与高频不写回 Gaussian | ~5 s + 0.25 s + **~4.8 min 2D** | 数据剔除显著动态 |
| **WorldSplat** | 多视图 pixel-aligned RGB+**metric depth**+semantic latent | 独立训练的 latent Gaussian decoder | **decoder 自带**：GS decoder 用 metric depth $\ell_1$ 监督训练，训生成器时冻结 | Gaussian，**但另有 video refiner 精修渲染视图** | 前馈 | 静态背景 + 动态物体分离后聚合 |
| **CVD-STORM** | **多视图视频 latent** | stage-1 与 VAE 联合训、stage-2 冻结的 GS decoder | **decoder 自带**：STORM-VAE 阶段引入，stage 2 起冻结 | 视频（主结果）；Gaussian 为并行分支 | 前馈 | 按预测速度变换 |
| **本方法** | 压缩的多层 DGGT feature lattice（**没有内建尺度**） | **已预训练冻结**的 DGGT depth/GS/instance heads | **被生成**：3 个场景级的数（尺度 + 两个视场角通道），逐 29 帧片段，LiDAR 标定 GT | Gaussian 本体；推理时**无第二个 2D 生成器** | 一次 ODE(35 步) + 一次 decode | 逐帧 dense Gaussian + `dynamic_conf` + 每帧 actor placement/velocity |

**这一列是本轮最重要的新增。** 它把"你为什么不用 voxel"从口味之争变成一个可陈述的权衡（见 D.1.5）。

### 三条可辩护的理由

**1. 解码链是训练好的，"三维"不必由一个任务专用的 tokenizer 从头学。**
ScenDi 的三维表达被 0.4 m voxel 与固定体积**在构造上**限制——这正是它自己的动机
（"relying solely on 3D diffusion → degradation in appearance details"），也是它必须接第二阶段 2D
扩散的原因。我们的 latent 继承了一条"从无位姿真实驾驶图像重建"训练出的解码链：camera、depth、
动态归属、Gaussian 属性都在图像分辨率上可解码，无固定体积、无 voxel 量化、无单目深度融合预处理。

**2. 因为这条解码链可微且冻结，解码之后的监督对生成器才成为可能——决定性的、非替代性的理由。**
voxel-VQ latent 的 decoder 含 codebook argmax 且分阶段训练；video-VAE latent（WorldSplat、CVD-STORM、
PhiGenesis）的几何只在另一个单独训练的 Gaussian decoder 之后才出现，而那个 decoder 在训练生成器时
是冻结的。**"选这个空间"不是"换一个 latent"，而是 D.3/D.4 的前提条件。**

**3. 相机与内容在同一状态里，因此相机是生成变量而不只是渲染时的 query。**
ScenDi 与 LSD-3D 的相机轨迹是在已生成场景上的**事后渲染查询**。我们在同一序列里联合生成
**9 维米制 Waymo camera state**（`waymo_metric_relative_se3_rot6d_v4`）与独立的 3 个尺度/视场角数；
请求轨迹与生成目标使用**同一参数化、同一 anchor/delta 角色、同一套统计**，因此"复现给定轨迹"
在参数化层面是恒等映射。

> **诚实的代价（D3 判定，必须写）**：训练 RGB render 使用 detached teacher pose，而不是生成相机。
> 理由不是"生成相机不够好"，而是 latent 的 flow 目标本身就是 teacher 空间的
> $\text{Encoder}(\text{direct tokens})$；用换算后的米制相机渲染会主动把几何往 Waymo 世界拉。
> 后果是**生成相机与生成几何之间没有光度耦合**，这一点由相机–几何一致性诊断来报告，
> 在它落地前由 `generated_static_geometry_reprojection_cycle_v1` 诊断顶上。

---

## D.1.5 米制从哪来：为什么它是我们的代价，而不是别人的短处

> ⚠ **2026-08-04 重写，方向完全反了过来。** 旧版主张"现有方法的米制是一个固定属性，
> 不能随片段自适应"，并把它当作 ¶2 的第二层缺口。**这条已被逐篇核实推翻，全文删除。**
> 本节现在的作用是：说清米制在各方法里究竟从哪来，从而说明**为什么我们这条路线必须自己生成它**。

### 逐篇核实：11 个驾驶前馈重建方法的米制来源

| 方法 | 输入 | 米制？ | 尺度从哪来 | 核实 |
|---|---|---|---|---|
| **STORM**（ICLR 25；CVD-STORM 的底座） | 图像 **+ 内参 + 外参** | ✅ | **输入外参**。$\boldsymbol\mu=\text{ray}_o+d\cdot\text{ray}_{dir}$；论文把"需要内外参输入"写成自己的 limitation | 全文 |
| **DrivingForward**（AAAI 25） | 环视（固定 camera-to-vehicle 变换） | ✅ | **rig 外参**。spatial photometric loss 靠相邻相机真实基线定尺；推理 $\mu_i=\Pi^{-1}(I_i,D_i)$ 用 $K_i,E_i$ | 全文 |
| **DrivingRecon** | $\{X_t,K_t,E_t\}$ | ✅ | 输入外参 + **LiDAR 投影深度监督** | 全文 |
| **DrivingScene** | 两帧环视 | ✅ | "transformed into a common world coordinate system using the **known extrinsic parameters**" | 摘要+正文片段 |
| **Omni-Scene**（CVPR 25） | 六张环视 | ✅（推定） | 环视已标定 + 固定体素 volume | ⚠ 403 未取全文 |
| **XYZCylinder** | 环视 | ✅（推定） | 环视已标定，cylinder lifting 需内外参 | ⚠ PDF 过大 |
| **VGD**（蒸馏 VGGT 先验，nuScenes 环视） | 环视 | 推定 | 环视已标定；**摘要不声称米制** | ⚠ 位姿输入未确认 |
| **ConFixGS** | 前馈 3DGS 的修复插件 | 继承宿主 | — | 略 |
| **FRUC** | 未标定协同视图，**推理显式排除内外参 / 位姿 / LiDAR** | ❌ | **没有**，输出是归一化/相对尺度 | 全文 |
| **DGGT**（本方法） | **无位姿图像** | ❌ | **没有**。实测 1 单位 = 25–64 米，逐片段变 | 自测 |
| **VGGT**（底座） | 无位姿无标定 | ❌ | 按点到原点平均距离归一，**并训练网络直接输出归一化坐标** | 论文 |

### 结论：米制只能从输入侧进来

> **在驾驶前馈三维重建里，米制尺度只能从输入进来**：已标定 rig 的外参、给定的位姿、
> 或 LiDAR 深度监督。只接受无位姿、未标定图像的系统，绝对尺度在数学上不可确定
> （未标定情况下场景只能恢复到相似/射影变换之内），因此必须把它归一化掉。

**FRUC 是这条规律的反向验证**：它是唯一明确丢掉标定的驾驶前馈方法，也恰好是唯一不米制的。
pose-free 那一支（PF3plat、PREF3R、TokenSplat、AnySplat）全部在 canonical frame 里重建，无一声称米制。

**因此不能写的**："三维重建系统都是尺度归一化的"——被 STORM / DrivingForward / DrivingRecon 直接反驳。
**可以写的**：上面方框那句。它不是经验概括而是几何事实的推论，**就算再冒出十篇米制方法，
它们也一定是靠输入拿到的**，规律不会被推翻。

### 为什么我们仍然选无位姿的那一支

**因为要让先验落在三维上，就只能用它。** 需要位姿的重建器（STORM 一路）在生成时没有被拍摄过的
场景，所谓"输入位姿"只能是自己指定的——三维于是被钉在给定坐标系里，模型做的是**在这个坐标系里
把生成出来的视图三角化**。CVD-STORM 走的正是这条路，这也解释了它的主结果为什么是视频。

**所以尺度不可确定不是我们选出来的麻烦，是"让几何成为生成对象"这个要求的必然入场费。**
这句话是 ¶3 那个"然而"的全部依据。

### 我们怎么付这个入场费

不是"多生成三个数"这么单薄，是一整套**重参数化**（见 A.5 的 16 通道表）：

| 机制 | 解决什么 |
|---|---|
| 16 维 placement：**11 个尺度不变通道 + 5 个标准化 log 幅值** | 未知尺度在条件里**只表现为同一个加性常数**，而不是把条件毁掉 |
| 那个常数**由同一次前向生成出来** | 于是它不再是未知量 |
| 两条内参链永不交叉（真实 Waymo K 管像素寻址，生成视场角管三维解码） | 像素被两套约定共用，横向不需要换算 |
| 距离通道用 z-depth 而非欧氏 range | 映射是各向异性的，只有 z 是纯标量 |

**把病态的条件问题变成良定义问题的重参数化——这才是可以写进 contribution 的技术内容。**

### 这条主张是可证伪的

判据写在训练与验证日志里：如果模型预测的尺度并不比"永远输出训练集均值"更准
（`gauge_vs_prior_gain` ≈ 0），说明它只是记住了一个常数，没有从图像内容里推断尺度。
**验证集上要报的是走完整采样的那一版**（`train_scene_flow_pretrain.py:7358` 的 `sample_gauge_*`），
不是训练时随机噪声水平下的 x-prediction（`:5676`）——后者是乐观估计。

> **当前观测**：训练集上 `gauge_vs_prior_gain ≈ 0.2`（单位是 log(米/单位) 的平均绝对误差之差）。
> 换算一下让这个数可读：尺度范围 25–64 m，$\ln(64/25)=0.94$，所以"永远猜均值"这个基线的平均
> 绝对误差约 0.23，也就是约 **23% 的相对尺度误差**；领先 0.2 意味着模型自己的误差约 0.03，
> 也就是约 **3%**。**论文里要同时报模型误差和基线误差两个数**（`gauge_log_scale_error` 与基线），
> 不要只报差值——只报差值读者算不出量级。

### 这个尺度自由度到底有多大（让 reviewer 相信问题真实存在的数字）

| 量 | 实测 | 含义 |
|---|---|---|
| 1 个 DGGT 单位 | **25–64 米** | 不是小扰动，是 2.5× 的量程 |
| 逐 **29 帧片段**变化 | 相邻片段漂移 mean 8.2% / p95 21.9% / max 30.8% | **不是场景常数**，同一场景的不同片段都不一样 |
| 会不会只是估计噪声？ | LiDAR 尺与相机尺的有符号漂移 41/44 同向，**Pearson 0.9711** | **不是**。是冻结 teacher 自己在不同片段选了不同尺度 |
| 机制 | `train.py:82-93`：`camera_head` 全程 `requires_grad=False`，沿用 VGGT 的逐场景尺度归一化 | 米制信息**从未进入**这个 head |
| 视场角 | 片段**内** std 0.262°/0.178°，片段**间** std 9.376°/6.624° | **视场角同样是逐片段的量，不是逐帧的量** |

### 最锋利的一句（Branch A）

> **teacher 自己预测的内参，比真实 Waymo 标定更好地解释它自己的几何。**

判据是静态区域的 **primitive-level leave-one-frame-out** 渲染（目标帧产生的 Gaussian means 完全排除，
同一候选 $K$ 同时用于源帧反投影与目标帧 rasterization）：`K_pred` − `K_Waymo` = **+0.472 dB**，
scene-bootstrap 95% CI **[+0.253, +0.741]**。且把视场角压成片段常量后，相对逐帧值只掉
−0.108/−0.150 dB，通过预注册的 −0.2 dB non-inferiority margin。

**所以视场角不是一个需要修的坏 head，而是这一段场景的两个属性。** 这是一个略反直觉、可复现、
且换掉任何 baseline 都仍然成立的经验发现——正是 ICLR 喜欢的那类小结果。

---

## D.2 联合控制的真正优势

### 边界先立

`text + 3D box + camera trajectory` 的联合条件**已被** MagicDrive3D、DriveGen3D、InfiniCube、
WorldSplat、X-Scene、CVD-STORM 覆盖，ScenDi 覆盖 layout/text。**"more controls" 一定被驳回。**

### 【重写】核心差异：不是"条件更多"，是"条件与几何终于在同一套单位里"

v8 的说法是"全部条件约束同一状态"。**那句话当时其实是站不住的**，因为在尺度被生成出来之前，
四族条件里有两族在量纲上是坏的——它们进了同一个注意力序列，但和解码几何**没有共同单位**：

| 条件族 | 改造前的实际状况 | 后果 |
|---|---|---|
| 相机 | 生成目标 = DGGT 位姿 = $s\times$（米制条件），$s$ 逐片段随机、CV **23.5%**、**不可观测** | 模型被要求学"输入乘一个看不见的随机数"。**相机可控性在数学上不成立** |
| actor placement | `center`/`log_size`/`velocity` 是**米制原值**，被**全局**统计归一化，而解码几何在 DGGT 单位 | $s=25$ 与 $s=64$ 的场景归一化后输入**完全相同**，却对应不同物理位置 |
| 文本 | 无量纲 | 正常 |
| actor 外观 | 无量纲（canonical token） | 正常 |

**所以"joint control"这个卖点在改造前有一半是空的。** 把它写进论文而不修，是会被扎穿的。

改造后（已落地）：

| 条件族 | 当前契约 | 与解码几何的关系 |
|---|---|---|
| 相机 | 9D 米制 Waymo state，请求与目标**同一参数化、同一 anchor/delta 角色、同一套统计** | **恒等映射** |
| actor placement | 16 维 `factorized_asset_v3`：**11 个尺度不变 passthrough**（方向、角尺寸、yaw、速度方向、`tanh(speed/z_depth)`、`in_frustum`）+ **5 个标准化米制 log 幅值**（`log_z_depth`、三个 `log_box_lwh`、`log_speed`） | 未知尺度只表现为这 5 个通道上的**同一个加性常数**，而这个常数正是同一次前向里生成的 `log_metric_scale` |
| 像素寻址 | `target_bbox_patch` / `in_frustum` 走**真实 Waymo K** 的全米制链路 | 像素被两套约定共用，**不需要换算** |
| 内参 | 生成出来的两个视场角通道；DGGT 链路（depth/render/sky）统一用这套内参 | 两条 K 链**永不交叉** |

> **写作句（比 v8 那句更强，因为现在有机制）**：
> *our contribution is not that the model accepts more conditions, but that every condition and the
> geometry it refers to are expressed in the same units, so that a condition can be checked against
> the decoded scene rather than merely attended to.*

### 【本轮补写】第二条腿：条件之间的依赖关系被写进了训练分布

单位一致解决的是"条件和几何能不能比较"。但还有第二个问题："这些条件之间是什么关系？"
大多数可控生成把条件当成一个**可任意开关的集合**，训练时独立随机丢弃。
**在驾驶事件里这是错的**：目标的位置只有在相机确定之后才有像素意义。

代码里实现的是一条**投影依赖链**（`train_scene_flow_pretrain.py:1539-1629`，详见 A.5）：

```
文本    →  规定整体语义与环境
米制相机 →  规定观察这个世界的射线族
目标条件 →  在这族射线里规定具体实例的外观与位置
```

因此只有三种组合是合法的，`actor-only` **在结构上被禁止**——它的 `target_bbox_patch`、
`in_frustum` 和 canonical-UV mRoPE 都要先经过指定相机投影才有定义。
推理端的层级 CFG 与之严格对应（$\Delta_{\text{cam}}=v_{\text{text+cam}}-v_{\text{text}}$，
$\Delta_{\text{actor}}=v_{\text{full}}-v_{\text{text+cam}}$），所以相机增量和目标增量可以分别调节，
而永远不会去评估那个非法组合。

> **写作句**：*we do not treat text, camera and actors as an unordered bag of conditions that can be
> dropped independently. An actor's image footprint, visibility and positional encoding are defined
> only once a camera is fixed, so we train and guide along the only nesting that is physically
> well-defined.*

**消融怎么设计**：合法层级 vs 独立随机 dropout，比较各控制指标。
同时要说明 **actor-only 不是一个"我们放弃了的公平任务"，而是一个定义不良的接口**——
不要把它当成对手在表格里让出的一格。

### 三条可守的差异（保留，第 3 条重写）

**1. 显式实例外观锚点。** 必须区分三种强度：① 类别或全局文本暗示；② 文本中的对象级描述；
③ **显式实例外观锚点（reference crop / image / identity embedding）**。只有 ③ 支撑"精确控制目标实例外观"：

| 方法 | 实例外观强度 |
|---|---|
| MagicDrive3D / DriveGen3D / CVD-STORM | ① 类别与文本 |
| X-Scene | ② 文本含 appearance 描述，非 reference identity |
| WorldSplat / InfiniCube / ScenDi | ①②，**无实例图像锚点** |
| DreamDrive | 单图可锚定**整场**外观，不锚定实例 |
| PhiGenesis / Envision4D | 由历史观测继承，非新实例指定 |
| **本方法** | **③**：per-instance canonical appearance token，来自**窗外**帧（`CanonicalAssetEncoder`，API 结构上无法接收目标 clip/轨迹/latent/bbox/mask），与该实例逐帧 metric placement 绑定 |

**2. 满足度可在解码世界中被度量，且度量是米。** 在 cascade 里，外观在 video model 决定、几何在
voxel/scaffold 决定、相机在渲染时决定；**只能审计最终像素**，那时几何、外观、相机误差已纠缠。
在我们这里，actor identity、actor metric z-depth、camera pose（米）、text alignment
**各自可在解码后的 Gaussian / depth / camera 上单独测量**。

**3. 条件寻址是 spatially typed 的，且三种"地址"互不冒充。**

```
appearance token  →  "是什么"（canonical，窗外，防泄漏）
placement 16D     →  "在三维哪里"（尺度不变量 + 由生成尺度决定的 log 幅值）
mRoPE 位置        →  "在图像哪里"（canonical UV → target projected bbox；出画对象放 reserved 带）
```

**诚实的代价（两条，都必须写）**：

1. 这仍是**全 attention 的软绑定**，代码没有结构性保证"一个对象条件只改变对应实例"。
   论文必须二选一或都做：(a) 用模块二的尺度对齐条件读取硬化绑定，(b) 用隔离实验实证。
   否则会被"这不就是多加了几个 condition token 的 conditional diffusion"击穿。
2. **上表第 1 条（显式实例外观锚点）目前没有任何东西验证过。** 我们声称能锚定实例外观，
   但训练里没有一项损失、评测里没有一个指标去检查渲染出来的确实是参考图里那个目标。
   这正是身份相似度评测（含 shuffled-reference 对照）要补的洞——**在它落地之前，第 1 条只能写成接口描述，不能写成能力主张**。

---

## D.3 现有方法是否有完整梯度链与联合训练

**不能写"现有方法没有 Gaussian/RGB/render loss"——这是错的。** 可守的表述是：

> **世界级误差在现有工作中普遍存在，但它训练的是下游 decoder 或表示、或拟合的是单个场景——
> 它被 stage boundary 或 per-scene optimization loop 与生成模型隔开。生成分布本身只对 latent
> 空间的目标负责。**

| 方法 | 世界级 loss 存在？ | 回到场景生成器？ | 一手依据 |
|---|---|---|---|
| **WorldSplat** | 是：GS decoder 用 render $\ell_1$+LPIPS、metric depth $\ell_1$、seg BCE | **否** | Sec. 3.1 "The three modules are trained independently" |
| **CVD-STORM** | 是：GS decoder 与 VAE 在 stage 1 联合训练（$\mathcal L_{\text{STORM}}$） | **否** | stage 2 "we freeze the encoder of STORM-VAE"；diffusion 损失只有 rectified flow。**摘要的 "jointly-trained" 指 VAE↔GS decoder，不是 diffusion↔world** |
| **ScenDi** | 是：VQ-VAE 用 occupancy BCE+VQ+render $\ell_1$+SSIM+fg mask | **否**，3D DiT 只有 Eq.(9) latent MSE | 三阶段分别训练；2D 阶段条件来自 VQ-VAE reconstruction 而非 diffusion sample |
| **PhiGenesis** | 是：Stage 1 RGB render / depth / seg | **否** | Stage 1 与 Stage 2 flow matching 分开 |
| **InfiniCube** | 是：两个 GS branch 用 photometric loss | **否** | "We train two branches separately" |
| **X-Scene** | **未报告** | 未报告 | 三个独立 diffusion denoising loss。只能写"论文未报告"，**不能武断说"确定没有"** |
| **MagicDrive3D / DreamDrive / LSD-3D** | 是 | 只回到**当前场景**的 Gaussian 参数 | 生成先验固定，不是 amortized generator training |
| **Envision4D** | 是：RGB MSE+LPIPS + VGGT camera/depth 伪标签 | **是**，贯通前馈网络 | 但它不是从噪声与条件生成新世界的生成器 |
| **本方法** | 是：L1/L2/L3 施加在 flow 自己预测的 $\hat z_0$ 上 | **是**（decode/heads 参数冻结但梯度穿过） | 生成器与世界反馈同一个优化目标 |

**梯度链已逐行确认贯通**：`compute_rgb_render_loss` 接收 flow 自己预测的 $\hat z_0$；
`decode_generated_dggt_geometry` 在 `autocast(enabled=False)` 下调用 tokenizer decoder 与三个 DGGT heads，
注释明确写 "Frozen DGGT/tokenizer parameters keep `requires_grad=False` but this module must not run
their decode/head calls under `torch.no_grad`"——**参数冻结，梯度穿过**。teacher 分支在 `torch.no_grad()` 内。

> **必须如实报告的两个零**（诚实性直接影响可信度）：
> `camera_grad_scale = 0.0`（`rgb_render_loss.py:663` 现在是**硬断言**，非零直接报错）、
> `sky_mask_grad_scale = 0.05`。D3 之后前者不再是可调旋钮，而是契约的一部分——
> 这一点由相机–几何一致性诊断报告，不要含糊过去。

**优势三条陈述：** (i) 训练信号在样本被消费的空间中测量；(ii) 生成器可把容量花在**对世界重要的
latent 方向**上（见 D.4）；(iii) 推理时**不需要第二个 2D 生成器**就能达到所报保真度，因此同一个
Gaussian 场景可从其他轨迹重新渲染——这是数据生成器必须具备而 per-clip 2D refiner 无法提供的性质。

---

## D.4 三层损失如何成为方法创新而非"多加三个 loss"

### 原则性陈述（保留，这仍是最漂亮的一段）

rectified-flow 目标度量的是 latent 欧氏误差 $\lVert\hat z_0-z_0\rVert^2$；我们关心的是解码之后的误差。
设 $F=R\circ D$，$\delta z=\hat z_0-z_0$：

$$
\lVert F(\hat z_0)-F(z_0)\rVert^2
\approx \delta z^\top J_F^\top J_F\,\delta z,
\qquad
\text{而 flow loss 用的是 }\delta z^\top I\,\delta z.
$$

**标准目标隐含假设 $J_F^\top J_F\propto I$。它不是。** 三层损失就是**在解码链的三个深度上对
$J_F^\top J_F$ 做 Monte-Carlo 估计**——我们让 flow 模型在**解码之后的误差**下被优化，
而不是在 latent 欧氏误差下被优化。

### $F$ 本身也必须先被测过，不能只被假设

上面那个论证有一个隐含前提：**$F$ 是可信的**。"在解码之后测量"只有当"解码之后"确实是物理空间时才成立。

**这条论证是结构性的，与任何具体 checkpoint 无关，因此永远成立。** 它也正是模块一（JST）
把一条**几何不变性约束写进 tokenizer 目标函数**的理由（`gaussian_scale_depth_similarity_loss`，
逐同像素要求编码–解码往返是相似变换，`gs_scale_sim_ratio` 须收敛到 1.0）——
**不是先测出问题再补救，而是把"解码之后可测量"变成压缩器自己的训练目标。**

> ⚠ **数值状态（2026-08-04，必须严格遵守）**：下表是 **tokenizer v1** 上的历史测量。
> **当前实现是 v2，尚未训练完成。** 下表**只作为"为什么需要那条约束"的动机证据**保留，
> **不能作为当前系统的 limitation 写进论文，也不能引用其中任何数字**。
> v2 训练结束后必须重测并整体替换本表。

| 【v1 历史测量】配对同像素，scene-balanced（30 个场景） | 值 |
|---|---:|
| `depth_recon / depth_direct` | 1.0307，CI [1.0208, 1.0421] |
| Gaussian 三轴几何平均 `scale_recon / scale_direct` | 0.8289 |
| 配对 `GS/depth`（相似性的必要条件，理想 = 1） | 0.7964，IQR 0.7876–0.8120，30/30 场景 < 1 |
| 帧内 depth ratio 的 log-MAD | 0.0122（空间上高度均匀） |

即：在 v1 上，冻结 tokenizer 的往返保住了"单位"，但没保住"各方向缩放一致"。
**这正是 v2 引入相似性约束的直接动因。**

这条发现有三个用途：

1. **它给 HDS 补上了缺的前提**，并且这个前提现在由模块一的目标函数承担，而不是靠事后校正。
2. **它是一个方法论教训，且是可复用的。** 我们最初想用 render PSNR 扫描一个校正常数 `c_gs`；
   D4 证明 **PSNR 对"尺寸类常数"是单调的**（扫到语义上界 2.5 仍在涨 +0.997932 dB），
   因为一个偏小的 splat 配上抬高的 opacity 渲染结果几乎一样。**这个目标函数没有内点极值。**
   我们据此**拒绝**了该校正，并把根因定位到 tokenizer 训练本身（附录 A）。
   > 写进论文的一句：*a rendering metric cannot identify a size-like constant; we report this as a
   > negative result rather than tuning against it.*
3. **它划出了按作用域的正确处置。** depth 有独立物理尺（LiDAR），所以米制边界可以校正
   （AbsRel `7.567% → 6.901%`，scene-bootstrap Δ CI `[0.052%, 1.225%]`）；
   GS 没有独立尺寸尺，所以**不做事后校正**——正确的出口是把约束放进 tokenizer 的训练目标，
   也就是 v2 做的事。

> **诚实边界**：v1 的测量只说明 `direct → recon` 在那个 checkpoint 上不是各方向等比缩放，
> **不证明 direct 的 Gaussian 本身就是以 Waymo 米制标定的**，
> 也不替代完整 renderer 的 opacity/quaternion/compositing 测试。
> **v2 的结论一律待训练完成后重测。**

### 两类损失：和真值比 vs 生成量之间互相比

HDS 的三层全部属于**和真值比**——它们把解码结果与一个**目标**比较（特征层和高斯层比 teacher 的解码，
图像层比真实像素）。这类损失只在训练时存在，**推理时没有目标可比**。

还有第二类损失，训练和推理**都成立**：让**同一次前向里生成的几个量互相解释**。
生成的相机运动必须和生成的几何对得上；解码出的目标表面必须落在条件给定的 3D box 里；
渲染出来的目标必须还是参考图那个目标。这类约束不需要任何真值，只需要生成结果自己。

| | 和谁比 | 训练时 | 推理时 |
|---|---|---|---|
| **HDS 三层** | 目标特征 / 真实像素 | ✓ | ✗ 没有目标 |
| **一致性指标** | 同一次前向里生成的其它量 | ✓ | **✓ 仍然可以算** |

目前这一类只落地了**一条诊断**（不是损失）：`generated_static_geometry_reprojection_cycle_v1`，
而且只接在推理输出里。**这一类现在全部作为评测指标报告，不作为损失**（见 E.4.4）：

```
生成相机   ↔  解码几何                        （已实现的 cycle 诊断）
解码几何   ↔  条件给定的米制 3D box            （2D IoU + 米制 z-depth，待接评测脚本）
渲染出的目标 ↔  参考图里的目标                  （identity similarity + shuffled 对照，待接）
```

这个区分不是分类游戏，它有两个直接用处：说明为什么满足度适合做评测而不是损失，
以及回答"推理时你还剩下什么一致性可查"——答案就是这三项。

### 小节命名

方法侧只有两个名字：**Hierarchical Decoding Supervision (HDS)**，三层就叫**特征层 / 高斯层 /
图像层**。另外两个模块（JointSceneTokenizer、统一的场景状态生成）用描述性名字，不造词。
不要把所有东西都叫 world loss，也不要再给这两类损失起第三个总称。

### 散点图必须升级为系统性测量

两个 sample 的轶事一定被攻击。除两个案例外**必须**给：

- 全验证集 feature MSE 与 depth / Gaussian / render error 的 **Spearman 相关系数**；
- **按 $\sigma$ 分层**（解码后的监督只在 $(1-\sigma)^2$ 权重下生效，分层能证明不是噪声效应）；
- **加入 HDS 前后**的相关性与离群点比例；
- **局部近邻内的 endpoint 方差**：把横轴切成窄条，报每条内 world error 的分布宽度。
  这比"挑两个点"强得多——它证明的是"横轴接近时纵轴仍然分散"这个一般性质；
- **Pareto front 的移动**：加入解码后的监督之后，(latent error, world error) 的前沿是否整体
  向左下移动。这是最能说明问题的一张，因为它同时排除了"只是把 latent 拟合得更好"这个解释；
- Gaussian error 按 position / scale / opacity / color / rotation **分解**；
- **把往返缩放的测量画进同一张图**：$x$ 轴 latent error，$y$ 轴 world error，
  再叠一条"若解码是各方向等比缩放，$y$ 应该长什么样"的参考线。这张图同时说明两件事：
  解码不保距（所以需要 HDS），且解码本身有系统畸变（所以需要先测）。
- 可选：calibration curve（给定 latent MSE 分位数，world error 的条件分布）。

> **这张图的解释必须限定在自己身上**：*The scatter does not diagnose competing methods. It shows
> that, in our representation, latent proximity alone gives an incomplete ranking of decoded scene
> quality.* 不要拿它当作对别人的指控——我们没有跑过别人的模型。

### 边界

不是"首个 render loss"（LSD-3D、ScenDi VQ-VAE、WorldSplat GS decoder、CVD-STORM STORM-VAE 都有）；
不是"首个多层特征"（PhiGenesis 已用冻结 video VAE 的多尺度 decoder features）；
不能无限定写 "complete gradient chain"（Envision4D 有完整 render gradient）。
**可以写的**：*the flow objective is blind to an anisotropic decoder; we make it see it, and we
first check that the decoder is worth seeing through.*

---

# Part E. 统一故事与模块地图（2026-08-02 重写）

## E.1 一句话故事

> 在现有的驾驶 Gaussian 生成里，最终的三维总是在生成目标之外产生的：由一个独立的重建阶段产生、
> 由图像平面的 latent 抬起来、或者藏在体素的离散解码后面。三条路各有各的机制，落点相同。
> **我们改在一个前馈重建模型的内部状态里生成**——它的先验落在三维上，又不引入格点。
> **然而**这类模型只接受无位姿图像，绝对尺度在数学上不可确定，尺度因此被归一化掉；
> 而驾驶的条件与标注全是米。所以我们把这一段的**米制尺度和视场角作为三个数与场景一同生成**，
> 并重参数化条件，使未知尺度只表现为一个加性常数；随后在**解码之后、用米**测量误差。

## E.2 三层论证（每层都是机制级，换掉任何 baseline 仍成立）

> **2026-08-04 重写。** 旧版把"米制是固定属性"当作第二层缺口，已删除（理由见 F.6 修订链条第 8 条）。
> 新版的 ¶2 是三个各有机制的缺点，它们合起来把 ¶3 的落点挤成唯一。

```
¶1   一个驾驶样本的价值，是在米制三维世界里被判定的
              │
              ▼
¶2   三条路线，三个各自的机制，落点相同：
     ├ 视频 → 重建     阶段之间没有反馈通道      → 重建不在生成目标之内
     ├ 视图对齐 latent  先验的来源仍在图像上       → 抬起来把不一致变成几何误差
     └ 体素 latent     先验在三维上，但格点粗 + 离散解码 → 渲染误差进不了生成目标
              │
              ▼   （前两条：不引入格点，但先验不在三维上）
              │   （第三条：先验在三维上，但引入了格点）
              ▼
¶3-a  于是只剩一个位置：先验在三维上，且不引入格点
      → 前馈重建模型的内部状态                                  ← 表示的选择
              │
              ▼
¶3-b  然而这类系统只接受无位姿图像，绝对尺度在数学上不可确定，
      它们因此把尺度归一化掉；而驾驶的条件与标注都是米制的      ← 这条路线的代价
              │
              ▼
¶3-c  把尺度与场景一同生成，并重参数化条件，
      使未知尺度只表现为一个加性常数；再在解码之后用米测量误差  ← 回答
              │
   ┌──────────┼───────────┬──────────────┐
   ▼          ▼           ▼              ▼
 D.1        D.1.5       D.2            D.3/D.4
 在哪个空间   用什么单位    条件怎么进来     误差在哪里测
 生成        尺度被生成    同单位 + 依赖    解码之后，
 decoder 可微 出来         关系正确        用这套单位
```

**依赖关系是真实的，不是修辞：**

- 没有 D.1（一个可微且冻结的 decoder），D.1.5 的尺度**无处安放**，D.3/D.4 的三层损失**写不出来**；
- 没有 D.1.5（生成出来的尺度），D.2 的四族条件里有两族**量纲是坏的**，D.4 的"米制误差"**没有定义**；
- 没有 D.3/D.4（解码之后的监督），D.1.5 生成的尺度**没有任何东西约束它**。

**三者互为前提，这就是"一个故事"而不是"三个卖点"的证据。**

> **叙事上最关键的一处**：¶3-b 的"然而"是被 ¶3-a **逼出来的**——先验要落在三维上，就只能用
> 无位姿的重建系统；无位姿，尺度就在数学上不可确定。
> **所以尺度问题不是我们自找的，是这条路线的必然入场费。** 这比 v9 的"我们选了个没单位的空间"
> 强得多，也让 ¶2 完全不需要提米制。

## E.3 措辞库

**核心句（可放 Abstract 第 2 句与 Intro ¶3 首句）：**

> In driving Gaussian generation the final 3D is produced outside the generative objective: by a
> separate reconstruction stage, by lifting an image-plane latent, or behind a discrete voxel
> decoding. We generate instead in the internal state of a feed-forward reconstruction model, whose
> prior is over 3D and which introduces no grid, and supervise the generator through its frozen
> decoder at the feature, Gaussian and image levels.

**第二核心句（¶3 的转折，建议放 Intro ¶3 或 contribution 第一条）：**

> Such a model takes unposed images, where absolute scale cannot be determined, so we generate the
> metric scale of the clip together with the scene and reparameterize the conditions so that the
> unknown scale appears only as a single additive constant.

> **2026-08-04 替换记录**：旧的两句核心句都建立在"他们的米制来自一个生成器碰不到的部件"上，
> 而那条已被调研推翻（见 D.1.5）。新版第一句改成三条路线共同的结构事实（三维在生成目标之外产生），
> 第二句把米制放回它真正的位置——**我们这条路线的代价与解法**。

**标题候选（都不玩文字游戏）：**

1. *Metric-Scale Generation of Controllable Driving Gaussian Scenes*
2. *Generating Driving Gaussian Scenes Together with Their Metric Scale*
3. *Predicting Scene Scale for Controllable Driving Gaussian Generation*
4. *Driving Gaussian Scene Generation with Decoder-Level Supervision and Explicit Metric Scale*

推荐 **1**（最短，且"metric-scale generation"一眼就知道在讲什么）或 **2**（把"和场景一起生成"
这个关键点写进标题，更准确但长一些）。

**用语纪律：方法侧只留两个名字。** SceneDirector 全文只造了一个模块名（MGRA），它的核心概念
*structural reliability* / *semantic completion* / *uncertainty-aware allocation* 全是普通词组。

| 角色 | 表述方式 |
|---|---|
| 失败机制（一） | 不造词：*the 3D scene is produced after generation, by a separate stage, and never enters the generative objective.* |
| 失败机制（二） | 不造词：*their metric scale is fixed inside a component the generator never trains, so it cannot adapt to the clip being generated.* |
| 正面主张 | 不造词：*we generate the scene's metric scale and field of view jointly with the scene, as three scene-level numbers.* |
| 方法名（一） | **Hierarchical Decoding Supervision (HDS)**，三层叫**特征层 / 高斯层 / 图像层** |
| 模块名（其余两个） | **JointSceneTokenizer**、**统一的场景状态生成**——描述性，不造词 |

> **不要再用的词**：米制承诺、表示保持未承诺、可审计 / 可被核对、规范量、scene gauge（正文里
> 直接说"场景的米制尺度和视场角"）、readout、fidelity / coherence、世界级监督。
> 代码标识符 `scene_gauge` / `gauge_*` / `pullback_*` 保留，但只在提到代码时出现。

## E.4 模块地图：三个核心模块（2026-08-04 重写）

### E.4.0 模块与核心创新的对应（先看这张表）

核心创新是 ¶3 的三步，三个模块一一对应：

| 核心创新的哪一步 | 模块 | 类型 |
|---|---|---|
| **¶3-a** 在前馈重建模型的内部状态里生成（先验在三维上、无格点） | **一、JointSceneTokenizer** | 架构 |
| **¶3-b/c** 付清这条路线的代价：尺度不可确定 | **二、统一的场景状态生成** | 架构 |
| **¶3-d** + ¶2 缺点一/三：在解码之后测量误差 | **三、Hierarchical Decoding Supervision** | 损失 |

> **判据（本轮新增，用来挡住模块膨胀）**：一个模块如果**指不出**它关的是 ¶2 的哪个缺点或 ¶3 的哪一句，
> 它就不是核心模块。按这条判据，Actor Slots 与 Epipolar Attention 已被移出（理由见 E.4.5）。
>
> **三个模块里只有一个是损失。** 这一点必须保持——"我们加了五个 loss"撑不起方法章节，
> 而 SceneDirector 的三个模块（scaffold / texture bank / mask-gated attention）**全是架构**，
> 它的轨迹精度 ATE/AOE 是**评测指标**，不是训练损失。

---

### E.4.1 模块一：JointSceneTokenizer

> **代码**：`dggt/models/joint_scene_tokenizer.py`（758 行）。
> **当前实现是 tokenizer v2，尚未训练完成**——见本节末尾的状态方框。

**唯一叙事职责**：让"在重建模型的状态里生成"从一个想法变成可行的事。

**问题的规模是具体的**：DGGT 的四层三流特征是每 patch $4\times3072=12288$ 维；一个 10 帧、
$25\times37$ patch 的窗口就是 $10\times925\times12288\approx1.14\times10^8$ 个数。
**这个东西不可能直接做扩散。** JST 把它压到每 patch **1024** 维，**12:1**。

但"压缩"不是这个模块的创新点，**怎么压才是**。全部设计决策服从同一条总纲：

> **被压缩的对象有结构，压缩器就按那个结构设计。**

这正是它区别于"拿一个通用 ViT-VAE 去压"的地方：

| DGGT 特征的结构 | JST 的对应设计 | 不这样会怎样 |
|---|---|---|
| 三条流：DINO（语义）、frame（单帧几何）、global（跨帧关系） | **每流独立 LayerNorm + 独立线性投影**，且**逐层各有一套**（`stream_norms` / `stream_proj` 都是 `ModuleList` over layers，`:454-477`） | 三流的统计量差异很大，共享归一化会让一路淹没另一路；四层角色不同，共享投影等于假设它们可互换 |
| 四个深度（4/11/17/23）高度冗余但不等价 | 层轴上先 `LayerAttnStack`（depth 2）让四层互看（`:480-490`），再用 `LearnedQueryPool(n_query=1)` 以**一个学习到的 query** 聚成一个 token（`:491`） | 平均会抹掉深层几何，拼接则维度爆炸；learned query 让网络自己决定每个 patch 该从哪一层取什么。**层轴上的 4:1 压缩在这里完成** |
| DGGT 主干本身是逐帧 / 全局注意力交替 | JST 主干用**同构的** `FrameGlobalBlockPair`×3（`:492-508`），2D RoPE 管 patch、1D RoPE 管时间 | 用同一套归纳偏置去压缩，而不是拿通用 ViT 去压一个几何模型的内部状态 |
| patch 是 14× 下采样的，高频容易丢 | **仅 encoder 侧**加一条 `DetailConvBranch`（两层 3×3 conv，出 128 维拼成 1152，`:322-353`） | 车道线、杆状物、目标边界在纯 token 域会被抹平；conv 在 patch 网格上补回局部结构 |
| 四层要还原成不同的东西 | 解码端 `LearnedQueryPool(n_query=4)` 反池化（`:635`）+ **每层一个独立的 `PerLayerDecoderHead`**（`:648-664`，各自 `pre_proj` 到 3072 再过一个带 2D RoPE 的 Block） | 共享 head 会把四层拉平，等于白做层轴注意力 |

**一处刻意的非对称，值得单独写一句**：detail 分支**只在 encoder**——decoder 里明写
`del detail_dim  # decoder does not have a detail branch`（`:609`）。
细节在编码时被**捕获**，解码时由四个 per-layer head 各自**重建**。
这是一个信息瓶颈设计，不是遗漏；若解码端也放一条 conv，瓶颈就被绕开了。

#### 最能支撑故事的一条：目标函数里有一条几何不变性约束

`train_tokenizer.py::gaussian_scale_depth_similarity_loss`（`:1011`）是一条**逐同像素配对**的约束，
要求编码–解码往返是一个**相似变换**：解码出的高斯尺寸与解码出的深度之比，
必须与直接解码的一致；日志量 `gs_scale_sim_ratio` 是被审计量的 $\exp$，**必须收敛到 1.0**。
`--lambda_gs_scale_sim 0` 可退回只优化特征保真的旧目标。

**为什么这条直接服务核心创新，而不是又一条正则**：

> ¶3-d 说"在解码之后、用米测量误差"。但**特征域的重建误差小，并不意味着解码之后的几何是等比的**——
> 一个只优化特征保真的 tokenizer 完全可能把深度和高斯尺寸缩放成不同的倍数，
> 那样"在解码之后测量"这句话本身就失去意义。
> **所以它不是额外的约束，它是 ¶3-d 成立的前提，并且被写进了 tokenizer 自己的目标函数。**

这也是这个模块在论文里的落点：**压缩器不只要保住特征，还要保住解码后几何的可测量性。**
据我们所知，把这样一条几何不变性写进特征 tokenizer 目标的做法，在重建特征空间生成这一路线上尚无先例。

> ⚠ **状态（必须严格遵守）**：当前实现是 **tokenizer v2**，`--lambda_gs_scale_sim` 已实现，
> **但尚未训练完成**。因此论文里**只能写目标函数包含这一项**，
> **不能写任何 `gs_scale_sim_ratio` 的数值，也不能写任何往返畸变的实测结论**。
> 早期在 **v1** checkpoint 上的历史测量描述的是一个**已被替换的模型**，
> 不能用来描述 v2，也不能作为 limitation 写进论文。**训练完成后重新测量并回填。**

**消融（每条检验一个设计决策）**：
1. 三流独立 LayerNorm + 独立投影 **vs** 直接把 12288 维拍平；
2. 层注意力 + learned query pooling **vs** 层平均 / 层拼接；
3. 有无 detail 分支（看高频区域的解码误差）；
4. 每层独立 decoder head **vs** 共享 head；
5. 有无相似性约束（看 `gs_scale_sim_ratio` 与解码后几何误差的变化）。

---

### E.4.2 模块二：统一的场景状态生成

**唯一叙事职责**：**解释这个场景所需要的一切——相机、天空、单位——都和它一起被采样，而不是从外面供给。**

这一句直接对上 ¶2 缺点二（那些方法的观察几何来自条件而非表示）与 ¶3-b/c（代价与如何付清）。

**包含什么**：

| 成分 | 代码落点 | 在这个模块里承担什么 |
|---|---|---|
| 五路联合生成序列 `[scene \| camera \| sky \| gauge]` | `dggt/models/scene_flow.py`，typed mRoPE | 四类量共用同一次去噪、同一组 $\sigma$，不是四个模型拼起来 |
| 9 维米制相机状态 | `dggt/utils/camera_generation.py` | 相机是**被解出来的生成变量**，不是外部供给的条件 |
| 3 维尺度与视场角 | `dggt/utils/scene_gauge.py` + `scene_flow.py:1121-1128,3052-3085` | 付清 ¶3-b 的代价：这条路线唯一缺的那个量 |
| 天空分支（片段常量视场角） | `train_scene_flow_pretrain.py` atlas 装配 | 远景留在生成状态内，不交给第二个二维阶段 |
| 16 维 `factorized_asset_v3` | `dggt/utils/factorized_asset_condition.py` | **11 个尺度不变通道 + 5 个标准化 log 幅值**，未知尺度只表现为同一个加性常数 |
| 合法任务层级 | `train_scene_flow_pretrain.py:1539-1629`；推理端 `:1886-1934` | 条件之间的投影依赖被写进训练分布与 CFG |

**三处真耦合（必须举证，否则会被读成"塞进一个 transformer"）**：

1. **尺度 hidden state 注入 scene decoder 的 conditioning**（`scene_flow.py:3465-3472`）：
   几何显式以生成的尺度为条件，同时 video flow 的梯度回流进尺度 token——**双向**，不是旁路输出。
2. **条件与生成量是同一个方程的两半**：placement 那 5 个 log 幅值与真值只差一个加性常数，
   而那个常数正是同一次前向生成的 `log_metric_scale`。
3. **相机是解出来的而不是给定的**，天空是状态的一部分而不是第二个二维阶段。

#### 本模块唯一待新增的部件：尺度对齐的条件读取

placement 的 5 个标准化通道与解码几何严格满足

$$\log(\text{metric}) = \log(\text{DGGT}) + \log s_{\text{metric}}$$

因此在条件进入注意力**之前**，直接减掉当前步生成的 $\hat g_{\text{scale}}$：

$$c'[\dots,\{3,4,5,6,13\}] \;=\; c[\dots,\{3,4,5,6,13\}] - \hat g_{\text{scale}}$$

**价值不在省了一个损失，在于它制造了一条梯度回路**：条件用得对不对，梯度会顺着这条减法
回流进尺度分支——**尺度因此被条件本身监督，不需要任何额外的一致性损失**。
它也是"未知尺度只表现为一个加性常数"这句话的**字面实现**。

**实现约束**：高噪声步的 $\hat g$ 很差，减掉它会引入噪声；按 $\sigma$ 退火或只在低噪声区启用，
这条必须写进方法小节，不要藏起来。

**消融**：减 / 不减 $\hat g$；placement v3 **vs** 米制原值；固定均值尺度 / 只预测但不注入 scene decoder / 完整。
中间那一臂是验证 `scene_flow.py:3465-3472` 那个耦合的**唯一**实验。

---

### E.4.3 模块三：Hierarchical Decoding Supervision (HDS)

**唯一叙事职责**：把解码之后的三维带进生成目标。**¶2 的三个缺点里有两个由它回答**
（缺点一"重建不在生成目标之内"、缺点三"离散解码把渲染误差挡在生成目标之外"），
这是全文分量最重的一处。

**做法**：误差穿过冻结的 tokenizer decoder 与 DGGT heads 直到 rasterizer，在三处测量——
**特征层 / 高斯层 / 图像层**（`dggt/losses/reconstruction_feedback_loss.py` + `rgb_render_loss.py`）。

**为什么不是显然的替代方案**（这一段必须写，否则读起来像"多加三个 loss"）：

- *"只用最终 render loss 不就够了？"* → 渲染梯度稀疏且不稳定；特征层是稠密的，三层一起才有条件数。
- *"为什么不把 decoder 一起训？"* → 那样 decoder 会吸收生成器的误差，增益无法归因到生成器。
  **冻结是必要条件，不是省事。**
- *"解码之后就是物理空间吗？"* → 不一定。这正是模块一里那条相似性约束存在的理由——
  **先保证解码不畸变，再在解码之后测量。** 两个模块在这里咬合。

#### 唯一待新增的部件：置信度加权

用**冻结解码头自己输出的** `depth_conf` / `gs_conf` 给高斯层和图像层的损失做逐像素加权：
解码可信的地方反馈强，解码本身就不确定的地方（天空、远景、细结构）反馈弱。

**当前代码状态（已核对）**：`reconstruction_feedback_loss.py` 目前把 `depth_conf`/`gs_conf` 当作
**被监督的目标通道**（`:264-322`），但 `_dense_weight`（`:103-140`）只用 sky mask 与 patch weight，
**没有把 teacher conf 乘进权重**。两件事不冲突，可以同时做。

**实现约束**：
1. 用 **teacher** 的 conf，不是学生自己的——否则模型把自己不确定的地方权重全调低就能摆烂，
   这是真实存在的退化解，论文里要点明已避免；
2. conf 逐样本归一化（除以其均值）后再乘，否则等价于给整个 batch 改学习率；
3. 权重要 detach。

**消融**：无 HDS / 只特征层 / +高斯层 / +图像层；加权 vs 不加权 + 权重图可视化。

---

### E.4.4 满足度全部作为评测指标，不作为损失

这是本轮最重要的一处方法论调整，也是与 SceneDirector 对齐的做法——
**它的三个模块全是架构，轨迹精度 ATE/AOE 是评测指标**。

| 量 | 处置 | 现状 |
|---|---|---|
| 目标 2D IoU + 米制 z-depth 误差 | **评测指标** | 待接评测脚本 |
| 目标 identity similarity（含 shuffled-reference 对照） | **评测指标** | 待接评测脚本 |
| 生成静态几何的重投影 cycle EPE | **评测指标** | **已实现**：`generated_static_geometry_reprojection_cycle_v1`，接在 `inference_scene_flow_pretrain.py:1381` |
| 相机位姿误差（米） | **评测指标** | 已可算 |

> 这样处置有三个好处：方法侧只剩一个损失族；¶2 缺点二的对照有了直接的数字（cycle EPE）；
> 而且**不需要用一个损失去强制"我们的先验在三维上"**——那样反而像在承认表示没兑现承诺。
> 代价是：只能写"我们测了，它是一致的"，不能写"我们保证它一致"。**这取决于那个数跑出来好不好。**

### E.4.5 明确不做的（记下理由，避免以后反复）

| 曾考虑 | 不做的理由 |
|---|---|
| **一致性损失族**（相机–几何、目标–box、目标身份三项） | 会让方法侧变成"五个损失"；三项的职责已分别由模块二的梯度回路与评测指标承接 |
| **Metric Actor Slots**（持久 slot + FiLM + 足迹门控 + 读回） | **指不出它关 ¶2/¶3 的哪一句**。目标控制是**能力**，按既定判断放 Intro 最后一段，不占核心模块位 |
| **Epipolar Attention** | 半对（关缺点二），但代价大：稠密偏置在 $N\approx9300$ 下需约 173 MB/样本，且 `F.scaled_dot_product_attention` 一旦传入任意 additive mask，flash 后端失效。可行版本是沿对极线稀疏采样的独立分支，属于可选增强，不进核心 |

---

### E.4.6 依赖与阻塞

| 项 | 状态 | 谁依赖它 |
|---|---|---|
| **tokenizer v2 训练** | **进行中，未完成** | **模块一的全部实验数字**；HDS 的"解码不畸变"前提；任何往返畸变的结论 |
| 尺度生成 + 米制相机 + placement v3 + 五路联合序列 | **已落地**（回归 732 passed, 1 skipped） | 模块二；G.1 的米制指标；米制导出 |
| 合法任务层级 | **已落地**（`train_scene_flow_pretrain.py:1539-1629`，工作树） | 模块二；D.2 的第二条腿；层级 CFG |
| 离线尺度 GT 表 | **已完成**：training 4787/4787 片段 / 798 场景；validation 1212/1212 / 202 场景 | 尺度的直接监督 |
| HDS 三层 | **已落地** | 模块三 |
| HDS 的置信度加权 | **已实现，未训练/未消融** | 回答"teacher 自己有误差" |
| 尺度对齐的条件读取（减 $\hat g$） | **未实现** | 模块二的梯度回路 |
| 满足度评测脚本（2D IoU / 米制 z-depth / identity + shuffled 对照） | **未实现** | E.4.4 的四行 |

> **克制建议**：三个模块到此为止。再加就会回到"功能清单"。
> 每一个待新增部件都满足同一条：**调节一份本来就存在的信息**
> （置信度用解码头自己的输出，减 $\hat g$ 用同一次前向生成的量），没有凭空的 learned gate。

---

# Part F. Introduction 逐段写作（2026-08-02 按新故事重写）

## F.0 与用户 ¶1/¶2 框架的对应关系

用户给的框架是三段：数据需求与三维职责 / 驾驶 Gaussian 方法的演进与缺口 / 引出方法。
新故事**不改前两段的骨架**，只做两处手术：

| 位置 | 用户框架里的写法 | 本轮修订 |
|---|---|---|
| ¶1 结尾 | "若这些量只在 2d 视图中看起来正确，生成数据仍可能给感知与规划提供错误的几何监督" | 再推一步，**点出"标注"这个具体后果**：尺度没定下来，样本连标签都写不出来。这是 ¶3 的接口 |
| ¶2 的 "但这些方法存在xxx问题" | 留空 | 填**三个各有机制的缺点**（见下表），它们合起来把 ¶3 要找的东西定义得只剩一个位置 |
| ¶2 中的 Envision4D | 与 LSD-3D / ScenDi 并列 | **移出核心谱系**（违反准入 2：依赖目标场景观测）。只在 Related Work 与 D.3 的边界论证里出现 |
| ¶2 中的 PhiGenesis | 与 WorldSplat 并列 | **同上移出**；否则整段会被读成"这些都是 source-free 生成器" |
| ¶2 第二类成员 | LSD-3D、ScenDi、Envision4D | **WorldSplat、CVD-STORM（视图对齐）+ ScenDi、LSD-3D（体素）**，且**视图对齐在前、体素在后** |

### 【2026-08-04 定稿】三个缺点，三种机制

| 路线 | 机制 | 缺点 |
|---|---|---|
| 第一类（视频 → 重建） | **阶段之间没有反馈通道** | 视频难重建不受惩罚——视频模型被优化的是画面真实，不是之后恢复出的几何 |
| 视图对齐 latent | **先验的来源仍在图像上** | 抬起来只是把视图之间的不一致变成几何误差 |
| 体素 latent | **先验终于在三维上，但格点太粗，且离散解码挡住梯度** | 细节受限；渲染误差被挡在生成目标之外 |

**三条构成递进链，每一步修好上一步的毛病又带来新的**：第一类的三维是下游产物 → 视图对齐把它变成生成变量了，但先验没走出图像平面 → 体素让先验落到三维上了，但换来格点与断链。

**于是 ¶3 要找的东西只剩一个位置：先验落在三维上（体素做到了，前两个没有），同时不引入格点（前两个做到了，体素没有）。** 三维重建系统的表示恰好同时满足。

> **本轮从 ¶2 移出的内容**：v9 曾把"米制尺度是一个生成器碰不到的固定属性"写进 ¶2 当作第二层缺口。
> **这条已删除**，理由见 D.1.5：现有方法的米制来自输入侧（已标定 rig 外参、给定位姿、LiDAR），
> 按构造是对的，把它说成缺点立不住。米制现在只出现在 ¶3，作为**我们这条路线的代价**，不是对手的短处。

## F.1 ¶1：保留，只改结尾

现有 ¶1 逻辑正确，已在做 SceneDirector ¶1 该做的事（卖任务，不卖模块）。唯一要改的是**结尾必须
埋下两颗种子**：(i) 样本的价值在三维中被判定，因此生成器被优化的对象也应当在三维中被度量；
(ii) **而"三维"必须是米**——尺度没定下来的场景，无法为检测器写出一行标签。

第一颗种子接 ¶2 的三个缺点（三者都是"最终三维不进入生成目标"），第二颗种子接 **¶3 的转折**
（重建表示没有尺度）。**没有第二颗，¶3 的"然而"会显得突兀。**

**¶1 结尾英文定稿**（2 句 / 44 词）：

> A generated sample is judged by its 3D content, so that is where a generator should be measured.
> In driving, this content must also be metric: a scene without a determined scale cannot be
> annotated, and its geometry cannot supervise a detector or a planner.

> **改动记录（2026-08-03 用语优化）**：`is judged in the metric 3D world` 是中式表达，改成
> `is judged by its 3D content`；删掉句首的 `And` 与引号里的 `"3D"`（都偏口语）；
> `cannot even be annotated, let alone used to` 的递进语气像博客，改成两个并列的具体后果。

## F.2 ¶2 结构（2026-08-04 重排）

严格按此顺序。**排序不是随意的：视图对齐必须排在体素之前**，因为它和第一类同属"在图像平面上生成"，
两句可以顺着说；体素是唯一走出图像平面的一支，放最后才能让 ¶3 的落点唯一。

1. **第一类：视频 → 重建**（MagicDrive3D、DriveGen3D；InfiniCube、X-Scene 加三维支架）
   → 缺点：**重建不在生成目标之内**；
2. **第二类之一：视图对齐 latent**（WorldSplat、CVD-STORM），不再经过显式视频
   → 缺点：**先验仍在图像上**；
3. **第二类之二：体素 latent**（ScenDi、LSD-3D），几何是显式的
   → 缺点：**格点太粗 + 离散解码挡住梯度**。

> **不要把三个缺点合并成一句共同结论。** 每条各有机制，合并会让读者以为是同一个抱怨说三遍；
> 分开写，读者自己会看出它们指向同一个空位，而那个空位就是 ¶3。

## F.3 ¶2 定稿：中英逐句对照

> **本表为 2026-08-04 定稿。** 全段 8 句、约 150 词，平均每句 19 词，无破折号。
> **编号已重排**（旧的 0–6 编号作废，其余章节的"第 5/6 句"等引用一律按本表更新）。

| # | 中文 | English | 词数 |
|---|---|---|---|
| 1 | 近期工作把驾驶场景生成扩展到 3D 与 4D Gaussian，主要形成两个方向。 | Recent work extends driving-scene generation to 3D and 4D Gaussians in two directions. | 13 |
| 2 | 第一类方法依托视频生成先验：一些工作先合成多视图视频，再将其重建为 Gaussian。 | The first builds on video generative priors: MagicDrive3D~\cite{magicdrive3d} and DriveGen3D~\cite{drivegen3d} synthesize multi-view videos and reconstruct Gaussians from them, | 19 |
| 3 | 另一些工作引入显式三维支架，来同时支持视频生成与 Gaussian 构建。 | while InfiniCube~\cite{infinicube} and X-Scene~\cite{xscene} add voxel or occupancy scaffolds to guide both stages. | 14 |
| **4（缺点一）** | 但重建不在生成目标之内：视频模型被优化的是画面是否真实，而不是之后从中恢复出的几何。 | However, reconstruction lies outside the generative objective: the video model is optimized for realistic frames, not for the geometry recovered from them. | 22 |
| 5 | 第二个方向直接从生成变量解码出 Gaussian。WorldSplat 与 CVD-STORM 在视图对齐的 latent 上直接抬出 Gaussian，不再经过显式的视频； | The second direction decodes Gaussians directly from the generated variable. WorldSplat~\cite{worldsplat} and CVD-STORM~\cite{cvdstorm} lift them from a view-aligned latent without an intermediate video, | 26 |
| **6（缺点二）** | 但先验仍是图像上的先验，而不是三维场景上的先验，抬起来只是把视图之间的不一致变成几何误差。 | but the prior remains one over images rather than 3D scenes, and lifting might turn inconsistency between views into geometric error. | 21 |
| 7 | ScenDi 与 LSD-3D 转到体素 latent，几何因此是显式的； | ScenDi~\cite{scendi} and LSD-3D~\cite{lsd3d} move to a voxel latent, where geometry is explicit, | 13 |
| **8（缺点三）** | 但格点定死了场景的分辨率，细节因此受限，离散的解码又一次把渲染误差挡在生成目标之外。 | but the grid fixes the scene's resolution, capping detail, and its discrete decoding leaves rendering error outside the generative objective. | 21 |

**三个缺点的措辞纪律**

- **只做可从对方论文查证的结构性陈述**：ScenDi 的 0.4 m 体素与固定体积（$32\times128\times192$，
  覆盖 $[-25.6,25.6]\times[-20,56.8]\times[-3,9.8]$ m）、它自陈的"外观细节受三维表示分辨率限制"、
  VQ codebook 与 occupancy 判定、LSD-3D 的 NKSR 表面抽取、WorldSplat 的三模块分训、
  CVD-STORM 的 stage-2 冻结 STORM-VAE——每一条都可查。
- **缺点三个各写各的机制，不要合并成一句共同结论**（见 F.2 的方框）。
- 第 4 句与第 8 句**共用 `the generative objective` 这个词组**：一个是阶段分开被放在外面，
  一个是不可微所以进不来。词组必须一致，机制不同才有味道；不要在其中一处换成
  `the training loss`、`the diffusion objective` 之类的同义词。
- 第 6 句用 **`might`** 而不是 `does`——我们没有测过他们的几何误差，只能说这条路径会把不一致
  转成几何误差，不能断言它一定发生。这与 v8 已确立的"声称无上界而非误差很大"是同一条原则。
- 第 7 句的 `where geometry is explicit` **是必须保留的让步**。体素是唯一真正把几何做成显式变量的
  路线；先表扬再指出代价，"但太粗、但断链"才像遗憾而不是抹黑，也才能让 ¶3 的落点唯一。

> ⚠ **本轮删除的一整条论证**：v9 的第 5/6 句曾写"这些场景之所以是米制的，靠的是一个生成器从不
> 训练的部件……尺度因此是固定属性而不是生成变量"。**已删除。** 调研确认现有方法的米制来自输入侧
> （STORM 的输入外参、DrivingForward 的 rig 外参、DrivingRecon 的 LiDAR 深度监督），按构造是对的；
> 把它写成缺点会被当场反驳。米制现在只出现在 ¶3，作为**我们这条路线的代价**。详见 D.1.5。
>
> **同时被这次删除解决的旧隐患**：那个反问"你自己选了一个没有单位的空间，才有这个问题"——
> 现在 ¶2 根本没提米制，¶3 主动把它作为代价说出来，反问失去了落点。

## F.4 ¶3 定稿：中英逐句对照

> **本节 2026-08-04 分成两半。前半（第 1–3 句）用户已定稿，后半（第 4–9 句）是本轮草稿。**
> ¶2 的三个缺点把 ¶3 的落点定义得只剩一个位置，所以第 1 句**直接落到那个位置**，
> 两个限定词放在句尾回指缺点二与缺点三；第 2 句是转折（尺度不可确定），
> 第 3 句把这个代价翻译成驾驶任务上的**两个**具体后果——**后半段就是逐一回答这两个后果**。

**前半（问题）：用户已定稿，2026-08-04，以下为用户原文，不要再改。**

| # | 中文 | English | 词数 |
|---|---|---|---|
| **1（承接三个缺点）** | 我们因此在前馈重建模型的内部状态上生成：同一个状态解码出 Gaussian 与相机位姿，先验因而落在三维上，也没有格点来定死场景的分辨率。 | We thus generate in the internal state of a feed-forward reconstruction model~\cite{dggt}, which decodes to Gaussians and camera poses together: a prior over 3D, and no grid fixing the resolution. | 30 |
| **2（转折：代价）** | 这类系统以无位姿图像为输入，绝对尺度不可确定，因而被归一化掉。 | Such models~\cite{vggt, cut3r, anysplat} take unposed images, where absolute scale cannot be determined, and normalize it away. | 15 |
| **3（代价的后果）** | 而驾驶场景生成的条件是以米给出的：轨迹或目标框无法施加在一个没有尺度的状态上，生成的结果也无从度量。 | Driving-scene generation, however, is conditioned in metres: a trajectory or a box cannot be imposed on a scale-free state, nor can the result be measured. | 25 |

> **用户在本轮把旧的第 2 句拆成了两句**（模型的事实 / 驾驶任务的后果），并在第 3 句末尾放了**两个**
> 抱怨：`cannot be imposed` 与 `nor can the result be measured`。**后半段必须把这两个各还一次**，
> 且用同样的词回指——这与 ¶2→¶3 用 `a prior over 3D` / `no grid` 缝合是同一手法。
> 第 1 句的 `which decodes to Gaussians and camera poses together` 也是后半段第 5 句
> "decoder 已预训练且可微" 的锚点。

> ⚠ **2026-08-05：后半草稿（第 4–9 句）已作废，下表只保留修订理由。** 用户已把 ¶3 的方法句定稿，
> 顺序是：命名句 → **SGT**（每 patch 很宽的 token 状态压到适合扩散的维度）→ 几何不变性约束
> （每像素把高斯尺寸与深度绑定）→ **JSF**（场景/天空/相机/尺度由文本与可选轨迹、目标一次生成）
> → 四者共用同一噪声水平、逐层互相注意 → 尺度是解码几何的单位 → 分层解码监督（HDS）。
> **正文以用户定稿为准。**
>
> 两处与旧纪律不一致，记下以免以后打架：
> 1. **自造名字从一个变成三个**（ChoraGen + SGT + JSF + HDS）。F.6 第六条原写"方法侧只造 HDS 一个词"，
>    已被这次定稿覆盖；要回退必须整体回退，不能只改其中一个。
> 2. **¶4 不再是"从 tokenizer 起"的方法后半段**，而是能力段（见 F.4.5）；F.5 相应为 ¶5。

**后半（方法介绍）：草稿，2026-08-04，已作废，仅作修订记录。**

| # | 中文 | English | 词数 |
|---|---|---|---|
| **4（命名 + 核心一招）** | 为此我们提出 ChoraGen，它把这一段的米制尺度与视场角作为三个数，与场景一同生成。 | To address this, we introduce ChoraGen, which generates the metric scale and field of view of the clip, three numbers, together with the scene. | 24 |
| **5（模块一）** | 一个场景 tokenizer 把这个状态压缩成我们真正生成的 latent，并约束它的编码解码往返是一个相似变换：压缩可以改变解码几何的尺度，但不能使它畸变。 | A scene tokenizer compresses this state into the latent we generate in, and constrains its round trip to be a similarity: compression may rescale the decoded geometry, but not distort it. | 31 |
| **6（模块二之一）** | 场景、相机、天空与这三个数在同一次前向中一起采样。 | The scene, the camera, the sky and these three numbers are sampled in a single pass. | 16 |
| **7（模块二之二）** | 条件被重参数化，未知的尺度因此只表现为一个加性常数，而这个常数正是同一次前向生成出来的。 | The conditions are reparameterized so that the unknown scale enters as a single additive constant, which that same pass produces. | 20 |
| **8（模块三）** | 由于 tokenizer 的 decoder 与重建头都是预训练且可微的，我们把监督加在解码之后的三层：特征、Gaussian 与渲染图像。 | Because the tokenizer decoder and the reconstruction heads are both pretrained and differentiable, we supervise the generator after decoding, at three levels: features, Gaussians, and rendered images. | 28 |
| **9（收尾，逐字回指第 3 句）** | 轨迹与目标框因此可以施加在这个状态上，生成的结果也可以用米与条件比对。 | A trajectory or a box can therefore be imposed on the state, and the result can be measured against it in metres. | 22 |

**后半的三条纪律**

- **建议在第 4 句之后分段。** 前半 70 词 + 后半 141 词 = 211 词，一段太长。切法：¶3 收在命名句
  （4 句 / 94 词：问题 → 选择 → 代价 → 答案），¶4 从 tokenizer 起（5 句 / 117 词：三个部件 + 收尾），
  两段都与 ¶2 的 150 词同量级。
- **第 5 句是模块一唯一进 Intro 的内容，它同时干两件事**：交代压缩这一步（否则读者会问
  每 patch 12288 维怎么生成），并说明**压缩只能改尺度、不能改形状**——而尺度恰好是第 4 句
  另外生成的那个量。**这是模块一与模块二真正咬合的地方**，也是第 8 句"在解码之后监督"的前提。
  三流独立归一化、层轴 learned query pooling、detail 分支只在 encoder 这些结构设计一律留给方法节。
- **第 9 句是收窄后的 ¶3-d。** 旧版写 *"we measure how well each condition is met ... and propagate
  that error back to the generator"*——**超发**：HDS 测的是对 teacher 解码结果的保真度，
  **条件满足度现在全部是评测指标**（E.4.4）。现在的 `can be measured against it in metres` 是
  **能力陈述 + 我们确实会报的数**（2D IoU、米制 z-depth、identity、cycle EPE），没有声称它进了损失。

> **缝合线（每一处都不能换同义词）**
> - **第 1 句句尾那两个词组必须与 ¶2 逐字一致**：`a prior over 3D` ← ¶2 第 6 句的 `a prior over
>   images rather than 3D scenes`，`no grid fixing the resolution` ← ¶2 第 8 句的 `the grid fixes
>   the scene's resolution`。一字未改，读者不需要推导就知道这句在回答谁；换成 `3D-aware`、
>   `grid-free` 之类的同义词，呼应立刻断掉。
> - **第 9 句必须与第 3 句逐字一致**：`imposed on` ← `cannot be imposed on`，
>   `measured` ← `nor can the result be measured`，`a trajectory or a box` 原样重复。
>   这是后半段唯一的收尾，它的全部作用就是把第 3 句提出的两个抱怨各还一次。
> - **第 2 句是全段的枢轴**：它是被第 1 句逼出来的（先验要在三维上 → 必须用无位姿系统 →
>   尺度不可确定），不是我们自找的麻烦。`cannot be determined` 是数学事实，不是设计选择，
>   这个措辞必须留着。
> - **第 7 句的重参数化是方法侧真正的技术内容**（16 维 placement 里 11 个尺度不变通道 +
>   5 个 log 幅值），此前一直埋在 A.5 的通道表里当实现细节。

**九句的落点依次是：需求 → 代价 → 后果 → 命名 → 模块一 → 模块二 → 模块二 → 模块三 → 收尾。**
三段共用一组词：

```
¶1  judged by its 3D content → cannot be annotated
¶2  outside the generative objective → a prior over images rather than 3D scenes → outside the generative objective
¶3  a prior over 3D and no grid → scale cannot be determined → imposed / measured → we generate it → imposed / measured
```

> `the generative objective` 在 ¶2 出现两次、`prior over images / over 3D` 在 ¶2 与 ¶3 各一次、
> `imposed / measured` 在 ¶3 内部一问一答各一次——**这三组词是全篇的缝合线，
> 任何一处换同义词都会让呼应断掉。**

## F.4.5 ¶4 定稿：能力段（2026-08-05）

> **段落职责**：对标 SceneDirector ¶4（PDF 第 2 页 093–102 行）——**能力枚举 → 每一级的接口 →
> 直接进 contributions，没有 payoff 收尾**。本段只回答"要给什么、能做什么"，不解释机制，不给数字。

| # | 中文 | English | 词数 |
|---|---|---|---|
| **1（三级控制）** | ChoraGen 可以只根据文本生成一段动态驾驶场景，并进一步控制相机轨迹，以及其上的目标。 | ChoraGen generates a dynamic driving scene from text, and enables further control over the camera trajectory and then the actors. | 20 |
| **2（相机这一级）** | 轨迹以逐帧的米制相机位姿给出，场景就按从这些位姿看出去的样子生成。 | A trajectory is given as one camera pose per frame in metres, and the scene is generated as seen from those poses. | 22 |
| **3（目标这一级）** | 目标随后由一张参考图给定外观，以及一条写在第一帧相机系里的米制三维框轨迹。 | An actor is then specified by one reference image for its appearance and a 3D box track in metres in the first camera frame. | 24 |

**全段 3 句 / 66 词**（¶2 为 8 句 150 词）。

**层次怎么搭起来的（三条，都不靠连接词堆）**

1. `then` 直接取自第 1 句的 `and then the actors`，两处呼应，读者看到的是一级一级加上去。
2. 第 2、3 句同一句式（**给什么 → 定什么**），并列结构本身就是台阶。
3. 递进落在内容上：**轨迹定下视点，目标是在这些视点里放进去的实例**——这正是代码里的嵌套依据
   （`train_scene_flow_pretrain.py:1539-1629`，A.5:335-339）：`target_bbox_patch`、`in_frustum`、
   canonical-UV mRoPE 都要先由相机投影才有定义，所以 **actor-only 在结构上不存在**，不是我们让出的一格。

**接口描述的代码依据（写之前逐条核过）**

| 写进句子的内容 | 依据 |
|---|---|
| 三条通路 text / text+camera / text+camera+actor | `inference_scene_flow_pretrain.py:350-357`（`none`/`cam`/`asset_cam`）；训练分布 `:1539-1629` |
| 轨迹 = 逐帧米制相机位姿，首帧为锚 | 9 维米制 state（translation 米 + rot6D），`datasets/dataset.py:1631`、`dggt/utils/camera_generation.py` |
| 目标 = 一张参考图 + **逐帧** 3D 框轨迹，写在首帧相机系 | ≤32 canonical appearance tokens（窗外帧 RGBA）；16 维 placement 逐对象逐帧，中心/尺寸/速度在 anchor 系，`factorized_asset_condition.py:717-763`、`datasets/dataset.py:2014`；≤5 对象 `:714` |

**为什么第 3 句必须把外观单独点出来（`for its appearance`）**：D.2 的表（本文 1462–1472 行）里，
MagicDrive3D / CVD-STORM / WorldSplat / InfiniCube / ScenDi 的目标外观只到**类别或文本级**（①②），
**实例级图像锚点（③）这条线上没有**。`text + box + camera` 的联合条件是别人也有的，
**"每个实例的外观由一张参考图给定"才是这一级里唯一不共有的东西**，措辞上不能被压缩掉。
> ⚠ 但按 H.1：身份相似度评测（含 shuffled-reference 对照）落地之前，
> 这里**只能写接口**（外观由参考图给定），**不能写"能保住那个实例的身份"**。

### 本轮从 ¶4 删掉的两句，记下理由（避免以后又加回来）

1. **滑动窗那句（原计划第 4 句）**。采样时融合重叠窗是长视频扩散的通行做法，**不是创新**；
   ¶4 是能力段，帧数属于 scope；写在这里等于把"窗口只有 10 帧（1 秒）"这个最容易被挑的点
   放到全段最显眼的位置，却换不到任何主张。**唯一属于本方法的只有半句**：尺度是 scene-global 的，
   跨窗按 cosine 覆盖融合后只做一次全局 Euler（`train_scene_flow_pretrain.py:4394-4620`，
   `tests/test_sliding_window_v2.py`），因此整条序列共用一套单位，不会每窗换一个米/单位——
   它是"尺度是一个 scene-global token"的必然结果，不是另一个模块。**已移到 F.5 的 scope。**
2. **米制导出收尾句（原计划第 5 句）**。理由是 D.1.5 那条：**米制是这条路线的入场费，不是卖点。**
   STORM / DrivingForward / DrivingRecon 一路的米制来自输入侧标定、ScenDi 的体素本来就有物理尺寸，
   把"导出是米制的、可以和条件比对"放在收尾位置，会被读成在主张一个**大家都有的属性**；
   而且 ¶3 已经把尺度定性成代价，¶4 再把它当成果展示，两段语气打架。
   **事实移到 F.5 的 scope，论点交给实验节的米制指标表。**

> **留在 ¶5 / 方法节、不进 ¶4 的**：≤5 actors、patch grid、层级 CFG（相机增量与目标增量可分别调节，
> `combine_pretrain_cfg_prediction`）、相机内参只生成不可条件、外部 manifest 未端到端接通。

## F.5 ¶5（scope + contributions）

> **旧的"¶4（方法）四句骨架"已作废（2026-08-04）。** 它的内容现在由 F.4 后半的第 4–9 句
> 在句子级落实，两处并存只会打架。骨架里**没有被后半段吸收的只有两项**，它们都不是核心模块，
> 按既定判断（能力 ≠ 方法创新，见 F.6 第三条）归到本节的 contribution bullet：
> **条件的投影层级**（文本/相机/目标的嵌套依赖）与**实例外观锚点**。
>
> **【2026-08-05 编号已定】** ¶3 是方法段（定稿）、**¶4 是能力段**（F.4.5 定稿三句），
> 因此本节确定为 **¶5**。上面那两项里，**条件的投影层级与实例外观锚点已经在 ¶4 以接口出现**
> （三级控制、参考图给定外观）；本节的 contribution bullet 写的是**机制与训练分布**
> （嵌套任务分布与层级 CFG、per-instance canonical token 的防泄漏构造），两处不重复。

**先 scope 后 bullets：**

- scope：前视单相机、10 帧窗口、≤5 actors、patch grid 25×37、逐帧 dense Gaussian、
  无持久 canonical 4D field。

  **【2026-08-05 更新】scope 里要写成正文的两句**（都是从 ¶4 移下来的，写成事实，不写成主张）：

  | 中文 | English | 词数 |
  |---|---|---|
  | 十帧窗口；更长的序列由重叠窗口覆盖，并共用同一个尺度。 | Windows of ten frames; longer sequences are covered by overlapping windows that share one scale. | 15 |
  | 输出为逐帧稠密高斯，在生成相机的坐标系里按米缩放。 | Outputs are dense per-frame Gaussians, scaled to metres in the generated camera frame. | 13 |

  - 第一句同时答掉"只有 1 秒吗"，又不把重叠窗抬成创新；`share one scale` 是唯一属于本方法的部分
    （scene-global gauge，跨窗一次全局 Euler）。**但 `longer sequences` 必须有一条真跑出来的
    ≥29 帧序列才能写**，否则删掉后半句只留 `Windows of ten frames`。
  - 第二句的 `scaled to metres in the generated camera frame` 是 H.5 允许的原话；
    **不能**写成 Waymo-canonical metric world。`dense per-frame Gaussians` 顺带把 H.2 要求主动交代的
    view-indexed / 无持久 4D field 说清楚。
  - **米制在 Intro 只以 scope 事实出现，不作为卖点**（D.1.5）：对手的米制来自输入侧标定，按构造是对的。
- bullets（每条回指 ¶2 的一个缺点或 ¶3 的一步）：
  1. **在一个前馈重建模型的表示中生成驾驶场景**——先验落在三维上，且不引入格点；
     并给出让它在米制规约下可用的重参数化：**未知尺度只表现为一个加性常数，而这个常数被生成出来**；
  2. **条件的投影层级**：按驾驶事件里条件之间的物理依赖组织训练任务与 CFG，
     实现目标外观 / 位置与相机轨迹的联合控制；
  3. **HDS**：通过冻结的多层 decoder、Gaussian heads 与 renderer，把解码之后的误差回传给生成器；
     并把条件满足度作为米制评测指标报告（目标 2D IoU + z-depth、身份相似度、重投影 cycle）；
  4. 一次**冻结重建先验的尺度标定研究**（Branch A / LiDAR 主尺 / 冻结解码器往返审计），
     其结论对任何在重建基础模型特征空间上做生成的工作都可复用。

> **第 1 条 bullet 的措辞在 2026-08-04 改过。** 旧版写成"把米制尺度做成生成变量而不是表示的
> 固定属性"——那个对比预设了"别人的固定属性是缺点"，而 ¶2 已经不再这样主张了。
> 新版把重点放在**表示的选择**（先验在三维上、无格点）与**重参数化**上，米制作为让这条路线
> 可用的必要一步出现，不再是与他人的对立面。

> 第 4 条 bullet 最容易被低估。**在方法论文里放一个可复用的测量结果，是抬高接收概率的低成本
> 手段**——它把"我们调了个参"变成"我们弄清了一件事"。
>
> ⚠ **模块一的实验数字全部等 tokenizer v2 训完。** 在那之前，摘要与贡献列表里
> **不能出现任何往返保真或几何不变性的数值**，只能写目标函数包含该约束。见 H.5 的证据分级表。

## F.6 六条写作纪律（第六条本轮新增）

**（一）三类断言必须各归其位。**

| 断言 | 证据来源 | 允许出现的位置 |
|---|---|---|
| 重建是独立阶段；ScenDi 的体素分辨率与体积；VQ 与 occupancy 判定；三模块分训 | **对方论文可查证** | ¶2 ✓ |
| 视频模型被优化的是画面而不是之后恢复的几何；渲染误差进不了生成目标 | **纯逻辑，不需数据** | ¶2 ✓ |
| 外观细节受三维表示分辨率限制 | **对方论文自己陈述**（ScenDi 的动机段） | ¶2 ✓，但必须能引到原句 |
| 视图不一致**会**变成几何误差 | **未测过** | ¶2 只能写 `might`，不能写 `does` |
| 这个落差在实测中确实很松（latent error 与 world error 相关性差）；尺度自由度是 25–64 m | **只在本文自己的模型 / teacher 上成立** | ¶4 / 实验，且必须写明 `in our setting` |
| **现有方法的米制有问题 / 不能自适应 / 是固定属性** | **不成立**——他们的米制来自输入侧且按构造正确 | **任何位置都不能写**（2026-08-04 删除） |

> 散点图与尺度量程图都是**本文自己模型上的测量**，不能用来指控别人的 decoder。
> ¶2 不引任何本文实验。**这条在 v6 已经修过一次，不要再犯。**

**（二）缺口要有机制，不能只有现象或对手性能断言。**
声称"无上界 / 无保证 / 无法被检验"，不要声称"误差很大"。前者从机制推出，后者永远可以被
一张更好的表反驳。

**（三）一个必须避免的自摆乌龙（v8 已删除，本轮继续禁止）。**
曾经写过 *"the ego-trajectory is given rather than generated, and an actor enters through its
placement alone."* 两处都错在论证方向：

1. **"轨迹被给定而非被生成"不是缺陷**——读者的反应是"轨迹当然该给定"。更危险的是，
   **"相机该不该被生成"恰好是审稿人要问本方法的问题**。把它当作别人的缺点写进 ¶2，
   等于把这个质疑主动递到自己面前。
2. **"actor 只以位置进入"讲的是控制能力（缺实例外观锚点），不是联合建模。**
   混进 ¶2 的一致性论证会让两条线互相稀释。

**正确归属：实例外观锚点是本文的贡献，不是别人的缺口。** 它出现在 ¶4 或 contribution 列表。

**（四）修订链条（八版，记下避免再犯）。**

1. **v1：** "存在一个没人用过的表示" → "X hasn't been tried" 型 gap，换个人做一遍就消失。
2. **v2：** `camera, scene geometry and Gaussians` 三者不并列；且"条件由不同阶段分别确立"对
   WorldSplat、CVD-STORM 不成立。
3. **v3：** "never enforced during generation" 是硬错误——别人当然在生成过程中约束。
4. **v4：** "the actor still lands metres away" 是**性能断言**，审稿人可用自己的 ATE 反驳。
5. **v5：** "不在训练目标之内" 只是**描述**——不在目标里不代表会出问题。
6. **v8→v9：** 只写第一层缺口（共同解释未被建模）时，"你自己选了个没单位的空间"
   这个反问**无处安放**。新增第 5/6 句把它前置成对方的结构性属性，并在 ¶3 第二句给出对偶答案。
   **缺口从一层变两层，方法从"换空间"变成"表示不带尺度、尺度另外生成"。**
7. **v9→v9.1：** 第 6 句原来写的是"米制无法被检验、也无法自适应"。
   **"无法被检验"这半句删掉**——审稿人不会认为"别人的数不能被核对"是一个缺点，
   也很难理解为什么需要核对。
8. **v9.1→v9.2（2026-08-04，本轮，改动最大）：** 把整条米制批评从 ¶2 删除。
   起因是逐篇核实了 11 个驾驶前馈重建方法（见 D.1.5）：**它们的米制全部来自输入侧**——
   STORM 的输入外参、DrivingForward 的 rig 外参、DrivingRecon 的 LiDAR 深度监督——
   **按构造是对的**。唯一未标定的 FRUC 也确实不米制，反向印证了这条规律。
   把一个按构造正确的东西说成缺点，是站不住的。
   > **教训（第三次犯同类错误了）**：v4 的 "the actor still lands metres away"、
   > v5 的 "不在训练目标之内"、这次的 "尺度是固定属性"——**每次都是先想好要论证什么，
   > 再去找理由**。正确顺序是先核实对方到底怎么做的，再决定能主张什么。
   >
   > 替换后的 ¶2 是三个各有机制的缺点（阶段无反馈 / 先验在图像上 / 格点粗且断链），
   > 米制移到 ¶3 作为我们这条路线的代价。

**（五）其它格式纪律。** ICLR 正文避免破折号，¶2 与 ¶3 定稿中零破折号；
避免 `these elements` 类前指代词，直接点名三要素；
`by a separate stage` 而非 `separately trained decoder`（LSD-3D 是逐场景优化而非 decoder）。

**（六）不造抽象词，句子要能一遍读懂。**
这是本轮加的一条，因为 v9 违反得最厉害。判据很简单：**任何一个词，如果读者需要回到定义处才能
读懂这句话，就换掉它。** 具体禁止的写法：

| 不要写 | 改成 |
|---|---|
| metric commitment / 米制承诺 | the metric scale of the scene / 场景的米制尺度 |
| the representation stays uncommitted | the representation has no built-in scale |
| auditable / accountable / 可被核对 | 直接说报告什么误差：*we report metric depth error against LiDAR* |
| gauge / 规范量（正文） | scale and field of view / 尺度与视场角 |
| readout | decoder / heads |
| fidelity term vs coherence term | losses against a target vs losses between generated quantities |
| world-level feedback / 世界级监督 | supervision after decoding |

**一个判断标准**：SceneDirector 全文只造了 MGRA 一个模块名，其余全是普通词组，
所以它读起来像论文而不像技术文档。本文对标这个密度：方法侧只造了 HDS 一个词，另外两个模块名都是描述性的。

## F.7 可选加强

- **CVD-STORM 那半句仍然有用**（"trained jointly with the VAE and then frozen"），它用对手自己的
  措辞证明了阶段边界的存在。但注意：**它只能支撑"分阶段"，不能再用来支撑任何米制主张**。
- **ScenDi 的自陈动机是第 8 句（体素缺点）最好的注脚**（"relying solely on 3D diffusion →
  degradation in appearance details"）：它自己说明了格点分辨率的代价。**这句话说的是分辨率，
  不是尺度**，引用时不要偷换。
- WorldSplat 也用增强 video diffusion 精修从 Gaussian 渲染出的 novel view，为"外观在视图空间被
  决定"提供第一类最强成员的直接自证。
- 若空间允许，可加"直接进入三维状态这一支目前均为静态或需事后合成动态"：LSD-3D 事后插入资产、
  ScenDi 剔除动态。**但 WorldSplat 与 CVD-STORM 有动态，所以这句只能限定到 LSD-3D + ScenDi。**

---

# Part G. 实验闭环（2026-08-02 更新）

## G.0 两张 motivation 图（都必须标 `in our setting`）

**Fig. 1 — 尺度自由度的量程（本轮最便宜也最有说服力的一张）。**
横轴：90 个 29 帧片段；纵轴：每个 DGGT 单位对应多少米。散点 25–64 m，相邻片段用线连起来显示
8–31% 的漂移，并用第二个 y 轴叠加视场角的片段内 std（0.26°）与片段间 std（9.4°）。
**这张图一秒钟说清"尺度自由度真实存在且很大"**，且它只是对 teacher 的测量，不涉及任何训练。

**Fig. 2 — latent error 与解码后误差的失配。**
在只有 latent MSE 的 ablation 上画散点：$x$ = $\hat z_0$ 的 feature MSE，$y$ = depth / Gaussian /
render error。附全集 Spearman + 按 $\sigma$ 分层 + 局部近邻内的方差 + 加 HDS 前后的 Pareto front
+ 一条"若解码保距应长什么样"的参考线（详见 D.4 末尾的清单）。

> **作用域纪律（v6 修正，仍然有效）：这两张图只在本文自己的模型 / teacher 上成立，
> 不能作为对其他方法的指控。**
> - **不要**把它们作为 ¶2 缺口的证据引用；¶2 只讲可查证的结构事实与逻辑上的"无保证"。
> - **应当**在 ¶4（方法）或实验中引用，并写明 `in our setting`。
> - 对手的经验性指控需要其权重，而 ScenDi 为 "Code coming soon"、LSD-3D 为 "Code (tba)"，
>   不具备条件；**不要尝试。**

## G.1 主表：五类指标分开报

| 组 | 指标 |
|---|---|
| 外观/真实性 | FID、FVD |
| 几何 | 对 pseudo-GT 的 metric depth error；±1–2 m novel-view PSNR |
| **控制满足度** | 目标 identity similarity（含 shuffled-reference 对照）；目标 **2D IoU + 米制 z-depth error** + coverage/leakage；**相机位姿误差（米）**；text alignment |
| **尺度标定**（已落地） | **模型自己的尺度误差与常数基线误差两个数分开报**（`gauge_log_scale_error` 与基线，不要只报差值）；视场角误差（度）；米制换算后 depth error vs LiDAR（`metric_depth_rel_err`）；相机尺/LiDAR 尺一致性；checkpoint-bound metric profile 的 LiDAR AbsRel gate；配对 `GS/depth` 审计 |
| **生成量之间的一致性**（评测指标，非损失） | 目标–box 的米制残差与 coverage/leakage；生成静态几何的重投影 cycle EPE 与 z-depth log 残差；目标 identity similarity；退化 support 比例 |

> **尺度那一行必须报两个数，不要只报领先量。** 只写 `gauge_vs_prior_gain = 0.2` 读者算不出量级；
> 写成"常数基线 0.23（≈23% 相对误差）→ 本方法 0.03（≈3%）"就一眼可读。
> 而且**验证集上要报走完整采样的那一版**（`sample_gauge_*`），不是训练时随机噪声水平下的
> x-prediction——后者是乐观估计。

> **禁止再用 render PSNR 选择尺寸常数。** D4 已证实它对 `c_gs` 单调奖励覆盖/模糊，且对
> `c_depth` 无法回答米制正确性；render 固定 identity，metric `c_depth` 由 LiDAR gate 决定。

> ⚠ **两条口径更正（仍然有效）：**
> 1. **不要报裸 3D IoU / 3D center error。** 米制→DGGT 是**各向异性**映射
>    （$\mathrm{diag}(k_x,k_y,1)/s$，$k_x=0.748$），DGGT 空间里的车横向天生窄 25%。
>    报 **2D IoU（像素，两套约定共用）+ 米制 z-depth 误差**——这一对本来就完备地确定了位置。
>    若必须报 3D IoU，先套 `metric_box_to_dggt`。
> 2. **"三路尺子互相一致性"不是主指标。** actor 尺仅 29/90 可用且有灾难性离群，已降为诊断；
>    主尺是 29 帧 LiDAR。相机尺仍作为移动样本上的高精度交叉验证
>    （29 帧 `0.99995 ± 0.02639`，corr 0.99296）。

> **camera pose error 现在才真正可比**：改造前目标混着一个 CV 23.5% 的不可观测标量，
> 这个数在不同 clip 之间没有共同单位；改成米制之后它就是米。

| 下游效用 | 在生成数据上训练/评测检测器 |

## G.2 逐环 ablation（一环一实验，与 E.2 的四层一一对应）

| 环 | 实验 | 状态 |
|---|---|---|
| 在哪个空间生成（D.1） | DGGT 特征 lattice **vs ScenDi 式固定 voxel latent，等算力/等数据**。唯一能把"这个空间更好"从假设变成结论的实验 | 未做，最贵 |
| **用什么单位（D.1.5）** | **三臂**：① 固定成训练集均值尺度 ② 只预测尺度但**不注入 scene decoder** ③ 完整。相机平移误差分解为"尺度误差"与"轨迹误差"；米制换算精度；尺度误差 vs 基线 | **代码已就绪，可跑** |
| **条件怎么进来（D.2）** | ① 合法层级 vs 独立随机 dropout（**本轮新增，验证层级本身**）；② 单条件 vs 四条件联合下各控制指标退化幅度（见 G.4）；③ placement v3 vs v1 米制原值 | 可跑 |
| 误差在哪里测（D.3/D.4） | 无 HDS / 只特征层 / 特征+高斯 / 三层全开；并报告 latent 与解码后误差的相关性变化 | 可跑 |
| HDS 的置信度加权 | 不加权 vs 加权 + 权重图可视化 | **已实现，未训练/未消融** |
| **模块一各设计决策** | 三流独立投影 vs 拍平；层注意力+learned query pooling vs 平均/拼接；有无 detail 分支；每层独立 head vs 共享；有无相似性约束 | **等 v2 训完** |
| **尺度对齐的条件读取** | 减 / 不减 $\hat g$ → 目标 **2D IoU + 米制 z-depth 误差**、**跨实例泄漏**、**尺度误差的变化** | 待实现 |
| **满足度评测**（非消融） | identity similarity **必须配 shuffled-reference 对照**；cycle EPE 报退化监控（静止片段占比、support 比例） | 待接脚本 |
| 相机诊断 | 生成相机 vs 输入相机渲染；相机等变扫描（固定 text/asset seed 扫轨迹，测世界内容是否保持） | 可跑 |

> **第二行的中间那一臂（只预测不注入）是本轮新增，别漏掉。** 它是验证
> `scene_flow.py:3465-3472` 那个 `gauge_context` 耦合的**唯一**实验——没有它，
> "尺度参与了场景生成"这句话只是一个实现细节，不是一个被验证过的设计。

## G.3 证明"不需要第二个生成器"（最锋利的对照实验）

把**同一个**生成 Gaussian 场景从 ≥3 条未参与生成的相机轨迹**直接渲染**，不接任何 2D refiner，
报告跨轨迹一致性。加强版：接一个 2D refiner，展示 FID 改善但**跨轨迹一致性变差**——
直接把与 ScenDi / WorldSplat / CVD-STORM 的对照变现。

## G.4 "统一"用干扰实验证明

照抄 SceneDirector 的 Obj.Only vs Obj.+Traj. 做法：单条件 vs 四条件联合下各控制指标的退化幅度。
退化可忽略才有权说条件之间不互相干扰。

## G.5 生成的尺度到底对不对：三个低成本实验

1. **导出后量一条已知长度。** `--export_units metric` 导出 PLY，量车道宽度或车长，对照 Waymo 真值。
   **这是"我们生成了米制尺度"最直白的证据**，一张图就够。
2. **尺度扰动实验。** 人为把生成的 `log s` 加 $\delta$，展示米制导出等比变化而像素渲染不变——
   证明尺度确实是一个独立的量，没有被吸收进别处。
   （`tests/test_gauge_similarity_invariance.py` 已冻结这个语义。）
3. **跨片段一致性 limitation。** 主动报告 teacher 尺度跨片段漂移 8–31%，说明 ≤29 帧生成干净，
   长序列会得到一个比 teacher **更**自洽但**无 GT 可比**的全局尺度。

## G.6 反事实实验组（本轮新增，全部很便宜，但作用最大）

前面的消融证明"去掉某项会变差"，**反事实实验证明"某项确实在起它声称的作用"**。
这一组是把"功能组合"升级成"因果职责"的关键，SceneDirector 也是靠同类实验站住的。

| 实验 | 做法 | 预期 | 它排除了什么 |
|---|---|---|---|
| **相机 shuffle** | batch 内交换相机条件，其余不变 | 目标投影与相机–几何一致性显著恶化 | 排除"模型其实忽略了相机条件" |
| **尺度 shuffle** | 交换生成的尺度 | 米制深度与重投影恶化，但语义/外观可能仍合理 | 证明尺度确实只承担尺度，没有偷偷编码内容 |
| **reference shuffle** | 交换目标的参考图，几何条件不变 | identity similarity 显著下降，位置不变 | 排除"encoder 只是在认车这个类别" |
| **placement 单调扰动** | 对米制 box 施加已知平移/旋转 | 输出目标位置**单调**响应 | 排除"条件只是让输出变模糊" |
| **teacher-pose 渲染 vs 相机–几何一致性** | 前者去掉，后者作为指标观察 | 两者反映的问题不同 | 说明 RGB 重建与相机一致性是两件事，不是重复 |

> 最后一行专门用来挡住"你为什么不直接把 `camera_grad_scale` 打开"这个追问——
> 如果两者影响的指标集重叠，那这个追问就是对的；如果不重叠，分开的理由就有了实证支持。

## G.7 基线选择

优先 **WorldSplat、CVD-STORM、ScenDi**（同为摊销式 latent-to-Gaussian，是最直接对手），
其次 **LSD-3D**（endpoint-aware 但逐场景优化，构成矛盾的另一端），
再次 **InfiniCube / DriveGen3D**（第一类代表）。
CVD-STORM 比较时须注明其主结果用 3 帧 reference，若要公平比 source-free 设定应用其 0-reference 配置。

---

# Part H. 表述边界与风险（2026-08-02 更新）

## H.1 一定不能写

- 首个 text-to-3DGS / text-to-driving-GS / camera-controllable Gaussian scene / GFM feature diffusion
- 首个在生成 Gaussian 上用 render loss
- 首个在驾驶场景直接生成 3D Gaussian（ScenDi 已从 3D latent 采样后解码 3DGS）
- 无限定的 "complete gradient chain"（Envision4D 有完整 render gradient）
- "现有方法缺少 Gaussian/RGB/render loss"
- "直接用输入相机渲染"
- **"现有方法的米制是错的 / 误差多少"**。可写的只有结构性陈述：**尺度固定在生成器不训练的部件里，
  因此不能随所生成的片段调整**，以及**这个固定有表示上的代价（对方自己陈述的）**。
  我们没有任何测量他人米制精度的证据
- **"他们的米制无法被检验"**——这句话本轮已删除。它需要读者先接受"米制需要被检验"这个前提，
  而这个前提在论文里没有位置，读起来像在给自己的不确定性找理由
- **不带尺度/往返标定 provenance 的任何米制几何数字。** 代码能力 ≠ 模型效果；
  必须等新 checkpoint 的验收
- **"相机内参可由外部指定 / 可控焦距"**。当前视场角是那 3 个生成数里的两个，
  不存在外部条件路径
- "生成的相机已经精确复现请求相机"——9 维米制目标已消除未知尺度，但仍须由新 checkpoint 的
  camera-guidance 实验验收；D3 的 ~0.2°/~1.3% 是 teacher/Waymo **控制保真度上界诊断**，
  **不是推理画质损失**
- **任何关于 tokenizer 往返保真或几何不变性的数值**——当前实现是 v2 且**尚未训练完成**，
  只能写目标函数包含相似性约束；v1 上的历史测量描述的是已被替换的模型，不能引用
- **"生成的相机与生成的几何是一致的"**——目前**没有任何损失**约束这一点
  （`camera_grad_scale` 硬断言为 0），只有一条 no-grad 诊断。因此这一条永远
  这句话只能写成 evaluation 结果，不能写成方法性质
- **"我们能锚定实例外观"**——接口上成立，但**训练和评测里目前没有任何东西验证过身份被保住**。
  在身份相似度评测（含 shuffled 对照）跑出来之前，只能写接口描述
- "推理时无需第二个二维生成阶段"**不加限定词**（WorldFlow3D 是反例）。准确表述：
  *首个在带动态归属的时序驾驶 Gaussian 状态上、以解码之后的监督训练生成分布、且推理时不依赖
  第二个二维生成器的方法*——每个限定词都必须留着
- X-Scene "确定没有"世界梯度（只能写"论文未报告"）
- "直接进入三维状态这一支均为静态"（WorldSplat、CVD-STORM 有动态；只能限定到 LSD-3D + ScenDi）

## H.2 必须作为 scope 主动写出

- latent 仍是 **view-indexed**（frame/patch aligned），不是 canonical 3D grid
- **没有**持久 canonical 4D Gaussian field、**没有**显式 deformation field、**没有**输出级 instance
  identity；导出是逐帧 PLY → 不能讲 persistent world 类叙事
- 前视单相机、10 帧窗口、≤5 actors、patch grid 25×37
- **与 Waymo 物理世界不同尺度，已标定（A.8）**：1 DGGT 单位约 25–64 米，逐 29 帧片段变化。
  camera center 与 direct z-depth 在约 2.5% 离散度下近似共尺，但 FOV/横向各向异性
  （$k_x=0.748$、$k_y=0.772$）。**tokenizer 往返的自洽性待 v2 训练完成后重测**
- **相机内参只生成、不条件**：旧冻结 CameraHead 的 FOV 与 Waymo 标定差 −11.7°±9.7°，
  因此当前把 teacher FOV 放进那 3 个生成数里的两个通道；9 维 camera state 不再含 FOV 段
- **训练 render 使用 teacher pose，`camera_grad_scale` 硬断言为 0**（D3）。
  这是刻意选择，代价是生成相机与生成几何无光度耦合；能报告的只有
  `generated_static_geometry_reprojection_cycle_v1` 这个**诊断**顶着
- **长序列没有一致的 GT 尺度**：teacher 尺度跨片段漂移 8–31%，超过一个 trunk 的滑动窗生成，
  其全局尺度无 GT 可比。≤29 帧不受此限
- 外部 manifest 分支未端到端接通 render/export
- raw-validation 的 asset / track / camera 来自同一 clip → 跨场景外观重组、反事实位置、
  显著 OOD 轨迹、prompt 组合泛化**均未证明**
- tokenizer 是本仓库对 DGGT stack 的扩展，不是公开 DGGT 自带
- 只 adapt 了 Cosmos 3 的 token 组织与 mRoPE 约定，**没有**加载 Cosmos 3 权重或 video VAE；
  同理未加载 RAEv2 生成权重

## H.3 八个最大风险

1. **解码之后的监督可能不显著。** 它是 every-2 步、$(1-\sigma)^2$ 权重、`max_samples=1` 的低占空比
   信号。若消融差异在噪声内，第二条贡献就空了。**前置验证：先只跑 HDS on/off 消融到能看出差异的
   步数，再决定是否写成主要贡献。**
2. **"这个空间更好"缺等算力 voxel-latent 对照就只是假设。** G.2 第一行最贵但最关键。
   跑不动就把 D.1 降级为"一个**足以**支撑解码后监督的选择"，而不是"最优空间"。
3. **软绑定。** 没有尺度对齐的条件读取或强隔离实验，D.2 无法防守；没有身份相似度评测，D.2 的第 1 条差异
   （实例外观锚点）只是接口描述。
4. **尺度可能学不好——但这条风险本轮已经下调。**
   前置验证**已完成**：D2 确认 target 本身自洽（LiDAR 尺 90/90 有效，逐帧 robust CV mean 0.688%）。
   模型端的观测也已出现：**训练集上模型比常数基线领先约 0.2**（log 单位），
   换算成相对尺度误差约为"基线 23% → 本方法 3%"。
   > **但这不是能进论文的那个数。** 训练时算的是**随机噪声水平下的 x-prediction**
   > （`train_scene_flow_pretrain.py:5676`），sigma 小的时候答案几乎是白给的，是乐观估计。
   > 论文要报的是**验证集上走完整 ODE 采样**的那一版（`:7358` 的 `sample_gauge_*`）。
   >
   > **判据不变**：若验证集上模型不显著优于"输出训练集均值"，说明它只是记住了一个常数，
   > D.1.5 就要退回"我们预测一个先验并报告误差区间"。**仍然建议把它列为最先跑的实验**——
   > 不是因为最危险，而是因为它最便宜、且整条故事的第一句都建立在它上面。
5. **解码之后的空间是否等比，目前是未知数。** HDS 的说服力建立在"解码之后的量是物理量"上；
   v2 的目标函数已包含逐同像素的相似性约束（`gs_scale_sim_ratio` 须收敛到 1.0），
   但**训练尚未完成，没有任何可引用的数字**。
   若 v2 收敛失败，fallback 只能用会惩罚模糊的 LPIPS/SSIM，**不能再用 PSNR 选择尺寸常数**
   （D4 已证实 PSNR 对尺寸类常数单调）。
6. **cycle EPE 作为指标会被退化情形污染。** 零相机运动 + 平坦 depth 的 cycle error 也是 0。
   报告时必须排除静止/纯旋转片段，并同时给出退化 support 的比例；
   **只报一个漂亮的平均值等于没报。**
7. **尺度对齐的条件读取可能反过来污染尺度。** 模型可以通过调整生成的尺度去迎合条件，
   而不是把几何做对。**缓解办法是始终把尺度对 LiDAR 的误差单独报告**
   （`metric_depth_rel_err`）——它和目标绑定误差是两个独立判据，不能互相顶替。
8. **身份相似度可能什么都没证明。** 冻结 encoder 对同类车辆的相似度本来就高，
   一个"总是渲染一辆普通轿车"的模型也能拿到不错的分数。
   **shuffled-reference 对照是这一项的存在前提**，没有它就不要写进论文。

## H.4 与最近邻方法的边界（必须正面写在 Related Work）

- **CVD-STORM**：最好的缺口证据，但也是最需要小心的近邻——同样是驾驶、同样可 source-free、
  同样输出可自由渲染的动态 Gaussian、同样用 rectified flow。
  **两条决定性差别**：(i) 世界监督是否跨越 stage 边界；(ii) 米制来自 stage-1 冻结的 STORM-VAE
  还是被生成。必须写清楚，不要靠模糊表述。
- **WorldSplat**：最强 latent-to-GS 直接对手；差别在（i）中间 latent 仍是相机/像素对齐的
  RGB/depth/seg 多模态 latent，(ii) 三模块独立训练，(iii) 它也用 video diffusion 精修渲染结果，
  (iv) metric depth 由独立训练的 GS decoder 承担。
- **ScenDi**：最强 3D-first 对手；差别在（i）latent 是单目深度融合的规则彩色体素的量化压缩，
  **受固定体积与 0.4 m voxel 限制**，(ii) render loss 只训练 VQ-VAE，(iii) 高频与远景在第二阶段
  2D 中产生且不写回 Gaussian，(iv) 数据剔除显著动态。
  **它是 D.1.5 里"把米制焊进表示"的教科书成员，且代价由它自己陈述。**
- **LSD-3D**：world feedback 确实约束 Gaussian，但逐场景 6000 步优化——
  **这是最清晰的一条区分线，应主动承认它有 world feedback，再指出代价。**
- **Envision4D**：有完整 render gradient 的前馈反例；贡献必须限定为"在 **controllable generative**
  world model 中的 world-aligned training"，绝不能泛称"首个完整梯度链"。
  **它也是唯一一个同样建立在重建先验（VGGT 伪标签）上的近邻**，因此
  **D.1.5 关于尺度的论证对它同样适用**——但它是 observation-conditioned 外推，
  尺度由输入观测锚定，所以它不需要生成尺度。这条区别要主动写。
- **通用域 GFM 生成（GLD / OneWorld / Gen3R / PixWorld / Geometry Forcing）**：
  必须承认"在几何基础模型特征空间做生成"这一架构在通用域已有先例。
  **本文的领域缺口正落在它们不必解决的那一环上**：通用域 3D 生成没有米制规约，
  而驾驶场景的条件（米制 box、米制 ego 轨迹、米制尺寸）与下游消费者（检测器、规划器）都是米制的。
  **把这条写清楚，"通用域已有先例"就从威胁变成本文领域贡献的定义。**

## H.5 【新增】证据分级表：每条主张现在允许怎么写

**写摘要之前必须再核对一遍这张表**，它唯一的作用是防止把计划写成结果。

| 内容 | 当前状态 | 论文里允许的措辞 |
|---|---|---|
| 场景尺度 + 视场角作为 3 个数被生成 | 已实现，有 production artifact | *we explicitly generate* |
| 9D 米制相机生成 | 已实现 | *we jointly generate metric camera motion* |
| 真实 Waymo K / 生成视场角 两条投影链 | 已实现 | *we maintain separate metric-control and decoding projection chains* |
| factorized actor v3 + 窗外 reference | 已实现 | *we condition on appearance and metric placement without target-window visual leakage* |
| 合法任务层级 `text ⊂ text+cam ⊂ text+cam+actor` | **工作树已实现，尚未进入所引 commit** | 固定代码快照 / 重训之后才可写 *we formulate* |
| HDS 特征 / 高斯 / 图像三层 | 已实现 | *we backpropagate through frozen decoders* |
| HDS 的置信度加权 | **已实现，未训练/未消融** | 可写实现事实：*we weight the Gaussian- and image-level losses by the frozen decoder's own depth confidence*；不能写任何数值或效果结论 |
| 冻结解码器往返标定（v1，按作用域） | 已冻结 | *we calibrate and report* |
| **RGB loss 更新生成相机** | **未实现，且被硬断言显式禁止** | **不能声称，任何形式都不行** |
| 静态相机–几何 cycle | 已实现，但是 **no-grad 诊断** | 只能写 evaluation / diagnostic |
| tokenizer v2 的几何不变性约束 | **已实现，训练中** | 只能写目标函数包含该约束；**不能写任何数值** |
| 尺度对齐的条件读取（减 $\hat g$） | **待新增** | 完成训练与消融前只能写 proposal |
| 满足度评测（2D IoU / z-depth / 身份 / cycle） | **待接脚本** | 只能写 evaluation，不能写成方法性质 |
| 米制 PLY 导出 | 已实现 | 可写 *metrically scaled in the generated camera frame*；**不能**写 *Waymo-canonical metric Gaussian world* |
| 外部 manifest 任意轨迹渲染 | **未端到端接通** | 不能声称任意用户轨迹已验证 |
| tokenizer v2 训练 | **进行中** | 在完成前不能写任何往返保真 / 几何不变性的**数值** |
