# ICLR 全场景生成：代码梳理、相关工作谱系、SceneDirector 叙事迁移与故事设计

> 版本：2026-08-02（第 8 次修订：完成 tokenizer-v1 Phase 1–8 production artifact、full-pass stats 与 CUDA direct/sliding wiring 验收；完整重训科学 gate 继续单列）
> 实测脚本（`conda activate dggt`）：`lyy_tools/verify_camera_gauge.py`（A.8.1–A.8.6）、`lyy_tools/verify_window_scale.py`（A.8.7）、`lyy_tools/verify_fov_consistency.py` / `tools/retest_scene_flow_gaussian_gauge.py`（A.8.9）、`tools/calibrate_tokenizer_pullback.py`（A.8.10）
> 阅读范围：`train_scene_flow_pretrain.py`、`inference_scene_flow_pretrain.py`、`dggt/models/{scene_flow,joint_scene_tokenizer,canonical_asset_encoder}.py`、`dggt/utils/factorized_asset_condition.py`、`dggt/losses/{flow_losses,reconstruction_feedback_loss,rgb_render_loss}.py`、`pretrain_single_node.sh`；`docs/ICLR_scene_generation_story_codex.md` §1–2（仅代码与来源部分）；`docs/ICML_SceneDirector.pdf` 全文；`paper/read/` 下一手审计。
> 主动回避：既有 story 文档中创新点与故事设计章节。
> 外部核验：LSD-3D、ScenDi、SEM-ROVER、WorldFlow3D、WorldSplat、CVD-STORM、Envision4D、GaussianCity、PrITTI、Urban Architect、InfiniVerse、AnyScene、GaussianDWM 均已用网络检索独立复核。文中标 ⚠ 的是未逐行核实、写作时需再确认的细节。

> **快照边界**：A.1–A.7 的 11D camera / 12D placement 描述是改造前 baseline 的代码快照，用于解释
> A.8 的问题从何而来；当前正式契约是 9D Waymo metric camera、16D placement、3D scene-global gauge，
> 以及 tokenizer-v1 production pullback。实现与验证清单以
> `docs/metric_scale_camera_redesign_plan.md` 为准。

---

## 目录

- [0. 结论先行](#0-结论先行)
- [Part A. Scene Flow Pretrain 代码梳理](#part-a-scene-flow-pretrain-代码梳理)
- [Part B. SceneDirector 的叙事为什么成立](#part-b-scenedirector-的叙事为什么成立)
- [Part C. 相关工作谱系（准入标准与逐篇判定）](#part-c-相关工作谱系准入标准与逐篇判定)
- [Part D. 四个核心问题的回答](#part-d-四个核心问题的回答)
- [Part E. 统一故事与模块地图](#part-e-统一故事与模块地图)
- [Part F. Introduction 逐段写作](#part-f-introduction-逐段写作)
- [Part G. 实验闭环](#part-g-实验闭环)
- [Part H. 表述边界与风险](#part-h-表述边界与风险)

---

## 0. 结论先行

**一句话故事（不用 bridge 句式）：**

> 现有驾驶 Gaussian 生成方法**在一个空间里生成，却在另一个空间里被使用**——生成发生在 video latent、voxel latent 或 proxy geometry 中，而样本的价值在 metric 三维世界中被消费；两者之间隔着一个解码器或一个逐场景优化循环，**没有任何训练信号穿过它**。我们把生成变量放在一个"读出即世界"的重建基础模型状态里，让全部条件约束这同一个状态，并让训练误差在**解码之后**测量，从而使生成分布对它最终产出的三维世界负责。

**命名策略：全文只造一个术语。** SceneDirector 也只造了一个模块名（MGRA），其核心概念 *structural reliability* / *semantic completion* / *uncertainty-aware allocation* 全是普通词组而非新造词——造得越少，越不像在包装。

| 角色 | 表述方式 |
|---|---|
| 失败机制 | **不造术语**，直接用机制句：*camera, scene geometry and Gaussians are established at different stages, and nothing constrains them to describe the same metric world.* |
| 正面主张 | **不造术语**，直接用：*a single generative state from which camera, depth and Gaussians are decoded together; because the decoding is differentiable, that consistency becomes trainable.* |
| 唯一术语（方法小节名） | **Hierarchical Decoding Supervision (HDS)**，三层为 *representation / primitive / observation fidelity* |

> 早期草稿曾用 `readout` 系术语（readout-blind generation、readout accountability gap、Hierarchical Readout Feedback）。已全部弃用：`readout` 作名词属工程口吻，不像 ICLR 论文用语；且用 *decode* 系可与 ¶2/¶3 的 "decoded together" 保持同一词根。

**三个环节、三个模块，互为前提：**

| 环节 | 问题 | 模块 | 状态 |
|---|---|---|---|
| **状态** | 生成变量是否存在一个可信、可微、已预训练的世界读出？ | Reconstruction-Grounded Scene State（DGGT lattice + JointSceneTokenizer） | 已有 |
| **规约** | 条件是否约束同一个状态，且满足度能在读出中被度量？ | Typed Specification Binding（typed mRoPE + factorized asset + joint camera state） | 已有，需硬化 |
| **目标** | 训练误差是否在解码之后测量？ | **Hierarchical Decoding Supervision（三层解码监督）** | 已有，是 headline |

依赖关系是真实的，不是修辞：**没有可微冻结的预训练读出，三层损失根本写不出来；没有单一状态，条件满足度只能在像素上审计。**

**建议新增的三个轻量模块：**

- **A. Confidence-Gated Decoding Supervision**：用 teacher 的 `depth_conf`/`gs_conf` 门控世界反馈，回答"你的 teacher 本身有误差"这一必问质疑。
- **B. Explicit Metric Scale Generation（`log s`）**：把 DGGT 空间与米制世界之间那个**逐窗口变化的标量**变成被生成、被监督的量。回答"轨迹已给定为何还生成相机"与"你报的米凭什么是米"。
- **C. Actor Geometry Binding（依赖 B）**：在已解码的 depth + 生成 camera 上、**换算成米之后**监督 actor 的位置与动态性，把"条件计数"变成"条件绑定"；其残差同时校准 B。

> **优先级：B > A > C 的实现顺序，但 B 与 C 是一个闭环，应一起设计。** B 是 C 的前提——没有 `log s`，C 的损失是量纲错误的（见 E.4）。

> ⚠ **2026-08-01 实测更新（见 A.8，尤其 A.8.9）**：Waymo 相机参数与 DGGT
> camera state **不是确定性关系**。camera center 与 direct z-depth 近似共享逐 29 帧
> trunk 的一维 gauge，但 FOV 差异会造成横向各向异性，tokenizer 往返也不保持
> depth/GS 的共同相似尺度。D3 进一步实测米制换算相机替换 teacher render pose 的损失为
> 1.41–1.44 dB，因此“完整三维只差标量”与“渲染一定不失配”均已撤回。这个数只是在
> D3 完整 29 帧、指定 source/target 协议下的**控制保真度上界诊断**，不是推理画质损失；
> 推理没有同一真实图像上的 teacher/metric 两臂可比。

**Intro 核心谱系（一句话名单）：**

> MagicDrive3D、InfiniCube、DriveGen3D、X-Scene、LSD-3D、WorldSplat、CVD-STORM、ScenDi（带 urban-road 限定）。其余全部进 Related Work。

---

# Part A. Scene Flow Pretrain 代码梳理

## A.1 任务定义

尽管文件与类名保留 `scene_flow`、`WanSceneFlow` 等历史命名，**这一阶段解决的不是两帧之间的 scene-flow estimation**。`build_full_scene_bundle`（`train_scene_flow_pretrain.py:4125`）设置

$$M_{\text{preserve}}=0,\quad M_{\text{source}}=0,\quad M_{\text{dest}}=1,\quad z_{\text{splat}}=0,$$

即整个目标窗口从纯噪声生成，既不是 inpainting，也不是局部编辑；代码中保留的 preserve/boundary loss 在该模式下数学上恒为零。

任务是：**在冻结 DGGT 特征空间中，从噪声与结构化条件出发，条件生成一段完整的驾驶场景状态，并由冻结 DGGT heads 解码为可渲染的三维场景。**

## A.2 输入输出

**数据**：Waymo 前视，默认 10 帧 @ 10 FPS。冻结 DGGT aggregator 对完整 29 帧 clip **只跑一次**，再抽取 10 帧目标窗口作为 teacher target——同一 trunk 内不同窗口共享同一 clip-global 几何上下文，但生成器只生成当前窗口。

Patch grid 25×37 = 925，scene latent $z\in\mathbb{R}^{B\times 10\times 925\times 1024}$。

**条件（全部 soft condition，进入同一 full attention 序列）：**

| 条件 | 表示 | 关键设计 |
|---|---|---|
| 全局文本 | 冻结 Qwen3-0.6B 编码的 caption tokens | 直接进入 full attention |
| 物理相机 | 20 维 Waymo condition：相对 anchor 的平移/旋转、相邻帧运动、平面 FOV | 与"生成相机"是两个不同量，见 A.5 |
| 目标外观 | 每对象 ≤32 个 canonical appearance tokens | 从**目标窗口之外**一帧取 RGBA reference，冻结 DGGT + tokenizer 以 `S=1` 编码（`CanonicalAssetEncoder`）；API 结构上无法接收目标 clip/轨迹/latent/bbox/mask——有价值的防泄漏设计 |
| 目标位置与运动 | 每对象每帧 12 维 placement state | $[\text{center}_3,\log\text{size}_3,\sin\theta,\cos\theta,\text{velocity}_3,\text{in-frustum}]$，≤5 对象；另由 3D box + camera 产生 target projected bbox |

**随机变量（噪声初始化，联合生成）：**

- scene latent $z$
- **camera generation target**：11 维 DGGT camera state（translation 3 + rotation-6D + log-FOV 2），第 0 帧绝对、其余帧相邻 $SE(3)$ delta；target 来自**冻结 DGGT CameraHead**，不是 Waymo GT
- **sky**：目标视图转成 32×64 上半球方向 atlas，2×2 RGB patch 打包成 12 维 token；未观测方向低权重监督
- patch 级与 refined dense 级 sky mask（`SkyMaskRefineDecoder`）

**最终可执行输出：**

$$\hat z_{1024}\xrightarrow{D_{\text{JST}}}4\times\hat F_{3072}\xrightarrow{\text{frozen DGGT heads}}\{\text{depth},\text{GS attrs},\text{dynamic conf}\}\xrightarrow{\text{gsplat}}\text{RGB}/\text{PLY}$$

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
gen  = [ scene tokens (S·P) | camera-gen tokens (S) | sky-gen tokens (K) ]
cond = [ timestep | text | camera-cond | asset | edit-control ]
full = concat(gen, cond)     ← 一次 full attention
```

只有 scene span 经 DDT head 预测 1024 维 clean latent；camera 与 sky 用独立轻量 decoder。

**四类 condition 共用三维 mRoPE 但位置定义不同**，这是 typed addressing 的实现：

| span | mRoPE 位置 |
|---|---|
| scene | 全局 frame / $y$ / $x$ |
| camera | 每帧图像中心 |
| **asset** | **canonical UV 线性映射到该帧 target projected bbox**（`scene_flow.py:1965-1982`）；出画对象放到 reserved 保留坐标带 |
| sky | 球面方向坐标 + 独立时间偏移区间 |

asset token 值为 `appearance + placement_MLP(12维) + slot_embed + modality_embed`（`scene_flow.py:1939-1944`），另加 per-frame summary token。**即：外观是"什么"、placement 是"在三维哪里"、RoPE 是"在图像哪里"，三者分解后相加。**

### Rectified flow

$$z_\sigma=(1-\sigma)z_{\text{clean}}+\sigma\epsilon,\qquad v^\star=\frac{z_\sigma-z_{\text{clean}}}{\max(\sigma,0.05)}$$

$\sigma\ge0.05$ 时 $v^\star=\epsilon-z_{\text{clean}}$。默认 `prediction_type=x`：预测 clean endpoint $\hat z_0$ 再转 pseudo-velocity。Waver 时间采样、shift 10、`mode_scale` 1.29。

## A.4 三层世界反馈（方法最独特的部分）

`reconstruction_feedback_loss.py` + `rgb_render_loss.py`。

**梯度链已确认贯通**：`compute_rgb_render_loss` 接收 flow 自己预测的 `z_clean_pred_n`（$\hat z_0$），`decode_generated_dggt_geometry` 在 `autocast(enabled=False)` 下调用 tokenizer decoder 与三个 DGGT heads，注释明确写 "Frozen DGGT/tokenizer parameters keep `requires_grad=False` but this module must not run their decode/head calls under `torch.no_grad`"——**参数冻结，梯度穿过**。teacher 分支在 `torch.no_grad()` 内。

| 层 | 名称 | 实现 | teacher | 度量 |
|---|---|---|---|---|
| L1 | representation fidelity | `_level_consistency` | $D_{\text{JST}}(z_{\text{clean}})$ 四层 | layernorm-L1 + cosine，both-zero 保护 |
| L2 | primitive fidelity | `_head_error_maps` | 同上经冻结 depth/GS/instance heads | log-depth smooth-L1、log1p(conf)、GS rgb/opacity/log-scale、quaternion $1-\lvert q_s\cdot q_t\rvert$、dynamic |
| L3 | observation fidelity | `_render_one_sample` + gsplat | **真实图像** | Charbonnier + LPIPS(spatial, 0.01) |

L1/L2 的 teacher 是 $D(z_{\text{clean}})$——**相对于目标世界**；L3 对齐真实像素——**绝对**。

**渲染装配严格对齐部署路径**：means 来自 `torch_unproject_depth(生成 depth, 生成 camera)`；static/dynamic 按 `dynamic_conf<0.5` 划分；static opacity 乘 $(1-p_{\text{dyn}})$ 与 $(1-p_{\text{sky}})$ 并按 `gs_conf` 做 $\exp(\ln 0.1\cdot\Delta t^2/\text{conf}^2)$ 时间衰减；背景用生成 sky atlas 可微投影（`sky_tokens_to_background`，align_corners 约定与 validation 严格一致）。

**调度（对论文诚实性很重要）**：`rgb_render_start_step=5000`、`rgb_render_every=2`、`warmup=5000` 线性 ramp、per-sample 权重 $(1-\sigma)^2$、`max_samples=1`。世界反馈是**低占空比、只在低噪声区生效**的信号——这也是它算力可负担的原因。`camera_grad_scale=0`（forward 用生成相机但不回梯度）、`sky_mask_grad_scale=0.05`。默认 $\lambda_{\text{rgb}}=\lambda_{\text{level}}=\lambda_{\text{head}}=0.1$。

## A.5 其余监督

- 主 scene rectified-flow loss（`masked_flow_edit_loss`，按 `M_edit` 掩蔽）
- 第 8 层 early/base head loss，`base_model_coeff=0.25`
- REPA（$\lambda=0.5$）：中间 trunk feature 投影后与 clean tokenizer latent 做 MSE——**仍在 latent 空间**
- camera flow(0.1) + 绝对/相对 $SE(3)$ + FOV + 二阶平滑，`lambda_camera_pose=0.5`
- sky flow(0.1) + patch mask + refined mask BCE/Dice/boundary
- CFG：text / asset / camera **独立** dropout，采样时分别算 delta 再组合

**两种相机量必须分清**：20 维 Waymo 物理 condition（请求）vs 11 维 DGGT camera state（生成，来自冻结 CameraHead 而非 Waymo GT）。当前语义是"给定请求的物理相机条件，同时生成一条 DGGT-space camera trajectory，后者用于最终几何解码与渲染"。**不能写成"直接用输入相机渲染"。**

> ⚠ **两者不是确定性关系，已实测确认（见 A.8）。** 旋转与轨迹形状一一对应（0.168° / 0.52%），但**平移尺度**与 **FOV** 都不是 Waymo 参数的函数：DGGT 沿用 VGGT 的逐窗口尺度归一化，1 DGGT 单位在不同窗口等于 24.8–64.2 米。因此
> - 20 维条件里的米制平移**无法**决定 11 维 target 的平移幅度；
> - 12 维 placement 的 `center`/`log_size`/`velocity` 全是**米制原值**（`factorized_asset_condition.py:685-695`，无归一化），与解码几何**不同量纲**；
> - 由 3D box + Waymo 相机产生的 **target projected bbox 不受影响**（box 与相机同为米制，投影到像素时尺度约掉），所以 RoPE 寻址与 mask 今天是正确的。
>
> 修复方案是把这个标量显式生成（`log s`），见 E.4 模块 B。

## A.6 推理流程与当前缺口

标准 raw-validation 路径：噪声 → scene/camera/sky 联合 ODE 采样（脚本内显式 Euler，$\sigma:1\to0$，35 步；`WanSceneFlow.sample()` 本身未实现）→ 补零 special tokens → 冻结 heads 以 `images=None` 解码 → gsplat 渲染 / 逐帧 PLY。长序列用 10 帧窗口、stride 7、cosine overlap 融合。

**缺口（必须写进 limitation）**：外部 manifest 分支目前 `return_camera=False`，只保存 normalized latent/sky/mask `.pt`，未接通 render/export——"任意外部 appearance/location/camera → Gaussian"的接口意图清楚但**未端到端验证**。raw-validation 的 asset、track、camera 都来自同一 Waymo clip，因此跨场景外观重组、反事实位置、显著 OOD 相机轨迹、prompt 组合泛化**均未证明**。输出是逐帧 image-grid-aligned dense Gaussians，**没有持久 canonical 4D field、没有显式 deformation field、没有 instance identity**。

## A.7 各模块如何配合

```
29 帧 RGB ──(冻结 DGGT aggregator, 跑一次)──> 四层三流 feature lattice
                                                    │
        窗外 RGBA reference ─(CanonicalAssetEncoder)─┤
                                                    ↓ (E_JST, 冻结)
  text ─(Qwen)─┐                          z_clean [B,10,925,1024]
 20D camera ───┤                                    │
 12D placement─┼──> typed mRoPE tokens ──> RAEVideoSceneFlow ──> ẑ₀ , ĉamera(11D) , ŝky(atlas)
  sky spherical┘        (28×1440 + DDT)             │
                                                    ↓ (D_JST, 冻结；梯度穿过)
                                      四层 3072 维 feature ──> 冻结 depth/GS/instance heads
                                                    │              │
                                            L1 level loss    L2 head loss
                                                                   ↓ (gsplat, 生成 camera + 生成 sky)
                                                              L3 render loss vs 真实像素
```

优势不来自任何单一模块，而来自"**表示、生成、执行**"三者相连：

$$\text{driving specification}\to\text{reconstruction-grounded latent distribution}\to\text{DGGT-executable state}\to\text{Gaussian scene}$$

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

$$s=\frac{\text{DGGT unit}}{\text{metric metre}}.$$

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
   0/7/14/21/28。分支统计仅用 29 帧最大两两相机中心跨度 $>2$ m 的 trunk，并先在
   scene 内平均 trunk，再做 10,000 次 scene-cluster bootstrap。
   aggregator 仍看过目标 RGB，因此这是 decoded geometry 的跨视角自洽性测试，
   不是 target-masked encoder 的 novel-view 泛化实验。
3. **D2 三把尺度尺**（`verify_gauge_gt.py`）：
   - Lidar 主尺：每帧取 `median(DGGT depth / lidar depth)`，只用 $1<d<80$ m
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

$$R_{\text{DGGT}}[t]\approx R_{\text{Waymo}}[t],\qquad
t_{\text{DGGT}}[t]\approx s\cdot t_{\text{Waymo}}[t]$$

**尺度不是每场景的常数，而是每 29 帧窗口的常数**：scene 301 三个 trunk 给出 0.0281 / 0.0343 / 0.0407（CV 15.0%），scene 302 给出 0.0234 / 0.0307 / 0.0325（CV 13.6%）。

**关键的正面结果——相机与深度共用同一把尺子：**

$$s_{\text{cam}}/s_{\text{depth}}=1.0073\pm0.0442,\qquad \mathrm{corr}(s_{\text{cam}},s_{\text{depth}})=0.980$$

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

$$s_{\text{cam}}/s_{\text{depth}}=1.00729\pm0.04422,\qquad
\mathrm{corr}=0.98013.$$

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

$$\Delta_{\text{cam}}=\operatorname{PSNR}(\text{teacher pose})-
\operatorname{PSNR}(\text{metric-converted pose}).$$

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

$$z_0=\mathrm{depth}_{recon}/s_{lidar},\qquad
c(z_0)=\exp\left[-0.0405706428+0.0146570329
\log\frac{\mathrm{clamp}(z_0,0.5,80)}{20}\right].$$

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

作用域必须分开：训练/推理 **render 仍用 identity**；loglinear 只服务米制导出、模块 C 的
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

### A.8.11 Phase 1–8 当前实现核对点（不覆盖 Phase 0 provenance）

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

### A.8.12 D1–D4 post-review 审计与修复（2026-08-02）

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

|  | 只有 latent 监督 | 有世界级监督 |
|---|---|---|
| **摊销式条件生成器** | WorldSplat、**CVD-STORM**、ScenDi、DriveGen3D、InfiniCube、X-Scene | **← 空缺，本方法所在** |
| **逐场景优化 / 非生成** | — | LSD-3D、MagicDrive3D(FTGS)、DreamDrive（逐场景）；Envision4D、DGGT（重建而非生成） |

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

# Part D. 四个核心问题的回答

## D.1 为什么用 DGGT 特征空间

### 安全且准确的抽象

> **reconstruction-grounded multi-view scene state**：一个 pose-free feed-forward 4D 重建模型的中间状态——其中跨视图对应、teacher-gauge depth、相机关系、动态归属与 pixel-aligned Gaussian 属性已被**联合编码**，并可由**已预训练的可信 heads 联合解码**。只有结合逐 trunk 的 LiDAR gauge 与 checkpoint-bound pullback 后，depth 才跨入米制边界。形式上 $z\in\mathbb R^{S\times P\times1024}$ 是四层三流 DGGT feature lattice 的 12:1 压缩；$D_{\text{JST}}\circ\text{heads}$ 是一条**预训练的世界读出**。

**不可写**：view-independent 3D space、canonical 3D grid、world-space geometry token、直接 Gaussian diffusion。**必须承认**：latent 仍是 view-indexed（frame/patch aligned），作为 scope 而非隐瞒——这也是 C.3.1 建议不要把分类轴建在"latent 是否三维"上的原因。

### 与核心谱系的对照

| | 生成变量 | 谁定义"世界" | 高保真外观最终落在哪 | 单场景代价 | 动态 |
|---|---|---|---|---|---|
| **LSD-3D** | voxel occupancy（latent diffusion）；Gaussian 参数是**逐场景优化变量** | NKSR proxy mesh + 带深度条件的 2D image diffusion teacher | 逐场景优化出的 2D oriented planar splats（+可选 deferred T2I） | **6000 步 ~2h/H100** | 事后插入外部资产 |
| **ScenDi** | 单目深度融合得到的彩色 voxel grid 的量化 latent（0.4 m，固定体积） | 从头训练 300k 步的 Voxel-to-3DGS VQ-VAE | **第二阶段 2D video diffusion**；远景与高频不写回 Gaussian | ~5 s + 0.25 s + **~4.8 min 2D** | 数据剔除显著动态 |
| **WorldSplat** | 多视图 pixel-aligned RGB+depth+semantic latent | 独立训练的 latent Gaussian decoder | Gaussian，**但另有 video refiner 精修渲染视图** | 前馈 | 静态背景 + 动态物体分离后聚合 |
| **CVD-STORM** | **多视图视频 latent** | stage-1 与 VAE 联合训练、stage-2 冻结的 GS decoder | 视频（主结果）；Gaussian 为并行分支 | 前馈 | 按预测速度变换 |
| **本方法** | 压缩的多层 DGGT feature lattice | **已预训练冻结**的 DGGT depth/GS/instance heads + 生成相机 | Gaussian 本体；推理时**无第二个 2D 生成器** | 一次 ODE(35 步) + 一次 decode | 逐帧 dense Gaussian + `dynamic_conf` 划分 + 每帧 actor placement/velocity 条件 |

### 三条可辩护的理由

**1. 读出是预训练且可信的，"世界"不必由任务专用 tokenizer 从头学。**
ScenDi 的世界观念被 0.4 m voxel 与固定体积**在构造上**限制——这正是它自己的动机（"relying solely on 3D diffusion → degradation in appearance details"），也是它必须接第二阶段 2D 扩散的原因。我们的 latent 继承了一个"从无位姿真实驾驶图像重建"训练出的读出：camera、teacher-gauge depth、动态归属、Gaussian 属性都在图像分辨率上可解码；再由生成 gauge 与 checkpoint-bound pullback 把 depth 映到米制边界，无固定体积、无 voxel 量化、无单目深度融合预处理。

**2. 因为读出可微且冻结，世界级监督对生成器才成为可能——这是决定性的、非替代性的理由。**
voxel-VQ latent 的 decoder 含 codebook argmax 且分阶段训练；video-VAE latent（WorldSplat、CVD-STORM、PhiGenesis）的几何只在另一个单独训练的 Gaussian decoder 之后才出现，而那个 decoder 在训练生成器时是冻结的。**所以"选这个空间"不是"换一个 latent"，而是 D.3/D.4 的前提条件。这一句是 D.1 与 D.3/D.4 的接缝。**

**3. 相机与内容在同一状态里，因此相机是生成变量而不只是渲染 query。**
ScenDi 的相机轨迹是在已生成静态场景上的**事后渲染查询**。当前改造在同一序列里联合生成
**9 维米制 Waymo camera state** 与独立的 3 维 scene gauge；请求/目标的相机参数化因此是恒等映射，
未知 teacher 尺度不再塞进相机通道。
**诚实的代价**：D3 判定训练 RGB render 必须使用 detached teacher pose，而不是生成相机；否则会主动
把 teacher latent 几何拉向 Waymo 世界。这使 camera 与 geometry 不再有光度耦合，生成相机的职责是
“可执行、可度量的联合状态输出”，而不是暗中承载未知 gauge。论文必须用模块 C 或
generated static-geometry reprojection/cycle diagnostic 把潜在分叉变成可观测的代理诊断；它不等价于独立光流真值，也不能单独证明两者绝无分叉。

## D.2 联合控制的真正优势

### 边界先立

`text + 3D box + camera trajectory` 的联合条件**已被** MagicDrive3D、DriveGen3D、InfiniCube、WorldSplat、X-Scene、CVD-STORM 覆盖，ScenDi 覆盖 layout/text。**"more controls" 一定被驳回。**

### 三条可守的差异

**1. 显式实例外观锚点。** 必须区分三种强度：① 类别或全局文本暗示的外观；② 文本中的对象级描述；③ **显式实例外观锚点（reference crop / image / identity embedding）**。只有 ③ 支撑"精确控制目标实例外观"：

| 方法 | 实例外观强度 |
|---|---|
| MagicDrive3D / DriveGen3D / CVD-STORM | ① 类别与文本 |
| X-Scene | ② 文本含 appearance 描述，非 reference identity |
| WorldSplat / InfiniCube / ScenDi | ①②，**无实例图像锚点** |
| DreamDrive | 单图可锚定**整场**外观，不锚定实例 |
| PhiGenesis / Envision4D | 由历史观测继承，非新实例指定 |
| **本方法** | **③**：per-instance canonical appearance token，来自**窗外**帧，与该实例逐帧 metric placement 绑定 |

**2. 全部条件约束同一状态，且满足度可在解码世界中被度量——最核心的一条。**
在 cascade 里，外观在 video model 决定、几何在 voxel/scaffold 决定、相机在渲染时决定；**只能审计最终像素**，而那时几何、外观、相机误差已纠缠。在我们这里，actor identity、actor metric center、camera pose、text alignment **各自可在解码后的 Gaussian / depth / camera 上单独测量**。

> **写作句**：our contribution is not that the model accepts more conditions, but that all conditions are resolved against a single state whose decoded geometry makes their satisfaction measurable in 3D rather than only in pixels.

**3. 条件寻址是 spatially typed 的。** appearance token 携带 canonical UV → target projected bbox 的 RoPE 地址；placement v2 是独立 16 维状态加到同一 token（5 个标准化 log-幅值 + 11 个 passthrough）；camera 有 per-frame 图像中心位置；sky 在球面坐标 + 独立时间带。

**诚实的代价**：这仍是**全 attention 的软绑定**，代码没有结构性保证"一个对象条件只改变对应实例"。论文必须二选一或都做：(a) 加轻量模块硬化绑定（新增模块 C），(b) 用隔离实验实证。否则会被"这不就是多加了几个 condition token 的 conditional diffusion"击穿。

> ⚠ **07-31 baseline 的硬伤及当前处置**：旧 `placement_state` 的 `center`/`log_size`/`velocity`
> 是米制原值，而解码几何是 DGGT 单位。placement v2 已改成 `log_z_depth`、`log_box_lwh`、
> `log_speed` 与尺度不变量；像素级 `target_bbox_patch` 继续走真实 Waymo K，metric depth 则由生成 gauge
> 与 v1 production pullback 对齐。模块 B 仍是可报告米制 actor error 的前提。

## D.3 现有方法是否有完整梯度链与联合训练

**不能写"现有方法没有 Gaussian/RGB/render loss"——这是错的。** 可守的表述是：

> **世界级误差在现有工作中普遍存在，但它训练的是下游 decoder 或表示、或拟合的是单个场景——它被 stage boundary 或 per-scene optimization loop 与生成模型隔开。生成分布本身只对 latent 空间的目标负责。**

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
| **本方法** | 是：L1/L2/L3 施加在 flow 自己预测的 $\hat z_0$ 上 | **是**（decode/heads 参数冻结但梯度穿过；`camera_grad_scale=0`、`sky_mask_grad_scale=0.05` 须如实报告） | 生成器与世界反馈同一个优化目标 |

**优势三条陈述：** (i) 训练信号在样本被消费的空间中测量；(ii) 生成器可把容量花在**对世界重要的 latent 方向**上（见 D.4）；(iii) 推理时**不需要第二个 2D 生成器**就能达到所报保真度，因此同一个 Gaussian 场景可从其他轨迹重新渲染——这是数据生成器必须具备而 per-clip 2D refiner 无法提供的性质，也是与 ScenDi / WorldSplat / CVD-STORM 最锋利的对照。

## D.4 三层损失如何成为方法创新而非"多加三个 loss"

### 原则性陈述

rectified-flow 目标度量 latent 欧氏误差 $\lVert\hat z_0-z_0\rVert^2$；我们关心的是读出之后的误差。设 $F=R\circ D$，$\delta z=\hat z_0-z_0$：

$$\lVert F(\hat z_0)-F(z_0)\rVert^2\approx\delta z^\top J_F^\top J_F\,\delta z,\qquad\text{而 flow loss 用的是 }\delta z^\top I\,\delta z .$$

**标准目标隐含假设 $J_F^\top J_F\propto I$。它不是。** 三层损失是**在读出链三个深度上对这个 pullback metric 的 Monte-Carlo 估计**——我们让 flow 模型在**世界诱导度量**下而不是 latent 欧氏度量下被优化。

这把"我们加了三个 loss"变成可检验的技术命题，散点图随之变成**这一小节的动机图**。

### 小节命名

**Hierarchical Decoding Supervision (HDS)**，三层分别叫 *representation / primitive / observation fidelity*。不要把三层都叫 world loss。

### 散点图必须升级为系统性测量

两个 sample 的轶事一定被攻击。除两个案例外**必须**给：

- 全验证集 feature MSE 与 depth / Gaussian / render error 的 **Spearman 相关系数**；
- **按 $\sigma$ 分层**（世界反馈只在 $(1-\sigma)^2$ 权重下生效，分层能证明不是噪声效应）；
- **加入 HDS 前后**的相关性与离群点比例；
- Gaussian error 按 position / scale / opacity / color / rotation **分解**；
- 可选：calibration curve（给定 latent MSE 分位数，world error 的条件分布）。

### 边界

不是"首个 render loss"（LSD-3D、ScenDi VQ-VAE、WorldSplat GS decoder、CVD-STORM STORM-VAE 都有）；不是"首个多层特征"（PhiGenesis 已用冻结 video VAE 的多尺度 decoder features）；不能无限定写 "complete gradient chain"（Envision4D 有完整 render gradient）。**可以写的**：*the flow objective is blind to an anisotropic decoder; we make it see it.*

---

# Part E. 统一故事与模块地图

## E.1 四点串成一条逻辑

```
   驾驶样本的价值在 metric 三维世界中被判定
                    │
     ┌──────────────┴───────────────┐
     │  但现有方法在另一个空间生成    │  ← 领域缺口（机制级）
     └──────────────┬───────────────┘
                    │
        "让生成变量与被判定的世界重合，并让训练误差在解码之后测量"
                    │
   ┌────────────────┼────────────────┐
   ↓                ↓                ↓
 D.1 状态        D.2 规约         D.3/D.4 目标
 读出存在        条件约束同一状态   误差在读出后测量
 且可微          → 满足度可在3D度量  → 在世界诱导度量下优化
   │                │                │
   └── D.1 是 D.3/D.4 的前提 ────────┘
       D.1 + D.2 使满足度可测
       D.3/D.4 使 D.1 的选择兑现
```

## E.2 措辞库

**核心句（可放 Abstract 第 2 句与 Intro ¶3 首句）：**

> Driving-scene generators are optimized for what they predict, not for what they produce. We generate a single scene state that decodes jointly to camera, gauge-calibrated metric depth and Gaussians, bind every condition to that same state, and supervise the generator *after* decoding — at the representation, primitive and observation levels — so that the quantities defining a driving event are consistent by construction and that consistency is trainable.

**标题候选：**

1. *A Plausible Latent Is Not a Valid World: Decoding-Supervised Generation of Driving Gaussian Scenes*
2. *One State, One World: Jointly Decoded Generation of Controllable Driving Gaussian Scenes*
3. *Reconstruction-Grounded State Generation with Hierarchical Decoding Supervision for Driving Scenes*

推荐 1（把缺口直接写进标题，且不是 bridge 句式）或 2（更简洁，且 "one state" 正是 ¶2→¶3 的枢轴）。

## E.3 模块地图：每个模块的唯一叙事职责

> **2026-08-01 按 Phase 0（A.8.5 / A.8.6）与独立复测重写。** 上一版把 `log s` 写成
> "在 11 维 camera state 上加一维"，并把 actor box 列为主尺、把 FOV 划出范围——三条都被实测推翻。
> 实现细节见 `docs/metric_scale_camera_redesign_plan.md`。

**已有模块（不改代码，只需在论文中赋予职责）：**

| 模块 | 唯一叙事职责 | 对应 Intro 的哪一句 |
|---|---|---|
| DGGT lattice + **JointSceneTokenizer**（冻结） | 让"世界读出"成为一个**可生成的接口**；世界不必重新学 | "读出必须已存在且可信" |
| **RAEVideoSceneFlow + typed mRoPE** | 让四类规约在**同一状态**上共同求解，各有时空地址 | "规约必须落在单一状态上" |
| **Factorized asset condition** | 把实例外观（canonical、窗外、防泄漏）与 placement **分解**后再绑定 | "外观与位置是两个可独立指定的变量" |
| **Sky atlas branch** | 远景留在生成状态内，而不是交给第二个 2D 生成器 | 与 ScenDi"远景完全交给 2D"的直接对照 |
| **Hierarchical Decoding Supervision** | **headline**：在世界诱导度量下优化生成分布 | "误差必须在解码之后测量" |

**结构性改造（本轮新增，见 E.4-B）：**

| 模块 | 唯一叙事职责 | 对应 Intro 的哪一句 |
|---|---|---|
| **Metric camera-state generation**（11D DGGT → **9D 米制 Waymo**） | 读出所依赖的坐标系本身是被生成、被问责的量，**且它的单位是米** | "相机不是渲染 query，而是状态的一部分" |
| **Explicit scene gauge**（3 维 scene-global token） | 冻结 teacher 留下的**规范自由度**（尺度 + 内参）本身是被生成、被监督的量；它是一切米制断言的兑现条件 | 同上一行的兑现条件 |
| **Frozen-decoder pullback calibration** | HDS 的**前置条件**：在解码之后测量之前，先证明"解码之后"这个空间是可信的 | "误差必须在解码之后测量"——那就得先审计解码本身 |

> 最后一行是本轮最容易被低估的一条。HDS 的全部说服力建立在"解码后的量是物理量"上；
> 实测发现冻结 tokenizer 的往返**保住了单位却没保住相似性**（depth ×1.031、Gaussian 三轴 ×0.829，
> 配对 `GS/depth` = **0.796**，30/30 场景 < 1）。不审计就用，HDS 就是在一个悄悄畸变的空间里测误差。
> 这条不是"又加一个模块"，是**给 headline 补上它缺的地基**。

## E.4 三个新增模块

> **2026-08-01 重写。** 上一版标题是"三个轻量模块"。Phase 0 之后 **B 不再轻量**——它从
> "在相机状态上加一维"升级为一次结构性改造（相机换空间、新增 scene-global 流、冻结解码器标定），
> 已独立成篇 `docs/metric_scale_camera_redesign_plan.md`。A 仍然轻量；C 换了角色。
>
> 实现优先级 **B > A > C**（B 是 C 的量纲前提，这一点没变）。

### A. Confidence-Gated Decoding Supervision

> 叙事上 A 是 HDS 最直接的加固，故仍列首位。本节内容未受 Phase 0 影响。

**回答的必问质疑**：*"你的 teacher 是一个冻结的重建模型，它自己的误差就变成了你的目标。"* 不回答，accountability 论证有一个洞。

**做法**：冻结 heads 本身输出 `depth_conf` 与 `gs_conf`。用 **teacher 的** confidence 作为 primitive/observation 两层的逐像素可靠性权重——读出可信处世界反馈强，读出自身不确定处（天空、远景、细结构）反馈弱。

**为什么它是 SceneDirector uncertainty mask 的真正结构对应物**：SceneDirector 的 gate 站得住是因为**先验可靠性信息本来就存在**（sensor-verified / inferred / asset / void）。我们同样有天然的可靠性先验（head confidence + $\sigma$），所以这不是为凑模块而加。

**落点**：`reconstruction_feedback_loss.py::_dense_weight` 与 `_head_error_maps`（把 teacher conf 乘进 `dense_weight`）、`rgb_render_loss.py` 的 `weight` 构造处。代价：一次 elementwise multiply，teacher 张量已在手。

**实验**：naive HDS vs confidence-gated HDS + gate map 可视化（呼应 SceneDirector Fig. 7）。

### B. Explicit Scene Gauge + Metric Camera（结构性改造）

> **2026-08-01 第三版。** 07-30 版叫 *Camera Gauge Consistency*（"拟合一个相似变换并证明它是恒等"）；
> 07-31 版改成"在 11 维相机状态上加一维 `log s`"。**两版都不对。**
> Phase 0（D1/D2，90 个 trunk 实跑）与一份不共享任何代码的独立复测给出了三条修正，
> 每一条都改变了做法而不只是数字。完整实现见 `docs/metric_scale_camera_redesign_plan.md`。

**回答的质疑**：*"轨迹已给定，为何还要生成相机？"* 以及更硬的一条：*"你报的米制数字凭什么是米？"*

#### B.1 被推翻的三条与实测

| 07-31 版的说法 | 实测 | 后果 |
|---|---|---|
| 「未知量只有一个标量」 | **不止**。除以 $s_{\text{cam}}$ 后仍余 camera-XYZ **2.452 m**，其中 **1.287 m 纯由 FOV 差异产生**；射线夹角 3.286°（CI [2.483°, 4.197°]） | 规范量是 **3 维**（尺度 + 两个 FOV），不是 1 维 |
| 「actor box 是**最稳**的尺子，建议作主信号」（表里甚至没有 LiDAR） | **最弱**。90 个 trunk 中仅 29 个形成有效 actor point，3 个静止样本给出 0.9425 / **0.5405** / **0.5434** 的灾难性离群；覆盖率 68.97%；且与 LiDAR 共享 DGGT depth，**不统计独立** | 主尺换成 **29 帧 LiDAR 深度尺**（90/90 有效，逐帧 robust CV mean 0.688%）；actor 降为诊断字段 |
| 「**不解决 FOV**，FOV 是另一回事」 | D1 的 leave-one-frame-out 渲染判定 **Branch A**：teacher 自己的 $K$ 比真实标定**更好**地解释它的几何，**+0.472 dB**（scene-bootstrap CI [+0.253, +0.741]） | FOV **不是坏 head，是规范量的一部分**，进 gauge 的两个通道 |

两条数字更正：`±4.4%` 不再当作"最好情况下界"（同口径两个独立估计器均给 ~2.5%，且 tokenizer 另加 ~3% 系统偏移）；
相机尺 vs LiDAR 尺的 29 帧一致性是 `0.99995 ± 0.02639`，corr 0.99296。

#### B.2 做法（两件独立的事，务必分清）

**(i) 生成的规范量：3 维 scene-global gauge token**
`[log(米/DGGT单位), log tan(FOVx/2), log tan(FOVy/2)]`，与 sky 同类（**不是**挂在相机状态上）。
同时相机生成目标从 11 维 DGGT 位姿改为 **9 维米制 Waymo 位姿**——FOV 移出相机流，尺度移进 gauge。

**(ii) 标定的常量：冻结解码器的 pullback**
这是 07-31 版完全没有的一层，也是本轮最有分量的发现。训练与推理的几何**全部**经过
tokenizer 往返（`rgb_render_loss.py:206,212` 的 `depth_head`/`gs_head` 作用在重建 token 上）：

| 配对同像素，scene-balanced | 值 |
|---|---:|
| `depth_recon / depth_direct` | **1.0307**，CI [1.0208, 1.0421] |
| Gaussian 三轴几何平均 scale ratio | **0.8289** |
| **配对 `GS/depth`（相似性的必要条件，理想 = 1）** | **0.7964**，30/30 场景 < 1 |

两条后果必须分作用域。depth 的约 3% 系统偏移会污染**米制导出**；A.8.10 已用独立
LiDAR gate 证明 v1 loglinear profile 可把 AbsRel 从 7.567% 降到 6.901%。但 D4 同时证明
该校正对 renderer 没有增益，因此不能再把“20 m 约 11 px”写成已证实的 render bug。
GS 相对 depth 偏小约 20% 仍是完整相似性缺陷；PSNR 扫描无法给出物理 `c_gs`，只能由
tokenizer v2 从源头修复。gauge 表不依赖 tokenizer，但任何 metric profile 都必须绑定 tokenizer 哈希。

#### B.3 硬性实现约束（不遵守会静默出错）

1. **target 用整段 29 帧估计，不用 10 帧窗口。** 10 帧窗口的估计 CV mean 3.19% / max 9.94%，静止 trunk 发散到 23–30×。
2. **做成 scene-global token**，用现成的 `scene_global_window_weight` 融合。不能挂 anchor 帧（delta-only 窗口无处安放），也不能做成逐帧通道。推理的滑动窗融合在 ODE 循环内部、整条序列共享一个 anchor，因此**滑动窗本身不会破坏 gauge**。
3. **静止片段排除相机尺子**（20/90 按定义无效），但 LiDAR 尺在这 20 个上**全部有效**——这正是换主尺的直接理由。
4. **metric `c_depth` profile 是 checkpoint 与窗口长度的属性**，加载时必须同时断言 tokenizer
   hash 与 `window_len`；render 强制 identity，`c_gs=1.0`。原始 v1 诊断产物不可加载，当前只允许
   加载 A.8.10.1 的严格 production artifact。
5. **placement v2 是 16 维，算术必须写对**：标准化通道 `{3,4,5,6,13}` 共 **5 个**；
   passthrough `{0,1,2,7,8,9,10,11,12,14,15}` 共 **11 个**。后者是方向、角尺寸、yaw、
   速度方向、速度/深度与 `in_frustum`；前者是 `log_z_depth`、三个 `log_box_lwh` 与 `log_speed`。
6. **两条 K 链永不交叉**：DGGT depth/render/sky 用 gauge K；米制 bbox/`in_frustum`/
   `target_bbox_patch` 用真实 Waymo K。不能把“gauge K 统一 DGGT 消费者”误写成“全系统只剩一个 K”。

#### B.4 它解决了什么（精确表述，别夸大）

- **它没有消灭尺度的不确定性**——模型仍要从场景内容推断。它做的是把这个未知量从 10 帧 × 3 个平移通道共 30 个数里拎出来，收进一个可解释的 3 维量。当前损失分不清"尺度猜错"和"轨迹画错"；拎出来之后轨迹部分由条件**完全决定**。
- **相机可控性从"数学上不成立"变成恒等映射**。这是最硬的一条：改造前目标 = $s \times$ 条件，$s$ 逐 trunk 随机且不可观测，**这个实验根本做不了**。
- **v1 单位换算得到独立支持**：先算
  `z0=exp(log_metric_scale)·depth_recon`，再用 `c_depth(z0)`；A.8.10 的 LiDAR gate
  将 AbsRel 相对降低 8.81%。它已作为 v1 checkpoint-bound production contract 正式启用；
  v2 必须重测且允许改变结论。
- **可证伪的机制假设**：仅由米制轨迹形状**无法解析** teacher 自选的 gauge，但图像内容（车辆尺寸、车道宽度）可能提供统计线索——这正是 gauge token 与全部 video token 做全注意力的理由。若假设为真，gauge 预测应显著优于"输出训练集均值"的 marginal-prior baseline。**训练日志强制打印这个对比。**
- **一条必须主动写的 limitation**：teacher 的 gauge 跨 trunk 漂移 mean 8.2% / max 30.8%（LiDAR 与相机同向，41/44 对，Pearson 0.9711）。≤29 帧生成干净；长序列滑动窗会得到一个全局尺度，它比 teacher 更自洽，但**不存在一致的 GT 尺度可供评测**。

#### B.5 落点与实验

**落点**：新文件 `dggt/utils/scene_gauge.py`（常量、归一化、米制↔DGGT、pullback 唯一实现）+
`tools/compute_dggt_scene_gauge.py`（离线 GT 表）+ `tools/calibrate_tokenizer_pullback.py`；
`camera_generation.py` 11→9 维米制；`scene_flow.py` 新增 gauge 流并接进 DDT 头的 `cond`；
`scene_gauge.py` 的 shared helper 只在 metric boundary 施加 profile，render 分支断言 identity。

**实验**：gauge 预测误差 vs marginal-prior baseline（机制假设的证伪器）；
米制换算后的 depth error vs LiDAR；**相机可控性扫描**（headline）；
tokenizer v2 的 `GS/depth` paired ratio（若仍失败，才用 LPIPS/SSIM 作 `c_gs` fallback）；
相机等变实验（固定 text/asset seed 扫请求轨迹，测世界内容是否保持）。

**表述**：写成 *the decoder's metric gauge — scale and intrinsics alike — is itself a generated and
supervised quantity, and the pullback through the frozen decoder is calibrated rather than assumed*。
最耐审的一句是 Branch A：**teacher 的内参不是要修的 bug，而是规范量的一部分，我们用
leave-one-frame-out 渲染证明了它**。这是个略反直觉、可复现、且换掉任何 baseline 都仍然成立的经验命题。

### C. Actor Geometry Binding（依赖模块 B）

> ⚠ **本节两次修正。** 07-30 版是量纲错误（米制 `center_anchor` 去对齐 DGGT 单位的解码 depth，
> 中间差着 24.8–64.2 倍）。07-31 版修了量纲，但把 C 定位成"让 $s$ 可被监督的**主要来源**"——
> **D2 推翻了这一点**（actor 是三把尺子里最弱的）。08-01 版换掉的是**角色**，不是做法。

**回答的质疑**：*"你的条件绑定是软的，凭什么说 actor 条件只改变那个 actor？"*

**做法**：对每个 in-frustum 的 conditioned actor $k$，用**已解码的** depth + **生成的** camera 取其
target projected bbox 内的三维点，经 `c_depth` 与 $\exp(\widehat{\log s})$ 换算为米，再监督

1. 这些点的稳健（中位数/trimmed）**z-depth** 对齐 `placement_state` 的 `log_z_depth`；
2. bbox 内 `dynamic_conf` 对运动 actor 为高。

仍是在 HDS 已算出的张量上做一次 masked reduction，**几乎免费**。

> **必须用 z-depth，不能用欧氏 range。** 米制→DGGT 的映射是**各向异性**的：
> 相机系下 $p_{\text{dggt}} = \mathrm{diag}(k_x,k_y,1)\cdot p_{\text{metric}}/s$，
> $k_x = \tan(\text{FOV}^{\text{dggt}}_x/2)/\tan(\text{FOV}^{\text{waymo}}_x/2) = 0.748$（$k_y = 0.772$）。
> **只有 z 分量是纯标量**；欧氏 range 是 $\sqrt{k^2X^2+k^2Y^2+Z^2}/s$，随离轴角变化。
> 这也是 `placement_state` 的距离通道改成 `log_z_depth` 的原因（见 plan Phase 7）。
> **横向那一半不需要换算**——它由 `target_bbox_patch` 的像素直接钉死，两套约定共用同一张像素网格。

**角色更正：C 是推理期的一致性约束，不是训练期的 GT 来源。**

| | 07-31 版的说法 | 08-01 版 |
|---|---|---|
| 训练期 $\log s$ 的 GT | actor box（"最稳，主信号"） | **离线 29 帧 LiDAR 表**；actor 仅 29/90 可用、3 个灾难性离群，降为诊断 |
| C 的价值 | 让 $s$ 可被监督的主要来源 | **推理期没有 LiDAR**——那时 actor 条件确实是约束生成尺度的机制 |

这个换位反而让 C 更站得住：它不再需要声称自己是最好的尺子（那条已被证伪），
只需要声称**在没有传感器真值的推理场景里，条件本身就是唯一的米制参照**。B↔C 仍是闭环，
只是闭环发生在推理期：

```
B 生成 gauge  →  C 用它把 actor 条件拉进解码后的世界  →  C 的残差反过来约束 gauge
```

**一致性检查是必需的，不是可选的**：训练时 gauge 来自真实片段，天然与 $z$ 配套；
**生成时 $\hat z$ 与 $\widehat{\log s}$ 都来自噪声，没有任何东西强迫二者一致**。
这一项本身就是"解码之后测量"，直接并入 HDS，不是外挂的第四件事。

在 C 完整落地前，当前实现提供的代理必须准确叫
**generated static-geometry reprojection/cycle diagnostic**：用生成 `depth_t` 与生成相机前向
重投影，在 `t+1` 落点采样生成 depth，再反投影回 `t`，报告 flow-cycle EPE 与 z-depth log residual，
并剔除 sky/dynamic/遮挡/出视锥 support。它**不是**独立预测 optical flow 与“几何 flow”的比较，
不读 GT 图像；系统本来也没有独立 flow 或 correspondence head。纯静态相机应标为 degenerate，
不能用零位移冒充一致。schema 为 `generated_static_geometry_reprojection_cycle_v1`。

**为什么属于同一个故事**：把 SceneDirector 的 ATE/AOE 从**评测指标**变成**训练信号**，且在**解码后的世界**里做——完全落在 accountability 概念内。

**落点**：`reconstruction_feedback_loss.py` 新增 actor-masked 项；bbox 与 placement 已在 `FactorizedAssetCondition` 中。

**实验**：actor **2D IoU（图像）+ 米制 z-depth 误差**；**跨实例泄漏**（只改 actor $k$ 的 reference，测其他实例的外观变化）。

> **评测口径警告**：不要报裸 3D IoU。DGGT 空间里的车横向天生窄 25%，
> 裸 3D IoU 会把一个正确结果判成差——那是约定差异，不是模型误差。
> 若必须报，先套上面的各向异性映射（`metric_box_to_dggt`）。G.1 主表需同步更正。

> **克制建议**：到此为止，不要为凑模块再加 learned gate。SceneDirector 的 gate 有价值是因为它调节的是**已存在的先验信息**；三个模块都满足这个条件（A 用 head confidence，B 用 LiDAR 实测的规范量与冻结解码器的可测偏差，C 用条件里已有的 box）。

---

# Part F. Introduction 逐段写作

## F.1 ¶1：保留，只改结尾

现有 ¶1 逻辑正确，已在做 SceneDirector ¶1 该做的事（卖任务，不卖模块）。唯一要改的是**结尾必须埋下 accountability 的种子**：把"若这些量只在 2D 视图中看起来正确，生成数据仍可能给感知与规划提供错误的几何监督"再推一步——**样本的价值在三维中被判定，因此生成器被优化的对象也应当在三维中被度量**。这一句是 ¶2 缺口与 ¶3 insight 的接口。

## F.2 ¶2 结构（按 Part C 的谱系）

严格按此顺序：

1. **间接构建 Gaussian**：MagicDrive3D → InfiniCube、DriveGen3D、X-Scene；
2. **直接进入三维状态**：LSD-3D（三维 proxy 上逐场景优化）；WorldSplat、CVD-STORM（生成 latent 前馈解码动态 Gaussian）；ScenDi（urban-road voxel latent 前馈解码静态 3DGS）；
3. **导出缺口**：现有快速 latent generators 的 world losses 通常止于表示或 decoder，而直接接受 endpoint guidance 的方法又依赖逐场景优化。

## F.3 三处必须的修改（相对用户 2026-07-30 双语稿）

| # | 问题 | 修订 |
|---|---|---|
| 1 | 把 PhiGenesis 与 WorldSplat 并列，会让整段被读成"这些都是 source-free 生成器" | PhiGenesis **移出核心谱系**（违反准入 2），只在末句作为"GFM 重建状态被用于外推"的例证出现 |
| 2 | "最终 Gaussian 场景的构建仍依赖视图空间的图像或特征"对 InfiniCube（独立 voxel GS branch）与 X-Scene（fuse occupancy and images）不成立 | 把断言从"整个 Gaussian"移到**外观**，并说明 voxel/occupancy 是**语义/几何支架、本身不承载外观**——对四个成员全部成立，且更锋利 |
| 3 | 末句只有"表示"这一个 gap，会让 ¶3 不得不引入全新问题 | 末句给出**两个** gap（表示 + 监督），并为"未来外推"补上 `phigenesis,envision4d` 的实证支撑 |

## F.4 英文草稿（可直接改写使用）

新增 bib key：`cvdstorm`、`envision4d`（后者只在末句用）。

**¶1 — task and value.** Autonomous driving requires large, diverse, structurally specified data, yet real collection is expensive and biased toward nominal conditions, leaving long-tail roads, object instances and interaction states rare. Video diffusion has broadened this supply, with driving-video generators conditioned on text, HD maps and ego actions. But the value of a driving sample is not decided in two dimensions: the surrounding environment, the metric position and appearance of every actor, and the path the camera takes through the scene together define a three-dimensional driving event. If these quantities are only *apparently* correct in the rendered views, the generated data can still supply wrong geometric supervision to perception and planning. **A sample is judged in the metric 3D world; what a generator is optimized for should therefore be measured there too.**

**¶2 — two paradigms（定稿，中英逐句对照见 F.6）.** Recent work has extended driving-scene generation to 3D/4D Gaussians along two paradigms. The first builds on video generative priors: MagicDrive3D~\cite{magicdrive3d} and DriveGen3D~\cite{drivegen3d} synthesize multi-view videos and reconstruct Gaussians from them, while InfiniCube~\cite{infinicube} and X-Scene~\cite{xscene} add voxel or occupancy scaffolds to support both steps. This progressively tightens the coupling between generation and 3D reconstruction, yet the Gaussian scene remains a downstream product of a separate reconstruction stage rather than the generated variable itself. The second paradigm generates the scene state itself: LSD-3D~\cite{lsd3d} and ScenDi~\cite{scendi} sample a proxy geometry or a voxel latent and derive Gaussians from it, while WorldSplat~\cite{worldsplat} and CVD-STORM~\cite{cvdstorm} decode pixel-aligned Gaussians from a view-aligned latent. In both paradigms, however, the 3D scene that would jointly explain scene semantics, actors and trajectory is produced only after generation, by a separate stage. It never enters the generative objective, so their joint consistency is left to emerge rather than modelled.

**¶3 — our position.** We therefore model that scene directly: environment, actors and the ego-trajectory constrain a single state, the internal representation of a feed-forward driving reconstruction model, whose heads jointly decode camera, teacher-gauge depth, dynamic membership and pixel-aligned Gaussians, while an explicit generated gauge maps the geometry to metric units. Because this decoding is pretrained and differentiable, the joint explanation is not merely represented but supervised: condition satisfaction is measured on the decoded geometry and back-propagated into the generator.

> **三次修订的教训，按顺序记下，避免再犯：**
>
> 1. **v1 的 gap 是 "X hasn't been tried" 型**（"存在一个没人用过的表示"）。换个人做一遍就消失，正是 SceneDirector 刻意避开的那类。
> 2. **v2 犯了范畴错误**：`camera, scene geometry and Gaussians` 不并列，Gaussian 就是场景几何加外观。把**规约族**（要控制什么：环境 / 相机 / actor，与 ¶1 一致）和**输出模态**（解码出什么）混进了一个并列。
> 3. **v2 还有事实错误**："条件由不同阶段分别确立"对 WorldSplat 与 CVD-STORM 不成立，它们的 text/box/map 都进同一个 diffusion。
> 4. **v3 的 "never enforced during generation" 是硬错误。** 别人当然在生成过程中约束（box/map/text 都是条件）。真正的区别不是**有没有约束**，而是**约束在哪个空间被度量**。
>
> **定稿的论证：** 条件确实被施加，但只施加在生成状态上；条件所指的 metric 几何由之后一个单独训练的 decoder 产生，二者之间的偏差无人惩罚。这对四个方法全部成立（LSD-3D 逐场景蒸馏在后、ScenDi 的 VQ decoder 单独训练且相机是事后渲染查询、WorldSplat 三模块独立训练、CVD-STORM 的 GS decoder 在 stage 1 训完即冻结）。
>
> **"联合解码"本身不是主张，是使能条件。** 主张是：只有当条件与可解码的几何落在同一个变量上，满足度才能从**被观察**变成**被优化**。这与 D.4 的各向异性论证是同一条道理的第二个落点（latent 空间的满足只是代理），因此故事是收紧的而不是并列的。
>
> 5. **v4 举的例子是性能断言**，v5 的"不在训练目标之内"又退化成纯描述。完整的修订链条与定稿理由见 F.6 的"第 5 句四个子句职责"表，此处不重复。
>
> **写作纪律：** 缺口要有**机制**，不能只有现象或对手性能断言；声称"无上界/无保证"而不是"误差很大"。问题段落给机制，解决段落再枚举三类规约（环境 / 相机 / actor）。ICLR 正文避免破折号，¶2 与 ¶3 定稿中零破折号。

**¶4 — method.** ⟨state / specification / objective 三句，逐项对应 ¶3⟩

**¶5 — scope and contributions.** ⟨先 scope 后 bullets：前视单相机、10 帧窗口、≤5 actors、逐帧 dense Gaussian、无持久 canonical 4D field⟩

## F.6 ¶2 定稿：中英逐句对照

| # | 中文 | English |
|---|---|---|
| 0 | 近期工作将驾驶场景生成扩展到 3D/4D Gaussian 表示，形成两类方法。 | Recent work has extended driving-scene generation to 3D/4D Gaussians along two paradigms. |
| 1 | 第一类依托视频生成先验：MagicDrive3D 与 DriveGen3D 先合成多视图视频，再从中重建 Gaussian；InfiniCube 与 X-Scene 则加入 voxel 或 occupancy 支架来支撑这两步。 | The first builds on video generative priors: MagicDrive3D~\cite{magicdrive3d} and DriveGen3D~\cite{drivegen3d} synthesize multi-view videos and reconstruct Gaussians from them, while InfiniCube~\cite{infinicube} and X-Scene~\cite{xscene} add voxel or occupancy scaffolds to support both steps. |
| 2 | 这逐步收紧了生成与三维重建的耦合，但 Gaussian 场景仍是一个独立重建阶段的下游产物，而不是生成变量本身。 | This progressively tightens the coupling between generation and 3D reconstruction, yet the Gaussian scene remains a downstream product of a separate reconstruction stage rather than the generated variable itself. |
| 3 | 第二类直接生成场景状态本身：LSD-3D 与 ScenDi 采样三维 proxy 或 voxel latent 并由此得到 Gaussian；WorldSplat 与 CVD-STORM 则从视图对齐的 latent 解码 pixel-aligned Gaussian。 | The second paradigm generates the scene state itself: LSD-3D~\cite{lsd3d} and ScenDi~\cite{scendi} sample a proxy geometry or a voxel latent and derive Gaussians from it, while WorldSplat~\cite{worldsplat} and CVD-STORM~\cite{cvdstorm} decode pixel-aligned Gaussians from a view-aligned latent. |
| 4 | 但在这两类方法中，条件是施加在这个状态上的，而它们所指的 metric 几何要由之后一个单独训练的 decoder 产生。 | In both cases, however, conditions are enforced on that state, while the metric geometry they refer to is produced afterwards by a separately trained decoder. |
| 4 | 但在这两类方法中，本该共同解释场景语义、actor 与轨迹的那个三维场景，要到生成之后才由一个单独的阶段产生。 | In both paradigms, however, the 3D scene that would jointly explain scene semantics, actors and trajectory is produced only after generation, by a separate stage. |
| 5 | 它从不进入生成目标，因此三者的共同一致性是被期待自然浮现的，而不是被建模的。 | It never enters the generative objective, so their joint consistency is left to emerge rather than modelled. |

### 与 ¶1 的铰接（v7 的核心改动）

用户 ¶1 定稿已给出两个关键短语：**"jointly define"** 与 **"cannot be jointly explained by a coherent 3D scene"**。¶2 只需说明现有方法从不建立这个 joint explanation，即自动呼应，无需新造概念。三段共用一组词：

```
¶1  jointly define  →  jointly explained by a coherent 3D scene
¶2  enter separately  →  left to emerge rather than being modelled
¶3  model that scene directly  →  not merely represented but supervised
```

**架构核查（为什么不能写"他们没有能力联合建模"）：** WorldSplat 与 CVD-STORM 的 text/box/trajectory 都进同一个 diffusion，这句话对它们为假。

| | 环境 | 相机作为条件 | actor 布局 | **实例外观锚点** | **相机作为被生成的量** |
|---|---|---|---|---|---|
| LSD-3D | ✓ scene prompt | ✗ 生成后自由渲染 | ✗ 外部资产事后合成 | 外部资产，在生成模型之外 | ✗ |
| ScenDi | ✓ text | ✗ 事后渲染查询 | ✓ boxes | ✗ | ✗ |
| WorldSplat | ✓ | ✓ | ✓ boxes | ✗ | ✗ |
| CVD-STORM | ✓ | ✓ | ✓ boxes | ✗ | ✗ |

上表的架构核查结论**不进 ¶2**，只用于 ¶4/¶5 和 Related Work。原因见下。

### ⚠ 一个必须避免的自摆乌龙（v8 删除的内容）

曾经写过：*"the ego-trajectory is given rather than generated, and an actor enters through its placement alone."* **两处都错在论证方向：**

1. **"轨迹被给定而非被生成"不是缺陷。** 读者的反应是"轨迹当然该给定，那就是条件生成的本意"。更危险的是，**"相机该不该被生成"恰好是审稿人要问本方法的问题**（当前方法生成 9 维米制 Waymo camera state，并以同参数化的请求轨迹作条件；见 A.5）。把它当作别人的缺点写进 ¶2，等于把这个质疑主动递到自己面前。
2. **"actor 只以位置进入"讲的是控制能力（缺实例外观锚点），不是联合建模。** 混进 ¶2 的一致性论证会让两条线互相稀释。

**正确归属：实例外观锚点是本文的贡献，不是别人的缺口。** 它应出现在 ¶4/¶5 作为方法能力（"actor 由参考图像锚定身份，而非仅由 box 指定"）或写进 contribution 列表。

**Intro 的分工因此是：¶2 只攻一件事（共同解释未被建模），¶4 才展示三类规约如何被联合施加、其中包含实例外观这一独有能力。**

### 第 4/5 句的措辞要点

- **直接点名三要素**（scene semantics, actors and trajectory），不用 `these elements` 之类前指代词——放在句首时读者会误以为指上一句的方法，且点名可与 ¶1 逐字呼应。
- **`by a separate stage` 而非 `separately trained decoder`**：LSD-3D 是逐场景优化而非 decoder，"阶段"两者都覆盖。
- **`the generative objective`** 限定更准：LSD-3D 的逐场景优化目标里确实有那个场景，但**生成模型**（voxel diffusion）的目标里没有。
- **刻意不点"全局语义"缺口**：四个方法都条件于 text，那里没有缺口。三要素只在"本该共同解释它们的场景"这一处整体出现。

**落脚点用 `left to emerge rather than being modelled`**：不说"他们做不到"，不说"结果不一致"，只说这件事没有被建模；¶3 第一句 `model that scene directly` 接同一个动词。

> **不要写 "truly complete" / "first complete driving scene generation"。** 过强的自评。功能性表达更有力也更安全：**我们建模那个共同解释，而不是等它浮现。** 完整性由此自证，不必自称。

### 三类断言必须各归其位（v6 修正的核心）

散点图是在**本文自己的模型**上做 with/without 对比，它只能证明"我们的 decoder 不保距、我们的方法需要这些损失"，**不能**用来推断别人的 decoder 也如此。把 `(Fig. 1)` 挂在关于对手的句子后面，是用自家证据指控他人，属无效推理。

| 断言 | 证据来源 | 允许出现的位置 |
|---|---|---|
| 条件施加在生成状态上；几何由之后一个单独训练的 decoder 产生；没有一项把二者关联 | **对方论文可查证** | ¶2 ✓ |
| 在状态上优化一致，对最终交付的几何**没有任何保证** | **纯逻辑，不需数据** | ¶2 ✓ |
| 这个落差在实测中确实很松（latent error 与 world error 相关性差） | **只在本文自己的模型上成立** | ¶4 / 实验，且必须写明 `in our setting` |

**第 5 句两个子句的职责：**

| 子句 | 作用 | 为何不可被攻击 |
|---|---|---|
| `carries no guarantee about the geometry finally delivered` | **逻辑后果** | 状态一致 ⇏ 几何正确；不需任何数据，也未声称误差大小 |
| `no term in the objective relates the two` | **结构事实** | 可查证：ScenDi 相机是事后渲染查询、3D DiT 只有 Eq.(9) latent MSE；WorldSplat 三模块独立训练；CVD-STORM 的 GS decoder 在 stage 1 训完即冻结；LSD-3D 相机是生成后自由渲染、actor 由外部合成 |

**"为什么没有保证是个问题"由 ¶1 供给**（样本的价值在 metric 三维中被判定）。所以这不再是 v5 那种悬空描述：¶1 说这个量重要，¶2 说没人保证它。

**Fig. 1 的正确归属是 ¶4（方法）或实验，并显式限定作用域：**

> In our setting, latent error alone is a weak predictor of the error in decoded depth and Gaussian placement (Fig. 1), which is why we supervise the generator after decoding rather than on the latent alone.

`in our setting` 五个字把作用域说清楚，反而**预先堵掉**"你只在自己模型上验证"这个质疑——因为从未声称更多。

> **现实提醒：不要尝试对对手做经验性指控。** 若要实测"WorldSplat 的 box 条件在解码几何里偏多少"，需要其权重；但 ScenDi 项目页为 "Code coming soon"、LSD-3D 为 "Code (tba)"，大概拿不到。¶2 停在"结构 + 无保证"，实测只对自己的模型讲，是当前证据条件下唯一站得住的分工。

> **修订链条（五版，记下避免再犯）：**
>
> 1. **v1：** "存在一个没人用过的表示" → "X hasn't been tried" 型 gap，换个人做一遍就消失。
> 2. **v2：** `camera, scene geometry and Gaussians` 三者不并列（Gaussian 就是几何加外观），把**规约族**（环境/相机/actor）与**输出模态**混进同一并列；且"条件由不同阶段分别确立"对 WorldSplat、CVD-STORM 不成立。
> 3. **v3：** "never enforced during generation" 是硬错误——别人当然在生成过程中约束，box/map/text 都是条件。
> 4. **v4：** "the actor still lands metres away" 是**性能断言**，审稿人可用自己的 ATE 反驳（SceneDirector 0.78 m、VACE 1.02 m、DriveEditor 0.86 m），我们也无证据支持该量级。
> 5. **v5：** "不在训练目标之内" 只是**描述**——不在目标里不代表会出问题，"没有又怎样"无法回答。
>
> **定稿的关键改动：把"目标里没有这一项"从主张降为最后一个子句。** 它单独出现时是描述；接在"因此没有上界"之后，就变成**补救缺位**（落差存在且无人弥合），而不是现象缺位。
>
> **两条一般原则：**
> - 缺口要有**机制**（为什么会出问题），不能只有**现象**（存在某个空白）或**断言**（对手很差）。机制层面的缺口换掉任何 baseline 仍然成立。
> - 尽量声称"**无上界/无保证**"而不是"**误差很大**"。前者从机制推出，后者永远可以被一张更好的表反驳。
>
> 6. **v5→v6 撤销了 `(Fig. 1)` 的引用。** 曾把散点图挂在关于对手的句子后面作为证据——但那张图只在本文自己的模型上做 with/without 对比，用它推断别人的 decoder 属无效推理。定稿的 ¶2 不引任何本文实验，只用可查证的结构事实与逻辑上的"无保证"；Fig. 1 移到 ¶4/实验并标 `in our setting`。详见上方"三类断言各归其位"表。
>
> **证据分工原则：** 关于**对手**只做可从其论文查证的结构性陈述；关于**效果**只在自己的设定内声称并标明作用域。二者不可互借。

### ¶3 前两句（承接 ¶2 第 5 句）

| # | 中文 | English |
|---|---|---|
| 1 | 我们改为让环境、相机与 actor 约束同一个状态：一个前馈驾驶重建模型的内部表示，其 heads 联合解码相机、teacher-gauge depth、动态归属与 pixel-aligned Gaussian，再由显式 gauge 映到米制。 | We instead let environment, camera and actors constrain a single state: the internal representation of a feed-forward driving reconstruction model, whose heads jointly decode camera, teacher-gauge depth, dynamic membership and pixel-aligned Gaussians, with an explicit gauge mapping geometry to metric units. |
| 2 | 由于这条解码已预训练且可微，条件的满足度可以在解码出的几何上测量并回传给生成器，使每个条件与它所指的几何处于同一个训练目标之下。 | Because this decoding is pretrained and differentiable, condition satisfaction can be measured on the decoded geometry and back-propagated into the generator, placing each condition and the geometry it refers to under one training objective. |

**精简手法（可复用）：**

- **第 4 句去掉了 voxel / video latent / proxy 的枚举**：第 3 句刚列过一遍，重复是纯冗余。改为 `that state` 反指，省 12 词。
- **第 5 句修了主谓**：旧版 "A box condition ... yet place the actor" 的主语是条件，而条件不会"放置"actor。改为 `the actor still lands`，主语正确且更具画面。
- **零破折号**：¶2 与 ¶3 定稿全部改用逗号与冒号。ICLR 正文极少用破折号，之前的版本每句都有，是明显的非论文口吻。
- **`back-propagated into the generator`** 一词替代旧版 "send that error back to the generator, so that ..." 整个从句，省 10 词且更技术化。

**关键改动记录：**

- **术语：全文已弃用 `readout` 作为名词**（`no built-in readout` 属工程口吻，不像论文用语），改用动词 **decoded jointly / decoded together**。方法小节名由 *Hierarchical Readout Feedback* 改为 **Hierarchical Decoding Supervision (HDS)**，与 ¶3 的 "decoded" 同词根；新增模块相应改为 *Confidence-Gated Decoding Supervision* / *Explicit Metric Scale Generation* / *Actor Geometry Binding*（后者原名 *Camera Gauge Consistency*，2026-07-31 因 A.8 的实测结果重新定义）。全文只保留 HDS 一个新造术语（见 §0 命名策略）。
- **第 4 句由"表示可用性"改为"因果机制"。** 旧版结尾（"这样的读出已存在，只是没被用于生成"）是 "X hasn't been tried" 型 gap，换个人做一遍就消失。新版指出的是：定义驾驶事件的量由不同阶段分别确立，无人保证一致——换掉任何 baseline 仍然成立。
- **第 4 句结尾兑现 ¶1。** ¶1 已写"全局环境、目标 metric position、外观与相机路径共同定义一个三维驾驶事件"；¶2 结尾指出现有方法**分别**决定它们，这个 payoff 之前一直空着。
- **"这样的状态已存在"从 ¶2 移到 ¶3。** ¶2 只负责问题，¶3 负责回答。
- 删去 Envision4D 引用与"用于观测下未来外推"整个从句。
- 第 3 句开头用 "generates the scene state itself"（而非 "generates scenes directly in 3D"）：后者对 WorldSplat/CVD-STORM 的视图对齐 latent 不成立；前者覆盖四个方法，并与第 2 句结尾 "the generated variable itself" 咬合。
- LSD-3D 用 **derive**：不误述为一次解码，也不引入已弃用的逐场景优化轴。
- **故事的边界（必须记住）：相机只是被联合解码的量之一，不是贡献。** 主张是 (i) 一致性由构造保证而非期望，(ii) 可微解码使一致性可训练——后者正是三层损失的位置，它是 ¶3 后半句的实现，不是并列的第四个卖点。

## F.5 可选加强

- **CVD-STORM 那半句是全段最有力的**（"trained jointly with the VAE and then frozen"），因为它用对手自己的措辞证明了缺口。若空间紧张，宁可删别的也留它。
- WorldSplat 也用增强 video diffusion 精修从 Gaussian 渲染出的 novel view，为"外观在视图空间被决定"提供第一类最强成员的直接自证。
- 若空间允许，可加"直接进入三维状态这一支目前均为静态或需事后合成动态"：LSD-3D 事后插入资产、ScenDi 剔除动态。**但 WorldSplat 与 CVD-STORM 有动态，所以这句只能限定到 LSD-3D + ScenDi**，不能覆盖整个第二类。

---

# Part G. 实验闭环

## G.0 latent error 与 world error 的失配（motivation 图，不需新模型）

在 latent-MSE-only 的 ablation 上画散点：$x$ = $\hat z_0$ 的 feature MSE，$y$ = depth / Gaussian / render error。附全集 Spearman + 按 $\sigma$ 分层 + 加 HDS 前后对比。

> **作用域纪律（v6 修正）：这张图只在本文自己的模型上成立，不能作为对其他方法的指控。** 它证明的是"我们的 decoder 不保距、我们的方法需要 HDS"，不能推断 ScenDi / WorldSplat / CVD-STORM 的 decoder 也如此。因此：
>
> - **不要**把它作为 ¶2 缺口的证据引用；¶2 只讲可查证的结构事实与逻辑上的"无保证"。
> - **应当**在 ¶4（方法）或实验中引用，并写明 `in our setting`。
> - 对手的经验性指控需要其权重，而 ScenDi 为 "Code coming soon"、LSD-3D 为 "Code (tba)"，不具备条件；**不要尝试。**

## G.1 主表：四类指标分开报

| 组 | 指标 |
|---|---|
| 外观/真实性 | FID、FVD |
| 几何 | 对 pseudo-GT 的 metric depth error；±1–2 m novel-view PSNR |
| **控制满足度** | actor identity similarity；actor **2D IoU + 米制 z-depth error**；**camera pose error（米制）**；text alignment |
| **规范量标定**（新，模块 B） | gauge 3 通道预测误差 **vs marginal-prior baseline**；米制换算后 depth error vs 激光雷达；相机尺/LiDAR 尺一致性；checkpoint-bound metric profile 的 LiDAR AbsRel gate；paired `GS/depth` audit |

> **禁止再用 render PSNR 选择尺寸常数。** D4 已证实它对 `c_gs` 单调奖励覆盖/模糊，且对
> `c_depth` 无法回答米制正确性；render 固定 identity，metric `c_depth` 由 LiDAR gate 决定。

> ⚠ **两条口径更正（2026-08-01，A.8.5/A.8.6 + 独立复测）：**
>
> 1. **不要报裸 3D IoU / 3D center error。** 米制→DGGT 是**各向异性**映射
>    （$\mathrm{diag}(k_x,k_y,1)/s$，$k_x=0.748$），DGGT 空间里的车横向天生窄 25%，
>    裸 3D 指标会把正确结果判成差。报 **2D IoU（像素，两套约定共用）+ 米制 z-depth 误差**——
>    这一对本来就完备地确定了位置。若必须报 3D IoU，先套 `metric_box_to_dggt`。
> 2. **"三路尺子互相一致性"不再是主指标。** actor 尺仅 29/90 可用且有灾难性离群，
>    已降为诊断；主尺是 29 帧 LiDAR。相机尺仍作为移动样本上的高精度交叉验证
>    （29 帧 `0.99995 ± 0.02639`，corr 0.99296）。
>
> **camera pose error 现在才真正可比**：改造前目标混着一个 CV 23.5% 的不可观测标量，
> 这个数在不同 clip 之间没有共同单位；改成米制之后它就是米。
| 下游效用 | 在生成数据上训练/评测检测器 |

## G.2 逐环 ablation

| 环 | 实验 |
|---|---|
| 状态（D.1） | DGGT lattice **vs ScenDi 式固定 voxel latent，等算力/等数据**。唯一能把"DGGT 空间更好"从假设变成结论的实验 |
| 目标（D.3/D.4） | no HDS / L1 only / L1+L2 / all three；并报告 latent-world 相关性诊断的变化 |
| 解码可信度（模块 A） | naive HDS vs confidence-gated HDS + gate map 可视化 |
| **尺度（模块 B）** | with/without `log s`：相机平移误差分解为"尺度误差"与"轨迹误差"、米制换算精度、生成 $\hat s$ 相对 GT 与训练集先验的增益；$\hat s$ 与 $\hat z$ 隐含尺度的自洽性。`log s` 只生成，不提供外部条件路径 |
| 条件绑定（模块 C） | with/without AGB → actor **2D IoU + 米制 z-depth error**（不是裸 3D center，见 G.1 口径更正）、**跨实例泄漏** |
| 相机（B 的诊断实验） | 生成相机 vs 输入相机渲染；相机等变扫描（固定 text/asset seed 扫轨迹，测世界内容是否保持） |

## G.3 兑现"不需要第二个生成器"（最锋利的对照实验）

把**同一个**生成 Gaussian 场景从 ≥3 条未参与生成的相机轨迹**直接渲染**，不接任何 2D refiner，报告跨轨迹一致性。加强版：接一个 2D refiner，展示 FID 改善但**跨轨迹一致性变差**——直接把与 ScenDi / WorldSplat / CVD-STORM 的对照变现。

## G.4 "统一"用干扰实验证明

照抄 SceneDirector 的 Obj.Only vs Obj.+Traj. 做法：单条件 vs 四条件联合下各控制指标的退化幅度。退化可忽略才有权说条件之间不互相干扰。

## G.5 基线选择

优先 **WorldSplat、CVD-STORM、ScenDi**（同为摊销式 latent-to-Gaussian，是最直接对手），其次 **LSD-3D**（endpoint-aware 但逐场景优化，构成矛盾的另一端），再次 **InfiniCube / DriveGen3D**（第一类代表）。CVD-STORM 比较时须注明其主结果用 3 帧 reference，若要公平比 source-free 设定应用其 0-reference 配置。

---

# Part H. 表述边界与风险

## H.1 一定不能写

- 首个 text-to-3DGS / text-to-driving-GS / camera-controllable Gaussian scene / GFM feature diffusion
- 首个在生成 Gaussian 上用 render loss
- 首个在驾驶场景直接生成 3D Gaussian（ScenDi 已从 3D latent 采样后解码 3DGS）
- 无限定的 "complete gradient chain"（Envision4D 有完整 render gradient）
- "现有方法缺少 Gaussian/RGB/render loss"
- "直接用输入相机渲染"
- **不带 gauge/pullback provenance 的任何米制几何数字。** v1 已能按逐 trunk gauge 与 metric-depth loglinear 导出米制结果，但在新的 metric-gauge Scene Flow checkpoint 验收前，不能把代码能力写成模型效果
- **"相机内参可由外部指定 / 可控焦距"**。当前 FOV 是 3 维 gauge 的两个生成通道，不存在外部 FOV 条件路径；旧 CameraHead 的 −11.7°±9.7° 只解释为什么不能直接用 Waymo K 替换 teacher K
- "生成的相机已经精确复现请求相机"——9 维米制目标已消除未知尺度，但仍须由新 checkpoint 的 camera-guidance 实验验收；D3 的 ~0.2°/~1.3% 是 teacher/Waymo 控制保真度上界诊断
- "推理时无需第二个二维生成阶段"**不加限定词**（WorldFlow3D 是反例）。准确表述：*首个在带动态归属的时序驾驶 Gaussian 状态上、以世界级监督训练生成分布、且推理时不依赖第二个二维生成器的方法*——每个限定词都必须留着
- X-Scene "确定没有"世界梯度（只能写"论文未报告"）
- "直接进入三维状态这一支均为静态"（WorldSplat、CVD-STORM 有动态；这句只能限定到 LSD-3D + ScenDi）

## H.2 必须作为 scope 主动写出

- latent 仍是 **view-indexed**（frame/patch aligned），不是 canonical 3D grid
- **没有**持久 canonical 4D Gaussian field、**没有**显式 deformation field、**没有**输出级 instance identity；导出是逐帧 PLY → 不能讲 persistent world 类叙事
- 前视单相机、10 帧窗口、≤5 actors、patch grid 25×37
- ~~与 Waymo 物理世界坐标是否严格同尺度**尚未标定验证**~~ → **2026-07-31 已标定（A.8）：不同尺度。** DGGT 沿用 VGGT 的逐窗口尺度归一化，1 DGGT 单位约 25–64 米，且同一场景不同 29 帧 trunk 也会变。camera center 与 direct z-depth 在约 2.5% 离散度下近似共尺，但 FOV/横向各向异性与 tokenizer 的 paired `GS/depth=0.796` 说明**完整系统并非相似自洽**。独立复测的轨迹形状残差为 1.334%（而非无条件沿用旧 0.52%）。在经验证的新 Scene Flow checkpoint 可用前，米制数字必须同时报告 gauge 与 v1 pullback provenance。
- **相机内参只生成、不条件**：旧冻结 CameraHead 的 FOV 误差为 −11.7°±9.7°，因此当前把 teacher FOV 放进 scene-global gauge 的两个生成通道；9 维 camera state 不再含 `state[9:11]`，接口也不允许外部指定焦距
- **长序列没有一致的 GT 尺度**（A.8.7）：teacher 的尺度跨 29 帧 trunk 漂移 10–25%，因此超过一个 trunk 的滑动窗生成，其全局尺度无 GT 可比。≤29 帧不受此限
- 外部 manifest 分支未端到端接通 render/export
- raw-validation 的 asset / track / camera 来自同一 clip → 跨场景外观重组、反事实位置、显著 OOD 轨迹、prompt 组合泛化**均未证明**
- tokenizer 是本仓库对 DGGT stack 的扩展，不是公开 DGGT 自带
- 只 adapt 了 Cosmos 3 的 token 组织与 mRoPE 约定，**没有**加载 Cosmos 3 权重或 video VAE；同理未加载 RAEv2 生成权重

## H.3 三个最大风险

1. **世界反馈可能不显著。** 它是 every-2 步、$(1-\sigma)^2$ 权重、`max_samples=1` 的低占空比信号。若 ablation 差异在噪声内，headline 就空了。**前置验证：先只跑 HDS on/off 消融到能看出差异的步数，再决定是否写成 headline。**
2. **"DGGT 空间更好"缺等算力 voxel-latent 对照就只是假设。** G.2 第一行最贵但最关键。跑不动就把 D.1 降级为"使 accountability 成为可能的**充分**选择"，而非"最优空间"。
3. **软绑定。** 没有模块 C 或强隔离实验，D.2 无法防守；而模块 C 又需要模块 B 先落地才有量纲上成立的损失。
4. **规范量可能学不好（2026-08-01 更新）。** 前置验证**已完成**：D2 确认 target 本身自洽
   （LiDAR 尺 90/90 有效，逐帧 robust CV mean 0.688%；移动样本上相机尺交叉验证
   `0.99995 ± 0.02639`，corr 0.99296）。剩下的风险转移到**模型端**：gauge 必须从图像内容推断，
   而"仅由米制轨迹形状无法解析 teacher 自选的 gauge"是已确认的。
   **判据换成 marginal-prior baseline**——若 gauge 预测不显著优于"输出训练集均值"，
   说明模型只是记住了先验、没从图像里学到尺度，米制换算这条卖点就要降级为"报预测误差区间"。
   > `±4.4%` 那个"下界"说法已撤回：它只是 direct camera/depth 的一种离散度
   > （同口径两个独立估计器均给 ~2.5%），既不是信息论下界，也不是可达误差；
   > 且 tokenizer 往返另有 ~3% 系统偏移；v1 LiDAR gate 支持在 metric boundary 用
   > loglinear `c_depth` 缓解，并已冻结成当前 production artifact；它仍是 v1 专属，v2 必须重拟合，
   > 不能称为已经永久吸收。
5. **冻结解码器的 pullback 不是相似变换（2026-08-01 新增）。** 配对 `GS/depth` = 0.796，
   30/30 场景 < 1。这意味着 HDS 所在的"解码后空间"本身带系统畸变。
   v1 只对有 LiDAR 物理尺的 metric depth 采用了 checkpoint-bound production profile；`c_gs` 已因
   PSNR 目标单调病态而拒绝，0.796 缺陷仍在。根治出口是 tokenizer v2（但它不阻塞当前 Phase 1–8）；若 v2 仍失败，
   fallback 只能用会惩罚模糊的 LPIPS/SSIM，而不能再用 PSNR 选择尺寸常数。

## H.4 与最近邻方法的边界（必须正面写在 Related Work）

- **CVD-STORM**：最好的缺口证据，但也是最需要小心的近邻——它同样是驾驶、同样可 source-free、同样输出可自由渲染的动态 Gaussian、同样用 rectified flow。**唯一但决定性的差别是世界监督是否跨越 stage 边界。** 必须把这条差别写清楚，不要靠模糊表述。
- **WorldSplat**：最强 latent-to-GS 直接对手；差别在（i）中间 latent 仍是相机/像素对齐的 RGB/depth/seg 多模态 latent，(ii) 三模块独立训练，(iii) 它也用 video diffusion 精修渲染结果。
- **ScenDi**：最强 3D-first 对手；差别在（i）latent 是单目深度融合的规则彩色体素的量化压缩，受固定体积与 0.4 m voxel 限制，(ii) render loss 只训练 VQ-VAE，(iii) 高频与远景在第二阶段 2D 中产生且不写回 Gaussian，(iv) 数据剔除显著动态。
- **LSD-3D**：world feedback 确实约束 Gaussian，但逐场景 6000 步优化——**这是最清晰的一条区分线，应主动承认它有 world feedback，再指出代价。**
- **Envision4D**：有完整 render gradient 的前馈反例；贡献必须限定为"在 **controllable generative** world model 中的 world-aligned training"，绝不能泛称"首个完整梯度链"。
- **通用域 GFM 生成（GLD / OneWorld / Gen3R / PixWorld / Geometry Forcing）**：必须承认"在几何基础模型特征空间做生成"这一架构在通用域已有先例，本文的领域缺口与贡献只在驾驶场景 + 世界级监督 + 联合规约上成立。
