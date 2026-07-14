# SceneFlow 模型设计

本文档只记录 SceneFlow 的模型结构、token 设计、条件设计、训练目标和采样策略。运行命令、路径、batch size、学习率等参数放在 `docs/scene_flow_cmd.md`。

## 1. 总体结构

SceneFlow 的主模型类是 `RAEVideoSceneFlow`，对外别名仍为 `WanSceneFlow`。整体沿用 RAEv2 T2I 的高维 latent diffusion/flow 设计：冻结 DGGT aggregator 和 scene tokenizer，把图像/视频编码成高维 tokenizer latent，再训练 DiT/RAE 风格的 full-attention latent generator。

模型输入被分成两类 token：

| 类别 | token | 说明 |
|---|---|---|
| generation state | `video z_t` | 当前噪声状态，shape 为 `[B,S,P,C]` |
| generation state | `camera_gen_tokens` | pretrain 使用的 normalized 11D relative-SE(3) camera 状态 |
| generation state | `sky_gen_tokens` | pretrain 使用的 scene-level sky atlas 状态 |
| condition | timestep tokens | RAEv2 Gaussian Fourier timestep tokens |
| condition | text tokens | Qwen text encoder 输出，经线性投影进入 full attention |
| condition | camera condition tokens | 每帧 camera pose summary，或 learned null camera tokens |
| condition | asset tokens | sparse asset patch/summary tokens，或 learned null/empty asset token |
| condition | edit-control tokens | 正式训练局部编辑时由 `z_splat/scaffold/masks` 构造 |

Forward 中先拼接 generation sequence，再拼接 condition sequence：

```text
full_seq = [video, camera_gen, sky_gen, timestep, text, camera_cond, asset, edit_control]
```

时间和长视频 camera 采用 clip-global 约定：raw pretrain 先对完整 29 帧 caption clip
运行一次冻结 DGGT，再从其输出切出 10 帧 tokenizer/camera target；Gaussian timestamp 恒为
`clip_local_frame_id / 4`，不再随训练窗口长度归一化；Waymo camera condition 的
`rel pose` 始终相对 clip frame 0，`delta pose` 始终相对真实前一帧。训练随机截取
10 帧时仍携带这两个全局上下文，因此与 offline 先编码完整轨迹再切 10 帧窗口完全
一致。窗口采样先以 `camera_anchor_window_probability=0.5` 决定取 frame-0 anchor
窗口还是非零起点的 delta-only 窗口，再在非零起点中均匀采样，避免自然采样导致
anchor 稀缺。`camera_anchor_context_dropout` 默认改为 `0`，因为 delta-only 窗口本身
已经覆盖长视频后续滑窗不包含 anchor 的输入分布；该参数仅保留为额外消融开关，且
只会作用于确实包含 anchor 的窗口。
默认长视频窗口为 10 帧、stride 7，即重叠 3 帧。
训练内 validation 的局部 loss 使用同一窗口表（10 帧时起点为
`0/7/14/19`），而生成验证在完整 29 帧上按训练 `sequence_length` 做 rollout；sampler
返回完整的 global anchor mask 和 delta-only 首窗所需的 DGGT previous-C2W。所有 loss、指标、
RGB/PLY 解码入口都必须显式消费这两个值，禁止根据局部 token 0 猜测 anchor。

所有可见 token 一起做 full self-attention。模型只对 generation spans 解码：video span 走 RAEv2 DDT head；camera/sky spans 分别走独立 decoder head。

## 2. Latent 与 Flow Matching

video token 来自 DGGT image tokens 的 selected pyramid levels，经 scene tokenizer 编码成 `z_clean`，再用 SceneFlow latent stats 标准化成 `z_clean_n`。pretrain 在线从 raw DGGT Waymo 图像构造 clean latent；正式训练从 flow cache 读取局部编辑相关 latent。

训练目标使用 rectified flow：

```text
z_t = (1 - sigma) * z_clean + sigma * eps
v_gt = (z_t - z_clean) / max(sigma, t_eps)
```

这里按 RAEv2 T2I 源码实现，而不是 Cosmos 的 velocity-only target 写法：RAEv2
`Transport` 使用 `t_eps=0.05`，在训练 target 和 `prediction_type=x` 的
`x_pred -> velocity` 转换中都做 `clamp_min(t_eps)`。Cosmos3 的 `RectifiedFlow`
直接返回 `dot_x_t = eps - x0` 作为 target，因此没有这个除法和 clamp。
SceneFlow 采用 RAEv2 的 DDT/x-pred 参数化，所以训练和采样都必须共用同一个
`scene_flow.config.t_eps`。

当前正式配置使用：

```text
prediction_type = x
weighting_scheme = waver
mode_scale = 1.29
shift = 10.0
```

`prediction_type=x` 对齐 RAEv2 T2I：DDT head 直接输出 clean latent，再按 `(z_t - x_pred) / max(sigma, t_eps)` 转换成 velocity 做 flow matching loss 和 ODE 采样。`shift=10.0` 是显式工程取值，不按视频 token 总维度继续放大，避免 `S*P*C` 使 shift 过大。pretrain、正式训练、训练内 validation 和离线 inference 必须保持同一组 flow/timestep 参数。

RAEv2 的 clamped pseudo-velocity 不能在 `sigma<t_eps` 的网格点被当作精确 ODE 导数。采样入口因此要求最后一个非零时间点满足 `sigma_last>=t_eps`；对 shifted uniform grid，安全条件为 `sample_steps <= shift/t_eps-shift+1`。默认 `shift=10,t_eps=0.05` 时上限为 191 步，常用的 35/50 步均在安全区间。

pretrain 是 full-scene prior：`M_preserve=0, M_source=0, M_dest=1`，从噪声生成完整 scene latent。正式训练把两类 mask 分开使用：连续的 `M_preserve/M_source/M_dest` 只表达几何覆盖和编辑语义；flow state 使用由 `M_source+M_dest` threshold 后再在 patch grid 上 dilation 得到的二值 `H_edit`。默认 `threshold=1e-4`、`dilation=1`，并由 train、训练内 validation 和 offline inference 共用同一个 helper。

正式阶段定义唯一的 conditional flow path：

```text
x_target = H_edit * z_clean + (1 - H_edit) * z_splat
z_sigma  = H_edit * ((1 - sigma) * z_clean + sigma * eps)
         + (1 - H_edit) * z_splat
v_gt     = H_edit * (z_base_sigma - z_clean) / max(sigma, t_eps)
```

采样只在初始化时把 noise 与 `z_splat` 组合；每步只更新 `H_edit` 子空间，再用同一张二值 mask hard-project。二值投影是幂等的，不允许把 soft coverage 当作每步 `alpha*z+(1-alpha)*z_splat` 的投影，否则结果会随采样步数按 `alpha^N` 收缩。preserve loss 使用 `1-H_edit`，boundary ring 位于 dilation 后的 `H_edit` 内，因此不会再和 clean/boundary target 冲突。

正式 checkpoint 写入 `formal_flow_domain_version=hard_binary_edit_domain_v1`。resume 和 offline inference 会拒绝缺少该标记的旧正式 checkpoint，避免把旧 soft-path 训练权重与新 sampler 静默混用；需要从 pretrain 权重重新进行 formal fine-tuning。

## 3. Timestep 注入

模型使用 RAEv2 的 Gaussian Fourier timestep embedder：

- `t_seq`：learned timestep tokens，作为 condition token 进入 full attention。
- `t_base`：DDT head 的 AdaLN conditioning。

这里不使用 Cosmos 的 noisy-token additive timestep 注入。连续时间变量仍是 `sigma in [0,1]`，`sigma=1` 接近纯噪声，`sigma=0` 接近 clean。

## 4. Text 编码

文本条件使用 `TextEncoder` 包装的 Qwen text encoder。`encode_text_condition` 返回：

```text
text_tokens: [B,T,qwen_dim]
text_attention_mask: [B,T]
```

SceneFlow 内部对 text 做 RMSNorm + Linear projection，再加 `text_modality_embed`。text token 的 RoPE position 为全零，这一点对齐 RAEv2：文本作为语义条件 token 参与 full attention，但不使用视频空间 RoPE。

训练 CFG dropout 时，text 不会被置为 `None`，而是替换为空 prompt `""` 的 text tokens，这样 conditional/unconditional 分支的文本编码路径一致。

## 5. Video State Embedding

DDT 的视觉 embedder 只吃当前噪声 latent `z_t`，不再吃旧实验中的 packed `[z_t, z_splat, scaffold, masks]`。局部编辑上下文作为 edit-control condition token 进入 full attention。

Edit-control 使用按 model forward 计的两级稀疏预算：每帧上限
`max_control_tokens_per_frame=128`，整个窗口上限 `max_control_tokens=1024`。
预算根据当前样本中具有 edit support 的 active frame 动态做 integer max-min fair
分配；support 较少的帧释放出的预算会重新分配，最后不足一轮的余量沿时间轴均匀
散布，不再把 T-major 展平后的前 1024 个 token 直接截断。默认 10 帧窗口在
所有帧均 active 时每帧得到 102 或 103 个 token；直接 29 帧 forward 时每帧得到
35 或 36 个。每帧内部仍使用确定性的空间均匀采样，因此同一滑窗在各 ODE step
使用一致的 control token 与 Cosmos mRoPE 位置。长视频滑窗的 1024 总预算按窗口
重新计算，不随完整视频长度增长而持续压缩。

每个 video token 额外注入 6 维 per-token state：

```text
[preserve, source, dest, edit, keep, sigma_eff]
```

其中 `preserve/source/dest/edit/keep` 仍保留连续覆盖语义，但正式阶段的 `sigma_eff = sigma * H_edit`，必须精确反映实际加噪域；pretrain 的全场景 `H_edit=1`。这样 preserve 区域在表示上明确是条件区域，edit 区域明确是需要生成的区域。

## 6. Cosmos-style mRoPE

SceneFlow 使用 Cosmos-style 3D mRoPE。当前正式从头训练配置：

| 模块 | head 配置 | mRoPE section | theta |
|---|---|---|---|
| encoder full attention | `20 heads x 72 head_dim` | `(12,12,12)` | `5e6` |
| DDT head | `16 heads x 128 head_dim` | `(24,20,20)` | `5e6` |

位置约定：

| token | position |
|---|---|
| video token | `(t, y, x)`，使用 patch grid |
| asset patch token | 对应 video patch 的 `(t, y, x)` |
| asset summary token | 对应 asset 覆盖区域的 `(t, mean_y, mean_x)` |
| asset null/empty token | `(0,0,0)`，表示 asset 条件缺失或显式 empty 目标 |
| camera condition token | 每帧 `(t, H//2, W//2)` |
| camera generation token | 每帧 `(t, H//2, W//2)` |
| text token | `(0,0,0)` |
| timestep token | `(0,0,0)` |
| sky token | `15000 + 8 * (dir_x, dir_y, dir_z)`，使用 scene-level 上半球方向 |

SceneFlow 不再暴露全局 `mrope_temporal_margin`。当前写死 A3 坐标设计，并在 checkpoint 的 `scene_flow_config` 中记录 `rope_layout_version=a3_camera_center_spherical_sky15000`：

- text/timestep 保持 RAEv2-style zero RoPE；文本顺序已经由 Qwen hidden states 表达，不额外注入视频空间坐标。
- video、asset、edit-control 共享真实视频 `(t,y,x)`，保留局部编辑所需的 patch 对齐归纳偏置。
- camera 是全局每帧几何条件，temporal 与对应 video frame 对齐，spatial 放在 patch grid 中心，避免把相机条件绑定到左上角 patch。
- sky 是 scene-level directional atlas，不是 image-plane patch；将上半球方向映射为以 `15000` 为中心的三轴 Cartesian RoPE 坐标。经度首尾方向因此在位置空间天然相邻，不使用 seam loss；同时它仍与 video、asset、edit-control、camera 的 `[0,15000)` 时间轴分离。`rope_max_position=16384` 会对越界位置 fail-fast。
- 旧 A1/A2 checkpoint 或仍记录全局 `mrope_temporal_margin` 的 checkpoint，其 sky 位置语义与 A3 不一致，不应直接续训、warm-start 或推理。

`frame_ids` 和 `fps` 可让 video/asset/camera temporal position 做 Cosmos 风格的时间缩放；默认路径使用固定 sequence order。

## 7. Asset Token 设计

Pretrain 的 asset 条件来自 clean latent 和 `dynamic_mask` 构造的 pseudo asset slots。每个场景最多保留 5 个 asset slot；每个 asset 每帧最多采样 32 个 patch token，并额外构造 summary token。

正式训练的真实 asset patch mask 固定使用目标位置 isolated render 的 `A_asset`：先以
`alpha >= 0.05` 二值化，再按与 DGGT patch 完全一致的非重叠 cell 做 max pooling；不使用
bbox、不使用联合 `M_dest`、不做 dilation。该 mask 与 `phase1_coverage` 和选中 slot 取交集。
cache 记录 `asset_patch_mask_version=alpha_max_t005_v1`，每个 asset 的轻量 mask 为 `[S,P]`；
缺失该字段的旧 Mode-A cache 必须拒绝。Scene tokenizer 仍先编码完整 isolated asset LUT，
随后才用 mask 做 sparse token 选择，这与 pretrain 的“先得到 clean latent、再按 object mask
选择 token”一致。`asset_position_mode=localized` 时 patch 和 summary 分别使用目标位置
`(t,y,x)` 与覆盖区域质心，保留用户指定目标位置的控制语义。

asset 使用四种互不等价的原语。统一简称为 `PAD / NULL / EMPTY / REAL`：

| 名称 | 含义 | 运行时表示 | attention 可见性 |
|---|---|---|---|
| `PAD`（容量空槽） | 固定容量 5 中未被真实 asset 占用的 slot；真实 asset 内未覆盖的帧/patch 也属于局部 padding | token 值补零，mask=false；它不是独立的 `asset_condition_kind` | 不可见 |
| `NULL`（条件缺席） | 样本本来存在可用 asset 条件，但用户未输入，或训练时对 asset 做 CFG dropout | `asset_condition_kind="asset_uncond"`，使用 1 个 learned `asset_null_condition_embed` | 仅该 learned token 可见 |
| `EMPTY`（显式空目标） | 用户明确要求删除目标，即 Mode-B deletion/empty；这是一个有效控制条件，不是缺少条件 | `asset_condition_kind="mode_b_empty"`，使用 1 个 learned `empty_asset_embed` | 仅该 learned token 可见 |
| `REAL`（真实资产） | 用户输入一个或多个 asset | `asset_condition_kind="mode_a"`，使用 sparse asset patch/summary tokens | 仅真实覆盖 token 可见 |

`PAD` 是张量批处理/稀疏化状态，`NULL` 和 `EMPTY` 是两个不同的可学习语义 token，`REAL` 是数据 token，四者不能互换。具体边界如下：

- 样本有 1--4 个真实 asset 时，kind 仍是 `mode_a`；未占用 slot 是 `PAD`，不会为每个空 slot 插入 learned token。
- 样本自然有 0 个 dynamic asset 时，kind 是 `none`，5 个 slot 全是 `PAD`，不插入 learned token。这是 `PAD` 的零资产边界，不等于 `NULL`。
- 用户主动省略本来可用的 asset 条件，或者训练 CFG 主动丢弃 asset 时，才使用 `asset_uncond`/`NULL`。
- pure deletion 使用 `EMPTY`；combined edit 使用 `mode_a_with_empty`，即 `REAL + EMPTY`。它是两种原语的组合，不是第五种原语，并且必须同时保留真实 asset token 和一个可见的 `empty_asset_embed`。
- real asset 内没有目标覆盖的帧/patch 同样 mask=false；这是 token 级 padding，不改变样本的 `mode_a` 条件语义。

`asset_null_condition_embed` 只插入 1 个 visible token。它表示“asset 条件这个模态未提供”，不是 5 个 asset slot 分别为空。optional-condition resolver 必须保留自然零资产的 `none`，只能把用户省略或 CFG dropout 的行标成 `asset_uncond`；不能仅凭 asset mask 全 false 就把 `none` 重写为 `asset_uncond`。

正式训练不启用 optional asset condition；局部编辑时 asset 条件必须由用户/样本给定。训练 dropout
和 CFG 的 no-asset 分支仍使用 `asset_uncond` learned null，这是为了给 guidance 一个稳定的
无 asset 基线，不表示正式推理接口允许缺失 asset 输入。

## 8. Camera Token 设计

Camera 有两条不同链路：

1. **camera condition tokens**：用户输入或 GT 模拟用户输入的 per-frame pose summary。
2. **camera generation tokens**：pretrain 中和 video 一起生成的显式 11D camera trajectory 状态。

camera condition summary 维度是 `CAMERA_POSE_SUMMARY_DIM`，当前为 20D。pretrain 和正式训练主链路都必须从 Waymo `camera_to_world_corrected + intrinsics` 构造该摘要，每帧一个 token，经 projection 后以 `(t,H//2,W//2)` RoPE position 进入 full attention。DGGT-space `pose_enc` 只能作为 pretrain camera generation target、pose loss target 或渲染相机来源，不能回灌成 SceneFlow 的 camera condition。

用户推理不提供 camera，或训练时 dropout camera condition 时，不传零姿态；使用：

```text
camera_condition_kind = "camera_uncond"
```

模型会为每帧插入一个 learned `camera_null_condition_embed`，mask=true，position 仍是 `(t,H//2,W//2)`。

camera condition 的版本固定为
`waymo_rel_delta_rot6d_fov20d_direct_v2`。camera generation 的版本固定为
`dggt_relative_se3_rot6d_logfov_v3`，它只能由冻结 DGGT CameraHead 的
`pose_enc_dggt=[t_w2c,q_xyzw,FOVy,FOVx]` 构造：先求 `W2C` 与 `C2W`，第 0 帧
取绝对 `C2W[0]`，后续帧取 `inv(C2W[t-1]) @ C2W[t]`，再写成每帧 11D
`[translation(3), rotation-6D(6), log(tan(FOVx/2)), log(tan(FOVy/2))]`。
Waymo camera 不做坐标转换，也不能进入这个入口。解码必须显式传全局 anchor mask；
零向量或共线 rotation-6D 也稳定投影为有限、右手、`det=+1` 的 SO(3) 矩阵。

全局 `camera_gen_anchor_mask` 只在 clip 第 0 帧为 true。滑窗只能切片这个 mask，
不能把窗口首帧提升成 anchor；因此相对轨迹始终在窗口融合结束后一次性积分。
delta-only 窗口的第一个 camera token 是相对全局前一帧的真实 delta；几何/RGB loss
只在解码 loss 坐标时使用该前一帧的冻结 DGGT `C2W` 作为积分初值，它不进入模型条件，
也不会把局部首 token 变成 anchor。训练中若包含 anchor 的窗口通过
`camera_anchor_context_dropout` 隐藏了 anchor，该行仍监督 camera delta flow，但从依赖
完整绝对轨迹的 RGB render loss 中排除。`rgb_render_max_samples` 在排除这种行之后才
生效，若无有效行则仅跳过当次 RGB render loss。
anchor 与 delta 分别使用 11D per-channel mean/std，stats 版本为
`dggt_camera_anchor_delta_per_channel_v4_global_context`，std 下限 `1e-4`。camera stats
始终统计完整 29 帧 DGGT 输出，而不是把随机窗口首帧重复计为 anchor。stats 同时记录 target
space/source、DGGT checkpoint SHA256 与 anchor/delta count；缺失、非有限、旧版本或
checkpoint hash 不匹配会直接终止。camera condition/generation 在统计归一化后只经过
保幅值的 `ChannelScale -> Linear`，不会被 per-token RMSNorm 消除尺度。训练包含 normalized-state flow loss，以及完整轨迹上的绝对
translation/SO(3) geodesic/log-FOV、相邻 relative pose、与 GT 二阶变化残差损失。

Waymo condition 的 FOV 使用主点感知公式 `atan2(cx,fx)+atan2(W-cx,fx)`（y 轴同理）。
DGGT generation target 的 FOV 直接读取 CameraHead `pose_enc`。原图尺寸
`raw_image_size_hw` 只对 Waymo condition 是必需元数据，禁止用 `2*cx/2*cy` 猜尺寸。

渲染用 camera 和 SceneFlow 输入 camera condition 是两个概念。即使用户不提供 camera condition，RGB render 仍需要 camera；它可以来自用户/default/generated camera，但不能在无条件采样时把 render camera 偷偷回灌成模型条件。

正式训练、训练时 validation 与正式 offline inference 始终给定同一 Waymo 20D condition，
永远不传 camera generation token。渲染相机始终由输入图像经冻结 DGGT CameraHead 得到；
edited/generated token 不重新预测或覆盖相机。

全局 text CFG 只作用于 video/sky/mask。camera generation 使用独立
`camera_text_guidance_scale`（默认 1），而 `camera_guidance_scale` 仍只缩放 camera
condition residual。因此扫描全局 CFG 1/2/4 不会把 11D camera 状态外推。

## 9. Sky Token 设计

sky generation 只属于 full-scene pretrain。sky token 是 scene-level directional atlas token：先构造 `32x64` RGB atlas，再用固定 `2x2` pixel-unshuffle 打包为默认 `16x32` grid，每个 token 12 维：

```text
2x2 atlas patch x [r, g, b]
```

训练时从输入图像、sky mask 和 DGGT-space camera 构造 sky token target。sky mask 只用于构造 RGB target 和逐输出通道的 `sky_gen_loss_weight [B,512,12]`，绝不能作为模型输入 attention mask；训练和开放推理都 pack 完整 `16x32` sky atlas。每个 atlas cell 对应上半球一个无穷远方向 bin；target 投影只使用 DGGT world-to-camera rotation，不使用 translation，和 renderer 的 camera-ray 环境贴图定义严格互逆。该方向投影到各帧后，在 GT sky mask 内采样 RGB，并选择置信度最高的可见帧，避免跨帧位姿误差模糊纹理。未观测方向使用带经度环绕的球面邻域补全，且只按默认 `0.05` 低权重监督；整段完全没有有效 sky 观测时仍为零权重。visibility 使用和 RGB 完全相同的 pixel-unshuffle 顺序打包，因此一个 token 内未观测的子像素不会因相邻子像素可见而被错误监督；可用 `--sky_unobserved_loss_weight` 调整低权重。

pretrain 采样时，sky token 和 video/camera 一起作为 generation state 更新；RGB validation 使用 generated sky directional atlas，并按 generated DGGT camera 逐帧渲染 sky background。正式训练、训练内采样和离线 inference 都不 pack generated sky token，也不计算 `loss_sky_flow`；正式编辑在图像空间执行 `GT_sky_mask * input_GT_RGB + (1-GT_sky_mask) * rendered_edit`，天空像素严格来自原始输入，不再经过 sky model 或窗口级 min-max。

## 10. Sky Mask 生成与 Refine

open-generation RGB render 不再使用 DGGT/VGGT 的 `semantic_head` 输出作为 sky mask。SceneFlow 自己预测 sky mask，避免未训练或域外 semantic head 在纯生成 latent 上给出全黑/错误天空区域。

sky mask 不新增 generation token：

| 输出 | shape | 用途 |
|---|---|---|
| `sky_mask_logits` | `[B,S,P,1]` | patch-grid 辅助监督和诊断 |
| `sky_mask_refined_logits` | `[B,S,1,Hr,Wr]` | refined dense mask，默认 `Hr=patch_h*4, Wr=patch_w*4`，用于 RGB render |

因此 sky mask 没有独立 RoPE 或 position id。它直接从 full-attention trunk 后的 video hidden 解码，每个 patch mask 继承对应 video token 的 `(t,y,x)` 3D mRoPE 语义位置。refined dense decoder 只在 trunk 后做卷积上采样，不改变 full-attention 序列长度。

mask decoder 输入是：

```text
sky_mask_cond = silu(enc_video + t_base + sky_context)
```

其中 `enc_video` 是经过 video/camera/sky/text/asset/control 全注意力交互后的 video span；`sky_context` 是 visible `sky_hidden` 的 masked mean，并 broadcast 到每个 video patch。这样 mask loss 不仅监督 video hidden，也会通过 trunk 和 pooled sky context 回传到 sky token 分支。refined decoder 还把 `base_feat` 作为 patch-grid skip feature，提供较早层的空间定位信息。

patch head 是轻量 MLP：

```text
RMSNorm -> Linear -> SiLU -> Linear(1)
```

refined head 是轻量 DeepLabv3+/U-Net 风格 dense decoder：

```text
[B,S*P,H] -> RMSNorm/Linear -> [B*S,C,patch_h,patch_w]
+ CoordConv(y,x)
+ base_feat skip
-> depthwise-separable residual conv
-> bilinear upsample x2 + depthwise-separable residual conv
-> bilinear upsample x2 + depthwise-separable residual conv
-> 1x1 conv -> [B,S,1,patch_h*4,patch_w*4]
```

相关设计参考：

- PointRend 把 segmentation 边界视为渲染/采样分辨率问题，说明低分辨率 mask 直接上采样会在边界过平滑；本实现不引入逐点 iterative sampler，但把监督分辨率提高到 patch grid 的 4 倍，并单独加 boundary band loss。
- DeepLabv3+ 使用简单 decoder 逐步恢复空间信息并强化 object boundary；本实现采用同样的 decoder-side refinement 思路，但保持输入来自 SceneFlow trunk 的 video token grid。
- SegFormer 证明 transformer encoder/trunk 后接轻量解码头是有效且高效的 segmentation 设计；本实现同样不向主序列新增高分辨率 mask token，而是在 trunk 后做轻量 dense decode。

参考链接：[PointRend](https://arxiv.org/abs/1912.08193), [DeepLabv3+](https://arxiv.org/abs/1802.02611), [SegFormer](https://arxiv.org/abs/2105.15203)。

使用 `bilinear upsample + conv`，不使用 transposed convolution，避免 checkerboard。默认 refined 输出为 `100x148`（当 patch grid 是 `25x37`），render 前再 bilinear resize 到固定模型渲染尺寸 `350x518`。这样 boundary 粒度从约 `14px` 提升到约 `3.5px`，但不引入 full-resolution mask token。

训练目标来自同一份 GT sky mask：

```text
sky_mask_clean         = area/adaptive pool GT sky mask -> [B,S,P,1]
sky_mask_refined_clean = area/adaptive pool GT sky mask -> [B,S,1,patch_h*4,patch_w*4]
```

patch loss：

```text
L_patch = BCEWithLogits(patch_logits, patch_target, pos_weight)
        + dice_weight * DiceLoss(sigmoid(patch_logits), patch_target)
```

refined loss：

```text
L_refine_region = BCEWithLogits(refined_logits, refined_target, pos_weight)
                + dice_weight * DiceLoss(sigmoid(refined_logits), refined_target)

boundary_band = dilate(refined_target > 0.5) - erode(refined_target > 0.5)
L_boundary = mean_BCE_on_boundary_band(refined_logits, refined_target)

L_sky_mask = lambda_sky_mask * L_patch
           + lambda_sky_mask_refine * (L_refine_region
             + sky_mask_refine_boundary_loss_weight
             * sky_mask_refine_boundary_weight
             * L_boundary)
```

默认权重：

```text
lambda_sky_mask = 0.05
lambda_sky_mask_refine = 0.10
sky_mask_dice_weight = 0.5
sky_mask_pos_weight_max = 10.0
sky_mask_refine_boundary_weight = 4.0
sky_mask_refine_boundary_loss_weight = 0.25
```

`pos_weight` 按当前 batch 的 sky/non-sky 比例动态计算并 clamp，避免天空正类比例低时被 BCE 淹没。Dice 约束区域重叠，boundary BCE 专门提高地平线、树冠、电线杆等边界附近的监督强度。

sky mask head 首先在和 video clean prediction 相同的随机 `sigma` denoising forward 上接受辅助监督。为了让最终渲染所用的 mask 与生成完成后的 clean scene 对齐，训练从 `sky_mask_endpoint_start_step` 开始，按 `sky_mask_endpoint_every` 周期把当前预测的 clean video、camera 和 sky state `detach` 后再做一次 `sigma=0` mask-only endpoint forward，并对 endpoint mask 施加同一份 GT mask 监督。RGB render 若在非 endpoint-supervision step 激活，也会按依赖关系执行这个 clean endpoint，但不会额外改变 `sky_mask_endpoint_every` 所定义的 BCE/Dice endpoint 监督频率。

采样时完成全部 ODE 更新后，对最终 clean video/camera/sky state 做一次 `sigma=0` mask-only endpoint forward；普通采样和滑动窗采样都只从该 endpoint 读取 `sky_mask_logits` 和 `sky_mask_refined_logits`。启用 factored CFG 时，endpoint 复用与生成完全相同的 text/asset/camera 分支及 scale，再对组合后的 logits 做 sigmoid。不能从最后一个非零去噪步读取最终 mask：在默认 `shift=10` 下，最后一个非零 `sigma` 仍可能较大，并不代表最终 clean scene。RGB render 优先使用 refined mask；如果只有 patch mask，则回退到 patch mask 上采样。render 在 `_render_gs_map_rgb` 内把 sky mask hard-threshold 成 non-sky Gaussian 选择，这与 DGGT 用 GT sky mask 排除 sky Gaussian、再由 rasterizer transmittance 合成背景的定义一致；mask target 不应替换成 renderer alpha。

## 11. DDT Head 与输出头

video span 通过 RAEv2 DDT decoder head 输出：

- `decoder_video_embed(z_t)` 作为 DDT 输入。
- `t_base` 作为 DDT AdaLN condition。
- final layer zero-init，训练初期输出稳定。

camera 和 sky 不走 video DDT final layer，而是使用各自的 lightweight decoder：

```text
camera_gen_decoder(hidden) -> camera_gen_dim
sky_gen_decoder(hidden) -> sky_token_dim
```

sky mask 也不走 DDT head。patch mask 使用 lightweight MLP decoder；refined mask 使用 trunk 后 dense conv decoder。DDT head 只服务高维 video latent 预测。

`base_final_layer` 是辅助 base prediction path，用于 `base_model_coeff` 相关训练项。

## 12. Attention Mask 与空 Token

所有 condition 和 generation token 都有显式 valid mask。无效 token：

- 不能作为 key/value 被其他 token 关注。
- query 输出会被 mask 置零。

空 token 语义必须严格区分：

| 情况 | token 可见性 |
|---|---|
| `PAD` capacity/patch padding | mask=false |
| natural zero-asset sample (`none`) | 5 个 slot 全为 `PAD`，无 visible asset token |
| `NULL` asset condition missing (`asset_uncond`) | 1 个 visible `asset_null_condition_embed` |
| camera condition missing | 每帧 visible `camera_null_condition_embed` |
| `EMPTY` explicit deletion/empty target | 1 个 visible `empty_asset_embed` |
| `REAL` asset condition | sparse patch/summary token mask=true，其余位置为 `PAD` |
| combined edit (`mode_a_with_empty`) | `REAL` token 与 1 个 `EMPTY` token 同时可见 |

这保证模型能区分“用户没有给条件”和“场景/任务本身为空”。

## 13. CFG 与 Optional Condition

Pretrain 支持 text、asset、camera 三类可选条件。训练时按样本独立 dropout：

```text
text_uncond_drop_prob
asset_uncond_drop_prob
camera_uncond_drop_prob
all_cond_drop_prob
```

dropout 只隐藏输入条件，不改变 video/camera/sky 的 clean target 和 loss。

Pretrain 默认 text CFG 对齐 Cosmos conditional generation：

```text
v_full         = text      + asset + camera
v_no_text_full = null text + asset + camera

v = v_full + (text_scale - 1) * (v_full - v_no_text_full)
```

因此 text conditional/unconditional 两个分支拥有完全相同的 asset 与 camera，仅文本不同。显式设置独立 control scale 时，再加入分解残差：

```text
v_text       = text + asset_null + camera_null
v_text_asset = text + asset      + camera_null

v += (asset_scale  - 1) * (v_text_asset - v_text)
v += (camera_scale - 1) * (v_full       - v_text_asset)
```

对应 CLI：

```text
--guidance_scale
--asset_control_guidance_scale
--camera_guidance_scale
```

三个 scale 默认都是 `1.0`，表示 no-op。pretrain 推理允许用户不输入 asset 或 camera 条件；采样端必须依据显式 condition kind，而不是只依据 valid mask 判断 optional condition：用户省略 asset 时使用 `asset_uncond`/`NULL`，自然零资产样本的 `none` 必须保持为零 visible token 的条件态。只有 `asset_uncond` 才表示 asset 模态缺席并使对应 scale 退回 `1.0`；`none` 与 `mode_b_empty` 都是已提供的有效条件语义。camera 整批缺失时对应 scale 强制退回 `1.0`。`asset_control_guidance_scale` 在正式训练中仍控制 asset + edit-control guidance；pretrain full-scene 没有局部 edit-control token，但保留同名参数以对齐两阶段采样接口。

默认的 CFG sweep 只改变 `text_scale`，asset/camera scale 保持 `1.0`。这与 Cosmos conditional generation 的两分支设计一致：clean visual/structural condition 在 conditional 与 text-unconditional 分支中都保留，CFG 只放大上下文相关的文本残差。同时放大三个 scale 会额外外推目标外观与相机轨迹，不应作为普通 `cfg2/cfg4` 的默认含义；需要研究控制强度时可显式单独设置 asset/camera scale。

正式训练不设计 optional asset/camera condition，因为正式训练是局部目标编辑，asset 条件和 camera
条件必须由用户/样本给定。正式训练里的 `asset_uncond` 只用于训练 dropout 和 CFG 分支，不作为用户
可省略 asset 的接口语义。

## 14. Validation 与 RGB Render

pretrain validation 采样从纯噪声同步生成：

```text
scene latent + camera_gen token + sky token
```

RGB render 不再把 GT image 送入 DGGT aggregator，也不再使用 validation batch 的 image-token 模板。SceneFlow 输出的 latent 经 tokenizer decode 成 selected DGGT patch levels；selected levels 的 special tokens 固定补零后送入 DGGT heads。render 尺寸由 `patch_grid * 14` 固定得到。

训练 RGB loss 与 validation/offline 共用同一个 generated-token DGGT 解码入口：`latent -> tokenizer.decode -> frozen depth/GS/instance heads`。frozen head 参数不更新，但解码过程不能放在 `no_grad` 中；generated depth 反投影与 point means 不允许 detach。主 RGB renderer 的接口不接受 teacher depth。

pretrain RGB 使用 `z_camera_pred -> denormalize_camera -> decode_camera_trajectory -> DGGT pose_enc`，并使用 generated sky atlas 与 predicted refined sky mask。正式训练固定使用输入图像的 DGGT camera 和 GT sky/sky mask。Waymo camera 始终只作为 SceneFlow condition。pretrain 的 GT sky mask 只作为 RGB/LPIPS 权重，renderer 的 sky split 使用 predicted mask，避免训练时 teacher-mask 捷径。

gsplat 在无 background 时返回 premultiplied RGB，因此合成公式固定为 `rendered_rgb + (1-alpha)*background`，禁止再次乘 alpha。LPIPS 使用 spatial 输出并应用与 Charbonnier 相同的 edit/sky 权重；正式训练 `sky_weight=0` 时天空不参与 LPIPS。

生成分支的 sky/non-sky split 来自 SceneFlow refined sky mask，而不是 `semantic_head`。generated sky atlas 只提供 sky background RGB；refined mask 决定哪些 image-plane pixels/points 作为非天空 Gaussian 参与渲染。

pretrain validation/offline 的 generated branch 必须使用 SceneFlow 生成的 DGGT camera；只有 clean/tokenizer-reconstruction 诊断分支可以使用 frozen DGGT teacher pose。任何 render camera 都不应在 optional camera condition 缺失时回灌成模型输入条件。

正式训练 validation 和离线 inference 不使用 generated camera/sky token。生成分支固定复用输入图像经 DGGT 预测出的 DGGT camera，并在 GT sky mask 内逐像素保留原始输入 RGB；非天空区域只使用 edited 3DGS render，避免把整张 GT 图作为 background 后泄漏原场景。该 DGGT camera 必须来自 cache 中对完整 29 帧输入一次性运行 CameraHead 得到的 `pose_enc`；10 帧训练 validation 或 `--render_per_window` 只按原始帧索引切片，禁止在局部窗口重新运行 CameraHead，否则窗口之间的上下文依赖会导致位姿不连续。`generated_pred_sky_mask` 是 edited latent 送入 DGGT `semantic_head` 得到的诊断图，不参与 sky/non-sky 合成，也不是 SceneFlow sky mask 输出。

## 15. 与运行参数文档的边界

本文件维护模型设计。以下内容放在 `docs/scene_flow_cmd.md`：

- 环境变量和路径。
- pretrain / 正式训练命令。
- batch size、学习率、warmup、EMA、validation 频率等运行参数。
- flow cache manifest 构建和转换命令。
