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

`prediction_type=x` 对齐 RAEv2 T2I：DDT head 直接输出 clean latent，再按 `(z_t - x_pred) / sigma` 转换成 velocity 做 flow matching loss 和 ODE 采样。`shift=10.0` 是显式工程取值，不按视频 token 总维度继续放大，避免 `S*P*C` 使 shift 过大。pretrain、正式训练、训练内 validation 和离线 inference 必须保持同一组 flow/timestep 参数。

pretrain 是 full-scene prior：`M_preserve=0, M_source=0, M_dest=1`，从噪声生成完整 scene latent。正式训练是 masked local edit：edit 区域加噪并监督 velocity，preserve 区域输入和采样都 clamp 到 `z_splat`。

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

每个 video token 额外注入 6 维 per-token state：

```text
[preserve, source, dest, edit, keep, sigma_eff]
```

其中 `sigma_eff = sigma * edit`。这样 preserve 区域在表示上明确是条件区域，edit 区域明确是需要生成的区域。

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
| sky token | `(128, sky_y, sky_x)`，使用 scene-level sky atlas grid |

SceneFlow 不再暴露全局 `mrope_temporal_margin`。当前写死 A1 坐标设计，并在 checkpoint 的 `scene_flow_config` 中记录 `rope_layout_version=a1_camera_center_sky128`：

- text/timestep 保持 RAEv2-style zero RoPE；文本顺序已经由 Qwen hidden states 表达，不额外注入视频空间坐标。
- video、asset、edit-control 共享真实视频 `(t,y,x)`，保留局部编辑所需的 patch 对齐归纳偏置。
- camera 是全局每帧几何条件，temporal 与对应 video frame 对齐，spatial 放在 patch grid 中心，避免把相机条件绑定到左上角 patch。
- sky 是 scene-level directional atlas，不是 image-plane patch；用独立 temporal offset `128` 与视频帧范围分离，但不采用 Cosmos3 `15000` 的 packed-segment大间隔。
- 旧 checkpoint 如果仍记录全局 `mrope_temporal_margin`，其 camera/sky 位置语义与 A1 不一致，不应直接续训、warm-start 或推理。

`frame_ids` 和 `fps` 可让 video/asset/camera temporal position 做 Cosmos 风格的时间缩放；默认路径使用固定 sequence order。

## 7. Asset Token 设计

Pretrain 的 asset 条件来自 clean latent 和 `dynamic_mask` 构造的 pseudo asset slots。每个场景最多保留 5 个 asset slot；每个 asset 每帧最多采样 32 个 patch token，并额外构造 summary token。

asset 有三种不同语义，不能混用：

| 语义 | 表示 |
|---|---|
| 真实 asset 条件 | sparse asset patch/summary tokens，可见 mask=true |
| 场景没有 dynamic asset | `asset_condition_kind="none"`，asset mask 全 false，不插 learned token |
| 用户未提供 asset / 训练 dropout asset | `asset_condition_kind="asset_uncond"`，插入 1 个 learned `asset_null_condition_embed` |
| 显式空目标/删除洞/Mode-B empty | `empty_asset_embed`，表示 conditional empty target |

`asset_null_condition_embed` 只插入 1 个 visible token。它表示“asset 条件这个模态未提供”，不是 5 个 asset slot 分别为空。padding token mask=false，不参与 attention。

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
anchor 与 delta 分别使用 11D per-channel mean/std，stats 版本为
`dggt_camera_anchor_delta_per_channel_v3`，std 下限 `1e-4`。stats 同时记录 target
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

sky generation 只属于 full-scene pretrain。sky token 是 scene-level directional atlas token，默认 grid 为 `16x32`，每个 token 3 维：

```text
[r, g, b]
```

训练时从输入图像、sky mask 和 DGGT-space camera 构造 sky token target。sky mask 只用于构造 RGB target 和 per-token `sky_gen_loss_weight`，绝不能作为模型输入 attention mask；训练和开放推理都 pack 完整 `16x32` sky atlas。每个 token 对应上半球一个无穷远方向 bin；target 投影只使用 DGGT world-to-camera rotation，不使用 translation，和 renderer 的 camera-ray 环境贴图定义严格互逆。该方向投影到各帧后，在 GT sky mask 内采样 RGB 并跨可见帧平均。低覆盖 atlas cell 保留 fallback RGB，loss weight 使用较小值（默认 `0.05`），避免伪 target 主导训练。

pretrain 采样时，sky token 和 video/camera 一起作为 generation state 更新；RGB validation 使用 generated sky directional atlas，并按 generated DGGT camera 逐帧渲染 sky background。正式训练、训练内采样和离线 inference 都不 pack generated sky token，也不计算 `loss_sky_flow`；正式编辑渲染保持 GT sky mask + sky model 背景。

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

sky mask head 和 video clean prediction 在同一个随机 `sigma` denoising forward 上训练。采样时从最后一个**非零、训练分布内**的去噪步直接读取 `sky_mask_logits` 和 `sky_mask_refined_logits`；启用 factored CFG 时，对 mask logits 使用同一组分支和 scale，再做 sigmoid。不能在采样结束后对 clean state 额外做 `sigma=0` forward，因为训练 timestep 不覆盖该输入面。RGB render 优先使用 refined mask；如果只有 patch mask，则回退到 patch mask 上采样。render 在 `_render_gs_map_rgb` 内把 sky mask hard-threshold 成 non-sky Gaussian 选择，这与 DGGT 用 GT sky mask 排除 sky Gaussian、再由 rasterizer transmittance 合成背景的定义一致；mask target 不应替换成 renderer alpha。

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
| padding | mask=false |
| no dynamic asset | 无 visible asset token |
| asset condition missing | 1 个 visible `asset_null_condition_embed` |
| camera condition missing | 每帧 visible `camera_null_condition_embed` |
| explicit empty asset | visible `empty_asset_embed` |

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

三个 scale 默认都是 `1.0`，表示 no-op。pretrain 推理允许用户不输入 asset 或 camera 条件；采样端会逐行检测有效 token，空行改为 `asset_uncond`/`camera_uncond`，整批缺失某类条件时对应 scale 强制退回 `1.0`。`asset_control_guidance_scale` 在正式训练中仍控制 asset + edit-control guidance；pretrain full-scene 没有局部 edit-control token，但保留同名参数以对齐两阶段采样接口。

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

生成分支的 sky/non-sky split 来自 SceneFlow refined sky mask，而不是 `semantic_head`。generated sky atlas 只提供 sky background RGB；refined mask 决定哪些 image-plane pixels/points 作为非天空 Gaussian 参与渲染。

pretrain validation 可用 GT/DGGT-space pose 模拟用户输入 render camera；这只用于渲染，不应在 optional camera condition 缺失时作为模型输入条件。

正式训练 validation 和离线 inference 不使用 generated camera/sky token。生成分支固定复用输入图像经 DGGT 预测出的 DGGT camera，并使用 GT sky mask 和 sky model 背景合成；`generated_pred_sky_mask` 是 edited latent 送入 DGGT `semantic_head` 得到的诊断图，不参与 sky/non-sky 合成，也不是 SceneFlow sky mask 输出。

## 15. 与运行参数文档的边界

本文件维护模型设计。以下内容放在 `docs/scene_flow_cmd.md`：

- 环境变量和路径。
- pretrain / 正式训练命令。
- batch size、学习率、warmup、EMA、validation 频率等运行参数。
- flow cache manifest 构建和转换命令。
