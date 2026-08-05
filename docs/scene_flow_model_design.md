# SceneFlow 模型设计

本文档只记录 SceneFlow 的模型结构、token 设计、条件设计、训练目标和采样策略。运行命令、路径、batch size、学习率等参数放在 `docs/scene_flow_cmd.md`。

## 1. 总体结构

SceneFlow 的主模型类是 `RAEVideoSceneFlow`，对外别名仍为 `WanSceneFlow`。整体沿用 RAEv2 T2I 的高维 latent diffusion/flow 设计：冻结 DGGT aggregator 和 scene tokenizer，把图像/视频编码成高维 tokenizer latent，再训练 DiT/RAE 风格的 full-attention latent generator。

模型输入被分成两类 token：

| 类别 | token | 说明 |
|---|---|---|
| generation state | `video z_t` | 当前噪声状态，shape 为 `[B,S,P,C]` |
| generation state | `camera_gen_tokens` | pretrain 使用的 normalized 9D Waymo 米制 relative-SE(3) camera 状态 |
| generation state | `sky_gen_tokens` | pretrain 使用的 scene-level sky atlas 状态 |
| generation state | `gauge_gen_tokens` | pretrain 使用的单个 scene-global 3D metric-gauge 状态 |
| condition | timestep tokens | RAEv2 Gaussian Fourier timestep tokens |
| condition | text tokens | Qwen text encoder 输出，经线性投影进入 full attention |
| condition | camera condition tokens | 每帧 camera pose summary，或 learned null camera tokens |
| condition | asset tokens | sparse asset patch/summary tokens，或 learned null/empty asset token |
| condition | edit-control tokens | 正式训练局部编辑时由 `z_splat/scaffold/masks` 构造 |

Forward 中先拼接 generation sequence，再拼接 condition sequence：

```text
full_seq = [video, camera_gen, sky_gen, gauge_gen, timestep, text, camera_cond, asset, edit_control]
```

时间和长视频 camera 采用 clip-global 约定：raw pretrain 先对完整 29 帧 caption clip
运行一次冻结 DGGT teacher，再按 clip-global frame id 切出 10 帧训练窗口；Gaussian timestamp 恒为
`clip_local_frame_id / 4`，不再随训练窗口长度归一化；Waymo camera condition 的
`rel pose` 始终相对 clip frame 0，`delta pose` 始终相对真实前一帧。训练随机截取
10 帧时仍携带这两个全局上下文，因此与 offline 先构造完整米制轨迹状态再切 10 帧窗口完全
一致。窗口采样先以 `camera_anchor_window_probability=0.5` 决定取 frame-0 anchor
窗口还是非零起点的 delta-only 窗口，再在非零起点中均匀采样，避免自然采样导致
anchor 稀缺。裸 argparse 的 `camera_anchor_context_dropout` 默认是 `0`；正式
`pretrain_*.sh` 有意设为 `0.25`，作为仅作用于确实含 anchor 窗口的额外生成分支鲁棒性训练。
D3 后 RGB render 使用 detached teacher pose，不会因这个 dropout 排除样本。
默认长视频窗口为 10 帧、stride 7，即重叠 3 帧。
训练内 validation 的局部 loss 使用同一窗口表（10 帧时起点为
`0/7/14/19`），而生成验证在完整 29 帧上按训练 `sequence_length` 做 rollout；sampler
返回完整的 global anchor mask 和 delta-only 首窗所需的 Waymo metric previous-C2W。所有 loss、指标、
RGB/PLY 解码入口都必须显式消费这两个值，禁止根据局部 token 0 猜测 anchor。

冻结 scene tokenizer 含时序 attention，其标定合同固定为 10 帧。所有正式的
29 帧 encode/decode（cache 生成、正式训练 feedback、validation 和 offline inference）
都通过公共 windowed helper 切成不超过 10 帧的重叠窗口，默认 `window=10,stride=7`，
然后用 cosine coverage 在 token 空间归一化融合。禁止把 29 帧直接送入一次
`scene_tokenizer.encode/decode`；这与 SceneFlow 采样滑窗是两层独立但相同上限的约束。

所有可见 token 一起做 full self-attention。模型只对 generation spans 解码：video span 走 RAEv2 DDT head；camera/sky/gauge spans 分别走独立 decoder head。

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
| encoder full attention | `20 heads x 72 head_dim` | `(14,11,11)` | `5e4` |
| DDT head | `16 heads x 128 head_dim` | `(24,20,20)` | `1e4` |

encoder 和 DDT 使用独立 theta：encoder 同时覆盖局部视频网格、条件位置和 15000 附近的
天空球面，因此保留较慢的 `5e4` 频谱；DDT 只对目标视频局部网格注入 RoPE，使用 `1e4`
提高短时空范围内的位置分辨率。`rope_theta` 仅作为旧 checkpoint 的单 theta 兼容入口，
新 checkpoint 显式记录 `encoder_rope_theta` 和 `ddt_rope_theta`。

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
| scene-gauge token | `(15100,15100,15100)`，单个 scene-global 位置 |

SceneFlow 不再暴露全局 `mrope_temporal_margin`。当前写死 A3 坐标设计，并在 checkpoint 的 `scene_flow_config` 中记录 `rope_layout_version=a3_camera_center_spherical_sky15000`：

- text/timestep 保持 RAEv2-style zero RoPE；文本顺序已经由 Qwen hidden states 表达，不额外注入视频空间坐标。
- video、asset、edit-control 共享真实视频 `(t,y,x)`，保留局部编辑所需的 patch 对齐归纳偏置。
- camera 是全局每帧几何条件，temporal 与对应 video frame 对齐，spatial 放在 patch grid 中心，避免把相机条件绑定到左上角 patch。
- sky 是 scene-level directional atlas，不是 image-plane patch；将上半球方向映射为以 `15000` 为中心的三轴 Cartesian RoPE 坐标。经度首尾方向因此在位置空间天然相邻，不使用 seam loss；同时它仍与 video、asset、edit-control、camera 的 `[0,15000)` 时间轴分离。`rope_max_position=16384` 会对越界位置 fail-fast。
- gauge 是整个 trunk 共享的单 token，不带帧索引。`15100` 与 sky 的
  `15000±8` 球面带、以及视频时间轴都分离，且仍在 `rope_max_position` 内。
- 旧 A1/A2 checkpoint 或仍记录全局 `mrope_temporal_margin` 的 checkpoint，其 sky 位置语义与 A3 不一致，不应直接续训、warm-start 或推理。

`frame_ids` 和 `fps` 可让 video/asset/control/camera temporal position 做 Cosmos 风格的
时间缩放；pretrain、formal train/validation 和 formal offline inference 都显式传
`fps=10.0`（formal 常量为 `FORMAL_SCENE_FPS`）。不存在 formal 用 `fps=None`、
pretrain 用 10 FPS 的时间 RoPE 口径差。theta/section 的调整不改变任何 token 的位置坐标语义。

## 7. Asset Token 设计

Pretrain 的 asset 条件来自 clean latent 和 `dynamic_mask` 构造的 pseudo asset slots。每个场景最多保留 5 个 asset slot；每个 asset 每帧最多采样 32 个 patch token，并额外构造 summary token。

placement 条件使用 factorized-v3 的 16D 表示（沿用 placement-v2 布局，并修正 camera-anchor
z-depth 统计与有界 motion ratio），不再直接把米制 xyz 和速度三通道混入全局统计：

| 通道 | 含义 | 归一化约定 |
|---|---|---|
| `0:3` | anchor 米制空间中的 `unit_direction_anchor` | 无量纲 passthrough |
| `3` | 目标在当帧相机光轴上的 `log_z_depth` | 统计标准化 |
| `4:7` | `log_box_lwh` | 统计标准化 |
| `7` | `log(box_diag / z_depth)` | 尺度不变 passthrough |
| `8:10` | `sin/cos(yaw)` | 无量纲 passthrough |
| `10:13` | `unit_velocity_dir`，零速时为零 | 无量纲 passthrough |
| `13` | `log_speed`，下限 `1e-3` | 统计标准化 |
| `14` | `tanh(speed / z_depth)` | 尺度不变、有界 passthrough |
| `15` | `in_frustum` | passthrough |

因此只有 `{3,4,5,6,13}` 五个 log 幅值通道做数据统计，其余 11 个通道固定
`mean=0,std=1`。`log_z_depth` 而不是欧氏 range，是因为 metric→DGGT 的横向映射受
FOV 引起的各向异性影响，只有光轴 z-depth 与 `exp(log_metric_scale)` 是纯标量关系。
`target_bbox_patch` 和 `in_frustum` 仍用米制 box + 米制 Waymo camera + 真实 Waymo K；
它们不得改用 gauge K。只有编辑算法确实要把米制 3D box 体积映射进 DGGT
空间时，才通过 `metric_box_to_dggt` 在相机坐标系施加 FOV 各向异性。

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
2. **camera generation tokens**：pretrain 中和 video 一起生成的显式 9D Waymo 米制 trajectory 状态。

camera condition summary 维度是 `CAMERA_POSE_SUMMARY_DIM=20`，版本为
`waymo_metric_rel_delta_rot6d_fov20d_stats_v3`：9D clip-anchor relative pose + 9D adjacent
delta pose + 2D Waymo FOV。pretrain 和正式训练主链路都必须从 Waymo
`camera_to_world_corrected + intrinsics + raw_image_size_hw` 构造该摘要，每帧一个
token，经 projection 后以 `(t,H//2,W//2)` RoPE position 进入 full attention。FOV 使用
主点感知公式 `atan2(cx,fx)+atan2(W-cx,fx)`（y 轴同理），禁止用
`2*cx/2*cy` 猜原图尺寸。DGGT `pose_enc` 不得回灌成 SceneFlow camera condition。

用户推理不提供 camera，或训练时 dropout camera condition 时，不传零姿态；使用：

```text
camera_condition_kind = "camera_uncond"
```

模型会为每帧插入一个 learned `camera_null_condition_embed`，mask=true，position 仍是 `(t,H//2,W//2)`。

camera generation 版本固定为
`waymo_metric_relative_se3_rot6d_v4`，target space/source 分别是
`waymo_metric_camera_to_world` / `waymo_gt_extrinsics`。它直接由真实 Waymo C2W 构造，
每帧 9D 为 `[translation_m(3), rotation_6d(6)]`；frame 0 是相对轨迹 anchor 的绝对位姿，
后续帧是 `inv(C2W[t-1]) @ C2W[t]` 的相邻增量。生成 camera 不含 FOV；
DGGT 内参的 FOV 是同一 trunk 共享的 gauge 通道，在组装 render `pose_enc`
时才由 `gauge_to_pose_enc_fov` 加入。零向量或共线 rotation-6D 也会稳定投影为
有限、右手、`det=+1` 的 SO(3) 矩阵。

全局 `camera_gen_anchor_mask` 只在 clip 第 0 帧为 true。滑窗只能切片这个 mask，
不能把窗口首帧提升成 anchor；因此相对轨迹始终在窗口融合结束后一次性积分。
delta-only 窗口的第一个 camera token 是相对全局前一帧的真实 delta；几何/RGB loss
只在解码 loss 坐标时使用该前一帧的 Waymo 米制 `C2W` 作为积分初值，它不进入模型条件，
也不会把局部首 token 变成 anchor。训练中若包含 anchor 的窗口通过
`camera_anchor_context_dropout` 隐藏了 anchor，该行的 delta token 仍接受 camera flow 监督，
隐藏的 anchor token 不监督，且需要完整绝对轨迹的 camera geometry loss 跳过该行。
RGB loss 现在使用 detached teacher camera，不再依赖生成 anchor，因此不因这个 dropout
排除该行；`rgb_render_max_samples` 直接作用于当前 batch。
anchor 与 delta 分别使用 9D per-channel mean/std，stats 版本为
`waymo_metric_camera_anchor_delta_per_channel_v5_global_context`，std 下限 `1e-4`。stats 始终按
完整 29 帧的 clip-global role 计数，不会把随机窗口首帧重复计为 anchor；其
representation、target space/source 和 provenance 缺失或不匹配都会 fail-fast。camera
condition 的 delta 半段和 generation target 共用这套 role-aware 统计，不再使用
`translation_scale=10`。训练包含 normalized-state flow loss，以及米制 translation、SO(3)
geodesic、相邻 relative pose 和二阶变化残差；不再有 camera FOV loss。

渲染用 camera 和 SceneFlow 输入 camera condition 是两个概念。即使用户不提供
camera condition，渲染也不得把所用位姿回灌成模型条件。

正式训练、训练时 validation 与正式 offline inference 始终给定 Waymo 20D
condition，不传 camera generation token。渲染相机始终是 cache 中对完整 29 帧一次性
运行冻结 DGGT CameraHead 得到的 full-context teacher 位姿；局部窗口只切片，
edited/generated token 不重新预测或覆盖相机。

pretrain 的 RGB 监督也使用 detached teacher-space trajectory，因为 tokenizer latent
的 clean target 就定义在该 teacher 空间。默认 FOV/尺度取离线 gauge GT，
`--render_use_predicted_gauge` 只是显式消融，它替换尺度/FOV gauge，不替换 teacher 的
旋转与轨迹形状。相机 render 梯度因此固定为零：`compute_rgb_render_loss`
会对任何非零 `camera_grad_scale` 拒绝运行，trainer 只传 `0.0`。相机分支由
确定的米制 target 和几何损失监督，而不是从 RGB render 受梯度。相应地，
开放生成的 camera/gauge 与生成几何间没有相机光度捷径；当前用
`generated_static_geometry_reprojection_cycle_v1` 的静态几何前后向重投影/cycle
诊断监测两者是否一致。

全局 text CFG 作用于 video/sky/gauge/mask，但 gauge 仍只是 generation state，不是可传入的
condition。camera generation 使用独立
`camera_text_guidance_scale`（默认 1），而 `camera_guidance_scale` 仍只缩放 camera
condition residual。因此扫描全局 CFG 1/2/4 不会把 9D camera 状态外推。

## 9. Scene-global Gauge Token 设计

gauge 是 pretrain 第四段 generation state，shape 始终为 `[B,1,3]`，与帧数 `S`
无关：

```text
[log_metric_scale, log_tan_half_fovx, log_tan_half_fovy]
```

`log_metric_scale = log(米 / DGGT 单位)`，representation 为
`dggt_teacher_log_metric_scale_logfov_v1`。GT 由完整 29 帧 teacher trunk 离线估计：
尺度主尺是 LiDAR，两个 FOV 通道是 trunk 内 teacher `log(tan(FOV/2))` 的常量统计。
数据集按 `(scene, start_idx // 29)` 严格查表，输出 `scene_gauge` 和逐通道
`scene_gauge_valid`；无效通道的物理消费者使用训练集均值 fallback，direct loss 仍用
valid mask 屏蔽。短窗不得在线重新估尺度。
用户决策固定为 **`log s` 只生成、不做条件路径**：没有 `gauge_condition_tokens`、
learned null-gauge condition 或外部强制 gauge 入口，CFG dropout/scale 也不新增 gauge 条件分支。

gauge token 用 `ChannelScale(3) -> Linear` 投影，RoPE 坐标是
`(15100,15100,15100)`，独立 decoder 输出 3D clean-state prediction。它与 video、camera、sky
和所有 condition token 一起做 full self-attention；gauge hidden 还显式广播加入 video DDT
的 conditioning：

```text
gauge_context = gauge_hidden                         # [B,1,H]
video_cond = s_projector(silu(enc_video + t_base + gauge_context))
```

因此 video token 可以为 gauge 提供车辆、车道等尺寸线索，video flow loss 也能反向
更新 gauge token；同时生成几何显式以它为条件。gauge 不加入 9D 米制 camera
解码条件，因为米制 camera 本就不应依赖 DGGT 规范。训练上它与 video 共用
sigma，同时使用 normalized flow MSE 和反归一化后物理 log 空间的 masked
Smooth-L1。`gauge_vs_prior_gain`、`metric_depth_rel_err`、`gauge_log_scale_error`、
`gauge_fov_error_deg` 和 `gauge_valid_frac` 是必须日志。
gauge 统计版本是 `scene_gauge_per_channel_v1`；metric-gauge v4 feature stats 必须同时
包含 1024D tokenizer latent、9D metric camera、3D gauge 和 16D placement 的统计及对应哈希
provenance，旧 stats 不作 fallback。

非滑窗 sampler 和 `_cfg_sample_pretrain_latents_sliding` 都维护同一个 global
`gauge_z`。滑窗时它不按帧切片，而是用 `scene_global_window_weight` 对各窗口
velocity 加权后每个 ODE step 只更新一次；因此 10 帧和 29 帧滑窗共用同一个
scene-global 尺度/FOV。

内参必须分成两条永不交叉的链路：

| 链路 | 内参 | 消费者 |
|---|---|---|
| DGGT 几何链 | trunk-constant gauge K | depth 反投影、RGB render、sky atlas 写入/读出 |
| 米制 Waymo 链 | 真实 Waymo K | metric box 投影、`target_bbox_patch`、`in_frustum` |

## 10. Sky Token 设计

sky generation 只属于 full-scene pretrain。sky token 是 scene-level directional atlas token：先构造 `32x64` RGB atlas，再用固定 `2x2` pixel-unshuffle 打包为默认 `16x32` grid，每个 token 12 维：

```text
2x2 atlas patch x [r, g, b]
```

训练时从输入图像、sky mask、冻结 DGGT teacher c2w 和离线 gauge GT 构造 sky token
target。teacher c2w 保留 camera-anchor 的 `-y-up` atlas 世界，FOV 由 trunk-constant gauge K
提供；因此 atlas 整个 29 帧 trunk 的写入内参不再跟随 teacher 逐帧抖动。sky mask
只用于构造 RGB target 和逐输出通道的 `sky_gen_loss_weight [B,512,12]`，绝不能
作为模型输入 attention mask；训练和开放推理都 pack 完整 `16x32` sky atlas。
每个 atlas cell 对应上半球一个无穷远方向 bin；target 投影只使用 world-to-camera
rotation，不使用 translation，和 renderer 的 camera-ray 环境贴图定义严格互逆。该方向
投影到各帧后，在 GT sky mask 内采样 RGB，并选择置信度最高的可见帧，避免跨帧
位姿误差模糊纹理。未观测方向使用带经度环绕的球面邻域补全，且只按默认 `0.05`
低权重监督；整段完全没有有效 sky 观测时仍为零权重。visibility 使用和 RGB 完全相同的
pixel-unshuffle 顺序打包，因此一个 token 内未观测的子像素不会因相邻子像素可见而
被错误监督；可用 `--sky_unobserved_loss_weight` 调整低权重。

pretrain 采样时，sky token 和 video/camera/gauge 一起作为 generation state 更新；
开放推理的 RGB 读出使用 generated sky directional atlas，并由生成的米制 camera
加生成 gauge 重组 DGGT `pose_enc`，因而 atlas 读出和深度/render 共用同一 gauge K。
正式训练、训练内采样和离线 inference 都不 pack generated sky token，也不计算
`loss_sky_flow`；正式编辑在图像空间执行
`GT_sky_mask * input_GT_RGB + (1-GT_sky_mask) * rendered_edit`，天空像素严格来自原始
输入，不再经过 sky model 或窗口级 min-max。

## 11. Sky Mask 生成与 Refine

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

其中 `enc_video` 是经过 video/camera/sky/gauge/text/asset/control 全注意力交互后的 video span；`sky_context` 是 visible `sky_hidden` 的 masked mean，并 broadcast 到每个 video patch。这样 mask loss 不仅监督 video hidden，也会通过 trunk 和 pooled sky context 回传到 sky token 分支。refined decoder 还把 `base_feat` 作为 patch-grid skip feature，提供较早层的空间定位信息。

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

sky mask head 首先在和 video clean prediction 相同的随机 `sigma` denoising forward 上接受辅助监督。为了让最终渲染所用的 mask 与生成完成后的 clean scene 对齐，训练从 `sky_mask_endpoint_start_step` 开始，按 `sky_mask_endpoint_every` 周期把当前预测的 clean video、camera、sky 和 gauge state `detach` 后再做一次 `sigma=0` mask-only endpoint forward，并对 endpoint mask 施加同一份 GT mask 监督。RGB render 若在非 endpoint-supervision step 激活，也会按依赖关系执行这个 clean endpoint，但不会额外改变 `sky_mask_endpoint_every` 所定义的 BCE/Dice endpoint 监督频率。

采样时完成全部 ODE 更新后，对最终 clean video/camera/sky/gauge state 做一次 `sigma=0` mask-only endpoint forward；普通采样和滑动窗采样都只从该 endpoint 读取 `sky_mask_logits` 和 `sky_mask_refined_logits`。启用 factored CFG 时，endpoint 复用与生成完全相同的 text/asset/camera 分支及 scale，再对组合后的 logits 做 sigmoid。不能从最后一个非零去噪步读取最终 mask：在默认 `shift=10` 下，最后一个非零 `sigma` 仍可能较大，并不代表最终 clean scene。RGB render 优先使用 refined mask；如果只有 patch mask，则回退到 patch mask 上采样。render 在 `_render_gs_map_rgb` 内把 sky mask hard-threshold 成 non-sky Gaussian 选择，这与 DGGT 用 GT sky mask 排除 sky Gaussian、再由 rasterizer transmittance 合成背景的定义一致；mask target 不应替换成 renderer alpha。

## 12. DDT Head 与输出头

video span 通过 RAEv2 DDT decoder head 输出：

- `decoder_video_embed(z_t)` 作为 DDT 输入。
- `t_base` 作为 DDT AdaLN condition。
- final layer zero-init，训练初期输出稳定。

camera、sky 和 gauge 不走 video DDT final layer，而是使用各自的 lightweight decoder：

```text
camera_gen_decoder(hidden) -> camera_gen_dim
sky_gen_decoder(hidden) -> sky_token_dim
gauge_gen_decoder(hidden) -> 3
```

sky mask 也不走 DDT head。patch mask 使用 lightweight MLP decoder；refined mask 使用 trunk 后 dense conv decoder。DDT head 只服务高维 video latent 预测。

`base_final_layer` 是辅助 base prediction path，用于 `base_model_coeff` 相关训练项。

## 13. Attention Mask 与空 Token

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

## 14. CFG 与 Optional Condition

Pretrain 支持 text、asset、camera 三类条件。文本独立 dropout；结构条件按样本采样三个合法任务：

```text
text_uncond_drop_prob
joint_generation_prob            # asset=NULL, camera=NULL
camera_controlled_prob            # asset=NULL, camera=REAL
asset_camera_controlled_prob      # asset=REAL, camera=REAL
```

dropout 只隐藏输入条件，不改变 video/camera/sky/gauge 的 clean target 和 loss。

Pretrain 默认 text CFG 对齐 Cosmos conditional generation：

```text
v_full         = text      + asset + camera
v_no_text_full = null text + asset + camera

v = v_full + (text_scale - 1) * (v_full - v_no_text_full)
```

因此 text conditional/unconditional 两个分支拥有完全相同的 asset 与 camera，仅文本不同。显式设置独立 control scale 时，再加入分解残差：

```text
v_text_base   = text + asset_null + camera_null
v_text_camera = text + asset_null + camera

v += (camera_scale - 1) * (v_text_camera - v_text_base)
v += (asset_scale  - 1) * (v_full        - v_text_camera)
```

对应 CLI：

```text
--guidance_scale
--asset_control_guidance_scale
--camera_guidance_scale
```

三个 scale 默认都是 `1.0`，表示 no-op。factorized asset placement 含 camera 投影得到的 bbox/RoPE，因此 asset 条件存在时必须同时提供匹配的 camera；推理会拒绝 asset-without-camera。用户省略 asset 时使用 `asset_uncond`/`NULL`；asset 与 camera 都省略时对应 `joint_generation`。自然零资产样本的 `none` 仍必须保持为零 visible token 的条件态。`asset_control_guidance_scale` 在正式训练中仍控制 asset + edit-control guidance；pretrain full-scene 没有局部 edit-control token，但保留同名参数以对齐两阶段采样接口。

默认的 CFG sweep 只改变 `text_scale`，asset/camera scale 保持 `1.0`。这与 Cosmos conditional generation 的两分支设计一致：clean visual/structural condition 在 conditional 与 text-unconditional 分支中都保留，CFG 只放大上下文相关的文本残差。同时放大三个 scale 会额外外推目标外观与相机轨迹，不应作为普通 `cfg2/cfg4` 的默认含义；需要研究控制强度时可显式单独设置 asset/camera scale。

正式训练不设计 optional asset/camera condition，因为正式训练是局部目标编辑，asset 条件和 camera
条件必须由用户/样本给定。正式训练里的 `asset_uncond` 只用于训练 dropout 和 CFG 分支，不作为用户
可省略 asset 的接口语义。

## 15. Validation 与 RGB Render

pretrain validation 采样从纯噪声同步生成：

```text
scene latent + 9D metric camera token + sky token + 3D scene-gauge token
```

RGB render 不再把 GT image 送入 DGGT aggregator，也不使用 validation batch 的
image-token 模板。SceneFlow 输出 latent 经 window-bounded tokenizer decode 成 selected DGGT
patch levels；selected levels 的 special tokens 固定补零后送入冻结 DGGT
depth/GS/instance heads。render 尺寸由 `patch_grid * 14` 固定得到。

训练 RGB loss 与 validation/offline 共用同一个 generated-token DGGT 几何解码入口：
`latent -> tokenizer.decode -> frozen depth/GS/instance heads`。frozen head 参数不更新，
但训练解码不能放在 `no_grad` 中；generated depth 反投影与 point means 不允许
detach。主 RGB renderer 的接口不接受 teacher depth。

渲染相机按作用域分开：

- **pretrain RGB 训练监督**：使用 detached full-29-frame DGGT teacher trajectory，
  内参由离线 trunk gauge GT 组成常量 gauge K。默认路径对 9D camera 分支没有
  render gradient，任何非零 `camera_grad_scale` 都 fail-fast。它使用 generated sky atlas
  与 predicted refined sky mask；GT sky mask 只作为 RGB/LPIPS 权重，renderer sky split
  使用 predicted mask，避免 teacher-mask 捷径。
- **pretrain validation / open-generation offline render**：将生成 9D camera 解码成米制
  C2W，先相对 metric trunk anchor 重基到首相机的 teacher-atlas 世界，再用生成
  `log_metric_scale` 换回 DGGT translation，
  再用生成 gauge FOV 组成 `[B,S,9] pose_enc`。generated sky atlas 的读出也使用
  这一 gauge K。该渲染用于生成结果展示/诊断，不改变上述训练相机政策。
- **formal edit**：不生成 camera/sky/gauge token，固定复用 cache 中完整 29 帧上下文
  预测的 DGGT teacher camera。窗口 render 只切片这条轨迹，禁止逐窗重跑
  CameraHead。天空像素用 GT sky mask 从原输入保留，非天空区域只用 edited
  3DGS render。

### 15.1 Tokenizer v2 production pullback（v1 仅作历史审计）

冻结 tokenizer 的 direct→reconstruction 往返偏差与 scene gauge 是两个正交问题。当前唯一
production artifact 是 `data/scene_gauge/pullback_d63b34f7.json`：

| 字段 | 冻结值 |
|---|---|
| tokenizer SHA256 | `d63b34f7b1193ed7da399f953db504cfadb4f98dce2519854227a0f44714c8e8` |
| DGGT SHA256 | `352652738a5480b8d3ee9dd521ce07b528e5a297bd3feca4d07427dac6d87def` |
| tokenizer window / patch grid | `10` / `25x37` |
| runtime contract | `metric_depth_profile_gs_same_factor_render_identity_v2` |
| Phase 1b 方案 | A：render identity、metric identity、`c_gs=1.0` |

两个 boundary 仍由共享 helper 的显式 `boundary` 参数区分，但 v2 方案 A 下都必须是精确 no-op：

- `boundary="render"`：depth 和 GS 返回 tokenizer 原生 reconstruction tensor；禁止在
  `rgb_render_loss` 中接入 metric profile。
- `boundary="metric"`：`c_depth=1`、`c_gs=1`；只在米制边界按
  `exp(log_metric_scale)` 转换 point means、depth 与 Gaussian scale，不再施加 tokenizer
  round-trip 校正。

pretrain、formal train、validation 和 offline inference 入口都必须通过严格 loader 验证
artifact schema、`eligible_for_training`、tokenizer/DGGT/artifact SHA、window length、patch grid
和 gauge representation。loader 只接受 `tokenizer_generation=t0_v2`、schema `2.0.0` 与上述 v2
runtime contract；`pullback_75e566ef.json` 及其 v1 loglinear 系数仅保留为不可变历史证据，不能进入
训练或推理。绑定 v1 tokenizer 的旧 feature stats 同样禁止复用；当前 v2 全量统计为
`logs/scene_flow_pretrain_1024/feature_stats_pretrain_v5.pt`，且没有复用 v1 latent moments。
SceneFlow checkpoint 必须继续绑定该 stats 与 pullback provenance。

gsplat 在无 background 时返回 premultiplied RGB，因此合成公式固定为 `rendered_rgb + (1-alpha)*background`，禁止再次乘 alpha。LPIPS 使用 spatial 输出并应用与 Charbonnier 相同的 edit/sky 权重；正式训练 `sky_weight=0` 时天空不参与 LPIPS。

生成分支的 sky/non-sky split 来自 SceneFlow refined sky mask，而不是 `semantic_head`。generated sky atlas 只提供 sky background RGB；refined mask 决定哪些 image-plane pixels/points 作为非天空 Gaussian 参与渲染。

## 16. 与运行参数文档的边界

本文件维护模型设计。以下内容放在 `docs/scene_flow_cmd.md`：

- 环境变量和路径。
- pretrain / 正式训练命令。
- batch size、学习率、warmup、EMA、validation 频率等运行参数。
- flow cache manifest 构建和转换命令。
