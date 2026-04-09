# FlowDGGT: 基于 Render-Encode-Harmonize 的 4D 驾驶场景目标编辑研究计划

> 目标会议: CVPR / NeurIPS 2027
> 核心约束: 单一模型; 目标编辑精确; 有限训练资源; 不依赖 SceneDirector 权重
> 范围: 仅目标编辑 (删除/插入/替换/重定位), 暂不考虑轨迹编辑

---

## 一、核心结论

FlowDGGT 的当前方案是: 先在 3DGS 空间中做确定性几何编辑, 再把编辑后的渲染结果送回同一个 DGGT 主干, 在统一的场景隐空间 `z_scene` 上做条件流匹配, 最后解码回 DGGT 原有的 dense heads, 联合协调 appearance、geometry 和 dynamicness。

当前设计只保留以下结论:

1. 精确编辑仍在 3DGS 空间完成, 删除、插入、替换、重定位都由 3D bbox 和高斯操作严格控制。
2. 生成变量是 DGGT 上游的 joint scene latent `z_scene`。
3. 训练只需要两种模式: 协调训练和补全训练。
4. 推理统一使用双掩码: `M_source` 负责背景补全, `M_dest` 负责外观协调。
5. `M_dest` 采用 SDEdit 式部分加噪初始化, `M_source` 从纯噪声开始。
6. 跨视角监督不使用 pseudo novel-view GT, 只用真实观测视图之间的深度引导重投影一致性, 再辅以 unseen-view 几何正则。

**关键澄清**

- 补全训练不是“删掉车辆后再把车辆恢复回来”。
- 补全训练是在无车背景区域上合成 deletion-shaped holes, GT 始终是原始背景本身。
- 因此补全分支学到的是 background completion, 不是 object restoration。

---

## 二、调研结论

### 2.1 扩散空间分类

| 类别 | 代表方法 | 扩散空间 | 对本问题的结论 |
|------|----------|----------|----------------|
| 像素或 latent 后处理 | Difix3D+ | SD latent | 可借助强 2D 先验, 但多视角一致性弱 |
| GS 参数空间扩散 | DiffGS, GaussianAnything | GS 参数 latent | 原生 3D 一致, 但训练成本高 |
| 多视角 latent 联合去噪 | DSplats, DiffSplat | 多视角图像 latent | 兼顾 2D 先验与 3D 一致, 但仍需大量 3D 数据 |
| 前馈重建加扩散增强 | GIFSplat, Leveling3D | ViT 特征或渲染图像 | 前馈高效, 但主要面向重建增强而非可控编辑 |

### 2.2 对本方案最关键的启发

1. **DriveEditor** 证明了“重建式训练可统一覆盖多种编辑”。训练时只学重建, 推理时通过输入构造切换删除、插入、替换、重定位。
2. **RegNeRF / FreeNeRF / 3DGIC** 的共同启发是: 不要把模型自己渲染出的 novel view 当 GT。更稳妥的做法是用真实观测视图之间的对应关系, 再配合几何正则。
3. **LDM / Latent Flow Matching / RAE / DINO-SAE / Perceiver IO / DUNE** 共同支持一个判断: 生成应发生在可解码、可抗噪、可服务多下游头的 latent 上, 而不是直接发生在任意 hidden state 上。

### 2.3 与 DGGT 直接相关的结构结论

当前 DGGT 中, `Aggregator` 已经产出了天然的联合场景状态:

| 流 | 通道 | 代码来源 | 下游 head |
|----|------|----------|-----------|
| `dino_tokens[l]` | `1024` | `dino_token_list[l]` | `instance_head`, `semantic_head` |
| `frame_tokens[l]` | `1024` | `frame_intermediates[l]` | 几何与局部外观 |
| `global_tokens[l]` | `1024` | `global_intermediates[l]` | 时序与全局上下文 |
| `image_tokens[l]` | `3072` | `concat(dino, frame, global)` | `gs_head` |

这意味着最终结果并不只依赖 `GaussianHead`。更自然的生成路径是:

```text
image_tokens_[4,11,17,23]
  -> split(dino, frame, global)
  -> JointSceneTokenizerEncoder
  -> z_scene
  -> SceneFlowMatching
  -> JointSceneTokenizerDecoder
  -> split back
  -> {gs_head, depth_head/point_head, instance_head}
```

结论是: 应该在 joint scene latent 上联合建模, 而不是只修补某一个 head 的输入。

---

## 三、方案设计

### 3.1 设计原则

1. 精确编辑在 3DGS 空间完成, 位置由 3D bbox 严格确定。
2. 协调对象是 joint scene state, 不是单一 appearance token。
3. 整个方法保持单一模型, 两次前向共享同一个 `DINO + Aggregator + dense heads` 主干。
4. 用双掩码拆分补全和协调, 避免不同任务共用一套更新逻辑。
5. flow 显式接收 clean context、rendered scaffold 和资产特征。
6. 解码后继续复用原有 dense heads, 不重写 DGGT 主体。

### 3.2 整体流程

```text
Pass 1: 原始视频 -> DGGT -> clean scene tokens / 相机 / 原始场景状态
3DGS Edit: 在 G_original 上执行删除、插入、替换或重定位 -> G_edited
Render: G_edited -> I_edited, D_edited, A_edited, dynamic_prior
Pass 2: I_edited -> 同一个 DGGT Aggregator -> edited scene tokens
Tokenizer: edited tokens -> z_edited, clean tokens -> z_clean
Flow: SceneFlowMatching(z_edited, z_clean, scaffold, F_asset, M_source, M_dest) -> z_hat
Decode: z_hat -> joint tokens -> split back
Heads: gs_head + depth_head/point_head + instance_head -> 最终 4D 场景状态
```

### 3.3 JointSceneTokenizer

`JointSceneTokenizer` 的任务是把 4 层 joint scene token 整理成可生成、可解码、抗噪的 `z_scene`。

**输入与输出**

| 项 | 形状 | 说明 |
|----|------|------|
| 输入 | `4 × [B, S, P=1369, C=3072]` | 取自 `image_tokens_[4,11,17,23]` |
| latent | `[B, S, P=1369, C_scene=768]` | 统一场景隐空间 |
| 解码输出 | `4 × [B, S, P=1369, C=3072]` | 再按通道拆回三类 token |

**推荐结构**

1. encoder 内部先按通道拆出 `dino / frame / global` 三个子流。
2. 各层分别做轻量投影后, 在 joint space 中做 cross-scale fusion。
3. 额外保留 shallow detail branch, 用于边界和局部纹理恢复。
4. decoder 输出仍是 `3072` 维 joint token, 再固定 split 回 `dino / agg / img`。

**推荐配置**

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| `C_scene` | `768` | 默认容量, 同时承载 appearance、geometry、dynamicness |
| encoder 主干宽度 | `640` | 便于三子流融合 |
| detail branch | `128` | 保留局部细节 |
| decoder trunk | `896` | 不小于 latent, 便于高维 joint token 重建 |

不建议把 `agg_tokens` 和 `dino_tokens` 再额外并联输入 encoder, 因为 `image_tokens` 已经包含这些信息。

### 3.4 SceneFlowMatching

`SceneFlowMatching` 在 `z_scene` 上进行条件流匹配。它不直接生成 RGB, 而是生成可解回各 dense heads 的 joint scene state。

**条件输入**

| 条件 | 来源 | 作用 |
|------|------|------|
| `z_clean` | Pass 1 | clean scene context |
| `D_edited / A_edited / dynamic_prior` | 编辑后渲染 scaffold | 提供几何、透明度和动态先验 |
| `F_asset` | 资产单独渲染后经冻结 DINO 提取 | 仅在 `M_dest` 提供资产外观与形状信息 |
| `M_source / M_dest` | 编辑掩码 | 指示补全区与协调区 |

**推荐配置**

| 配置项 | 推荐值 |
|--------|--------|
| `token_dim` | `768` |
| `hidden_dim` | `1024` |
| `num_block_pairs` | `3` |
| `num_heads` | `16` |
| `num_steps` | `6` |
| `state_dim` | `96` |

**初始化策略**

- `M_source`: 从纯噪声开始, 解决背景补全和 disocclusion 生成。
- `M_dest`: 对 `z_edited` 做部分加噪, 从中间时刻开始去噪, 保留资产几何和粗外观, 只做协调。

### 3.5 Edit-State Routed Flow

确定性 3DGS 编辑之后, token 的可靠性并不均匀。统一更新强度会导致两类错误:

1. 可靠区域被过度改写, 出现 texture washing 和边界漂移。
2. 空洞或 disocclusion 区域修改不足, 出现 ghosting 和补全失败。

因此 flow 需要基于 scene-state reliability 做软路由。

**状态特征**

```text
s_tok = concat(
  M_source,
  M_dest,
  |A_edited - A_original|,
  clip(|D_edited - D_original| / d0, 0, 1),
  vis_support,
  boundary_flag
)
```

**三类 residual experts**

| expert | 作用 | 推荐更新强度 |
|--------|------|--------------|
| `preserve` | 尽量保持原有表示 | `0.25` |
| `harmonize` | 做外观与局部几何协调 | `1.0` |
| `generate` | 负责补全与真正生成 | `1.5` |

router 不替代共享 trunk, 只在 shared trunk 之后提供 lightweight residual adaptation。

### 3.6 Dense Heads 的复用

解码后的 joint token 用固定 channel split 复用原有 heads:

```python
img_tokens_hat_l = T_hat_l
dino_hat_l, frame_hat_l, global_hat_l = img_tokens_hat_l.split([1024, 1024, 1024], dim=-1)
agg_hat_l = torch.cat([frame_hat_l, global_hat_l], dim=-1)
```

然后直接复用:

- `gs_head(img_tokens_hat)`
- `depth_head(agg_hat)` 或 `point_head(agg_hat)`
- `instance_head(dino_hat)`

主线推荐 `depth_head -> point_map` 作为 geometry 分支, `point_head` 可作为辅助或消融。

### 3.7 必须满足的约束

1. **四层联合处理**: `GaussianHead` 是 DPT 风格结构, flow 和 tokenizer 必须覆盖 `[4, 11, 17, 23]` 四层, 不能只改一层。
2. **时空一致结构**: flow block 需要同时包含帧内 attention 和跨帧 attention, 与 `Aggregator` 的时空建模方式对齐。
3. **双掩码分治**: `M_source` 和 `M_dest` 不能合并成单一编辑掩码。
4. **显式条件注入**: clean context、scaffold、asset feature 都需要显式进入 flow。
5. **统一解码目标**: 最终解码必须同时服务 `gs / depth / dynamic` 三类状态。

---

## 四、3DGS 编辑接口

### 4.1 编辑指令

```python
class EditInstruction:
    action: str        # "delete" | "insert" | "replace" | "reposition"
    bbox: Tensor       # [T, 8, 3]
    bbox_new: Tensor   # [T, 8, 3], reposition 时使用
    asset: Asset       # insert / replace 时使用
```

### 4.2 四种操作的统一表示

| 操作 | 3DGS 操作 | `M_source` | `M_dest` | `asset_images` |
|------|-----------|------------|----------|----------------|
| 删除 | 删除目标高斯 | 旧位置 | 空 | 无 |
| 插入 | 放入新资产 | 空 | 新位置 | 资产单独渲染 |
| 替换 | 删旧目标并放入新资产 | 可选 | 目标位置 | 新资产单独渲染 |
| 重定位 | 将原目标移动到新位置 | 旧位置 | 新位置 | 移动后的目标单独渲染 |

统一原则:

1. 删除和重定位的 source 区域都属于背景补全任务。
2. 插入、替换和重定位的 dest 区域都属于外观协调任务。
3. 同一次推理可以同时存在 `M_source` 和 `M_dest`。

---

## 五、训练设计

### 5.1 两种训练模式

| 模式 | 输入构造 | GT | 覆盖的推理任务 |
|------|----------|----|----------------|
| 协调训练 | 删除动态车辆后, 用同一车辆的 Trellis 资产重新放回场景 | 原始视频与原始场景状态 | 插入、替换、重定位的 `dest` |
| 补全训练 | 在无车背景区域上合成 deletion-shaped holes | 原始视频与原始场景状态 | 删除、重定位的 `source` |

**协调训练**

1. 用原始视频做 Pass 1, 得到 `G_original`、相机位姿和动态目标掩码。
2. 选取一个动态目标, 用该目标自己的 Trellis 资产替换真实目标, 得到 `G_edited`。
3. 渲染 `I_edited` 以及资产单独渲染 `I_asset`。
4. 用 `M_dest` 训练外观协调能力。

**补全训练**

1. 只选择 vehicle-free 背景区域。
2. 在纯背景上合成与真实删除任务尽量同分布的 holes。
3. 删除对应高斯并渲染 `I_deleted`。
4. 用 `M_source` 训练背景补全能力。

### 5.2 补全训练的 hole 合成

补全训练的 mask 分布必须接近真实删除任务。推荐分布如下:

| 策略 | 比例 | 目的 |
|------|------|------|
| vehicle track replay mask | `70%` | 让洞的形状、尺度、透视和时序接近真实车辆删除 |
| core + ring mask | `20%` | 强化边界、残影和接地区域鲁棒性 |
| generic irregular / rectangle mask | `10%` | 仅作鲁棒性补充, 避免过拟合单一形状 |

额外要求:

1. 显式覆盖 truncation、partial occlusion、far-object 和 multi-hole 情况。
2. 训练以 3DGS cutout 为主, 不以 2D 图像涂黑为主。
3. mask 在时间维度上要保持一致或平滑变化, 不能每帧独立随机。

### 5.3 损失函数

```text
L_total =
    λ_flow     * L_flow
  + λ_render   * L_render
  + λ_lpips    * L_lpips
  + λ_xview    * L_xview
  + λ_auxgeom  * L_auxgeom
  + λ_3d       * L_3d
  + λ_state    * L_state
  + λ_route    * L_route
  + λ_preserve * L_preserve
```

| 损失项 | 作用 |
|--------|------|
| `L_flow` | 编辑区域上的 latent flow matching 主损失 |
| `L_render` | 原始观测视角下的渲染重建 |
| `L_lpips` | 感知质量约束 |
| `L_xview` | 真实观测视图之间的深度引导重投影一致性 |
| `L_auxgeom` | 虚拟扰动视角上的几何先验和 floater 抑制 |
| `L_3d` | scene state 的直接 3D 参数约束 |
| `L_state` | router 的高置信状态监督 |
| `L_route` | 反向路由惩罚, 防止明显错误的 expert 选择 |
| `L_preserve` | 非编辑区域保持 |

**监督原则**

1. `L_xview` 只在真实视图对之间计算, 不使用 pseudo novel-view GT。
2. `L_auxgeom` 只提供几何正则, 不对虚拟视角做 RGB 回归。
3. `L_3d` 约束的是 pixel-aligned scene state, 不只是单独的高斯外观参数。

**推荐权重**

```text
λ_flow = 1.0
λ_render = 1.0
λ_lpips = 0.1
λ_xview = 0.25
λ_auxgeom = 0.05
λ_3d = 0.1
λ_state = 0.10
λ_route = 0.05
λ_preserve = 0.5
```

### 5.4 训练阶段

**Stage T0: Tokenizer 预训练**

- 冻结 `Aggregator + gs_head + depth_head + instance_head`
- 训练 `JointSceneTokenizerEncoder / Decoder`
- 目标是 joint reconstruction、head anchors 和 noisy decoding

**Stage T1: SceneFlow 训练**

- 固定 encoder
- 训练 `SceneFlowMatching`
- decoder 前期只部分解冻, 后期放开 `layer_heads + local_refine`
- router 相关损失先 warm up 再开启

**Stage T2: 小学习率联合微调**

- 联合微调 flow、decoder 和 encoder 最后一层 cross-scale block
- 混合协调训练和补全训练

### 5.5 推荐训练配置

| 项 | 推荐值 |
|----|--------|
| 优化器 | AdamW + cosine decay |
| SceneFlow 学习率 | `2e-4` |
| dense heads 学习率 | `5e-6` |
| batch | `4 clips/GPU × 5 帧/clip` |
| 精度 | BF16 |
| 训练资源 | `8 × A100/H100` |

---

## 六、代码落点

### 6.1 `dggt/models/vggt.py`

需要新增三个组件:

1. `self.scene_tokenizer`
2. `self.scene_flow`
3. `self.asset_encoder`

原有 `forward()` 保持不变, 仅新增 `forward_edit()` 用于编辑模式。

**核心前向接口**

```python
def forward_edit(
    self,
    images,
    images_edited,
    M_source,
    M_dest,
    edit_scaffold=None,
    asset_images=None,
    mode="inference",
):
    ...
```

**核心流程**

```text
images -> Aggregator -> img_tok_1 -> z_clean
images_edited -> Aggregator -> img_tok_2 -> z_edited
asset_images -> AssetEncoder -> F_asset
SceneFlow(z_edited, z_clean, scaffold, F_asset, M_source, M_dest) -> z_hat
Decode(z_hat) -> img_tok_hat -> split -> {dino_hat, agg_hat}
{gs_head, depth_head/point_head, instance_head} -> predictions
```

### 6.2 方案级伪代码

下面这段伪代码对应的是当前 `research_plan + DGGT` 结构, 不是简单照搬某个 image-editing 模板。它保留了 DGGT 里真实存在的三路 token 组织方式:

- `aggregated_tokens_list`: `frame + global`, 供 `camera/depth/point` 分支使用
- `image_tokens_list`: `dino + frame + global`, 供 `gs_head` 使用
- `dino_token_list`: 供 `instance/semantic` 分支使用

同时只在 DPT 实际会取用的四层 `[4, 11, 17, 23]` patch token 上做 harmonization, 再把结果回填回原有 head 接口。

```python
def forward_edit(
    self,
    images,
    images_edited,
    M_source,
    M_dest,
    edit_scaffold=None,
    asset_images=None,
    mode="inference",
):
    levels = [4, 11, 17, 23]

    # 1) 原始 clip 编码: 提供 clean context, 同时保留原始 pose/world anchor
    agg_clean_all, image_tok_clean_all, dino_clean_all, _, patch_start_idx = self.aggregator(images)

    # 2) 编辑后渲染 clip 编码: 提供待修正的 scene state
    agg_edit_all, image_tok_edit_all, dino_edit_all, _, _ = self.aggregator(images_edited)

    # 3) 只抽取四层 patch token 进入 joint scene latent
    image_tok_clean_4 = select_patch_pyramid(image_tok_clean_all, levels, patch_start_idx)
    image_tok_edit_4 = select_patch_pyramid(image_tok_edit_all, levels, patch_start_idx)

    z_clean = self.scene_tokenizer.encode(image_tok_clean_4)
    z_edit = self.scene_tokenizer.encode(image_tok_edit_4)

    # 4) 显式条件: 编辑渲染 scaffold + 资产外观 + dual-mask state
    scaffold_feat = self.pack_scaffold(edit_scaffold)  # depth / alpha / dynamic_prior / visibility
    asset_feat = self.asset_encoder(asset_images) if asset_images is not None else None
    route_state = self.build_edit_state(
        M_source=M_source,
        M_dest=M_dest,
        scaffold=edit_scaffold,
    )

    # 5) 条件 flow: source 区做生成补全, dest 区做保形协调
    z_init = self.scene_flow.initialize(
        z_edit=z_edit,
        M_source=M_source,
        M_dest=M_dest,
        mode=mode,   # M_source 纯噪声, M_dest 部分加噪
    )
    z_hat = self.scene_flow(
        z_init,
        cond_clean=z_clean,
        cond_scaffold=scaffold_feat,
        cond_asset=asset_feat,
        state=route_state,
        M_source=M_source,
        M_dest=M_dest,
        mode=mode,
    )

    # 6) 解码回四层 joint token, 再补回 special tokens 以兼容原 DPT heads
    image_tok_hat_4_patch = self.scene_tokenizer.decode(z_hat)
    image_tok_hat_4 = reattach_special_tokens(
        template_tokens=image_tok_edit_all,
        levels=levels,
        patch_start_idx=patch_start_idx,
        patch_tokens=image_tok_hat_4_patch,
    )

    # 7) 按通道拆回 DGGT 的三路输入
    dino_hat_4, frame_hat_4, global_hat_4 = split_joint_channels(
        image_tok_hat_4,
        dims=[1024, 1024, 1024],
    )
    agg_hat_4 = [
        torch.cat([frame_hat_4[i], global_hat_4[i]], dim=-1)
        for i in range(len(levels))
    ]

    # 8) 只替换四层, 其余层保持 edited pass 的原始结果, 因而 dense heads 主体无需重写
    image_tok_hat_all = replace_selected_levels(image_tok_edit_all, levels, image_tok_hat_4)
    dino_hat_all = replace_selected_levels(dino_edit_all, levels, dino_hat_4)
    agg_hat_all = replace_selected_levels(agg_edit_all, levels, agg_hat_4)

    predictions = {}

    # 9) appearance / geometry / dynamicness 全都从 harmonized token 读出
    gs_map, gs_conf = self.gs_head(image_tok_hat_all, images_edited, patch_start_idx)
    predictions["gs_map"] = gs_map
    predictions["gs_conf"] = gs_conf

    depth, depth_conf = self.depth_head(agg_hat_all, images=images_edited, patch_start_idx=patch_start_idx)
    predictions["depth"] = depth
    predictions["depth_conf"] = depth_conf

    pts3d, pts3d_conf = self.point_head(agg_hat_all, images=images_edited, patch_start_idx=patch_start_idx)
    predictions["world_points"] = pts3d
    predictions["world_points_conf"] = pts3d_conf

    dynamic_conf, _ = self.instance_head(dino_hat_all, images_edited, patch_start_idx)
    predictions["dynamic_conf"] = dynamic_conf

    # 相机与世界坐标系不因对象编辑改变, 仍锚定 clean pass
    predictions["pose_enc"] = self.camera_head(agg_clean_all)[-1]

    return predictions
```

其中:

- `select_patch_pyramid(...)` 表示从四层 `image_tokens_list[l]` 中去掉 `camera/register` special tokens, 只保留 patch token 给 tokenizer / flow。
- `reattach_special_tokens(...)` 和 `replace_selected_levels(...)` 的目的都是复用 DGGT 现有 `DPTHead / GaussianHead` 接口, 避免为编辑模式重写一套 dense head。
- `pose_enc` 继续来自 clean pass, 因为本方案只做对象编辑, 不改变 ego pose 和世界坐标系定义。

### 6.3 原有 dense heads 尽量不改

本方案的核心不是重写各个 head, 而是把生成过程放到它们的共同上游。

因此:

- `GaussianHead` 主体不改
- `DepthHead / PointHead` 主体不改
- `InstanceHead` 主体不改
- 主要新增逻辑集中在 joint latent 编解码和条件 flow 链路

---

## 七、推理流程

### 7.1 删除

1. Pass 1 得到 `G_original`、动态掩码和相机位姿。
2. 在目标区域删除高斯, 形成 `G_edited`。
3. `M_source = bbox 区域`, `M_dest = ∅`。
4. Pass 2 中 `M_source` 从纯噪声开始补全背景。

### 7.2 插入

1. 把资产转成高斯并放入目标 bbox。
2. `M_source = ∅`, `M_dest = bbox 区域`。
3. 资产单独渲染得到 `I_asset`, 经冻结 DINO 得到 `F_asset`。
4. Pass 2 中 `M_dest` 从部分加噪的 `z_edited` 开始做协调。

### 7.3 替换

1. 删除原目标并放入新资产。
2. 通常只需要 `M_dest`。
3. 若旧目标移除后有明显背景空洞, 可额外启用 `M_source`。

### 7.4 重定位

1. 提取原目标高斯并移动到新位置。
2. `M_source = 原位置`, `M_dest = 新位置`。
3. source 分支负责补背景, dest 分支负责协调移动后的目标外观。

### 7.5 推理开销

| 方法 | 推理时间 | 前馈 |
|------|----------|------|
| DGGT 纯重建 | `~0.5s` | 是 |
| FlowDGGT | `~1.5s` | 是 |
| SceneDirector | `~15min` | 否 |
| DriveEditor | `~2min` | 否 |

---

## 八、实验方案

### 8.1 数据与评估

- 训练: Waymo Open 训练集 + Trellis 资产
- 验证: WOD validation
- 泛化: nuScenes 零样本测试

| 任务 | 指标 |
|------|------|
| 删除 | FID, FVD, CLIP-I, 多视角一致性 |
| 插入 | FID, FVD, 多视角一致性, 3D bbox 精度 |
| 重定位 | FID, FVD, ATE |
| 无编辑重建 | PSNR, SSIM, LPIPS |
| 下游任务 | 3D 检测 mAP |

### 8.2 关键消融

| 实验 | 验证目标 |
|------|----------|
| 直接渲染编辑后 GS, 不加 flow | post-edit harmonization 是否必要 |
| 单掩码 vs 双掩码 | 双掩码是否提升删除与重定位 |
| `gs-only` latent vs `joint scene` latent | 联合场景隐空间是否必要 |
| `C_scene = 512 / 640 / 768 / 896` | latent 容量权衡 |
| 无 `F_asset` | 资产条件是否必要 |
| `M_dest` 纯噪声初始化 vs 部分加噪初始化 | SDEdit 式初始化是否有效 |
| 无 `L_xview` | 真实视图间重投影监督是否必要 |
| 无 `L_auxgeom` | unseen-view 几何正则是否必要 |
| 无 `L_3d` | 直接 3D 监督是否必要 |
| 无 router | reliability routing 是否必要 |
| hard routing vs soft routing | 软路由是否更稳 |
| 只用最深层 vs 四层联合 | DPT 多层特征是否必须保留 |
| 去掉跨帧 attention | 时序一致模块是否必要 |

---

## 九、风险与应对

| 风险 | 应对 |
|------|------|
| Trellis 资产与真实车辆域差大 | 增强资产扰动, 保留 `F_asset`, 提高协调分支容量 |
| 两次 Aggregator 前向带来额外开销 | 当前目标仍是前馈方案; 后续可蒸馏为更少步数 |
| 跨视角重投影在边界处有噪声 | 仅在 mutually visible 区域计算, 对边界做 dilation |
| `M_source` 和 `M_dest` 可能冲突 | 重叠时优先视为协调区, 训练阶段避免重叠样本 |
| 资产预计算成本高 | 离线缓存 Trellis 资产, 训练阶段直接加载 |
| router 伪标签有噪声 | 只监督高置信 core 区域, 先 warm up 再开启路由损失 |

---

## 十、论文主张

本文的核心主张可以压缩为四点:

1. **Joint Scene-Latent Correction**: 生成变量是 DGGT 的 joint scene latent, 不是图像域后处理, 也不是单 head 修补。
2. **Dual-Mask Conditional Flow**: 同一模型同时处理背景补全和外观协调。
3. **Reliability-Aware Routing**: flow 的更新动力学由 token 的 scene-state reliability 决定。
4. **Self-Supervised 3D Consistency**: 训练依靠原始视频、真实视图间重投影一致性和几何正则, 不依赖 pseudo novel-view GT。

---

## 十一、参考文献

### 核心

- DGGT
- VGGT
- SceneDirector
- DriveEditor

### 扩散与 3DGS

- DiffGS
- DiffSplat
- GaussianAnything
- GIFSplat
- Leveling3D
- GaussianEditor

### latent 与 flow

- LDM
- Flow Matching in Latent Space
- RAE
- DINO-SAE
- Perceiver IO
- DUNE

### 3D 一致监督

- RegNeRF
- FreeNeRF
- 3DGIC
