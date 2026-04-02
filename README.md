# Mini-VLA: ViT + Q-Former + Qwen-2.5VL + Action Head

> 从图像像素到机器人动作的完整 VLA (Vision-Language-Action) 系统实现

---

## 目录

1. [系统架构总览](#1-系统架构总览)
2. [数据流维度追踪](#2-数据流维度追踪)
3. [模块详解](#3-模块详解)
   - 3.1 [ViT 视觉编码器](#31-vit-视觉编码器)
   - 3.2 [Q-Former 视觉桥接器](#32-q-former-视觉桥接器)
   - 3.3 [Qwen-2.5VL 多模态主干](#33-qwen-25vl-多模态主干)
   - 3.4 [Action Head — 动作生成器](#34-action-head--动作生成器)
4. [Action Head 四大变体](#4-action-head-四大变体)
5. [训练策略](#5-训练策略)
6. [VLA 方案对比](#6-vla-方案对比)
7. [优缺点分析](#7-优缺点分析)
8. [后续优化方向](#8-后续优化方向)
9. [代码结构与使用](#9-代码结构与使用)
10. [测试](#10-测试)

---

## 1. 系统架构总览

本系统是一个完整的 **Vision-Language-Action (VLA)** 架构，将视觉感知、语言理解和动作生成统一在一个端到端框架中。核心思路是：

```
Raw Input → ViT → Q-Former → Qwen2.5VL → Action Head → Robot Action
```

### 完整推理数据流 (Inference Pipeline)

```
┌──────────┐     pixels      ┌──────────┐    patch tokens    ┌──────────┐    visual tokens   ┌───────────┐   hidden states   ┌───────────┐    actions     ┌─────────────┐
│ Raw Input│ ───────────────→ │   ViT    │ ────────────────→  │ Q-Former │ ─────────────────→ │ Qwen2.5VL │ ────────────────→ │Action Head│ ──────────── │Robot Action │
│ 🖼 图像   │                  │(N_p,d_vit)│                   │ (32,768) │                    │ (T,3584)  │                   │(T_a, 7)   │              │ 6DoF+夹爪   │
│ 📝 指令   │                  │          │                   │          │                    │           │                   │           │              │ (16, 7)     │
│ 🤖 状态   │                  └──────────┘                   └──────────┘                    └───────────┘                   └───────────┘              └─────────────┘
└──────────┘
```

### 关键超参数 (以 ViT-L + Qwen2.5VL-7B 为基准)

| 符号 | 含义 | 典型值 | 说明 |
|------|------|--------|------|
| H, W | 输入图像分辨率 | 448 × 448 | 多分辨率支持 |
| P | ViT patch size | 14 | ViT-L/14 |
| N_p | patch token 数 | 32×32=1024 | (H/P)×(W/P) |
| d_vit | ViT hidden dim | 1024 | ViT-L |
| N_q | Q-Former query 数 | 32 | 压缩视觉信息的 query 数 |
| d_q | Q-Former hidden dim | 768 | BERT-base 维度 |
| d_llm | LLM hidden dim | 3584 | Qwen2.5-7B |
| T_txt | 文本 token 数 | 64–512 | 指令长度 |
| T_a | 动作 chunk 长度 | 16–50 | 一次预测多步 |
| d_a | 动作维度 | 7 | 6DoF + 夹爪 |

---

## 2. 数据流维度追踪

以单张 448×448 图像 + 文本 T=64 为例，逐步维度变换：

| 步骤 | 阶段 | 维度 | 说明 |
|------|------|------|------|
| ① | 原始图像 | `(3, 448, 448)` | RGB 像素 |
| ② | Patch Embedding | `(1024+1, 1024)` | +1 = [CLS] token, patch_size=14 |
| ③ | ViT Encoder Out | `(1024, 1024)` | 去掉 CLS，保留 patch tokens |
| ④ | Q-Former Queries | `(32, 768)` | 可学习 query，固定数量 |
| ⑤ | Q-Former Output | `(32, 768)` | 视觉信息压缩到 32 tokens |
| ⑥ | 线性投影 → LLM 维 | `(32, 3584)` | 768→3584 线性层 |
| ⑦ | 文本 Token Emb | `(64, 3584)` | 指令文本 embedding |
| ⑧ | LLM 输入序列 | `(96, 3584)` | concat [visual(32) + text(64)] |
| ⑨ | LLM Hidden State | `(96, 3584)` | 最后一层 hidden，取最后 token |
| ⑩ | Action Condition | `(1, 3584)` | LLM 最后 token = 动作条件 |
| ⑪ | 机器人状态 | `(1, 7)` | 当前关节角度 q_t |
| ⑫ | 动作输出 | `(16, 7)` | 16步动作序列，每步7维 |

**关键观察：**
- ViT 输出 1024 个 patch tokens → Q-Former 压缩到 32 个，**压缩比 32:1**
- Q-Former 768维 → LLM 3584维，通过线性投影对齐
- LLM 的最后 token hidden state 作为 Action Head 的条件信号

---

## 3. 模块详解

### 3.1 ViT 视觉编码器

**Vision Transformer (ViT-L/14, CLIP)**
- 参数量：307M, 24 层, 16 头
- 功能：将图像切成固定大小的 patch，经 Transformer 编码得到视觉特征序列

#### 3.1.1 Patch Embedding

- **输入：** `(B, 3, H, W)` — RGB 像素值，归一化到 [-1,1]
- **输出：** `(B, N_p, d_vit)` — N_p = (H/14)² 个 patch，每 patch 展平后线性投影

```
x_i = Linear(flatten(img[i*P:(i+1)*P, j*P:(j+1)*P]))    // P=14
x = [CLS; x_1; x_2; ...; x_{N_p}]                        // prepend CLS token
x += PE                                                    // +位置编码
```

实现方式：`Conv2d(3, 1024, kernel_size=14, stride=14)` 等价于展平+线性投影

#### 3.1.2 Transformer Encoder × 24

- **结构：** Pre-LN Transformer Block
- **输入/输出：** `(B, N_p+1, 1024)`

```
x' = x + MHSA(LN(x))     // Multi-Head Self-Attention, heads=16, d_head=64
x  = x' + MLP(LN(x'))     // FFN: d_mlp=4096, GeLU
```

#### 3.1.3 特征提取策略

| 策略 | 形状 | 说明 |
|------|------|------|
| 末层 patch tokens | `(B, N_p, 1024)` | 去掉 CLS，保留全部 patch，常用于 Q-Former |
| CLS token only | `(B, 1, 1024)` | 全图语义摘要，信息压缩，简单场景可用 |

在 VLA 中，典型使用 448×448 → 1024 个 patch tokens，每个维度 1024。

> **Qwen2.5VL 的 ViT 改进：** 采用 NaViT 思路支持任意分辨率输入，使用 2D RoPE 代替绝对位置编码，patch token 数动态变化（最多 4096）。同时使用 SigLIP 初始化（对比学习目标更适合视觉-语言对齐）。

---

### 3.2 Q-Former 视觉桥接器

**Q-Former (Querying Transformer, BLIP-2 原创)**
- 参数量：~188M, 12 层, BERT-base backbone
- 功能：用 N_q 个可学习 Query 通过 Cross-Attention 从 ViT patch tokens 中提取信息，将任意数量的 patch tokens 压缩为固定数量（32）的视觉 token

#### 3.2.1 可学习 Query (Learnable Queries)

- **Query：** `Q ∈ (B, 32, 768)` — 32 个可学习向量，形状固定，不依赖输入图像
- **来自 ViT 的 KV：** `V_feat ∈ (B, 1024, 1024)` — 先通过线性层 1024→768 对齐维度

#### 3.2.2 Q-Former Transformer Layer × 12

每层包含**双重 Attention 结构**：

1. **Self-Attention (Q 内部)**：32 个 query 之间互相交流
   - `(B, 32, 768) → (B, 32, 768)`

2. **Cross-Attention (Q 查 V)**：Query 从 ViT 特征中提取信息
   - `Q:(32,768) × KV:(1024,768)`
   - 每个 query 学会关注不同图像区域

```
// Q-Former 层内部结构
Q'    = LayerNorm(Q + MHSA(Q, Q, Q))               // query 自注意力
Q''   = LayerNorm(Q' + CrossAttn(Q', K_v, V_v))     // query 查询 ViT 特征
Q_out = LayerNorm(Q'' + FFN(Q''))                    // FFN

// CrossAttn 公式：
K_v = V_feat · W_k  ∈ (B, 1024, 768)
V_v = V_feat · W_v  ∈ (B, 1024, 768)
Attn = softmax(Q'' · K_vᵀ / √768) · V_v            // (B, 32, 768)
```

#### 3.2.3 输出投影 → LLM 维度

- **Q-Former 输出：** `(B, 32, 768)` — 32 个 token，包含从 ViT 提炼的视觉语义
- **投影后：** `(B, 32, d_llm=3584)` — 线性层 768→3584，作为 LLM 的前缀 token

> **Q-Former 两阶段训练：**
> - 阶段一：冻结 ViT，Q-Former + ViT 联合预训练（ITC + ITG + ITM）
> - 阶段二：冻结 ViT 和 Q-Former，训练线性投影层 + LLM（或 LoRA）
> - VLA 中在阶段二之后再微调 Action Head

---

### 3.3 Qwen-2.5VL 多模态主干

**Qwen2.5VL-7B**
- 参数量：7B, 28 层, d=3584, 28 heads (GQA 4 KV heads)
- 功能：原生多模态架构，视觉 token 与文本 token concat 输入 LLM，输出 hidden state 作为 Action Head 的条件

#### 3.3.1 视觉编码器 (Qwen2.5VL 内置 ViT)

```
patch_tokens = ViT(img)  ∈ (N_p, d_vit)
merged = PatchMerge(patch_tokens)                    // 2×2 → 1, N_p/4 tokens
visual_emb = Linear(merged)  ∈ (N_v, 3584)          // 投影到 LLM 维度
```

#### 3.3.2 多模态序列拼接

输入组件按如下格式拼接：

```
[IMG_tokens]         // 视觉 token: (N_v, 3584)
[SYS_prompt]         // 系统提示
[TASK_instruction]   // 任务指令: (T_txt, 3584)
[ROBOT_state]        // 机器人状态 (可选): (1, 3584)
[ACTION_token]       // 动作特殊 token
```

总长度：`T_total = N_v + T_txt + T_state`

#### 3.3.3 Qwen2.5 Decoder Layers × 28

关键特性：

| 组件 | 参数 | 说明 |
|------|------|------|
| GQA | 28Q / 4KV heads | KV Cache 显存 ÷7 |
| SwiGLU FFN | d_ffn=18944 | SwiGLU(x) = Swish(xW₁) ⊙ (xW₂) |
| 2D RoPE | theta=1000000 | 图像用 (row,col) 2D坐标，文本用 1D 序列 |
| RMSNorm | ε=1e-6 | Pre-Norm，无均值中心化 |

```
// Qwen2.5VL Decoder Layer
h' = h + GQA-Attn(RMSNorm(h), rope=rope_2d)
h  = h' + SwiGLU-FFN(RMSNorm(h'))

// GQA 细节
Q = h · W_q  ∈ (T, 28, 128)    // d_head = 3584/28 = 128
K = h · W_k  ∈ (T, 4, 128)     // 4 个 KV head
V = h · W_v  ∈ (T, 4, 128)
O = GQA(Q, K, V)  ∈ (T, 28, 128) → (T, 3584)
```

#### 3.3.4 动作条件提取

- **LLM 完整输出：** `(B, T_total, 3584)`
- **送往 Action Head：** `(B, 1, 3584)` — 取 [ACTION] token 或最后 token 的 hidden state

---

### 3.4 Action Head — 动作生成器

**Diffusion Action Head (DiT 架构)**
- T_a=16 steps, d_a=7, noise steps=100 (DDIM)
- 条件扩散模型，以 LLM 特征为条件去噪

#### 3.4.1 输入组件 (3 路输入)

| 输入 | 形状 | 说明 |
|------|------|------|
| LLM 条件 c | `(B, 3584)` | LLM 最后 token hidden state → MLP 投影到 d_act |
| 机器人状态 s | `(B, 7)` | 6 关节角度 + 夹爪开合 |
| 噪声动作 aₜ | `(B, T_a, 7)` | 扩散步骤 t 的带噪样本 |

#### 3.4.2 时间步嵌入 (Diffusion Step Embedding)

```
t_emb = SinusoidalEmb(t)  ∈ (d_model,)
t_emb = MLP(t_emb)  ∈ (d_act,)    // Linear-SiLU-Linear
```

#### 3.4.3 DiT Blocks × N (条件化去噪)

使用 **AdaLN-Zero** 作为条件化机制：

```
// 条件信号 = LLM条件 + 时间步嵌入
cond = MLP(c + t_emb)  ∈ (B, 6×d_act)    // 6 = 3×(γ,β,α)
γ₁,β₁,α₁, γ₂,β₂,α₂ = chunk(cond, 6)

// DiT Block
x' = x + α₁ · Attn(γ₁ · LN(x) + β₁)     // 条件自注意力
x  = x' + α₂ · FFN(γ₂ · LN(x') + β₂)     // 条件 FFN
```

#### 3.4.4 输出层

```
// 训练损失 (简单 MSE)
ε ~ N(0, I),  aₜ = √ᾱₜ·a₀ + √(1-ᾱₜ)·ε     // 加噪
L = ||ε - ε_θ(aₜ, t, c)||²

// 推理 (DDIM 加速，10-50步)
a_T ~ N(0, I)
for t = T,...,1:  a_{t-1} = DDIM_step(a_t, ε_θ(a_t, t, c))
return a_0                                       // 去噪动作序列
```

---

## 4. Action Head 四大变体

### Variant A: MLP 回归头 (最简单)

```
IN:  (B, 3584)
OUT: (B, T_a × d_a) → reshape (B, T_a, d_a)
Loss: MSE(pred, gt_action)
```

- ✅ 推理快
- ❌ 无法建模多模态动作分布

### Variant B: GMM 混合高斯头 (概率建模)

```
IN:  (B, 3584)
OUT: K × (μ:(T_a,d_a) + σ:(T_a,d_a) + π:scalar)
Loss: -log Σπ_k · N(a|μ_k, σ_k)
```

- ✅ 能建模多模态分布
- ⚠️ K 固定限制表达能力

### Variant C: 扩散式 DiT 头 (主流方案)

```
IN:  (B, T_a, d_a) noisy + c:(B,3584) + t:scalar
OUT: ε̂:(B, T_a, d_a) 预测噪声
推理: DDIM 10-50步去噪
```

- ✅ 表达力强（Diffusion Policy / π₀ 均使用）
- ❌ 推理较慢

### Variant D: Flow Matching 头 (新兴方案)

```
IN:  aₜ=(B,T_a,d_a), t∈[0,1], c:(B,3584)
OUT: v_θ(aₜ,t,c) ∈ (B,T_a,d_a) 向量场
Loss: ||v_θ - (a₀-ε)||² CFM目标
```

**核心数学 (π₀ / RDT 使用)：**

```
// Conditional Flow Matching (CFM) 训练
t ~ Uniform(0, 1),  ε ~ N(0, I)
aₜ = (1-t)·ε + t·a₀         // 线性插值: t=0 纯噪声, t=1 真实动作
u_t = a₀ - ε                  // 条件向量场 (目标方向)
L = ||v_θ(aₜ, t, c) - u_t||² // 训练向量场网络

// 推理 (ODE求解, Euler method, 10步)
a₀ ~ N(0, I)
for i = 0,...,9:
    t = i/10,  dt = 1/10
    aₜ₊ᵈᵗ = aₜ + dt · v_θ(aₜ, t, c)
return a₁                     // 终点即预测动作
```

- ✅ 推理快（10步 Euler）
- ✅ 训练更稳定

---

## 5. 训练策略

### 三阶段训练策略 (常见 VLA 实践)

#### 阶段 1: 视觉-语言预训练

- **冻结：** 无（或 ViT 冻结）
- **训练：** Q-Former + 线性投影
- **数据：** 图文对（LAION/CC3M）
- **目标：** ITC + ITG + ITM
- **目的：** 让 Q-Former 学会从图像中提取语言对齐的视觉特征

#### 阶段 2: 视觉指令微调

- **冻结：** ViT
- **训练：** Q-Former + 投影 + LLM (LoRA)
- **数据：** VQA / 指令跟随数据集（LLaVA-1.5 格式）
- **目标：** 语言建模 loss（下一个 token 预测）
- **目的：** 让 LLM 学会理解视觉输入，遵循多模态指令

#### 阶段 3: 机器人动作微调

- **冻结：** ViT + Q-Former（或轻量解冻）
- **训练：** LLM (LoRA) + Action Head (全量)
- **数据：** 机器人演示轨迹（图像+动作）
- **目标：** 扩散去噪 loss / Flow Matching loss
- **目的：** 让 Action Head 学会从 LLM 条件生成动作轨迹

---

## 6. VLA 方案对比

| 方案 | 视觉编码 | 视觉桥接 | LLM | Action Head | 特点 |
|------|----------|----------|-----|-------------|------|
| **本文方案** | ViT-L/14 (CLIP) | Q-Former | Qwen2.5VL-7B | Diffusion DiT | 解耦视觉压缩，支持任意分辨率 |
| π₀ | ViT-H (SigLIP) | 直接 concat | PaliGemma-3B | Flow Matching | 端到端，动作专家混合 |
| OpenVLA | ViT-L (SigLIP) | 直接 proj | LLaMA-2-7B | MLP离散化 | 动作离散化为 token，自回归生成 |
| RoboFlamingo | ViT-L (CLIP) | Flamingo Xattn | LLaMA-7B | Diffusion Policy | 冻结 VLM，只训练 action head |
| RDT-1B | SigLIP-400M | 直接 proj | DiT (无 LLM) | Flow Matching | 无 LLM，纯 DiT 端到端 |
| LEROBOT (ACT) | ResNet-50 | 无 | 无 LLM | CVAE + Chunking | 轻量，实时控制，无语言 |

**本方案核心优势：** Q-Former 将 1024 个 patch token 压缩为 32 个，LLM 输入长度可控且多分辨率友好；Qwen2.5VL 视觉理解能力强（物体、位置、文字感知优于 LLaMA 系列）；Diffusion DiT 建模多模态动作分布，适合复杂操作任务。

---

## 7. 优缺点分析

### 优点

1. **模块化解耦设计**
   - ViT、Q-Former、LLM、Action Head 各司其职，可独立优化
   - 各模块可替换（如 ViT 换 SigLIP，Action Head 换 Flow Matching）
   - 三阶段训练使微调代价小

2. **高效视觉信息压缩**
   - Q-Former 将 1024 个 patch tokens 压缩为 32 个，LLM 计算量大幅降低
   - 相比直接 concat patch tokens（如 LLaVA），推理速度和显存占用显著优化
   - 压缩过程可学习，保留任务相关的关键视觉信息

3. **强大的语言理解主干**
   - Qwen2.5VL 是原生多模态模型，视觉理解能力超强
   - 支持多分辨率、多帧视频输入
   - GQA 降低 KV Cache 显存 7×，长序列友好

4. **多模态动作建模**
   - Diffusion/Flow Matching Action Head 能建模复杂多模态动作分布
   - Action Chunking (T_a=16) 提高时间一致性，减少抖动
   - 相比 MLP 回归头，在歧义场景下表现更好

5. **利用预训练知识**
   - ViT (CLIP/SigLIP) 提供强大视觉表征
   - Qwen2.5VL 提供 NLP + 多模态推理能力
   - 海量图文预训练知识迁移到机器人领域

### 缺点

1. **系统复杂度高**
   - 四个独立模块 + 三阶段训练流程，工程难度大
   - 模块间接口需要精心设计（维度对齐、梯度传播）
   - 调试困难，问题定位需要逐模块排查

2. **推理延迟**
   - ViT (307M) + Q-Former (188M) + LLM (7B) + Diffusion Head (多步去噪)
   - 扩散式 Action Head 需要 10-50 步迭代，延迟不可忽略
   - 实时控制场景（>10Hz）可能受限

3. **Q-Former 信息瓶颈**
   - 32 个 query 可能丢失精细的空间信息（抓取精度要求高的任务）
   - 压缩比固定（32:1），不同任务的最优压缩比不同
   - 与直接 projection 方案相比，引入了额外的架构复杂度

4. **训练资源需求大**
   - 三阶段训练需要大量图文对 + VQA数据 + 机器人数据
   - LLM LoRA 微调仍需要多卡 GPU
   - Q-Former 预训练数据需求量大（百万级图文对）

5. **动作空间限制**
   - 当前 d_a=7 (6DoF+夹爪) 限制了灵巧操作
   - Action Chunk 长度固定 (T_a=16)，不同任务最优长度不同
   - 缺乏对力/力矩等反馈信号的建模

---

## 8. 后续优化方向

### 8.1 视觉编码优化

- **动态分辨率 ViT：** 采用 NaViT/Qwen2.5VL 的动态分辨率方案，避免固定 resize 丢失信息
- **多尺度特征提取：** 融合 ViT 多层特征（浅层局部 + 深层语义），类似 FPN
- **轻量化视觉编码：** 使用 EfficientViT/MobileViT 替代 ViT-L，降低推理延迟
- **视频流编码：** 支持时序帧的高效编码（帧差、光流辅助、时序注意力）

### 8.2 Q-Former 改进

- **自适应 Query 数量：** 根据场景复杂度动态调整 query 数量（简单场景用 8，复杂用 64）
- **层级注意力：** 不同 Q-Former 层关注不同 ViT 层的特征（低层→空间细节，高层→语义）
- **去掉 Q-Former：** 参考 π₀ 直接 concat方案，用 PatchMerge 替代（更简单但 token 更多）
- **Cross-Attention 稀疏化：** Top-k 或 Local Attention 降低 Q-Former 计算量

### 8.3 LLM 主干优化

- **更小模型：** Qwen2.5VL-3B/1.5B 适合低延迟场景
- **推测解码 (Speculative Decoding)：** 小模型草稿 + 大模型验证加速
- **量化部署：** INT4/INT8 量化 + Flash Attention 降低显存和延迟
- **KV Cache 优化：** 结合 GQA + PagedAttention 进一步降低长序列显存

### 8.4 Action Head 进化

- **Flow Matching 替代 Diffusion：** 10步 Euler 比 DDIM 50步更快，训练更稳定
- **Token-based Action：** 参考 OpenVLA，将动作离散化为 token，利用 LLM 自回归能力
- **层级动作生成：** 高层任务规划 + 低层轨迹细化，分级 Action Head
- **力/力矩建模：** 扩展 d_a 到包含力反馈，支持力控任务
- **可变 Chunk 长度：** 动态决定 T_a，短任务少步、长任务多步

### 8.5 训练策略改进

- **端到端微调：** 减少阶段数，第3阶段解冻全部模块联合微调
- **数据增强：** 图像增强 + 动作扰动 + 语言改写
- **在线学习：** 支持部署后在线适应新任务/新场景
- **仿真预训练：** 先在大规模仿真数据上预训练 Action Head，再迁移到真实

### 8.6 工程部署优化

- **ONNX/TensorRT 加速：** 各模块独立优化，流水线并行
- **Edge 部署：** 模型蒸馏（7B → 1B），适配机器人端侧芯片
- **多机器人适配：** 统一动作空间，支持不同机器人形态 (morphology-agnostic)

---

## 9. 代码结构与使用

### 项目结构

```
mini_vla/
├── README.md
├── mini_vla/
│   ├── __init__.py
│   ├── config.py           # 配置定义
│   ├── vit.py              # ViT 视觉编码器
│   ├── qformer.py          # Q-Former 视觉桥接器
│   ├── llm_backbone.py     # LLM 主干 (模拟/真实 Qwen)
│   ├── action_heads.py     # Action Head 4种变体
│   └── model.py            # 完整 VLA 模型
└── tests/
    ├── test_vit.py
    ├── test_qformer.py
    ├── test_action_heads.py
    └── test_model.py
```

### 快速使用

```python
import torch
from mini_vla.config import MiniVLAConfig
from mini_vla.model import MiniVLAModel

# 创建配置
config = MiniVLAConfig(
    img_size=224,
    action_head_type="flow_matching",  # mlp / gmm / diffusion / flow_matching
)

# 创建模型
model = MiniVLAModel(config)

# 准备输入
images = torch.randn(2, 3, 224, 224)
text_tokens = torch.randint(0, 1000, (2, 64))
robot_state = torch.randn(2, 7)

# 前向推理
actions = model.predict(images, text_tokens, robot_state)
print(actions.shape)  # (2, 16, 7)
```

---

## 10. 测试

```bash
# 运行所有测试
cd /hdd/home/zhangh77/LLM/mini_vla
python -m pytest tests/ -v

# 运行单个模块测试
python -m pytest tests/test_vit.py -v
python -m pytest tests/test_qformer.py -v
python -m pytest tests/test_action_heads.py -v
python -m pytest tests/test_model.py -v
```
