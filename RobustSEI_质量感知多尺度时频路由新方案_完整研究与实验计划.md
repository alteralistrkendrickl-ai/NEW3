# RobustSEI 质量感知多尺度时频路由新方案

## 完整研究说明、论文检索与实验计划

> 文档版本：2026-08-03
> 当前定位：旧教师恢复方案终止后的新研究路线定义稿
> 暂定方法名：`RobustSEI_QCRouter`
> 英文暂定名：Quality-Calibrated Multi-Scale Time-Frequency Routing for Low-SNR Specific Emitter Identification

---

## 1. 一页结论

本项目仍然研究低信噪比条件下的特定辐射源识别（Specific Emitter Identification，SEI），但不再继续教师网络、特征恢复、三视图或渐进式 SNR 课程路线。

已有公平实验已经说明：

- 教师网络和恢复损失没有带来可测量的独立收益；
- 单层恢复和多层恢复没有建立有效贡献；
- 三视图相对双视图只提高约 `0.05` 个百分点，属于随机波动；
- 渐进式 SNR 课程不优于固定 SNR 混合；
- 真正稳定有效的是多尺度时频编码器和低 SNR 样本参与身份监督训练。

因此，新方案不再尝试把噪声特征“恢复成干净特征”，而是解决另一个更直接的问题：

> 当噪声破坏不同时间尺度、时域特征和频域特征的程度不同时，模型如何判断哪些特征仍然可靠，并针对每个输入动态选择可信的指纹证据？

新方案保留 `MSFTFNet`，将其升级为具有以下能力的模型：

1. 显式保留不同尺度、时域和频域的候选特征；
2. 不依赖推理阶段真实 SNR 标签，自主估计输入质量；
3. 根据样本质量和分支统计量动态分配尺度与域的权重；
4. 使用身份分类损失和一个轻量的质量顺序损失完成训练；
5. 在完全相同的 AWGN 样本暴露下，与固定融合和当前二路门控公平比较。

新方案是否成立，不看训练准确率，也不看某一次测试结果，而看质量路由相对强 AWGN 基线的独立增益、重复稳定性、未见 SNR 泛化和第二数据集复现。

---

## 2. 研究问题与适用范围

### 2.1 研究问题

输入是一段长度为 256 的复数 IQ 信号，以 I、Q 两个实数通道表示。目标是在接收信号受到强 AWGN 污染时，仍然识别其发射设备身份。

可写为：

```text
x = s_y + n
```

其中：

- `s_y` 是设备 `y` 发出的信号；
- `n` 是噪声；
- 当 SNR 降低时，`n` 会掩盖由硬件非理想产生的细微指纹。

真正的困难不是普通信号分类，而是：

- 设备之间差异细微；
- 噪声可能先破坏局部高频和瞬态特征；
- 不同尺度特征在不同 SNR 下可靠性不同；
- 一个固定融合规则需要同时处理 `-10 dB` 到 `20 dB`，容易产生折中。

### 2.2 当前主任务边界

新方案第一阶段只解决：

```text
已知设备闭集识别 + AWGN低SNR鲁棒性
```

第一阶段暂不把以下问题混入核心训练：

- Rayleigh 或 Rician 衰落；
- 多径传播；
- 接收机变化；
- 跨日期采集；
- 开集设备识别；
- 少样本增量学习。

这些问题与实际部署有关，但必须在 AWGN 主问题和核心机制证明成立后，作为外部泛化实验逐步加入。否则论文会同时改变太多因素，无法判断提升来自哪里。

### 2.3 “干净信号”的准确含义

ManyTx 中的原始样本并不一定是物理意义上的无噪声信号。因此后续论文和代码建议使用：

```text
reference view / unaugmented view
```

而不是绝对意义上的 `clean signal`。

如果 `x_r` 是数据集原始样本，`x_n` 是在同一样本上继续加入 AWGN 得到的视图，那么我们只知道：

```text
x_r 的附加噪声强度低于 x_n
```

这足以构造相对质量顺序，但不能宣称 `x_r` 完全无噪声。

---

## 3. 旧方案的最终实验结论

### 3.1 已经完成的关键消融

| 实验 | 低 SNR 平均 Acc | 全 SNR 平均 Acc | 结论 |
| --- | ---: | ---: | --- |
| `TriView-NoRestore` | 约 50.48% | 约 73.82% | 不含恢复的三视图版本 |
| `PairView-Curriculum` | 50.43% | 73.79% | 与三视图几乎相同 |
| `TriView-FixedMix` | 50.60% | 73.87% | 固定混合不弱于课程 |
| V4 Multi-Level Restore | 约 50.4% | 约 73.8% | 恢复目标无独立收益 |

`PairView-Curriculum` 的五次评估为：

| SNR | Acc |
| ---: | ---: |
| -10 dB | 11.94% +/- 0.11% |
| -5 dB | 53.47% +/- 0.19% |
| 0 dB | 85.87% +/- 0.04% |
| 5 dB | 90.68% +/- 0.04% |
| 10 dB | 91.36% +/- 0.03% |
| 15 dB | 91.54% +/- 0.02% |
| 20 dB | 91.69% +/- 0.01% |

`TriView-FixedMix` 的五次评估为：

| SNR | Acc |
| ---: | ---: |
| -10 dB | 11.99% +/- 0.05% |
| -5 dB | 53.82% +/- 0.22% |
| 0 dB | 86.00% +/- 0.05% |
| 5 dB | 90.72% +/- 0.05% |
| 10 dB | 91.35% +/- 0.02% |
| 15 dB | 91.54% +/- 0.02% |
| 20 dB | 91.68% +/- 0.01% |

### 3.2 能够下出的结论

#### 第二个噪声视图没有贡献

三视图相对双视图的低 SNR 平均差距约为：

```text
50.48% - 50.43% = 0.05%
```

该差距小于评估波动，不能作为有效贡献。

#### 渐进课程没有贡献

固定混合的低 SNR 平均反而略高：

```text
固定混合 50.60% > 渐进课程 50.48%
```

因此不能声称 `0 dB -> -5 dB -> -10 dB` 的顺序带来了收益。

#### 恢复路线没有贡献

恢复版本与不恢复版本基本持平，说明模型提升主要来自低 SNR 身份监督，而不是教师或恢复约束。

### 3.3 旧方案应该如何处理

旧方案不是删除实验记录，而是作为完整负结果保留：

- 它证明了“加噪训练”和“恢复机制”必须分开归因；
- 它证明了更多视图不等于更强鲁棒性；
- 它为新方案确定了强基线和公平协议；
- 它不再作为论文主方法继续调参。

---

## 4. 当前 MSFTFNet 到底已经有什么

在 `models/MSFTFNetFeature.py` 中，当前编码器已经包含以下结构。

### 4.1 时域分支

时域输入为：

```text
[B, 2, 256]
```

其中两个通道分别是 I 和 Q。

时域主干包含多个尺度：

- 卷积核 3；
- 卷积核 5；
- 卷积核 9；
- 卷积核 3、膨胀率 2。

这些分支用于捕获短时局部变化、中尺度结构和更长感受野信息。

### 4.2 频域分支

模型先将 I、Q 组成复信号，执行复数 FFT，再构造：

- `log(1 + magnitude)`；
- `cos(phase)`。

频域分支随后使用卷积和多尺度特征块提取频谱指纹。

### 4.3 当前已经存在的自适应结构

当前模型不是完全固定融合，它已有：

1. `AdaptiveTFEnhancer`：从时域和频域特征的均值、标准差生成两个可靠性系数，并控制残差修正强度；
2. `TimeFrequencyFusion`：根据时域和频域全局均值产生两个 softmax 权重；
3. Transformer：在融合后的序列上建模长距离依赖；
4. 平均池化与最大池化联合输出 1024 维特征。

这意味着以下表述不能再作为新创新：

```text
“根据输入动态融合时域和频域特征”
```

因为当前代码已经这样做了。

### 4.4 当前门控的不足

当前结构仍存在可以研究的问题：

- 多个时间尺度先拼接再固定投影，尺度可靠性没有显式输出；
- 时频门控只有两个权重，粒度较粗；
- `reliability` 只是一个自由学习的门值，没有质量监督或顺序校准；
- 无法证明门值会随 SNR 合理变化；
- 没有评价门控是否在低 SNR 下抑制了不可靠分支；
- 当前门控的提升尚未与固定平均融合公平隔离。

新方案正是从这些不足出发，而不是简单添加另一个注意力模块。

---

## 5. 新方案：质量校准的多尺度时频路由

### 5.1 暂定名称

中文：

```text
面向低信噪比特定辐射源识别的质量校准多尺度时频路由网络
```

英文：

```text
Quality-Calibrated Multi-Scale Time-Frequency Routing Network
for Low-SNR Specific Emitter Identification
```

方法简称暂定：

```text
RobustSEI_QCRouter
```

名称在投稿前还可以调整，目前主要用于区分实验目录和代码。

### 5.2 核心思想

不同噪声水平不会等比例破坏全部特征。例如：

- 极低 SNR 下，微弱局部频谱细节可能最先失真；
- 某些中长尺度时域结构仍可能保持稳定；
- 高频细节在高 SNR 下有区分力，但在低 SNR 下可能成为噪声；
- 对每个输入固定使用相同融合权重并不理想。

因此，模型不再恢复一个假定的干净表示，而是：

```text
产生候选特征 -> 估计当前质量 -> 评估各分支可靠性 -> 动态选择证据
```

### 5.3 整体流程

```text
输入 IQ 信号 x
      |
      +-----------------------------+
      |                             |
      v                             v
多尺度时域特征                  幅度/相位频域特征
T3, T5, T9, Td                 F3, F5, ...
      |                             |
      +-------------+---------------+
                    |
                    v
       分支统计与跨域一致性描述
       mean/std/max/cosine agreement
                    |
                    v
          无真实SNR输入的质量估计器
                    |
                    v
       质量条件的尺度与域可靠性路由器
                    |
                    v
            加权融合的特征序列
                    |
                    v
               Transformer
                    |
                    v
            mean/max pooling + ID头
                    |
                    v
                设备类别
```

### 5.4 多尺度候选特征

设模型获得 `K` 个候选分支：

```text
B = {B1, B2, ..., BK}
```

第一版建议只使用已有结构自然产生的分支，避免扩大网络：

- 四个时域尺度分支；
- 一个频域分支，或频域内部两个尺度分支；
- 不新增第二套独立编码器。

所有分支首先投影到相同通道数和序列长度，再参与加权融合。

### 5.5 分支描述量

对每个分支 `Bk` 提取轻量统计：

```text
sk = concat(mean(Bk), std(Bk), max(Bk))
```

还可以加入时频域之间的余弦一致性，但第一版不建议加入过多人工统计。最小实现使用均值和标准差即可。

### 5.6 质量估计器

质量估计器接收全部分支统计，输出标量：

```text
q_hat = sigmoid(MLP(concat(s1, ..., sK)))
```

`q_hat` 表示模型估计的相对信号质量，而不是精确 SNR 数值。

重要约束：

- 推理时不输入真实 SNR；
- 不允许根据测试文件名选择权重；
- 不允许使用测试集标签；
- 合成训练时的 SNR 只用于构造相对顺序监督。

### 5.7 质量条件路由器

每个分支的路由分数为：

```text
ak = Router(sk, q_hat)
```

归一化权重为：

```text
alpha_k = softmax(a1, ..., aK)
```

融合特征为：

```text
Z = sum(alpha_k * Project(Bk))
```

建议为路由权重设置一个很小的下限或使用温度参数，防止训练初期所有权重塌缩到单一分支。第一版不使用稀疏专家、Top-K 路由或额外负载均衡损失。

### 5.8 为什么它与当前门控不同

| 当前 MSFTFNet | 新 QCRouter |
| --- | --- |
| 多尺度先拼接，尺度权重不可见 | 显式保留并加权每个尺度 |
| 只在时域和频域之间输出两个权重 | 同时输出尺度级和域级权重 |
| reliability 没有质量校准 | 使用同一样本的附加噪声顺序校准质量 |
| 未检查权重随 SNR 的变化 | 必须报告权重-SNR曲线和分支稳定性 |
| 很难解释低 SNR 下使用了什么证据 | 可以观察不同 SNR 下的路由行为 |

---

## 6. 训练数据与损失函数

### 6.1 只使用双视图

三视图已经被证明没有额外收益，因此新训练只使用：

```text
参考视图 xr + 一个附加AWGN视图 xn
```

这样可以减少计算量，也使质量顺序定义清楚。

### 6.2 固定低 SNR 混合

不再使用渐进课程。建议沿用已验证稳定的固定混合：

```text
-10 dB : -5 dB : 0 dB = 6 : 7 : 2
```

正式比较中，基线和新方法必须复用完全相同的采样函数、比例和随机种子。

### 6.3 身份分类损失

参考视图和噪声视图都进行身份分类：

```text
L_ID = CE(p_r, y) + lambda_n * CE(p_n, y)
```

第一版建议 `lambda_n = 1.0`。

### 6.4 质量顺序损失

因为 `xn` 是在 `xr` 上继续加入噪声得到的，所以有相对关系：

```text
quality(xr) >= quality(xn)
```

使用简单 margin ranking：

```text
L_QR = max(0, margin - (q_r - q_n))
```

完整损失：

```text
L = L_ID + lambda_q * L_QR
```

建议初始范围：

```text
margin = 0.1
lambda_q = 0.05 或 0.1
```

只允许在预先定义的小范围内选择一次，不进行大规模参数搜索。

### 6.5 为什么不直接监督真实 SNR 分类

不把质量估计器训练成 3 类或 7 类 SNR 分类器，原因是：

- 目标是判断相对可靠性，不是精确测量 SNR；
- 数据集原始样本本身可能已有噪声；
- 精确 SNR 标签只对合成 AWGN 成立；
- 顺序约束更容易推广到未见的中间 SNR；
- 可以减少一个分类头和一项复杂损失。

---

## 7. 两个候选创新点

### 创新点一：面向细粒度射频指纹的多尺度时频表示

这部分建立在 MSFTFNet 已有工作上，不能简单宣称“首次使用多尺度或时频融合”。合理表述应该是：

> 针对低 SNR 下不同尺度指纹退化不一致的问题，构建显式保留局部、中尺度、长感受野及时频互补信息的细粒度指纹候选空间。

最终能否作为独立创新，取决于与：

- CVTSLANet；
- WiSigCNN；
- 单时域 MSFTFNet；
- 单频域 MSFTFNet；
- 当前固定拼接 MSFTFNet；

的消融结果。

### 创新点二：SNR不可知的质量校准尺度路由

这是新方案的主要创新：

> 使用同一样本的参考/附加噪声顺序作为弱监督，在不依赖推理阶段真实 SNR 的情况下学习输入质量，并动态选择当前样本中可靠的尺度和时频证据。

它必须通过以下证据成立：

- 相同 AWGN 暴露下优于固定平均融合；
- 相同 AWGN 暴露下优于当前二路时频门控；
- `QCRouter + CE` 已有结构收益；
- 加入 `L_QR` 后进一步提升且不牺牲干净准确率；
- 估计质量与 SNR 呈合理单调关系；
- 路由权重不是恒定或单分支塌缩；
- 未见 SNR 上仍然有效。

### 不应声称的内容

不得声称：

- 首次提出注意力或动态融合；
- 首次将时域和频域结合；
- 模型完成了噪声去除或恢复；
- 原始视图是绝对无噪声真值；
- 仅凭 ManyTx 一个数据集达到普适鲁棒性；
- 仅凭五次测试噪声重复证明训练稳定性。

---

## 8. 公平实验矩阵

### 8.1 第一阶段：机制验证

所有实验采用同一数据划分、初始化、训练轮数、AWGN采样、分类头和随机种子。

| 编号 | 模型 | 融合 | 质量顺序损失 | 目的 |
| --- | --- | --- | --- | --- |
| A0 | MSFTFNet | 固定平均/固定投影 | 无 | 最基础融合基线 |
| A1 | 当前 MSFTFNet | 当前二路门控 | 无 | 现有门控强基线 |
| A2 | MSFTFNet-QCRouter | 尺度级质量路由 | 无 | 单独验证路由结构 |
| A3 | MSFTFNet-QCRouter | 尺度级质量路由 | 有 | 完整方法 |

这四组都使用双视图固定混合 AWGN。不得让 A2/A3 多看一个噪声视图，也不得使用不同 SNR 比例。

### 8.2 第二阶段：结构消融

只在 A3 第一阶段通过后运行：

| 实验 | 删除内容 | 验证问题 |
| --- | --- | --- |
| B0 | 删除频域分支 | 频域证据是否必要 |
| B1 | 删除多尺度，仅单尺度 | 多尺度是否必要 |
| B2 | 删除质量标量，只用分支统计 | 质量校准是否必要 |
| B3 | 真实SNR只作为上界，不作为正式方法 | 理想质量信息的性能上限 |

`B3` 只能作为 oracle 上界，不能进入正式方法结果。

### 8.3 第三阶段：泛化验证

训练 SNR 为：

```text
-10, -5, 0 dB
```

未见 SNR 建议测试：

```text
-12, -8, -3, 2, 7, 12, 17 dB
```

还应报告：

- 每个 SNR 的 Acc 和 Macro-F1；
- 低 SNR 平均；
- 全 SNR 平均；
- 最坏 SNR 准确率；
- 从高 SNR 到低 SNR 的性能下降面积或曲线面积；
- 推理参数量、FLOPs 或 MACs、单批推理时间；
- 质量估计与 SNR 的 Spearman 相关系数；
- 各尺度平均权重随 SNR 的变化曲线；
- 路由熵，检查是否塌缩。

### 8.4 随机性要求

五次 `evaluate_snr.py` 重复只改变测试噪声，它不能替代独立训练。

论文主结果至少需要：

```text
训练种子：2024、2025、2026
每个训练模型：固定测试种子集合
最终报告：跨训练种子的 mean +/- std
```

关键方法差异建议使用配对统计检验或至少报告逐种子差值和置信区间。

### 8.5 第二数据集

第二数据集不是现在立即加入，但在主方法和关键消融稳定后必须加入。它用于证明结果不是 ManyTx 数据划分或特定信号格式造成的。

选择第二数据集时优先满足：

- 公开可下载；
- 设备数不少于 10；
- 有原始 IQ 或可重复的预处理；
- 有多个采集时段、接收机、距离或信道条件更好；
- 能构造与 ManyTx 一致的 SNR 评估协议。

---

## 9. 成功、部分成功与失败标准

### 9.1 第一阶段继续门槛

完整方法 A3 相对强基线 A1，应满足：

- 低 SNR 平均 Acc 提高至少 `1.0` 个百分点；
- `-10 dB` Acc 提高至少 `1.5` 个百分点；
- `-5 dB` 或 `0 dB` 至少一个档位稳定提高；
- 无额外噪声 Acc 下降不超过 `0.5` 个百分点；
- 五次测试噪声重复的提升大于标准差；
- 三个独立训练种子方向一致。

这些是项目内部继续门槛，不是通用的“高水平期刊录用线”。论文是否有竞争力还取决于公开基线、第二数据集、理论动机、统计完整性和写作质量。

### 9.2 完全成功

满足第一阶段门槛，并且：

- A2 明显优于 A1，证明结构本身有效；
- A3 优于 A2，证明质量校准有效；
- 未见 SNR 仍然领先；
- 路由权重随质量发生合理变化；
- 第二数据集复现主要趋势。

### 9.3 部分成功

- A2 优于 A1，但 A3 不优于 A2：保留路由，删除顺序损失；
- A3 只改善 `-10 dB`，但牺牲高 SNR：研究性能折中，不能直接宣称全面鲁棒；
- ManyTx 有效、第二数据集无效：只能定位为数据集特定方法，论文级别需要下调。

### 9.4 失败与停止规则

出现以下情况应停止该方向：

- 两次合理实现后，A2/A3相对A1低SNR平均提升仍小于 `0.5%`；
- 权重几乎恒定，或者所有样本都选择同一个分支；
- 提升完全来自更多参数而不是质量路由；
- 干净准确率下降超过 `1%`；
- 多个训练种子结果方向相反。

不要像旧方案一样无限增加门控、教师、恢复器和损失函数。

---

## 10. 与高水平论文的关系

### 10.1 直接相关的射频指纹论文

#### Multi-Channel Attentive Feature Fusion for Radio Frequency Fingerprinting

链接：<https://arxiv.org/abs/2303.10691>

该工作融合 IQ、载波频偏、FFT 和 STFT 等多种信号表示，并使用注意力完成特征融合。它与本方案非常接近，必须优先精读，用来确定：

- 当前方案与已有多通道注意力融合的差异；
- 是否需要 STFT、CFO 等额外输入；
- 对照网络和消融如何设置；
- 不能重复声称的创新内容。

#### SinFormer: A Tailored Transformer for Robust RFFI

链接：<https://arxiv.org/abs/2605.24389>

该工作使用多尺度自注意力并采用两阶段训练提高低 SNR 和信道变化下的鲁棒性。它适合用于比较多尺度建模和鲁棒训练逻辑，但其发布时间很新，引用时要核对正式发表状态、代码和实验可复现性。

#### Radio Frequency Fingerprint Identification for Security in Low-Cost IoT Devices

链接：<https://arxiv.org/abs/2111.14275>

该工作明确研究低 SNR RFFI，并讨论在线增强与多包推理。它提醒我们：AWGN 增强本身就是很强的基线，必须与新结构严格分开归因。

#### A Receiver-Agnostic RFFI Approach in Low SNR Scenarios

链接：<https://ieeexplore.ieee.org/document/10757464/>

该工作将低 SNR 与接收机无关性结合。当前第一阶段不做跨接收机，但后期外部泛化实验和论文问题定义可以参考其协议。

#### RF Fingerprinting Identification in Low SNR Scenarios for AIS

链接：<https://ieeexplore.ieee.org/document/10188589/>

该方向与低 SNR 指纹识别直接相关，适合补充不同信号类型下的低 SNR 方法和评价指标。

### 10.2 可迁移的可靠性融合论文

#### Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion

链接：<https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Embracing_Unimodal_Aleatoric_Uncertainty_for_Robust_Multimodal_Fusion_CVPR_2024_paper.html>

该论文研究如何量化单模态噪声并进行稳健融合。可借鉴的是“先估计不可靠性，再控制融合”的逻辑，不建议照搬其完整对比学习和信息论损失。

#### Enhancing Testing-Time Robustness for Trusted Multi-View Classification

链接：<https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Enhancing_Testing-Time_Robustness_for_Trusted_Multi-View_Classification_in_the_Wild_CVPR_2025_paper.html>

该工作通过证据可靠性降低受损视图的影响。它可以帮助设计可靠性分析指标，但本方案的时域、频域和尺度分支不是独立传感器，不能直接套用多模态叙事。

#### Improving Robustness Against Common Corruptions With Frequency Biased Models

链接：<https://openaccess.thecvf.com/content/ICCV2021/html/Saikia_Improving_Robustness_Against_Common_Corruptions_With_Frequency_Biased_Models_ICCV_2021_paper.html>

该工作说明不同频率偏好的专家可能对不同腐蚀具有互补鲁棒性，并强调鲁棒性提升不能以明显损害干净性能为代价。它与“不同尺度在不同噪声下可靠性不同”的论证相关。

#### Residual Channel Boosts Contrastive Learning for RFFI

链接：<https://arxiv.org/abs/2412.08885>

该工作通过残余信道增强和轻量对比学习提升新环境泛化。当前不要立即加入对比损失，但可用于后期跨信道扩展和第二数据集设计。

### 10.3 阅读时要回答的问题

每读一篇论文，都应记录：

1. 它解决的是低 SNR、跨信道、跨接收机还是开集问题？
2. 输入是 IQ、FFT、STFT、星座图还是人工特征？
3. 它的多尺度或多域特征如何产生？
4. 融合权重是全局固定、样本自适应还是有可靠性监督？
5. 推理时是否使用真实 SNR 或信道标签？
6. 低 SNR 样本是否参与训练？
7. 提升究竟来自增强还是模型结构？
8. 是否有同数据暴露的公平消融？
9. 使用几个数据集、几个设备、几个训练种子？
10. 是否公开代码，代码与论文方法是否一致？

---

## 11. 建议检索关键词

检索时优先使用英文，并交替使用 `SEI`、`RFFI`、`RF fingerprinting` 和 `physical-layer authentication`。

### 11.1 第一优先：射频指纹中的质量感知融合

```text
"radio frequency fingerprint" "quality-aware fusion"
"RF fingerprint identification" "adaptive feature fusion"
"specific emitter identification" "reliability-aware"
"RF fingerprinting" "multi-channel attentive fusion"
"radio frequency fingerprint" "multi-scale feature fusion"
"specific emitter identification" "time-frequency fusion"
"physical layer authentication" "feature reliability"
"RF fingerprint identification" "dynamic routing"
```

### 11.2 第二优先：低 SNR 多尺度时频表示

```text
"low SNR" "multi-scale time-frequency" classification
"low SNR signal classification" "adaptive fusion"
"automatic modulation recognition" "time-frequency fusion" low SNR
"I/Q signal" "multi-scale transformer" noise robustness
"frequency-aware transformer" wireless signal classification
"phase-aware" "RF fingerprint identification"
"magnitude phase fusion" emitter identification
"multi-resolution feature fusion" signal classification
"weak signal classification" adaptive time-frequency network
```

### 11.3 第三优先：可靠性、质量估计和动态路由

```text
"reliability-aware feature fusion" noisy signals
"uncertainty-aware fusion" signal classification
"signal quality estimation" "dynamic feature fusion"
"quality-conditioned routing" neural network
"noise-aware gating" feature fusion
"corruption-aware mixture of experts" classification
"sample-adaptive feature fusion" noise robustness
"confidence-guided feature fusion" low quality signal
"modality reliability estimation" robust fusion
```

### 11.4 第四优先：避免只记住训练 SNR

```text
"unseen SNR" signal classification
"cross-SNR generalization" radio frequency fingerprinting
"continuous SNR evaluation" wireless signal classification
"out-of-distribution SNR" signal recognition
"noise severity generalization" neural network
"continuous corruption severity" classification robustness
"receiver-independent" radio frequency fingerprinting
"cross-day" RF fingerprint identification
```

### 11.5 带代码的检索方式

在 Google Scholar、IEEE Xplore、Web of Science 和 arXiv 搜索标题后，再使用：

```text
论文标题 GitHub
论文标题 code
方法简称 GitHub
site:github.com "radio frequency fingerprint" fusion
site:github.com "specific emitter identification" low SNR
```

筛选优先级：

1. 与 RFFI/SEI 直接相关；
2. 有公开代码和公开数据；
3. 报告逐 SNR 结果；
4. 有融合消融；
5. 有未见环境或第二数据集；
6. 近五年期刊或高水平会议。

不要因为一篇论文有代码就强行加入其所有模块。

---

## 12. 接下来几天的工作安排

### 第 1 天：理解与文献定位

目标：完全理解新问题，不跑正式训练。

需要完成：

- 阅读本文档第 1 至第 9 节；
- 阅读 `models/MSFTFNetFeature.py`；
- 画出当前 `MultiScaleTemporalBlock`、`AdaptiveTFEnhancer` 和 `TimeFrequencyFusion` 的数据流；
- 精读 McAFF、SinFormer、低 SNR LoRa RFFI 三篇直接相关论文；
- 建立文献表格，记录第 10.3 节的十个问题；
- 保存旧方案最终结果，不再启动恢复和课程实验。

当天产出：

```text
literature_review_qcrouter.xlsx 或 literature_review_qcrouter.md
current_msftfnet_flow.png
```

### 第 2 天：实现最小可验证版本

目标：只实现必要代码，不添加额外模块。

需要完成：

- 让多尺度块能够返回各尺度特征；
- 添加质量估计器；
- 添加尺度级路由器；
- 保留当前模型作为 A1，不能覆盖原实现；
- 新建独立方法名和运行目录；
- 添加形状、梯度、权重和数值稳定性测试；
- 输出每批次平均质量和路由权重用于诊断。

当天不跑 60/120 轮。

### 第 3 天：烟雾测试与机制检查

先运行 A0、A1、A2、A3 各 5 轮烟雾测试。

烟雾测试只判断：

- 程序是否正常；
- 是否出现 NaN；
- 路由权重是否全部相同；
- 是否塌缩到一个分支；
- 质量估计是否至少能区分参考视图和 `-10 dB` 视图；
- 训练时间和显存是否可接受。

5轮准确率不用于判定论文方法是否有效。

### 第 4 天：第一轮正式机制实验

烟雾测试通过后，使用一个训练种子完成 A0 至 A3 的 60 轮公平实验。

优先顺序：

```text
A1 当前门控基线
-> A2 新路由但无质量损失
-> A3 完整方案
-> A0 固定平均融合
```

如果 A2/A3 在验证阶段明显崩溃，停止并检查实现，不通过堆参数掩盖问题。

### 第 5 天：固定测试与决策

对 A0 至 A3 使用完全相同的五个测试噪声种子，汇总：

- clean/reference Acc 和 Macro-F1；
- 七个已见 SNR；
- 低 SNR 平均；
- 全 SNR 平均；
- 质量-SNR相关性；
- 路由权重曲线；
- 参数量和推理时间。

然后按第 9 节门槛作出一次明确决定。

### 第 6 至第 7 天：只在通过后继续

如果 A3 通过：

- 运行三个训练种子；
- 运行未见 SNR；
- 完成 B0 至 B2 消融；
- 开始准备第二数据集。

如果 A3 没通过：

- 最多允许一次基于诊断证据的结构修正；
- 第二次仍无进展则停止 QCRouter；
- 转向更明确的跨环境/接收机泛化问题，而不是继续堆融合模块。

---

## 13. 现在到底需要跑什么实验

### 当前立即执行

QCRouter、固定融合基线、质量顺序损失和诊断脚本现已实现。服务器先更新代码并运行自动化测试：

```bash
cd ~/yl/NP3MC/NEW3
git pull
conda activate p3mc

python -m unittest \
  tests.test_qc_router \
  tests.test_evaluate_snr \
  tests.test_low_snr_baseline \
  tests.test_manytx_protocols
```

测试通过后，顺序运行四组 5 轮烟雾测试：

```bash
chmod +x run_qc_smoke_sequence.sh
nohup bash run_qc_smoke_sequence.sh \
  > qc_smoke_sequence.log 2>&1 &

tail -f qc_smoke_sequence.log
```

四组定义为：

```text
A0 = MSFTFNet-Fixed + ID
A1 = 当前MSFTFNet门控 + ID
A2 = MSFTFNet-QCRouter + ID
A3 = MSFTFNet-QCRouter + ID + QUALITY_RANK
```

烟雾测试结束后汇总：

```bash
grep -H -E "Val Set|Router diagnostics|End. Best Record|Traceback|RuntimeError" \
  qc_a0_fixed_smoke.log \
  qc_a1_current_smoke.log \
  qc_a2_router_no_rank_smoke.log \
  qc_a3_router_smoke.log
```

### 当前不要执行

- 不再跑 V4/V5 恢复版本；
- 不再调教师 EMA 或固定教师；
- 不再改变三视图数量；
- 不再调课程阶段；
- 不立即跑 120 轮；
- 不立即加入 Rayleigh、Rician、多径或第二数据集；
- 不跳过 A0/A1 公平基线直接只跑完整方法。

### 实现完成后的正式顺序

```text
代码单元测试
-> 5轮 A0/A1/A2/A3
-> 检查质量和权重行为
-> 60轮单种子公平实验
-> 5次固定测试噪声评估
-> 继续/停止决策
-> 3个独立训练种子
-> 未见SNR
-> 第二数据集
-> 非AWGN外部泛化
```

烟雾测试通过后，用顺序脚本运行第一轮 60 轮正式训练：

```bash
chmod +x run_qc_full_sequence.sh
nohup bash run_qc_full_sequence.sh \
  > qc_full_sequence.log 2>&1 &

tail -f qc_full_sequence.log
```

该脚本严格按照 A0、A1、A2、A3 顺序运行，不会让四个程序同时占用一张 GPU。

完整 A3 训练后，除了常规 `evaluate_snr.py`，还必须运行路由诊断：

```bash
python evaluate_qc_router.py \
  -e MSFTFNet-QCRouter -d manytx \
  --method_name RobustSEI_CleanAnchor_QCRouter \
  --checkpoint best \
  --eval_seed 2024 \
  | tee qc_a3_router_diagnostics.log
```

诊断结果中应看到：

- `quality` 随 SNR 整体上升；
- `Quality-SNR rank correlation` 为明显正值；
- 五个路由权重不是始终完全相同；
- `route_entropy` 没有立即降到接近 0；
- 不同 SNR 下至少部分分支权重发生稳定变化。

---

## 14. 高水平期刊所需的证据链

仅把准确率提高一点并不足以构成高水平论文。完整证据链应包括：

### 问题证据

- 展示不同尺度和时频分支在不同 SNR 下的退化差异；
- 证明固定融合无法同时适应高低 SNR；
- 说明现有门控没有质量校准。

### 方法证据

- 质量估计与真实合成 SNR 有稳定相关性；
- 路由权重随 SNR 合理变化；
- 路由没有塌缩；
- 动态路由在相同参数量或受控参数量下优于固定融合。

### 性能证据

- 优于公开架构基线；
- 优于当前 MSFTFNet 二路门控；
- 优于简单 AWGN 配对训练；
- clean、低 SNR、全 SNR 三者不存在不可接受的交换；
- 未见 SNR 仍然提升。

### 稳定性证据

- 三个独立训练种子；
- 固定测试噪声集合；
- mean、std、逐种子差值；
- 必要的统计检验。

### 泛化证据

- 第二公开数据集；
- 后期至少一种非 AWGN 失真；
- 如果条件允许，增加跨日期、跨接收机或跨信道测试。

### 工程证据

- 参数量和计算量；
- 推理延迟；
- 路由模块增加的开销；
- 代码和配置可复现。

---

## 15. 风险与备选路线

### 风险一：当前门控已经足够

如果 A2 不优于 A1，说明当前二路门控已吸收大部分可获得收益。此时不要继续增加路由层数。

### 风险二：质量估计只学会识别合成噪声

通过未见 SNR、不同噪声实现和第二数据集检验。如果只在训练 SNR 有效，应降低论文主张。

### 风险三：极低 SNR 信息论上不足

ManyTx 有 90 类，`-10 dB` 极其困难。若全部模型都在约 10% 至 12% 附近，应检查混淆矩阵、信号长度和多片段推理上限，而不是无限增加损失。

### 风险四：提升来自参数增加

加入等参数量 MLP 对照，或者缩小主干以控制总参数量。

### 备选路线

如果质量路由经过两次合理实现仍失败，下一条更有价值的路线是：

```text
低SNR训练 + 跨接收机/跨日期域泛化
```

该路线研究噪声鲁棒性与真实环境变化的共同影响，并使用域泛化或信道残差增强。但它属于下一研究阶段，不能与当前 QCRouter 同时展开。

---

## 16. 最终执行原则

1. 每次实验只改变一个核心变量。
2. 增强收益与模型收益必须分开。
3. 不把验证准确率当作最终低 SNR 结论。
4. 五次测试重复不替代多个训练种子。
5. 推理阶段不得使用真实 SNR。
6. 不把数据集原始样本称为绝对干净真值。
7. 未通过消融的模块不写成创新点。
8. 先验证机制，再扩展第二数据集和真实信道。
9. 新方向最多允许一次有证据的修正，不无限改版。
10. 所有结果保留日志、配置、提交号和结构化 JSON。

---

## 17. 当前最终决策

旧的教师多层恢复路线已经结束。

下一阶段唯一主线为：

```text
双视图固定AWGN身份训练
+ 显式多尺度时频候选特征
+ SNR不可知的质量估计
+ 质量校准的尺度级动态路由
```

第一轮只允许两项损失：

```text
身份分类损失 + 质量顺序损失
```

当前不启动新的正式训练。下一步是实现 A0/A1/A2/A3 和诊断输出，然后从 5 轮烟雾测试开始。
