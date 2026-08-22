# TabPFN因果蒸馏4–72小时实验

本模块是论文的正式研究方向。它使用本站过去24小时的9项水质历史，同时预测未来
4、8、…、72小时，共18个直接时距。已完成的 `tabpfn_comparison` 仍是独立的4小时
先导基线，不会被这里的文件覆盖。

## 研究问题

第一阶段同时回答两个问题：

1. 严格时间前向OOF的Delta-TabPFN教师能否提升轻量GRU？
2. 在教师、输入、网络容量、随机种子和损失权重完全相同时，学生直接预测原值还是预测
   相对当前值的变化量更好？

四个学生消融为：

| 学生 | 真实监督目标 | 教师监督 | 最终绝对预测 |
| --- | --- | --- | --- |
| 原值监督GRU | 未来原值 | 无 | 直接输出 |
| 变化量监督GRU | 未来值−当前值 | 无 | 当前值＋输出 |
| 原值因果蒸馏GRU | 未来原值 | 当前值＋教师变化量 | 直接输出 |
| 变化量因果蒸馏GRU | 未来值−当前值 | 教师变化量 | 当前值＋输出 |

教师只训练一次Delta-TabPFN；原值和变化量学生共享同一份教师信息，避免把教师差异混入
表示方式消融。

## 时间协议

- 输入：6个4小时时间步，即过去24小时。
- 输出：18个直接时距，即未来4–72小时。
- 训练：2022–2023；验证：2024；测试：2025起，目前没有测试集运行入口。
- OOF折：2022下半年、2023上半年、2023下半年。
- 每个OOF折只能使用该折开始前已经完整观测到的72小时标签。
- 第一个半年是教师暖启动期，仅参与真实标签监督，不伪造OOF标签。
- 标签只使用质量侧表批准的原始观测。

## 1. 数据与泄漏预检

下面命令不训练模型，也不会检查2025年标签：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.preflight --stations "上仙屋" --all-targets
```

必须看到全部时距 `ready=True`，并且 `OOF因果性审计.csv` 中全部
`strictly_causal=True`，才允许生成教师。

## 2. 先运行无蒸馏的原值/变化量对照

这两组不依赖耗时的TabPFN教师，可以先检查18输出GRU是否正常：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.student --variant supervised_absolute_gru --stations "上仙屋" --targets "pH(无量纲)" --seeds 42
```

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.student --variant supervised_delta_gru --stations "上仙屋" --targets "pH(无量纲)" --seeds 42
```

两组完成后可以立即生成阶段性原值/变化量比较，不必等待教师：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.report --stations "上仙屋" --targets "pH(无量纲)" --seeds 42 --allow-partial
```

`--allow-partial` 只用于阶段检查；论文最终报告不使用该参数，缺少任何教师或学生文件都会
直接报错。

## 3. 生成因果教师

先从上仙屋pH的OOF教师开始。教师种子固定为42，每完成一个“OOF折×时距”都会写入
原子检查点，中断后直接重复命令即可续跑：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.teacher --stations "上仙屋" --targets "pH(无量纲)" --cache oof
```

不要在未审阅缓存前使用 `--force`；它会从头替换当前任务的教师进度。TabPFN v2只支持
单输出，因此18个时距仍需逐个拟合，本阶段会明显慢于GRU。可以用例如
`--horizons 4,24,48,72` 分阶段填写同一个缓存；最终训练蒸馏学生前仍须完成全部18个时距。

## 4. 训练完整学生消融

教师完成后运行全部四种学生和五个固定种子。已经完成的监督GRU会自动跳过：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.student --variant all --stations "上仙屋" --targets "pH(无量纲)" --seeds 17,42,73,101,202
```

## 5. 生成验证集教师并汇总

验证集教师只用于评价TabPFN自身，不参与学生训练：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.teacher --stations "上仙屋" --targets "pH(无量纲)" --cache validation
```

随后汇总：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.report --stations "上仙屋" --targets "pH(无量纲)" --seeds 17,42,73,101,202
```

输出位于 `outputs/TabPFN因果蒸馏长时距/`。报告不会跨不同量纲的指标直接平均原始
RMSE；原值/变化量和蒸馏/无蒸馏均按相同站点、指标、种子和时距配对比较。

## 6. OOF惯性门控

原生TabPFN在长时距可能不如持续性预测。本阶段不使用验证标签设置切换点，
而是对每个时距使用训练集严格因果OOF预测拟合 `alpha_h`：

```text
最终预测 = 当前值 + alpha_h * (原预测 - 当前值), 0 <= alpha_h <= 1
```

`alpha_h=1` 保留原预测，`alpha_h=0` 退回持续性。系数由逐时距过原点受限
最小二乘得到，验证集和测试集标签都不进入拟合。

先生成门控参数（只复用现有OOF缓存，不运行TabPFN）：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.inertia_gate --stations "上仙屋" --targets "pH(无量纲)"
```

然后对教师、监督GRU和蒸馏GRU应用同一门控并生成独立报告：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.inertia_report --stations "上仙屋" --targets "pH(无量纲)" --seeds 17,42,73,101,202
```

输出位于 `outputs/TabPFN因果蒸馏长时距/验证集/惯性门控/`。该阶段不修改或覆盖
原教师缓存、学生预测和原验证报告。

## 7. LSTM与XGBoost同协议基线

两个基线均使用本站过去24小时信息、相同训练/验证边界和18个直接时距，
并预测相对当前值的变化量。LSTM与GRU共享隐层宽度、当前值分支、100轮、
批量和学习率；XGBoost对18个时距分别直接拟合，不使用验证集早停。

首先安装冻结的Python 3.11兼容版XGBoost。它是同协议基线的可选依赖，
单独安装后不会改动TabPFN主环境的锁文件：

```cmd
uv pip install --python .\.venv-tabpfn\Scripts\python.exe xgboost==3.2.0
```

运行两类基线和五个种子：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.protocol_baselines --model all --stations "上仙屋" --targets "pH(无量纲)" --seeds 17,42,73,101,202
```

生成原模型、OOF惯性门控版本与门控蒸馏GRU的统一报告：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.baseline_report --stations "上仙屋" --targets "pH(无量纲)" --seeds 17,42,73,101,202
```

预测、运行时间和报告位于
`outputs/TabPFN因果蒸馏长时距/验证集/同协议基线/`，不覆盖已有结果。

## 8. XGBoost门控归因实验

当TabPFN-OOF惯性门控后的XGBoost成为最优基线时，必须排除“任意模型用自己
OOF校准都能得到相同改进”的解释。本实验为每个XGBoost种子生成三个严格时间前向
OOF折，然后用同一公式拟合XGBoost自身的18个惯性系数。

生成五种子XGBoost OOF与自门控参数：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.xgboost_gate_attribution --stations "上仙屋" --targets "pH(无量纲)" --seeds 17,42,73,101,202
```

共需拟合 `5种子 × 3个OOF折 × 18个时距 = 270` 个CPU XGBoost。每完成一个
“折×时距”就保存，中断后直接重复命令续跑；不要随意加 `--force`。

生成原始XGBoost、TabPFN-OOF门控XGBoost、XGBoost自OOF门控XGBoost和蒸馏GRU的
统一归因报告：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.xgboost_gate_report --stations "上仙屋" --targets "pH(无量纲)" --seeds 17,42,73,101,202
```

训练OOF和门控参数位于 `outputs/TabPFN因果蒸馏长时距/门控归因实验/`，验证报告
位于同级 `验证集/门控归因实验/`。两类门控均不读取验证或测试标签。

## 9. 因果TabPFN蒸馏XGBoost

门控归因实验表明XGBoost自OOF门控优于TabPFN门控，因此门控增益不能作为
TabPFN知识转移的证据。本实验把真正的TabPFN软标签蒸馏放入当前最强的XGBoost学生。

对每个训练样本和时距，冻结目标为：

```text
L = (prediction - true_delta)^2
    + 0.5 * (prediction - causal_TabPFN_OOF_delta)^2
```

只有通过质量掩码的真实标签进入第一项，只有严格时间前向教师预测进入第二项。
XGBoost每行只接收一个标签，代码将上述两个平方误差精确化为等价的加权合成标签，
并非近似或验证集调参。

训练五种子蒸馏XGBoost验证预测、三折严格因果OOF与各自的门控参数：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.distilled_xgboost --stations "上仙屋" --targets "pH(无量纲)" --seeds 17,42,73,101,202
```

命令共拟合 `5种子 × (18个验证模型 + 3折 × 18个OOF模型) = 360`
个CPU XGBoost。验证预测和OOF都逐时距保存，可直接重复命令续跑，不要随意使用
`--force`。

生成监督/蒸馏和原始/自门控的完整配对归因报告：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_distillation.distilled_xgboost_report --stations "上仙屋" --targets "pH(无量纲)" --seeds 17,42,73,101,202
```

报告位于 `outputs/TabPFN因果蒸馏长时距/验证集/蒸馏XGBoost实验/`。必须同时报告
原始模型与自门控模型，不允许只报告门控后的更好指标。
