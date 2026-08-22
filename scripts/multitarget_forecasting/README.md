# 五指标联合水质预测与提前预警

本模块把论文任务改为真正的多任务预测：每个站点只训练一个模型，一次前向同时输出
5项水质指标在未来4、8、…、72小时的结果，输出张量形状为 `18 × 5`，共90个值。

第一阶段同时筛选输入表示和输出表示，不使用测试集，也暂不生成昂贵的TabPFN教师。
四个严格配对的输入方案为：

| 参数 | 输入信息 |
| --- | --- |
| `24h` | 过去24小时原值、变化量和缺测掩码 |
| `72h` | 过去72小时原值、变化量和缺测掩码 |
| `7d` | 过去7天原值、变化量和缺测掩码 |
| `multiscale` | 过去24小时序列，加日/年周期、24小时至1年滞后和多尺度统计 |

每种输入分别训练“直接输出原值”和“输出变化量后加回当前值”两个联合模型，共8个模型。
各组使用完全相同的训练、验证样本和联合GRU结构。五个指标分别按训练集统计量标准化，
损失对每个“指标×时距”等权，并通过掩码忽略不合格标签。

训练轮数不再固定为100，也不使用2024验证集早停。每个候选先用截至2023年6月的训练
数据拟合、在2023年下半年选择轮数；最多500轮，每5轮评价一次，连续15次没有改善即
停止。随后从头使用完整2022至2023年数据训练到选定轮数，最后预测2024。训练损失和
内部验证损失曲线都会保存。

## 第一阶段：上仙屋单种子输入筛选

Windows CMD单行运行：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.run --stations "上仙屋" --contexts "24h,72h,7d,multiscale" --target-modes "absolute,delta" --seeds 42
```

八个模型完成后生成报告：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.report --stations "上仙屋" --contexts "24h,72h,7d,multiscale" --target-modes "absolute,delta" --seeds 42
```

预测文件的 `pred` 不是五个独立模型拼接，而是同一个模型的一次输出，形状为
`[验证样本数, 18, 5]`。模型权重同时保存到中文输出目录，后续可直接建立推理与预警入口。
新结果位于 `outputs/多指标联合水质预测/验证集/时间前向早停输入尺度消融/`，不会覆盖
先前固定100轮的失败诊断。

## 预警口径

当前报告使用训练集分位数构造方法学事件：pH使用双侧异常，溶解氧使用低值异常，
高锰酸盐指数、氨氮和总磷使用高值异常。它用于比较4至72小时的召回率、F1和误报率，
不是国家标准或断面考核阈值。正式应用前必须补充每个断面的水功能区类别和对应阈值。

## 全国控站输入尺度验证

25个国控站使用单种子运行全8种输入/输出表示，用于检查单站结论
是否能跨站复现：

```bat
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.run --all-stations --contexts "24h,72h,7d,multiscale" --target-modes "absolute,delta" --seeds 42
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.report --all-stations --contexts "24h,72h,7d,multiscale" --target-modes "absolute,delta" --seeds 42
```

该阶段共200个联合模型。不在架构筛选期直接扩展到5个种子；先锁定
预测头和输出表示，再对最终候选方案做多种子复验。

## 指标专属预测头消融

在全25个国控站输入尺度消融完成后，使用固定24小时输入比较共享
线性头、参数量匹配的共享非线性头和指标专属非线性头。混合输出
表示在所有站点上固定为 pH/溶解氧/氨氮变化量与高锰酸盐指数/总磷原值。

```bat
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.head_ablation_run --all-stations --seeds 42
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.head_ablation_report --all-stations --seeds 42
```

## 同协议XGBoost强基线

专属头消融后，在相同25站、24小时输入、五指标、18时距和评价样本上
分别训练原值与变化量XGBoost。XGBoost每站拟合90个标量回归器，用作准确率
强基线，不宣称为单模型联合输出。

```bat
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.xgboost_baseline --all-stations --target-modes "absolute,delta" --seeds 42 --device cuda
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.xgboost_report --all-stations --seeds 42 --device cuda
```

## 混合输出表示五种子确认

XGBoost同协议比较完成后，只复验共享线性GRU中已经观察到的混合输出表示收益。
统一变化量与混合表示除输出定义外完全一致，均运行冻结的5个随机种子。已有种子42
会自动续跑，不会重新训练；报告写入独立中文目录，不覆盖单种子预测头消融结果。

```bat
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.run --all-stations --contexts "24h" --target-modes "delta" --seeds 17,42,73,101,202
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.head_ablation_run --all-stations --variants "mixed_linear" --seeds 17,42,73,101,202
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.mixed_representation_multiseed_report --all-stations --seeds 17,42,73,101,202
```

该实验仍只使用2024验证集，只检验结论对训练随机性的稳定性。若5个种子方向一致，
才冻结候选结构并一次性进入未使用的2025测试集确认。

## 数据预处理消融

在保持24小时输入、共享线性头、指标混合输出和18×5联合预测不变的前提下，
比较以下四个预处理候选：

| 变体 | 稳健标准化＋Huber | 高锰酸盐指数/氨氮/总磷对数空间 | 软存疑标签与缺测时长 |
| --- | --- | --- | --- |
| `robust_huber` | 是 | 否 | 否 |
| `robust_huber_log` | 是 | 是 | 否 |
| `robust_huber_quality` | 是 | 否 | 是 |
| `robust_huber_log_quality` | 是 | 是 | 是 |

当前共享线性头混合表示结果直接作为基线，不重复训练。主分析保留全25站；
浦阳江出口、闸口和浮石渡的排除只在报告中做敏感性重算，不删除数据也不重训模型。

Windows CMD单行运行：

```bat
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.preprocessing_ablation_run --all-stations --seeds 42
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.preprocessing_ablation_report --all-stations --seeds 42
```

该实验仅物化2022—2024开发期窗口。对数空间预测会反变换到原始浓度单位后再计算
RMSE、NSE和预警指标；报告同时给出全量正式标签与排除软存疑标签两种口径。

不直接对稳健＋Huber组合进行5种子确认。先补齐单组件消融，再验证
预处理是否改变各指标的原值/变化量选择，最后只对冻结组合做5种子。

## 预处理组件拆分与输出表示机制

首先复用既有的统一原值/统一变化量结果，计算训练期归一化4小时波动
与验证集输出表示优势的关联。该命令不训练模型、不读取2025标签：

```bat
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.output_mode_dynamics_report --all-stations --seed 42
```

随后仅训练缺失的B、C：B为“中位数/IQR＋MSE”，C为“均值/标准差＋Huber”。
A原始模型、D稳健＋Huber、E稳健＋Huber＋对数均直接复用既有结果：

```bat
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.preprocessing_component_run --all-stations --variants "robust_mse,standard_huber" --seeds 42
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.preprocessing_component_report --all-stations --seeds 42
```

报告以A–E之间的严格配对RMSE为主，持续性仅作外部参照。得到组件结果后，
再进入预处理与输出表示交互消融。

## 预处理与原值/变化量表示交互消融

这一阶段不把“波动大就用原值、波动小就用变化量”当成已证明结论。
它保留当前五指标混合映射，每次只把一个指标从原值翻为变化量，
或从变化量翻为原值；其他四个指标、GRU结构、24小时输入和18个输出时距不变。

四条预处理链为：A原始均值标准化＋MSE，C均值标准化＋Huber，
D中位数/IQR＋Huber，E中位数/IQR＋Huber＋浓度指标对数变换。
A/C/D/E的当前混合映射基准直接复用已有结果，只训练缺失的逐指标翻转模型。

Windows CMD单行运行：

```bat
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.preprocessing_mode_interaction_run --all-stations --preprocessings "A,C,D,E" --flips "ph,do,codmn,nh3n,tp" --seeds 42
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.preprocessing_mode_interaction_report --all-stations --seeds 42
```

共25站×4预处理×5单指标翻转＝500个筛选模型。命令可断点续跑；
这些筛选模型不保存权重，只保存预测、轮数和训练诊断，减少磁盘占用。
报告以逐站、逐指标、逐时距的严格配对RMSE为主，用对称的配对
对数RMSE比汇总为几何变化百分比，同时报告NSE、
预警结果和按站点聚类重抽样的95%置信区间。该阶段仍只读取2022—2024，
不读取2025测试标签，也不扩展到5个随机种子。

## 最终预处理五种子确认

交互消融后冻结E方案：中位数/IQR标准化、Huber损失，对高锰酸盐指数、
氨氮和总磷做对数变换；输出仍为pH/溶解氧/氨氮变化量与高锰酸盐指数/总磷原值。
下列命令只补齐E的其余4个随机种子，已有种子42会自动复用；A原模型的5种子结果也直接复用。

```bat
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.preprocessing_ablation_run --all-stations --variants "robust_huber_log" --seeds 17,42,73,101,202
.\.venv-tabpfn\Scripts\python.exe -m scripts.multitarget_forecasting.preprocessing_ablation_report --all-stations --variants "robust_huber_log" --seeds 17,42,73,101,202 --report-folder "最终预处理五种子确认"
```

正式报告使用严格配对的对数RMSE比、分种子方向、NSE、预警指标和删站敏感性。
只有5/5个种子均相对A改善，才将E冻结为进入2025测试集的正式预处理。

## 多候选教师因果OOF初筛

冻结E预处理后，不直接假定TabPFN是最优教师。第一轮只在训练期对5个代表站点和
4、24、48、72小时锚点进行严格前向OOF筛选，比较TabPFN、逐输出XGBoost、
联合多输出CatBoost和轻量补丁时序Transformer。补丁Transformer是时序架构对照，
不是对官方PatchTST代码的复现；若时序架构入围，再补正式PatchTST候选。

每个“指标×时距”单独排名，因此允许短时距和长时距由不同教师指导。所有教师
共享24小时可见输入、E预处理、混合输出定义、样本和OOF折。2024和2025标签均不参与
本轮筛选。先运行成本较低的候选：

```bat
.\.venv-tabpfn\Scripts\python.exe -m scripts.teacher_screening.run --representative-stations --models "xgboost,patch_transformer" --horizons "4,24,48,72" --seeds 42 --device cuda
```

CatBoost是可选依赖，若环境还没有安装：

```bat
uv pip install --python .venv-tabpfn\Scripts\python.exe catboost
.\.venv-tabpfn\Scripts\python.exe -m scripts.teacher_screening.run --representative-stations --models "catboost_joint" --horizons "4,24,48,72" --seeds 42 --device cuda
```

最后单独运行最慢的TabPFN，命令可按站点断点续跑：

```bat
.\.venv-tabpfn\Scripts\python.exe -m scripts.teacher_screening.run --representative-stations --models "tabpfn" --horizons "4,24,48,72" --seeds 42 --device cuda
.\.venv-tabpfn\Scripts\python.exe -m scripts.teacher_screening.report --representative-stations --models "tabpfn,xgboost,catboost_joint,patch_transformer" --horizons "4,24,48,72" --seeds 42
```

报告会生成中文文件 `教师指标时距排名.csv`、`教师随时距变化诊断.csv` 和
`教师指标时距权重.csv`。若教师胜者随时距变化，下一阶段补齐入围教师的18个时距，
并严格比较单一教师、分时距硬选择和分时距软加权；同一OOF上估计的权重不作为
无偏性能结果，最终选择只在2024验证集完成，2025继续封存。
