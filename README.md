# paperv4 水质预测实验工程

本项目研究 4 小时时间尺度下的水质预测。当前主线先建立单站点变化量预测模型，再检验直接上游水质变化、物理传播时延、流量权重和事件门控能否为下游预测提供额外信息。

这份 README 是项目的运行入口。每个任务都说明了用途、输入输出和运行命令。除带命令行参数的图实验外，主要实验参数均集中在对应脚本顶部，修改固定变量即可，不需要在命令行传入大量参数。

## 数据

原始监测数据和处理后的建模数据不纳入 Git 仓库。请将本地保存的数据复制到项目根目录的
`data/` 下，再运行数据处理或模型命令。`data/` 已被 `.gitignore` 忽略，避免再次提交
大文件或 Git LFS 对象。

## 一、当前统一实验口径

| 项目 | 当前设置 |
| --- | --- |
| 数据起点 | 2022-01-01 |
| 时间粒度 | 4 小时 |
| 训练集 | 2022-01-01 至 2023-12-31 |
| 验证集 | 2024-01-01 至 2024-12-31 |
| 测试集 | 2025-01-01 至 2025-05-15 |
| 水质输入 | 9 个水质指标 |
| 预测目标 | pH、溶解氧、高锰酸盐指数、氨氮、总磷 |
| 单步任务 | 过去 36 小时预测未来 4 小时 |
| 多步任务 | 过去 24 或 36 小时，直接预测未来 4–36 小时 |
| 当前单站主线 | D：变化量 GRU + 当前目标值 MLP，输出未来变化量 |

正式数据位于 `data/processed/v2/`。正式实验结果位于 `outputs/experiments/v2_reprocessed_20260710/`，并按 `protocol/`、`baselines/`、`gru/`、`graph/`、`reports/` 分类。旧 V1 预处理与结果已清除，后续结果统一使用 V2 口径。

## 二、环境安装

在项目根目录执行：

```bash
cd /Users/zz/Applications/paperv4
uv sync
```

Apple Silicon Mac 会由 PyTorch 自动选择可用设备。模型脚本运行时会打印实际使用的 `mps` 或 `cpu`。

## 三、代码目录

| 目录 | 内容 |
| --- | --- |
| `scripts/data/` | 水质预处理、降雨/流量拆分、站点资料和地图 |
| `scripts/baselines/` | 持久性、Ridge、MLP、TCN、普通 GRU/LSTM 和论文基线 |
| `scripts/gru/` | A/C/D 变化量 GRU、输入窗口、多步预测和站点 embedding |
| `scripts/graph/` | 直接边、传播时延、流量约束、事件门控和连续子图 |
| `scripts/common/` | V2 协议、统一数据接口和公共建模函数 |
| `scripts/reports/` | 训练前预检与正式结果汇总 |

`tests/` 使用相同的分类目录。Python 任务统一从项目根目录使用 `python -m scripts.分类.模块名` 运行，不再依赖散落的脚本路径或临时 `PYTHONPATH`。

## 四、推荐运行顺序

第一次完整复现实验时，建议依次执行：

```bash
# 1. 重建 V2 数据
uv run python -m scripts.data.build_processed_quantity_data_v2

# 2. 检查数据、时间切分、目标掩码和因果协议
uv run python -m scripts.reports.run_v2_stage1_preflight

# 3. 跑单步基线
uv run python -m scripts.baselines.literature_baseline_models

# 4. 跑 A、C、D 三种变化量方案，必须按此顺序
uv run python -m scripts.gru.run_all_station_window_level_ablation
uv run python -m scripts.gru.run_all_station_step_level_ablation
uv run python -m scripts.gru.run_all_station_dual_branch_delta_gru

# 5. 检验共享参数、站点 embedding 和完全独立建模
uv run python -m scripts.gru.run_v2_station_parameter_sharing_ablation --seed 42
uv run python -m scripts.gru.run_v2_station_parameter_sharing_ablation --formal

# 6. 跑 4–36 小时直接多步预测
uv run python -m scripts.gru.run_all_station_multistep_self_gru_ablation

# 7. 比较小时均值、四小时端点和端点加窗口统计
uv run python -m scripts.data.build_hourly_representation_ablation
uv run python -m scripts.gru.run_v2_hourly_representation_multiseed

# 8. 按预测目标筛选自身表示与辅助历史输入
uv run python -m scripts.gru.run_v2_target_input_group_multiseed

# 9. 汇总 Stage 1–3 的正式结果
uv run python -m scripts.reports.build_v2_stage1_stage3_report
```

下面按任务解释每条命令具体做什么。

## 五、数据处理任务

### 任务 1：重建 V2 水质数据

**目的：** 从 `data/quantity/` 的原始站点文件重新生成统一的 4 小时水质数据。该脚本负责重复时间戳处理、已确认的异常值规则、因果 4 小时聚合、质量标记和人工审阅用短缺口重建。

**重要原则：** 正式观测表不写入插值值；重建结果单独保存，只供人工检查，避免模型把人工生成值当成真实标签。

```bash
uv run python -m scripts.data.build_processed_quantity_data_v2
```

主要输出：

- `data/processed/v2/quantity_4h_observed.csv`：正式建模观测值。
- `data/processed/v2/quantity_4h_quality.csv`：每个值的质量状态和目标可用标记。
- `data/processed/v2/quantity_4h_reconstructed_review.csv`：仅供人工审阅的重建结果。
- `data/processed/v2/preprocessing_metadata.json`：输入文件哈希、规则版本和处理统计。
- `outputs/quality/preprocessing_v2/`：重复值、覆盖率和站点指标质量报告。

如需修改预处理范围或短缺口上限，修改 `scripts/data/build_processed_quantity_data_v2.py` 顶部的 `START_DATE`、`RECONSTRUCTION_LIMIT_STEPS` 等固定变量。

### 任务 1B：小时数据表示消融

**目的：** 检验把小时指标聚合成 4 小时均值是否削弱变化信息。三个候选方案使用相同的四小时端点当前值和未来标签，只改变历史输入：四小时均值变化、四小时端点变化、端点变化加窗口 `mean/max/std/slope`。另设同维度的过去错位窗口统计，排除参数量增加带来的假提升。

```bash
# 从原始 XLS 一次性生成端点和因果窗口统计缓存，不覆盖正式 V2 主表
uv run python -m scripts.data.build_hourly_representation_ablation

# 25 站、24h 输入、未来 4h、5 个随机种子正式消融
uv run python -m scripts.gru.run_v2_hourly_representation_multiseed
```

数据缓存位于 `data/processed/v2/ablation_hourly_representation/`，正式结果位于 `outputs/experiments/v2_reprocessed_20260710/gru/stage3d_hourly_representation_ablation/formal_multiseed/`。

当前结论：端点加正确对齐窗口统计的五种子验证宏平均站点 RMSE 为 `0.565089`，优于均值历史 `0.571094`、纯端点 `0.570530` 和同容量错位对照 `0.568671`。收益主要来自 pH 和溶解氧；总磷没有收益。该结果支持把均值降为辅助统计，而不是用均值替代端点真值，但尚未覆盖原正式数据或追溯重跑全部图实验。

### 任务 1C：按目标筛选历史输入组

**目的：** 避免把任务 1B 的 all9 总体结论直接套到五个预测目标。模型输出仍只有 pH、溶解氧、高锰酸盐指数、氨氮和总磷；水温、浊度、电导率和总氮只作为候选历史协变量，不需要预测。第一阶段只判断 pH、溶解氧自身应使用均值、端点还是端点加本指标统计；三个原生 4 小时目标只使用原始 4 小时观测。第二阶段再逐目标判断其他目标历史、非目标辅助端点和辅助统计是否提供额外信息。

```bash
# 25 站、24h 输入、未来 4h、5 个真实随机种子；验证锁定后才打开测试集
uv run python -m scripts.gru.run_v2_target_input_group_multiseed
```

所有候选保持相同输入张量维度和模型容量，未启用通道同时清零数值与有效性掩码。窗口统计另设同容量的过去错位对照。正式门槛为：验证宏 RMSE 至少改善 `0.5%`、至少 `4/5` seeds 获益、至少 `15/25` 站点获益；统计方案还必须至少 `4/5` seeds 优于错位统计。

锁定结果：pH 使用“自身端点与自身统计 + 其他目标端点 + 辅助端点与统计”；溶解氧使用“自身端点与自身统计 + 其他目标端点 + 辅助端点”；高锰酸盐指数、氨氮和总磷只保留自身 4 小时历史。测试宏平均站点 RMSE 分别为 `0.143661`、`0.602910`、`0.573781`、`0.070053` 和 `0.015504`。相对持久性改善 `31.52%`、`34.68%`、`5.46%`、`3.92%` 和 `4.57%`。

结果目录：`outputs/experiments/v2_reprocessed_20260710/gru/stage3e_target_input_group_ablation/formal_multiseed/`。`stage1_decision.json`、`stage2_decision.json` 记录验证门槛，`locked_test_metrics.csv` 记录封存测试结果，`validation_reproduction_audit.json` 验证测试重训没有改变锁定的验证结果。

### 任务 2：V2 数据与因果协议预检

**目的：** 在训练前检查正式数据路径、输入哈希、训练/验证/测试时间边界、变化量标签覆盖率、质量掩码，以及重建值是否错误进入正式目标。

```bash
uv run python -m scripts.reports.run_v2_stage1_preflight
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/protocol/stage1_protocol/`。

只有预检全部通过，后续模型结果才具有可比性。

### 任务 3：重建连续国省控子图面板

**目的：** 将连续河段内的国控、省控水质和日流量整理为相同时间轴，用于连续子图实验。它不是 25 个国控站单站模型的必需步骤。

```bash
uv run python -m scripts.data.build_continuous_subgraph_panel
```

输出目录：`data/processed/v2/continuous_subgraph/`。

### 辅助数据整理工具

这些脚本不参与每次模型训练，只在原始站点资料更新时运行：

```bash
# 将水利厅日流量总表按站点拆分
uv run python -m scripts.data.split_water_resources_daily_flow

# 根据固定坐标台账与本地数据重建国控、省控、水文和降水站地图
uv run python -m scripts.data.build_station_location_map

# 转换分钟/小时最大降雨量旧表
uv run python -m scripts.data.convert_rainfall_extreme_tables

# 将水文日平均表按站点拆分
uv run python -m scripts.data.split_hydro_daily_by_station
```

固定输入、输出路径集中在各脚本顶部；原始 XLS/XLSX 不会被覆盖。

地图输出为 `outputs/maps/station_locations_map.html`。页面支持站点分类开关、站名/河流/站码搜索、站名标签、卫星底图、两点直线测距和未定位站点检查；虚线水文气象标记属于地名或扫描图近似位置，不应直接用于河道传播距离计算。

## 六、单步预测任务

### 任务 4：跑单步基线模型

**目的：** 在同一份 V2 数据和时间切分上比较持久性、Ridge、MLP、TCN、普通 GRU 和 LSTM。任务为过去 9 个时间步（36 小时）预测未来 1 个时间步（4 小时）的 5 个水质指标。

```bash
uv run python -m scripts.baselines.literature_baseline_models
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/baselines/stage2_baselines_9to1/seed_42/`。

这里的结果是**单步基线**，不能直接与 4–36 小时多步结果放在同一张表中比较。

### 任务 5：A 方案，纯变化量 GRU

**目的：** 对每个预测目标，仅输入训练集筛出的 corr-top3 历史变化量序列，检验“预测变化量”本身是否有效。

```bash
uv run python -m scripts.gru.run_all_station_window_level_ablation
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/gru/stage3_change_ablation/A_diff_only_9to1_seed42/`。

### 任务 6：C 方案，逐时间步原始值加变化量

**目的：** 每个历史时间步同时输入该步原始值和对应变化量，检验逐步状态与变化联合输入是否优于纯变化量。

```bash
uv run python -m scripts.gru.run_all_station_step_level_ablation
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/gru/stage3_change_ablation/C_step_raw_diff_9to1_seed42/`。

### 任务 7：D 方案，变化量 GRU 加当前状态 MLP

**目的：** 五个目标分别建模。对每个目标，历史 corr-top3 变化量序列经过 GRU 编码，该目标在预测时刻最近的原始值及其有效性标记经过小型 MLP 编码；两种表示拼接后预测该目标的未来变化量。该方案检验“历史变化规律”和“当前目标状态”分支建模是否更合理。

**站点口径：** 当前实现对每个站点独立前向计算，预测某站时不会读取另外 24 个站点的数据；但 25 个站点共享同一套 GRU、MLP 和输出层参数。因此它属于“全站共享参数的站点独立自回归”，不是 25 套完全独立训练的模型，也不是已经聚合跨站信息的图模型。

```bash
uv run python -m scripts.gru.run_all_station_dual_branch_delta_gru
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/gru/stage3_change_ablation/D_dual_branch_9to1_seed42/`。

注意：该脚本会读取 A、C 方案结果生成对比表，因此应先运行任务 5 和任务 6。

### 任务 8：站点参数共享消融

**目的：** 在数据、特征、缩放、损失函数和时间切分完全相同的条件下，比较三种 D 模型：25 站完全共享参数、共享参数加站点 embedding、每个站点单独训练一套参数。三者都只读取本站历史，不发生跨站消息传递。

```bash
# 预实验：三种方案都跑，包含 125 个独立站点-目标模型
uv run python -m scripts.gru.run_v2_station_parameter_sharing_ablation --seed 42

# 正式实验：只复核通过验证门槛的共享模型与站点 embedding
uv run python -m scripts.gru.run_v2_station_parameter_sharing_ablation --formal
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/gru/stage3b_station_parameter_sharing/`。

当前结论：完全独立建模验证集只改善 10/25 个站点，宏平均 RMSE 由 0.525018 变差到 0.526481，且训练量约增至 25 倍，因此不采用。站点 embedding 在 5 个随机种子的验证集宏平均 RMSE 上赢 4/5 次，均值由 0.523666 降至 0.521577；测试集均值由 0.362968 降至 0.361641。提升约 0.4%，可保留，但不能夸大。

## 七、多步预测任务

### 任务 9：直接预测未来 4–36 小时

**目的：** 比较 C、D 两种方案的直接多步预测。模型一次输出未来第 1–9 步，即 4、8、12、16、20、24、28、32、36 小时，不把前一步预测递归传给下一步。

```bash
uv run python -m scripts.gru.run_all_station_multistep_self_gru_ablation
```

默认设置：过去 9 步（36 小时）输入，未来 9 步（36 小时）输出。

输出目录：`outputs/experiments/v2_reprocessed_20260710/gru/stage3_direct_multistep/CD_9to9_seed42/`。

如需重新训练并比较 24 小时输入下的 LSTM、普通 GRU、主线 Full D、滚动 D、Ridge 和 MLP，使用：

```bash
uv run python -m scripts.gru.run_v2_long_horizon_baseline_comparison
```

该入口不会读取历史实验结果，也不生成 report 或 CSV；各模型的 4–36 小时结果会在终端中逐个输出。`rolling D` 训练单步增量，把上一步预测状态送入下一步 MLP，并将预测出的目标变化量滚入 GRU 历史；未来不可知的其他特征通道按缺失处理。正式五随机种子复核加 `--formal`。

### 任务 10：消融历史输入窗口长度

**目的：** 固定未来 4–36 小时输出，比较不同历史窗口对 D 模型验证集表现的影响。单种子脚本从 12、24、36、48 小时开始，并在最优点落在边界时继续扩展搜索。

```bash
uv run python -m scripts.gru.run_v2_direct_multistep_input_window_ablation
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/gru/stage3_direct_multistep/input_window_ablation_D_9out_seed42/`。

### 任务 11：输入窗口多随机种子复核

**目的：** 使用 5 个随机种子复核 36、48、72、96 小时历史窗口的稳定性，避免根据单次训练偶然选择窗口。

```bash
uv run python -m scripts.gru.run_v2_direct_multistep_input_window_multiseed
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/gru/stage3_direct_multistep/input_window_multiseed_D_9out/`。

该任务训练次数较多，耗时明显高于单种子消融。当前主线最终采用 24 小时输入，以减少训练时间；如需修改候选窗口或种子，修改脚本顶部的 `CANDIDATE_INPUT_STEPS` 和 `SEEDS`。

### 任务 11B：验证 24 小时历史分支是否真的起作用

**目的：** 固定 24 小时输入与未来 4-36 小时直接输出，比较仅当前值、仅历史、完整 D、零历史和错配历史，并加入持久性与直接 Ridge。零历史和错配历史保持完整 D 参数量不变，用于排除“只是参数更多”或“任意历史分布都有效”的解释。

```bash
uv run python -m scripts.gru.run_v2_multistep_history_branch_ablation
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/gru/stage3c_history_branch_ablation_24h/`。每个种子和变体都有独立完成标记，可中断后继续；正式结论只使用验证集制定门槛，测试集随后汇报。

## 八、图信息任务

图实验均以单站变化量模型为 self 基线。严格图必须在验证集上同时优于 self 和时间打乱、反向边、错误来源等负对照，才能作为图传播有效的证据。

### 任务 12：直接上下游站点图消息试验

**目的：** 在少量已确认直接边上，将物理时延对齐后的上游变化信息加入下游多步预测，并和 self、时间打乱及错误来源进行比较。

```bash
# 单随机种子试跑
uv run python -m scripts.graph.run_v2_direct_pair_graph_ablation --seed 42 --pilot

# 仅在验证门槛通过后运行正式多种子实验
uv run python -m scripts.graph.run_v2_direct_pair_graph_ablation --formal
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/graph/stage4_direct_pair_graph/`。

### 任务 13：延迟逐步图修正

**目的：** 将上游逐步变化按物理传播区间对齐，通过无偏置的 9→5 映射得到下游各预测步修正量，再累计到 self 预测上。该任务用于区分“已观测上游消息”和“需要预测的未来上游消息”。

```bash
# 单随机种子试跑
uv run python -m scripts.graph.run_v2_delayed_step_graph_ablation --seed 42 --pilot

# 正式多随机种子复核
uv run python -m scripts.graph.run_v2_delayed_step_graph_ablation --formal
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/graph/stage4b_delayed_step_graph/`。

### 任务 14：流量质量守恒权重

**目的：** 在多入边汇流位置使用前一完整日的流量估算各上游分支权重，并显式保留未监测支流或本地入流比例。比较不加权、旧分支归一化和下游流量约束三种方法。

```bash
# 只检查数据、流量映射和样本覆盖，不训练
uv run python -m scripts.graph.run_v2_flow_mass_balance_ablation --dry-run

# 单随机种子试跑
uv run python -m scripts.graph.run_v2_flow_mass_balance_ablation --seed 42 --pilot

# 仅在验证门槛通过后运行正式实验
uv run python -m scripts.graph.run_v2_flow_mass_balance_ablation --formal
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/graph/stage4c_flow_mass_balance/`。

### 任务 15：受约束的流量传输

**目的：** 在“横山 + 费垅 → 将军岩”上检验更强的物理约束。上游同指标变化乘以因果流量份额和限制在 `[0,1]` 的特征保留系数，再作为下游修正量。

```bash
# 单随机种子试跑
uv run python -m scripts.graph.run_v2_flow_constrained_transport_ablation --seed 42 --pilot

# 仅在验证门槛通过后运行正式实验
uv run python -m scripts.graph.run_v2_flow_constrained_transport_ablation --formal
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/graph/stage4d_constrained_flow_transport/`。

## 九、事件样本任务

### 任务 16：全图突发事件普查

**目的：** 不先训练图模型，而是在全部严格边上统计“上游先发生大变化、下游当时平稳、随后在物理时延窗口内响应”的事件。阈值只由训练集确定，验证集用于筛选，测试集响应结果保持盲态。

```bash
uv run python -m scripts.graph.run_v2_full_graph_event_census
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/graph/stage5_event_census/`。

### 任务 17：浮石渡事件门控图

**目的：** 针对“富足山 + 双港口 → 浮石渡”候选关系，只在上游变化超过训练集事件阈值后开启图修正，检验图信息是否只在突发时期有价值。

```bash
uv run python -m scripts.graph.run_v2_fushidu_event_graph --seed 42
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/graph/stage6_fushidu_event_graph/pilot_seed42/`。

## 十、连续国省控子图任务

### 任务 18：连续五节点子图与负对照

**目的：** 在位置和上下游关系经过单独审计的连续国省控子图上，共享同一个 self 骨干，比较严格图、反向图、随机图、错误支流图和时间打乱图。该任务用于判断提升是否真的来自河网方向，而不是参数量或普通跨站相关性。

先重建子图面板，再运行模型：

```bash
uv run python -m scripts.data.build_continuous_subgraph_panel
uv run python -m scripts.graph.run_continuous_subgraph_pilot
```

输出目录：`outputs/experiments/v2_reprocessed_20260710/graph/stage7_continuous_subgraph/pilot_seed42/`。

如果站点位置、直接边、流量映射或时间覆盖预检不通过，脚本会在训练前停止，不会用不完整图继续训练。

## 十一、报告与测试

### 任务 19：生成 Stage 1–3 汇总报告

**目的：** 合并预检、单步基线、A/C/D 消融和多步结果，生成统一口径的汇总报告。

```bash
uv run python -m scripts.reports.build_v2_stage1_stage3_report
```

输出文件：`outputs/experiments/v2_reprocessed_20260710/reports/stage1_stage3_report.md`。

### 任务 20：运行全部自动化测试

**目的：** 检查数据接口、时间切分、目标掩码、因果流量、图负对照、模型输入输出形状和报告生成逻辑。

```bash
uv run python -m unittest discover -s tests -v
```

只验证某一任务时，可以运行对应测试文件，例如：

```bash
uv run python -m unittest tests.reports.test_v2_stage1_preflight -v
uv run python -m unittest tests.gru.test_v2_direct_multistep_input_window_ablation -v
uv run python -m unittest tests.graph.test_v2_delayed_step_graph -v
uv run python -m unittest tests.graph.test_continuous_subgraph_model -v
uv run python -m unittest tests.gru.test_v2_station_parameter_sharing_ablation -v
uv run python -m unittest tests.gru.test_target_input_group_ablation tests.gru.test_target_input_group_multiseed -v
```

## 十二、如何查看结果

### 终端输出

所有 Python 任务统一使用 `scripts/common/terminal_output.py`。默认终端只显示阶段、设备、进度、最佳 epoch、关键指标摘要和结果目录：训练过程每 10 个 epoch 显示一次，宽表最多预览 10 行，单行最多 160 个字符。完整配置、训练历史、分站点结果和分指标结果仍写入实验目录，不因终端精简而丢失。

需要排查完整日志时使用：

```bash
PAPERV4_VERBOSE=1 uv run python -m scripts.gru.run_all_station_multistep_self_gru_ablation
```

也可以只调整某一项：

```bash
PAPERV4_EPOCH_EVERY=5 PAPERV4_TERMINAL_MAX_LINES=20 PAPERV4_TERMINAL_WIDTH=200 \
uv run python -m scripts.gru.run_all_station_multistep_self_gru_ablation
```

多随机种子入口默认隐藏内部重复训练日志，每个种子结束后只报告验证集胜出方案；详细内容保存在各 `seed_*` 子目录。

每个正式实验目录通常包含：

- `run_manifest.json`：数据哈希、代码哈希、随机种子和实验配置。
- `run_report.md`：本次实验的中文或英文结果摘要。
- `overall_summary.csv` 或 `summary.csv`：整体 MAE、RMSE、NSE。
- `feature_metrics.csv`：每个预测指标的结果。
- `station_metrics.csv`：每个站点的结果。
- `horizon_metrics.csv`：多步任务每个预测时长的结果。
- `history.csv`：训练过程和验证集选择记录。

查看当前研究结论和阶段是否通过，优先阅读：

- `docs/project_ledger.md`：项目总台账。
- `docs/experiment_registry.md`：各阶段实验结论和失败原因。
- `outputs/experiments/v2_reprocessed_20260710/reports/stage1_stage3_report.md`：单站主线结果。
- `outputs/experiments/v2_reprocessed_20260710/gru/stage3b_station_parameter_sharing/formal_embedding_multiseed/formal_report.md`：站点参数共享正式结果。
- `outputs/experiments/v2_reprocessed_20260710/gru/stage3e_target_input_group_ablation/formal_multiseed/formal_report.md`：按目标输入组正式结果。

## 十三、运行纪律

1. 先重建数据并通过 Stage 1 预检，再训练模型。
2. 训练集用于拟合参数，验证集用于选择模型，测试集只做最终报告。
3. 不使用未来降雨、未来流量、未来上游水质或居中插值生成输入。
4. 单步和多步、不同数据版本、不同目标掩码下的结果不得直接混合比较。
5. 图模型必须同时比较 self 和负对照；只优于持久性不能证明图有效。
6. 正式实验必须保留 `run_manifest.json`，并写入版本化输出目录。

当前可写结论是：单站变化量预测有效；pH 和溶解氧需要保留小时端点及本指标窗口统计，但其他历史输入应按目标筛选；站点 embedding 带来小幅稳定增益，完全独立建模不值得采用；直接多步预测有稳定预测能力；现有图增强、流量权重和事件门控尚未形成可复现的总体增益。
