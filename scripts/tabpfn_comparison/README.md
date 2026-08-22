# 单站短历史 TabPFN 基线

本目录检验一个明确的问题：在每个站点独立、只看过去 24 小时（6 个 4 小时步长）时，
TabPFN 预测水质**变化量**能否超过同输入的变化量 GRU。

该目录现作为论文的冻结基线，不在这里直接加入蒸馏逻辑。后续的时间前向 OOF 教师标签、
混合损失和不确定性消融将放入独立的 `scripts/tabpfn_distillation/`，避免改变已有
预测文件对应的代码身份。

它使用项目的正式 V2 观测表和质量侧表，不使用图结构、其他站点特征、重建审阅值或未来
信息。这里的“V2 数据”与 TabPFN 的 `ModelVersion.V2` 不是同一个概念。

## 固定协议

- 数据：`data/processed/v2/quantity_4h_observed.csv` 与质量侧表。
- 时间：2022–2023 训练、2024 验证、2025-01-01 起测试。
- 单位：每个“站点 × 预测指标”独立建模；本站以外的数据不会进入特征。
- 输入：过去 6 步的 9 个水质原值、对应一阶变化量及显式有效性掩码，另加当前预测目标值。
- 输出：未来 1 步（4 小时）的目标变化量；预测值由“当前值 + 预测变化量”还原。
- 标签：仅使用质量侧表批准的原始观测。质量不通过的当前值或未来标签一律不参与拟合或评分。
- 模型：持久性、固定容量的短历史 Delta-GRU、`tabpfn==8.1.0` 的 Delta-TabPFN-v2。

TabPFN 使用内存节省模式，并将验证预测按 16 个起点分批送入 GPU，因此可以在 8 GB
显存设备上运行；这只改变推理调度，不改变数据、模型或评价口径。

GRU 不在验证集上早停或调参；两个学习模型的输入、目标、时间切分和缺失处理完全一致。
验证集用于比较与方案选择，测试集必须在验证结论锁定后显式解锁。

## 安装

TabPFN 保留独立环境，以免改变主项目环境。请在项目根目录执行；Windows 路径请使用
PowerShell 的相应写法：

```bash
UV_PROJECT_ENVIRONMENT=.venv-tabpfn \
uv sync --locked
```

Windows PowerShell 等价命令为：

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$PWD\.venv-tabpfn"
uv sync --locked
$env:PYTHONPATH = "."
```

首次创建 TabPFN 模型会下载本地权重并要求接受 Prior Labs 的许可证。模型输入仍在本地
计算；代码关闭匿名遥测。

## 先运行测试

```bash
PYTHONPATH=. .venv-tabpfn/bin/python -m unittest discover -s tests/tabpfn_comparison -v
```

Windows PowerShell 请把解释器替换为
`.\.venv-tabpfn\Scripts\python.exe`，例如：

```powershell
.\.venv-tabpfn\Scripts\python.exe -m unittest discover -s tests\tabpfn_comparison -v
```

测试覆盖因果窗口、单站隔离、质量掩码、训练期缺失填充和预测文件元数据。

## 验证集试运行

先选择一个真实站点和一个目标。站点名称必须来自 V2 数据；以下用 `<站点名>` 和
`<指标名>` 表示：

```bash
PYTHONPATH=. .venv-tabpfn/bin/python -m scripts.tabpfn_comparison.run \
  --model all \
  --stations "<站点名>" \
  --targets "<指标名>" \
  --seeds 42 \
  --evaluation-split val
```

确认产物后，再执行全站点、全部五个目标和五个预设随机种子：

```bash
PYTHONPATH=. .venv-tabpfn/bin/python -m scripts.tabpfn_comparison.run \
  --model all \
  --all-stations \
  --all-targets \
  --evaluation-split val
```

已有的预测只有元数据完全相同才会自动续跑；确实需要替换时才传入 `--force`。

## 汇总验证结果

汇总命令必须使用与运行命令相同的站点、目标和种子范围：

```bash
PYTHONPATH=. .venv-tabpfn/bin/python -m scripts.tabpfn_comparison.report \
  --all-stations \
  --all-targets \
  --evaluation-split val
```

结果写入：

`outputs/单站短历史TabPFN对比/验证集/`

其中 `TabPFN与GRU对比.csv` 是主比较表；负数 `difference` 和 `relative_pct` 表示
TabPFN 的宏平均 RMSE 低于 GRU。`实验报告.md` 会列出数据、输入和汇总口径。
逐样本文件统一放在一层 `预测结果/` 中，文件名直接标明模型、随机种子、站点和指标。

## 测试集

只在验证集结论和设置锁定后执行。测试会用训练集加验证集重新拟合，但不会读取任何测试
标签作为训练信息：

```bash
PYTHONPATH=. .venv-tabpfn/bin/python -m scripts.tabpfn_comparison.run \
  --model all \
  --all-stations \
  --all-targets \
  --evaluation-split test \
  --test-approved
```

随后将汇总命令的 `--evaluation-split` 改为 `test`。不要把长上下文的 TabPFN-TS 结果
混入本实验的主结论；它回答的是不同的问题。
