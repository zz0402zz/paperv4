# paperv4：五指标联合水质长时距预测与因果蒸馏

本项目研究单站水质的长时距预测与提前预警。正式任务要求一个模型一次输出pH、
溶解氧、高锰酸盐指数、氨氮和总磷在未来4、8、…、72小时的全部结果，输出形状为
`18个时距 × 5个指标`。已有单指标TabPFN与蒸馏实验保留为先导证据。

当前论文主线是：

1. 保留单指标TabPFN、GRU和蒸馏XGBoost结果作为方法先导证据；
2. 用五指标联合GRU筛选24小时、72小时、7天和多尺度周期输入；
3. 在代表站点验证输入表示和站点差异；
4. 按严格时间前向OOF筛选TabPFN新版、时间序列基础模型或集成教师；
5. 将最佳教师因果蒸馏到一次输出90个结果的联合学生；
6. 同时评价连续值预测和提前4–72小时的异常事件预警；
7. 方法完全锁定后才运行2025年测试集。

旧 GAT、图传播、流量门控、长时距消融和无关基线代码已从主分支移除。它们仍可从 Git
历史恢复，但不再属于当前论文口径。

## 固定实验协议

| 项目 | 设置 |
| --- | --- |
| 建模数据 | `data/processed/v2/quantity_4h_observed.csv` |
| 质量侧表 | `data/processed/v2/quantity_4h_quality.csv` |
| 训练集 | 2022-01-01 至 2023-12-31；其中2023下半年只用于内部选轮数后再并回重训 |
| 验证集 | 2024-01-01 至 2024-12-31 |
| 测试集 | 2025-01-01 起，方法锁定前不使用 |
| 输入 | 待验证24小时、72小时、7天及多尺度周期表示 |
| 正式输出 | 单模型一次输出未来4–72小时的18时距×5指标 |
| 表示消融 | 直接预测原值 vs. 预测变化量后加回当前值 |
| 正式目标 | pH、溶解氧、高锰酸盐指数、氨氮、总磷 |

数据目录未参与本次代码清理。正式标签只使用质量侧表批准的原始观测，人工重建审阅值
不会进入模型。

## 目录

| 路径 | 用途 |
| --- | --- |
| `scripts/common/` | V2协议、数据读取和终端输出 |
| `scripts/data/` | 从国控原始工作簿重建V2观测表与质量侧表 |
| `scripts/tabpfn_comparison/` | 持久性、匹配Delta-GRU、Delta-TabPFN及结果汇总 |
| `scripts/tabpfn_distillation/` | 18时距因果教师、原值/变化量GRU学生及消融报告 |
| `scripts/multitarget_forecasting/` | 五指标联合输出、输入尺度消融与提前预警评价 |
| `tests/tabpfn_comparison/` | 因果窗口、单站隔离、质量掩码和切分测试 |
| `tests/tabpfn_distillation/` | 18时距、时间边界、OOF因果性和表示转换测试 |
| `tests/multitarget_forecasting/` | 五指标联合形状、输入配对、周期特征和预警测试 |
| `outputs/单站短历史TabPFN对比/` | 当前验证或测试结果、报告及逐样本预测 |
| `outputs/TabPFN因果蒸馏长时距/` | 正式长时距预检、教师缓存、学生预测和消融结果 |
| `outputs/多指标联合水质预测/` | 五指标联合模型、预测、输入消融和预警报告 |

输出采用浅层中文结构：

```text
outputs/单站短历史TabPFN对比/
└── 验证集/
    ├── 预测结果/                 # 模型__种子__站点__指标.npz
    ├── 实验报告.md
    ├── TabPFN与GRU对比.csv
    ├── 总体比较.csv
    ├── 分站点比较.csv
    ├── 分指标比较.csv
    ├── 分时距比较.csv
    ├── 分随机种子比较.csv
    ├── 站点指标时距明细.csv
    ├── 运行摘要.json
    └── 运行清单.json
```

正式测试时会在同级创建 `测试集/`，不会和验证结果混放。

## 环境

主实验使用独立的 `.venv-tabpfn`。在 Windows PowerShell 中执行：

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$PWD\.venv-tabpfn"
uv sync --locked
```

当前环境固定 `tabpfn==8.1.0`，使用本地 TabPFN v2 权重。完整说明见
[TabPFN实验说明](scripts/tabpfn_comparison/README.md)。

需要从保留的国控原始工作簿重新生成V2数据时，使用同一环境执行：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.data.build_processed_quantity_data_v2
```

## 运行当前基线

下面命令可直接在 Windows CMD 中单行执行：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_comparison.run --model all --stations "上仙屋" --targets "pH(无量纲)" --seeds 17,42,73,101,202 --evaluation-split val
```

```cmd
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_comparison.report --stations "上仙屋" --targets "pH(无量纲)" --seeds 17,42,73,101,202 --evaluation-split val
```

当前上仙屋先导验证中，TabPFN相对匹配短历史GRU的RMSE在pH、溶解氧、
高锰酸盐指数和氨氮上分别降低约 `25.45%`、`20.20%`、`31.20%` 和 `21.61%`，
且四个指标均在5/5个随机种子上获胜。它证明了研究方向可行，但不能替代正式的18时距、
因果蒸馏和跨站验证。

## 测试

```cmd
.\.venv-tabpfn\Scripts\python.exe -m unittest discover -s tests\tabpfn_comparison -v
```

正式长时距模块测试：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m unittest discover -s tests\tabpfn_distillation -v
```

五指标联合模块测试：

```cmd
.\.venv-tabpfn\Scripts\python.exe -m unittest discover -s tests\multitarget_forecasting -v
```

测试集入口虽然存在，但在蒸馏方法、消融和验证方案锁定前不得运行。

## 当前五指标联合实验

先运行一次输出五指标的输入尺度消融，命令和冻结口径见
[五指标联合预测说明](scripts/multitarget_forecasting/README.md)。本阶段只运行上仙屋和种子42，
得到结果后再决定保留哪两种输入，不直接扩展全部站点。

## 已完成的单指标长时距实验

实现、冻结协议和分阶段命令见
[TabPFN因果蒸馏4–72小时说明](scripts/tabpfn_distillation/README.md)。先执行只读预检，
再运行无蒸馏原值/变化量对照，随后生成教师缓存并训练因果蒸馏学生。当前版本没有测试集
运行入口，避免在方法锁定前意外读取2025年结果。
