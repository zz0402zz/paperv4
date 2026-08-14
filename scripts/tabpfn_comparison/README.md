# TabPFN 2024严格比较

本目录实现以下五个模型在同一2024验证协议下的比较：

1. 原变化量Delta-GRU（载入现有冻结检查点）；
2. 同输入近参数量Delta-GRU（载入现有冻结检查点）；
3. 官方当前TabPFN-TS特征管线 + TabPFN-v2主干对照；
4. 当前专用检查点TabPFN-TS-3；
5. Delta-TabPFN-v2。

固定7站、5指标、2020–2023训练或上下文、2024验证、4小时数据、过去24小时
输入和未来4–72小时全部18个时距。原生TabPFN-TS逐站逐指标滚动预测；每次预测
只能看到该预测起点及以前的标签。Delta-TabPFN与匹配GRU使用相同的原值、
`diff1`、有效性mask和当前值，但按站点、指标、时距分别执行TabPFN-v2回归。

原生零样本模型固定`seed=0`。把同一结果复制五次不会形成五种子证据；只有
Delta-TabPFN执行17、42、73、101、202五个预设种子。

论文报告的v2检查点标识是`2noar4o2`，当前维护版`tabpfn==8.1.0`通过公开接口
`ModelVersion.V2`解析v2权重。在没有文件哈希能够证明二者完全相同之前，本实验
把第3个模型严格标成“官方当前v2主干对照”，不把它写成论文指定权重的完全复刻。
这个限制只影响版本归因，不改变时间切分、输入可见性或评价方式。

## 安装

### Windows + RTX 4060 Ti（pip）

使用 **Python 3.11 x64**，并分别建立两个虚拟环境。不能合并：论文主线锁定
`pandas 3.x`，而 `tabpfn-time-series==1.2.0` 要求 `pandas<3`。两个 requirements
均锁定官方 PyTorch CUDA 12.6 wheel；4060 Ti 可用，不需要另装 CUDA Toolkit，但
NVIDIA 驱动必须足够新。

在 PowerShell、项目根目录运行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements\windows-mainline-cu126.txt

py -3.11 -m venv .venv-tabpfn
.\.venv-tabpfn\Scripts\python.exe -m pip install --upgrade pip
.\.venv-tabpfn\Scripts\python.exe -m pip install -r requirements\windows-tabpfn-cu126.txt
```

安装后必须先确认 GPU 真正可见；`True`和显卡名才表示后续 TabPFN 会使用 CUDA：

```powershell
.\.venv-tabpfn\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

如显示 `False`，先更新 NVIDIA 驱动，再删除两个 `.venv*` 环境后按上述命令重装；
不要在 CPU 环境中开始原生 TabPFN-TS 滚动实验。

PowerShell 中运行实验时：

```powershell
$env:PYTHONPATH = "."
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_comparison.run --model frozen_gru
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_comparison.run --model tabpfn_ts_v2 --batch-size 8
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_comparison.run --model tabpfn_ts3 --batch-size 8
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_comparison.run --model delta_tabpfn_v2
.\.venv-tabpfn\Scripts\python.exe -m scripts.tabpfn_comparison.report
```

`--batch-size 8` 只把彼此独立的预测起点拼成批次，不放宽时间可见性；首次可先用
`--batch-size 1` 运行一个已知可续跑的任务，再固定批量执行全量实验。

### macOS（uv）

```bash
cd /Users/zz/Applications/paperv4
UV_PROJECT_ENVIRONMENT=/Users/zz/Applications/paperv4/.venv-tabpfn \
uv sync --project scripts/tabpfn_comparison --locked
```

TabPFN使用独立环境，是因为官方`tabpfn-time-series 1.2.0`依赖`pandas<3`，而
现有论文主系统固定`pandas 3.x`。不要为安装TabPFN而降级或覆盖原`.venv`。

本地模式不需要云API key。首次初始化模型会连接Prior Labs，要求登录并接受相应
权重许可证；完成后权重缓存在本机。无图形登录时，可在
`https://ux.priorlabs.ai`接受许可证并设置`TABPFN_TOKEN`。这不是云推理API key，
模型输入仍在本地计算。实验代码显式关闭匿名遥测。

## 运行

先导出现有两个GRU的逐起点预测，不重新训练：

```bash
PYTHONPATH=. .venv-tabpfn/bin/python -m scripts.tabpfn_comparison.run --model frozen_gru
```

该入口会自动调用现有论文主环境`.venv`加载冻结检查点，避免因为TabPFN隔离环境
的依赖差异改变GRU推理；它只导出预测，不训练也不改写检查点。

分别运行三个TabPFN实验：

```bash
PYTHONPATH=. .venv-tabpfn/bin/python -m scripts.tabpfn_comparison.run --model tabpfn_ts_v2
PYTHONPATH=. .venv-tabpfn/bin/python -m scripts.tabpfn_comparison.run --model tabpfn_ts3
PYTHONPATH=. .venv-tabpfn/bin/python -m scripts.tabpfn_comparison.run --model delta_tabpfn_v2
```

原生滚动预测计算量很大。默认每批1个预测起点，这是最保守、最容易审计的严格
模式。官方实现声明不同`item_id`独立；如果先用小规模一致性检查确认批量与逐起点
结果相同，可显式增加，例如`--batch-size 8`，以换取速度。

也可顺序执行全部步骤：

```bash
PYTHONPATH=. .venv-tabpfn/bin/python -m scripts.tabpfn_comparison.run --model all
```

原生滚动模型默认每32个预测批次（以及最终批次）原子保存`.partial.npz`，再次
执行会逐项核对时间、真值、mask、当前值和协议元数据后续跑；可用
`--checkpoint-every-batches`调整频率。其他模型按站点×指标保存。已有结果与冻结
协议不一致时程序会停止，只有检查清楚且确实要覆盖时才使用`--force`。

全部预测完成后生成五模型报告：

```bash
PYTHONPATH=. .venv-tabpfn/bin/python -m scripts.tabpfn_comparison.report
```

调试时可以查看已完成部分，但部分报告不能用于论文结论：

```bash
PYTHONPATH=. .venv-tabpfn/bin/python -m scripts.tabpfn_comparison.report --allow-partial
```

结果写入`outputs/paper/tabpfn_2024_comparison/`。
