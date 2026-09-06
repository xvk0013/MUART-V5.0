# V6-D1：Train 录音均匀数据消融

## 唯一变量

原 Train 从“全部模板均匀”改为：

1. 在当前类别的冻结 Train 原始录音中均匀选一条；
2. 在该录音的 Train 模板中均匀选一个。

V6-A2 模型、s42、60 epochs、损失、增强、SL、ER、Bellhop、Wenz、
SIR 0–15 dB 分层和七集合样本数全部保持不变。Val 从原 bp12 数据集逐文件
复制，并由 `audit_d1_dataset.py` 做内容 SHA-256 核对。Test 不构造、不复制、
不读取。

逐录音 SL 按原主线的完整录音键排序和标签独立随机种子确定性重建，只保留
冻结 Train 录音，并与原 Train `all_info.txt` 中所有已出现录音逐值核对。
这覆盖了原 Train 随机生成时可能从未抽中的录音，同时不使用舍入后的 SL 表值。

## 为什么值得做

原 Train 每条录音的模板数差异很大：Cargo 中位 36.5、Tanker 13、Tug 79，
最大分别 118/70/212。按模板均匀会让长录音或高模板密度录音被反复抽中，
而 A2 在固定 Train 子集接近 100%、Val 约 81%，符合录音指纹记忆风险。

D1 可能改善 Cargo 的跨录音泛化，但不能保证解决 Cargo/Tanker 的本征重叠，
也不能直接消除传播对 Cargo 的非对称伤害。

## 判读标准

先要求数据审计 PASS：

- D1 候选和最终接受录音曝光均被记录；
- 三类的最终录音曝光 CV 均低于原 Train；
- 每条 Train 录音至少被使用一次；
- 七集合和四个流计数保持；
- 双目标 SIR 配额保持且不调整增益；
- D1 Val 与原 Val 内容哈希完全一致。
- 冻结源池的 Train/Val 模板、录音键和 `Parent_Folder` 代理重叠均为 0；当前
  mapping 没有真实 vessel ID，因此报告必须明确保留“无法证明物理船只身份
  互斥”这一限制，不能把录音互斥写成 vessel 互斥。

训练结果必须联合看：Cargo 单目标、Cargo→Tanker、Tanker→Cargo、两个含 Tug
类别对、总体 EMR、best-NLL/best-EMR 差异、TrainEval/Val gap、错误置信度和
Brier。总体 EMR 单独上涨不构成 D1 成功。

正式运行前冻结两级判据：

- 方向成立：Cargo 正确数高于 A2 的 147/227，单目标 Cargo/Tanker 互换低于
  87，含 Tug 的 Cargo/Tanker 互换低于 222，且总体正确数不低于 A2；
- 强突破：Cargo 至少恢复到先前 V6-A 的 155/227，同时两类互换分别不高于
  76 和 205，且总体不退步。

这些阈值只用于结果判读，不参与 checkpoint 选择；checkpoint 仍只按冻结 Val
EMR 选择。`val_metrics.json` 和 `comparison.md` 会自动写出 D1 verdict。

## 严格执行顺序

代码目录由 runner 自身位置确定，数据生成代码默认位于其下的
`data_gen_v2`。数据根和结果根不写死，运行时分别通过 `-DataRoot`、
`-OutputRoot` 指定；A2 结果不在结果根默认位置时使用 `-ReferenceRun`。

```powershell
cd "<MUART V5.0>"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_v6d1.ps1" -Mode Check -DataRoot "<DATA_ROOT>" -OutputRoot "<OUTPUT_ROOT>"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_v6d1.ps1" -Mode Generate -DataRoot "<DATA_ROOT>" -OutputRoot "<OUTPUT_ROOT>"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_v6d1.ps1" -Mode Audit -DataRoot "<DATA_ROOT>" -OutputRoot "<OUTPUT_ROOT>"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_v6d1.ps1" -Mode Smoke -DataRoot "<DATA_ROOT>" -OutputRoot "<OUTPUT_ROOT>"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_v6d1.ps1" -Mode Train -DataRoot "<DATA_ROOT>" -OutputRoot "<OUTPUT_ROOT>"
```

前一步未 PASS 就停止。正式输出目录非空时拒绝覆盖。

正式数据：

```text
<DATA_ROOT>\data_cargo_sl_sir15_d1_recording_uniform\SS2_bp12\dataset
```

正式训练结果默认使用带毫秒时间戳的全新目录，避免任何已有结果被覆盖：

```text
<OUTPUT_ROOT>\checkpoints_v6d1_recording_uniform_s42_YYYYMMDD_HHMMSS_mmm
```

训练结果目录会同时复制 `d1_audit.json`、`d1_audit.md` 和
`d1_protocol.txt`，传回时复制整个目录即可保留数据侧证据链。
