# `all_info.txt` 审计字段规范

适用入口：`main_cargo_sl_d1.m` → `gen_sl_dataset.m`。目标样本与 `noise` 样本使用同一套 60 列表头。旧字段保持原有顺序，新字段只追加在末尾。

## 基本约定

- 写盘前的 `y_mix`、`y_signal`、`n_wenz` 和 `y_channels` 使用物理声压单位 μPa。
- WAV 中保存的是乘以共同 `scale_c` 后的数字幅值，WAV 数字幅值本身不是 μPa。
- 近似恢复公式：`physical_pressure_uPa ≈ wav_amplitude / scale_c`。误差来自浮点运算与 WAV 编码量化。
- 字符型不适用值为 `none`；数值型不适用值为 `0`。纯噪声的 `snr_band_db` 为 `-inf`。
- `s1/s2/s3`、`sl_*_1/2/3` 和 `rms_*_1/2/3` 的下标对应三个模型输出通道；不活动通道填 `0`。

## 字段表

| 序号 | 字段 | 单位 | 语义及不适用值 |
|---:|---|---|---|
| 1 | `combIdx` | 无量纲 | split/type 内样本序号。 |
| 2 | `fileA` | 文件名 | 通道 1 的模板文件；不活动或 noise 为 `none`。 |
| 3 | `fileB` | 文件名 | 通道 2 的模板文件；不活动或 noise 为 `none`。 |
| 4 | `fileC` | 文件名 | 通道 3 的模板文件；不活动或 noise 为 `none`。 |
| 5 | `r1_km` | km | 通道 1 传播距离；不活动或 noise 为 `0`。 |
| 6 | `r2_km` | km | 通道 2 传播距离；不活动或 noise 为 `0`。 |
| 7 | `r3_km` | km | 通道 3 传播距离；不活动或 noise 为 `0`。 |
| 8 | `z_src1` | m | 通道 1 源深；不活动或 noise 为 `0`。 |
| 9 | `z_src2` | m | 通道 2 源深；不活动或 noise 为 `0`。 |
| 10 | `z_src3` | m | 通道 3 源深；不活动或 noise 为 `0`。 |
| 11 | `z_recv` | m | 接收深度；noise 无传播信道，填 `0`。 |
| 12 | `fs` | Hz | 采样率。 |
| 13 | `wind` | m/s | Wenz 风速参数。 |
| 14 | `sf` | 无量纲 | Wenz shipping factor。 |
| 15 | `n_rms_uPa` | μPa | Wenz PSD 积分得到的理论噪声 RMS。 |
| 16 | `snr_band_db` | dB | 写盘前带内 signal/noise 功率比；noise 为 `-inf`。 |
| 17 | `sl_1` | dB re 1 μPa @1m | 旧兼容字段，等于通道 1 的 `sl_used_1`；不活动为 `0`。 |
| 18 | `sl_2` | dB re 1 μPa @1m | 旧兼容字段，等于通道 2 的 `sl_used_2`；不活动为 `0`。 |
| 19 | `sl_3` | dB re 1 μPa @1m | 旧兼容字段，等于通道 3 的 `sl_used_3`；不活动为 `0`。 |
| 20 | `er_1` | dB | 通道 1 相对当前样本最弱活动源的实际 ER；不活动/单目标/noise 为 `0`。 |
| 21 | `er_2` | dB | 通道 2 相对当前样本最弱活动源的实际 ER；不活动/单目标/noise 为 `0`。 |
| 22 | `er_3` | dB | 通道 3 相对当前样本最弱活动源的实际 ER；不活动/单目标/noise 为 `0`。 |
| 23 | `scale_c` | 无量纲 | mix 与全部参考通道共用的写盘缩放系数。 |
| 24 | `pre_mix_peak` | μPa | 写盘缩放前 `max(abs(y_mix))`。 |
| 25 | `post_mix_peak` | WAV 数字幅值 | `y_mix * scale_c` 的峰值。 |
| 26 | `pre_rms_mix` | μPa | 写盘缩放前混合信号 RMS。 |
| 27 | `post_rms_mix` | WAV 数字 RMS | 混合信号乘 `scale_c` 后的 RMS。 |
| 28 | `pre_rms_signal` | μPa | 写盘缩放前全部目标和的 RMS；noise 为 `0`。 |
| 29 | `post_rms_signal` | WAV 数字 RMS | 全部目标和乘 `scale_c` 后的 RMS；noise 为 `0`。 |
| 30 | `pre_rms_noise` | μPa | 写盘缩放前本地 Wenz 噪声的实测 RMS；目标和 noise 样本均有效。 |
| 31 | `post_rms_noise` | WAV 数字 RMS | Wenz 噪声乘 `scale_c` 后的 RMS；目标和 noise 样本均有效。 |
| 32 | `pre_rms_1` | μPa | Bellhop 与包络处理后，通道 1 写盘缩放前 RMS；不活动为 `0`。 |
| 33 | `pre_rms_2` | μPa | Bellhop 与包络处理后，通道 2 写盘缩放前 RMS；不活动为 `0`。 |
| 34 | `pre_rms_3` | μPa | Bellhop 与包络处理后，通道 3 写盘缩放前 RMS；不活动为 `0`。 |
| 35 | `post_rms_1` | WAV 数字 RMS | 通道 1 乘共同 `scale_c` 后的 RMS；不活动为 `0`。 |
| 36 | `post_rms_2` | WAV 数字 RMS | 通道 2 乘共同 `scale_c` 后的 RMS；不活动为 `0`。 |
| 37 | `post_rms_3` | WAV 数字 RMS | 通道 3 乘共同 `scale_c` 后的 RMS；不活动为 `0`。 |
| 38 | `sl_raw_1` | dB re 1 μPa @1m | SL 表读取的通道 1 逐录音原始 SL；不活动为 `0`。 |
| 39 | `sl_raw_2` | dB re 1 μPa @1m | SL 表读取的通道 2 逐录音原始 SL；不活动为 `0`。 |
| 40 | `sl_raw_3` | dB re 1 μPa @1m | SL 表读取的通道 3 逐录音原始 SL；不活动为 `0`。 |
| 41 | `sl_used_1` | dB re 1 μPa @1m | 通道 1 实际送入声呐方程的 SL；不活动为 `0`。 |
| 42 | `sl_used_2` | dB re 1 μPa @1m | 通道 2 实际送入声呐方程的 SL；不活动为 `0`。 |
| 43 | `sl_used_3` | dB re 1 μPa @1m | 通道 3 实际送入声呐方程的 SL；不活动为 `0`。 |
| 44 | `er_raw_pair_db` | dB | 双目标时按组合中的两个活动目标顺序计算 `sl_raw(1)-sl_raw(2)`；单目标/noise 为 `0`。 |
| 45 | `er_used_pair_db` | dB | 双目标时按同一顺序计算 `sl_used(1)-sl_used(2)`；单目标/noise 为 `0`。 |
| 46 | `overlap_start_s` | s | 两个实际包络都大于 0 的第一个采样点时间；无有效重叠、单目标或 noise 为 `0`。 |
| 47 | `overlap_end_s` | s | 两个实际包络都大于 0 的最后一个采样点时间；无有效重叠、单目标或 noise 为 `0`。 |
| 48 | `overlap_samples` | samples | 两个实际包络同时大于 0 的采样点数量；无有效重叠、单目标或 noise 为 `0`。 |
| 49 | `sir_rx_global_db` | dB | 双目标整个 5 秒窗口的接收端非负强弱比；无效、单目标或 noise 为 `0`。 |
| 50 | `sir_rx_overlap_db` | dB | 双目标实际重叠采样点内的接收端非负强弱比；无效、单目标或 noise 为 `0`。 |
| 51 | `sir_global_valid` | 0/1 | `1` 表示两路全局接收 RMS 均为有限正数，global SIR 有效；否则为 `0`。 |
| 52 | `sir_overlap_valid` | 0/1 | `1` 表示存在实际重叠且两路重叠 RMS 均为有限正数；否则为 `0`。 |
| 53 | `sir_rx_global_signed_db` | dB | 按类别组合顺序计算的全带宽有符号 SIR：第一活动目标减第二活动目标。 |
| 54 | `sir_rx_overlap_signed_db` | dB | 按类别组合顺序计算的重叠区全带宽有符号 SIR。 |
| 55 | `sir_rx_global_band_db` | dB | `[20,4000] Hz` 内整个窗口的非负强弱比。 |
| 56 | `sir_rx_overlap_band_db` | dB | `[20,4000] Hz` 内实际重叠采样点的非负强弱比；最终配对范围以此字段为依据。 |
| 57 | `sir_rx_global_band_signed_db` | dB | 按类别组合顺序计算的 `[20,4000] Hz` 全局有符号 SIR。 |
| 58 | `sir_rx_overlap_band_signed_db` | dB | 按类别组合顺序计算的 `[20,4000] Hz` 重叠区有符号 SIR。 |
| 59 | `sir_global_band_valid` | 0/1 | `1` 表示两路全局带内 RMS 均为有限正数。 |
| 60 | `sir_overlap_band_valid` | 0/1 | `1` 表示存在实际重叠且两路重叠区带内 RMS 均为有限正数。 |

## `scale_c` 定义

令：

```text
common_peak = max(abs(y_mix), abs(s1), abs(s2), abs(s3))
scale_c = 0.95 / common_peak,  common_peak > 0
scale_c = 1,                   common_peak = 0
```

随后统一执行：

```text
mix_wav = y_mix * scale_c
s1_wav  = s1    * scale_c
s2_wav  = s2    * scale_c
s3_wav  = s3    * scale_c
```

不允许逐参考通道单独归一化。若 `scale_c > 1`，生成器保留该值并发出 warning，不会静默改写。

## `sl_raw`、`sl_used` 与 ER

- `sl_raw` 来自逐录音 SL 表。
- `sl_used` 是声呐方程实际使用的 SL。
- bp12 主线为 `er_policy=natural`，因此所有活动通道必须满足 `abs(sl_used-sl_raw) <= 1e-9 dB`。
- `er_raw_pair_db` 与 `er_used_pair_db` 是有符号双目标差值；bp12 natural 主线中二者必须在 `1e-9 dB` 内相等。
- `er_1/2/3` 是旧兼容字段，表示各活动通道相对最弱活动源的非负 ER，和有符号 pair 字段用途不同。

## Bellhop 后接收端 SIR

SIR 测量函数始终是只读的。生产主线使用 `sir_mode=stratified`：它只根据测得的 `sir_rx_overlap_band_db` 决定接受或重抽双目标候选，不修改 SL/ER，也不调整任何源信号或接收信号幅度。Train-only pilot 独立强制 `measure_only`，保留全部自然候选。

SIR 的测量位置为：source SL 定标 → Bellhop 多途卷积 → 去 DC → 应用实际时变包络 → **测量 SIR** → 叠加 Wenz → 计算共同 `scale_c`。因此 SIR 不包含 Wenz，且使用写盘缩放前的物理接收信号。

双目标整个窗口的全局 SIR：

```text
receive_rms_i = rms(y_channel_i)
sir_rx_global_signed_db = 20 * log10(receive_rms_first / receive_rms_second)
sir_rx_global_db = abs(sir_rx_global_signed_db)
```

双目标重叠 SIR 使用两个实际包络均大于 0 的采样点：

```text
overlap_mask = (envelope_1 > 0) AND (envelope_2 > 0)
overlap_rms_i = rms(y_channel_i(overlap_mask))
sir_rx_overlap_signed_db = 20 * log10(overlap_rms_first / overlap_rms_second)
sir_rx_overlap_db = abs(sir_rx_overlap_signed_db)
```

其中 `first/second` 严格跟随类别组合顺序，例如 `0_2` 中 first 为类别 0、second 为类别 2，与 `er_raw_pair_db=sl_raw(1)-sl_raw(2)` 的方向一致。

冻结的 SIR 审计频带为 `[20,4000] Hz`。带内 RMS 对整个窗口或实际重叠片段分别执行 FFT，并依据 Parseval 定理累计该频带内的单边谱功率；不依赖额外滤波器工具箱。带内 signed/unsigned SIR 使用与上式相同的顺序和绝对值关系。

`overlap_start_s=(first_sample-1)/fs`，`overlap_end_s=(last_sample-1)/fs`，`overlap_samples` 是 mask 中为真的采样点数。无实际重叠时三个 overlap 范围字段和 overlap SIR 都记 `0`，同时 `sir_overlap_valid=0`。

单目标和 noise 的全部 SIR 数值均记 `0`，对应的四个 valid 字段均为 `0`，从而与有效的 `0 dB` SIR 区分。常规日志不使用 NaN 或 Inf 表示无效 SIR。

mix 与参考通道共用 `scale_c`，所以内存缩放前后应满足：

```text
sir_before_scale ≈ sir_after_scale
```

该关系只用于交叉验证。SIR 不参与 `scale_c`、SL、ER 或信号幅度计算；生产主线只允许它参与双目标候选的接受/重抽决策。

## 生产双目标 SIR 条件抽样

生产数据集对每个类别对、每个 split 使用相同的冻结规则：

```text
接受范围: 0 <= sir_rx_overlap_band_db <= 15 dB
分层:     [0,5) / [5,10) / [10,15] dB
比例:     45% / 35% / 20%
```

整数配额使用确定性的最大余数法分配；余数并列时优先较低 SIR 层。候选在 source SL 定标、Bellhop、去 DC 和时变包络之后测量：越界、无效或所在层配额已满时丢弃整个候选并重新抽取；合格候选保持原样，随后才生成 Wenz、计算共同 `scale_c` 并写盘。最大候选数为目标样本数的 20 倍，达到上限仍未填满配额时立即报错，不得放宽范围或调整增益。

单目标与 noise 不应用 SIR 条件抽样。每个双目标输出目录写入 `sir_sampling_info.txt`，记录分层参数、目标/实际计数、总候选数和各类拒绝数。该数据集是“满足明确可识别性条件的物理样本集”，不代表未经条件化的自然 SIR 出现频率。
