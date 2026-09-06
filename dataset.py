r"""
M-UART v4.0 数据集加载器 — dual 双通道特征 + DEMON + 波形级增强 (多目标)

数据来源 (MATLAB data_gen/main.m):
  - concat 管线: BELLHOP 多途 + 相遇场景拼接 (+ 可选接收端 Wenz 噪声)
  - 按原始录音文件防泄漏划分
  - 目录结构: {data_combined}/{type}/{Train,Val,Test}/mix/combined_XXXXX.wav

特征:
  - 主输入 dual: ch0 = LOFAR 0-1kHz (1.95Hz/bin, 513 bins)
                 ch1 = 宽带 1-4kHz (7.8Hz/bin, 385→513 插值)
  - DEMON 谱: 带通50-1000Hz → 平方检波 → 低通100Hz → FFT, 0-50Hz, 去 DC
  - 全局归一化: (x - mean) / std (训练集统计量)

    抗过拟合增强 (波形级): 历史频率平移 / 噪声注入 / 时间拉伸 / 随机增益
"""

import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt as sp_filtfilt, welch


NORMALIZATION_MODES = {'global', 'per_channel'}


def normalize_feature(feature, mean, std, mode='global'):
    """用 Train 统计量归一化特征，不做逐样本标准化。

    ``global`` 要求单个标量 mean/std，严格复现 E1。``per_channel`` 要求
    每个输入通道各一个 mean/std，并仅沿频率和时间维广播。
    """
    if mean is None and std is None:
        return feature
    if mean is None or std is None:
        raise ValueError('normalization mean/std 必须同时提供')
    if mode not in NORMALIZATION_MODES:
        raise ValueError(
            f'未知 normalization_mode={mode!r}; '
            f'当前只支持 {sorted(NORMALIZATION_MODES)}'
        )

    mean_t = torch.as_tensor(mean, dtype=feature.dtype, device=feature.device).flatten()
    std_t = torch.as_tensor(std, dtype=feature.dtype, device=feature.device).flatten()
    if mean_t.numel() != std_t.numel():
        raise ValueError('normalization mean/std 元素数量不一致')
    if not torch.isfinite(mean_t).all() or not torch.isfinite(std_t).all():
        raise ValueError('normalization mean/std 包含 NaN 或 Inf')
    if torch.any(std_t <= 0):
        raise ValueError('normalization std 必须全部为有限正数')

    if mode == 'global':
        if mean_t.numel() != 1:
            raise ValueError('global normalization 要求标量 mean/std')
        return (feature - mean_t[0]) / std_t[0]

    n_channels = 1 if feature.dim() == 2 else feature.shape[0]
    if mean_t.numel() != n_channels:
        raise ValueError(
            f'per_channel normalization 需要 {n_channels} 组 mean/std，'
            f'实际得到 {mean_t.numel()} 组'
        )
    if feature.dim() == 2:
        return (feature - mean_t[0]) / std_t[0]
    broadcast_shape = (n_channels,) + (1,) * (feature.dim() - 1)
    return (feature - mean_t.reshape(broadcast_shape)) / std_t.reshape(broadcast_shape)


# ═══════════════════════════════════════════════════════════════
#  音频 I/O
# ═══════════════════════════════════════════════════════════════

def safe_load_audio(path):
    """鲁棒音频加载: torchaudio → soundfile, 始终返回 1D (T,)"""
    try:
        t, sr = torchaudio.load(path)
        if t.dim() > 1:
            t = t.mean(dim=0)
        return t, sr
    except Exception:
        pass
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype='float32')
        t = torch.from_numpy(data.copy()).float()
        if t.dim() == 2:
            t = t.mean(dim=1)
        return t, sr
    except Exception:
        pass
    raise RuntimeError(f"无法加载音频: {path}")


# ═══════════════════════════════════════════════════════════════
#  LOFAR 特征变换
# ═══════════════════════════════════════════════════════════════

class LOFARTransform:
    """波形 → LOFAR 时频图 (全局归一化)

    LOFAR (Low Frequency Analysis and Recording):
      - 长窗 STFT (n_fft=8192) → 高频率分辨率 (1.95 Hz/bin)
      - 频率范围 0-1000Hz → 聚焦船舶线谱
      - log 压缩 → 动态范围压缩
    """
    def __init__(self, cfg, global_mean=None, global_std=None,
                 n_fft=None, hop=None, win=None, n_bins=None, frames=None):
        self.cfg = cfg
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft or cfg.lofar_n_fft,
            hop_length=hop or cfg.lofar_hop,
            win_length=win or cfg.lofar_win,
            power=2.0,
            normalized=False,
        )
        self.n_bins = n_bins or cfg.lofar_n_bins  # 513
        self.frames_target = frames   # 时间中心裁剪目标 (宽带短窗帧数 77 → 71)
        self.global_mean = global_mean
        self.global_std = global_std
        self.normalization_mode = getattr(cfg, 'normalization_mode', 'global')
        if self.normalization_mode not in NORMALIZATION_MODES:
            raise ValueError(f'未知 normalization_mode={self.normalization_mode!r}')

    def __call__(self, waveform):
        """waveform: (T,) → lofar: (1, n_bins, frames)"""
        spec = self.stft(waveform)
        spec = spec[:self.n_bins, :]
        spec_log = torch.log(spec + 1e-6)  # (F, T)

        # 时间中心裁剪 (宽带 n_fft=2048 → 77 帧 → 71, 与 LOFAR/CWT 帧数统一)
        if self.frames_target is not None and spec_log.shape[-1] > self.frames_target:
            start = (spec_log.shape[-1] - self.frames_target) // 2
            spec_log = spec_log[:, start:start + self.frames_target]

        # 全局归一化
        if self.global_mean is not None and self.global_std is not None:
            spec_log = normalize_feature(
                spec_log, self.global_mean, self.global_std,
                mode=self.normalization_mode)

        if spec_log.dim() == 2:
            spec_log = spec_log.unsqueeze(0)  # (1, F, T)
        return spec_log


def make_feature_transform(cfg, global_mean=None, global_std=None):
    """按 cfg.feature_type 构造主输入特征变换 (统一 (C, F, 79) 接口)

    'lofar' : 0-1kHz 长窗 STFT, 1 通道 (单通道基线)
    'dual'  : 双通道互补 — ch0=LOFAR 0-1kHz + ch1=宽带高频 1-4kHz (主线)
    """
    ft = getattr(cfg, 'feature_type', 'dual')
    if ft == 'dual':
        return DualBandTransform(cfg, global_mean, global_std)
    return LOFARTransform(cfg, global_mean, global_std)


class DualBandTransform:
    """双通道互补特征: ch0 = LOFAR 0-1kHz (1.95Hz/bin) + ch1 = 宽带高频 1-4kHz (7.8Hz/bin)

    依据 (单特征 1-seed 筛屏, no-anchor):
      lofar-alone 0.7475 > wideband-alone 0.7377 > cwt-alone 0.7353
      → 低频高分辨率 (8192 窗) 是命根子不可弃; 高频 1-4kHz 作增量通道
      (diag_feat.py: 高频带对 {0_1,0_2}-vs-1_2 混淆对 AUC 0.62, 低频仅 0.58)

    两个 STFT 从同一波形计算 (hop=1024, center=True → 79 帧天然对齐);
    高频 385 bins (1-4kHz @7.8Hz/bin) 线性插值到 513 与 ch0 堆叠 → (2, 513, 79)。
    """
    def __init__(self, cfg, global_mean=None, global_std=None):
        self.lofar_tf = LOFARTransform(cfg)               # ch0, 无归一化
        high_n_fft = cfg.wideband_n_fft
        self.hz_per_bin = cfg.sample_rate / high_n_fft    # 2048 → 7.8 Hz/bin @16k
        self.f_lo = getattr(cfg, 'wideband_freq_lo_hz', 1000.0)
        self.f_hi = getattr(cfg, 'wideband_freq_max_hz', 4000.0)
        n_bins_high = int(round(self.f_hi / self.hz_per_bin)) + 1  # 覆盖到 f_hi 的 bin 数
        self.high_tf = LOFARTransform(cfg,                # ch1 高频源
                                      n_fft=high_n_fft,
                                      hop=cfg.wideband_hop,
                                      win=high_n_fft,
                                      n_bins=n_bins_high)
        self.global_mean = global_mean
        self.global_std = global_std
        self.normalization_mode = getattr(cfg, 'normalization_mode', 'global')
        if self.normalization_mode not in NORMALIZATION_MODES:
            raise ValueError(f'未知 normalization_mode={self.normalization_mode!r}')

    def __call__(self, waveform):
        lo = self.lofar_tf(waveform)[0]                   # (513, 79) raw log
        wb = self.high_tf(waveform)[0]                    # (513, 79) raw log 0-4kHz
        i0 = int(round(self.f_lo / self.hz_per_bin))      # 128
        i1 = min(int(round(self.f_hi / self.hz_per_bin)), wb.shape[0])
        hi = wb[i0:i1]                                    # (385, 79)
        hi = torch.nn.functional.interpolate(             # 沿频率轴插值 385 → 513
            hi.t().unsqueeze(0), size=lo.shape[0],
            mode='linear', align_corners=False)[0].t()    # (513, 79)
        x = torch.stack([lo, hi])                         # (2, 513, 79)
        if self.global_mean is not None and self.global_std is not None:
            x = normalize_feature(
                x, self.global_mean, self.global_std,
                mode=self.normalization_mode)
        return x


# ═══════════════════════════════════════════════════════════════
#  DEMON 特征变换 (scipy 滤波, torchaudio filtfilt 有 NaN bug)
# ═══════════════════════════════════════════════════════════════

class DEMONTransform:
    """波形 → DEMON 谱 (去 DC + 标准化)

    DEMON (Detection of Envelope Modulation On Noise):
      1. 带通滤波 (50-1000 Hz): 隔离螺旋桨调制频带
      2. 平方检波 (envelope²): 提取包络
      3. 低通滤波 (cutoff=100 Hz): 去除高频残余
      4. FFT → DEMON 谱 (0-50 Hz): 展示螺旋桨叶片率/轴频峰值
      5. 去 DC (bin 0-1) + per-sample 标准化

    验证结论: 去 DC 后 Tanker-Tug 可分性 0.03→0.09, 互补性 0.16-0.31
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_bins = cfg.demon_n_bins
        self.max_hz = cfg.demon_max_hz
        self.remove_dc = cfg.demon_remove_dc
        self.sample_rate = cfg.sample_rate
        self.seg_len = cfg.audio_len
        # [demon-slice 臂 2026-08-27] 探针裁决 (_diag_demon_probe): 包络谱 P1 0.4493
        # 接近 chance (模型无视它是理性的); 原始谱切片 P2 0.5536 (+10.4pp)。
        # slice 模式 = 输出 ch0 0-58.5Hz Welch 谱切片 (30 bins @1.95Hz/bin, 与探针
        # P2 同口径) 插值到 126 维 → 模型/损失/协议零改动 (接口长度不变)。
        # 默认 'demon' = 原包络谱行为 (主线不变)。
        self.slice_mode = getattr(cfg, 'demon_mode', 'demon') == 'slice'

        # 预计算滤波器系数 (scipy)
        nyq = self.sample_rate / 2
        self._bp_b, self._bp_a = butter(
            4, [cfg.demon_bp_low / nyq, cfg.demon_bp_high / nyq], btype='band')
        self._lp_b, self._lp_a = butter(
            4, cfg.demon_lp_cutoff / nyq, btype='low')

    def __call__(self, waveform):
        """waveform: (T,) torch tensor → demon: (1, n_bins_out,) torch tensor

        去DC后 n_bins_out = n_bins - 2; slice 模式输出同长度 (接口不变)
        """
        sig = waveform.numpy().astype(np.float64)

        # [demon-slice 臂] 0-58.5Hz 原始谱切片替代包络谱 (log + per-sample
        # 标准化, 保持边缘分布恒定性质; 长度 = n_bins-2 → 模型零改动)
        if self.slice_mode:
            f, p = welch(sig, fs=self.sample_rate, nperseg=8192,
                         noverlap=4096, window='hann', scaling='density')
            m = f <= 58.5                     # 30 bins @1.95Hz/bin (= 探针 P2)
            sl = np.log10(p[m] + 1e-12)
            sl = np.interp(np.linspace(0, len(sl) - 1, self.n_bins - 2),
                           np.arange(len(sl)), sl)      # 30 → 126
            sl = (sl - sl.mean()) / (sl.std() + 1e-8)
            return torch.from_numpy(sl.astype(np.float32)).unsqueeze(0)

        # 1. 带通滤波 (50-1000 Hz)
        bp = sp_filtfilt(self._bp_b, self._bp_a, sig)

        # 2. 平方检波 (envelope²)
        env = bp ** 2

        # 3. 低通滤波 (cutoff=100 Hz)
        env_lp = sp_filtfilt(self._lp_b, self._lp_a, env)

        # 4. FFT → DEMON 谱
        spec = np.fft.rfft(env_lp)
        power = np.abs(spec) ** 2

        # 取 0-max_hz 范围, 重采样到 n_bins
        hz_per_bin = self.sample_rate / len(sig)
        n_bins_raw = int(self.max_hz / hz_per_bin)
        power = power[:n_bins_raw]
        power = np.interp(
            np.linspace(0, n_bins_raw - 1, self.n_bins),
            np.arange(n_bins_raw),
            power,
        )

        # 5. log 压缩
        demon = np.log(power + 1e-10)

        # 6. 去 DC + per-sample 标准化
        if self.remove_dc:
            demon = demon[2:]  # 去 bin 0-1 (DC 分量)
        demon = (demon - demon.mean()) / (demon.std() + 1e-8)

        return torch.from_numpy(demon.astype(np.float32)).unsqueeze(0)  # (1, n_bins_out)


# ═══════════════════════════════════════════════════════════════
#  波形级数据增强 (抗过拟合, 对单/双/三目标通用)
# ═══════════════════════════════════════════════════════════════

class WaveformAugmentation:
    """波形级数据增强 — 分桶增强 (单目标温和, 双目标强增强)

    所有增强在波形层面操作, LOFAR 和 DEMON 特征从同一增强波形提取,
    保证特征一致性。

    分桶策略 (根据 sample_type):
      - 单目标 (stype=0/1): 温和增强, 保护微弱线谱
        noise_snr 15-25dB (信号弱, 不能加太多噪声)
        freq_shift ±5Hz (线谱位置是关键判别特征)
      - 双目标 (stype=2): 强增强, 频谱复杂可承受
        noise_snr 5-15dB, freq_shift ±10Hz, 更多 SpecAugment 掩码

    增强项:
      1. 频率平移: 频域平移 (物理问题: 破坏谐波整数倍关系, 但正则化效果 > 物理损失)
      2. 噪声注入: SNR 高斯噪声, 模拟不同环境
      3. 时间拉伸: ±5% 速度变化, 模拟多普勒/不同航速 (物理正确: 等比例缩放频率)
      4. 随机增益: ±3 dB, 模拟不同距离
    """
    def __init__(self, cfg):
        self.sr = cfg.sample_rate
        self.freq_shift_mode = getattr(cfg, 'freq_shift_mode', 'legacy')
        if self.freq_shift_mode not in {'legacy', 'off'}:
            raise ValueError(
                f"未知 freq_shift_mode={self.freq_shift_mode!r}; "
                "当前只支持 'legacy' 或 'off'"
            )
        self.noise_mode = getattr(cfg, 'waveform_noise_mode', 'white')
        if self.noise_mode not in {'white', 'off'}:
            raise ValueError(
                f"未知 waveform_noise_mode={self.noise_mode!r}; "
                "当前只支持 'white' 或 'off'"
            )
        # 标准增强参数 (双目标用)
        self.freq_shift = cfg.aug_freq_shift_hz
        self.noise_snr = cfg.aug_noise_snr_db
        self.time_stretch = cfg.aug_time_stretch
        self.gain_db = cfg.aug_gain_db
        # 分桶增强参数 (单目标温和)
        self.freq_shift_single = getattr(cfg, 'aug_freq_shift_hz_single', (-5, 5))
        self.noise_snr_single = getattr(cfg, 'aug_noise_snr_db_single', (15, 25))
        # [spd 臂] 速度扰动 (Doppler 对口): 100% 概率, 频率等比缩放 ±10%
        self.use_speed = getattr(cfg, 'use_speed_perturb', False)
        self.speed_range = getattr(cfg, 'aug_speed_perturb', (0.9, 1.1))
        self._speed_rs = {}   # r (0.01 网格) → Resample 变换缓存 (核预计算)

    def __call__(self, waveform, stype_idx=None):
        """waveform: (T,) torch tensor → augmented (T,) torch tensor

        Args:
            stype_idx: sample_type 索引 (0=noise,1=single,2=double,3=triple)
                       None 时用标准增强; 0/1 时用温和增强

        每次随机选择 1-2 种增强 (不是全部同时应用, 避免过度增强)
        """
        # [spd 臂] 速度扰动最先施加: 后续 op (尤其噪声注入的功率/SNR 语义)
        # 作用于最终长度的波形; r 采样到 0.01 网格 (Resample 核按档缓存)
        if self.use_speed:
            waveform = self._speed_perturb(waveform)

        # 分桶选择增强参数
        is_single = stype_idx is not None and stype_idx <= 1
        if is_single:
            freq_shift_range = self.freq_shift_single
            noise_snr_range = self.noise_snr_single
        else:
            freq_shift_range = self.freq_shift
            noise_snr_range = self.noise_snr

        ops = ['freq_shift', 'noise', 'time_stretch', 'gain']
        n_ops = torch.randint(1, 3, (1,)).item()  # 随机选 1-2 种
        selected = np.random.choice(ops, size=n_ops, replace=False)

        for op in selected:
            if op == 'freq_shift':
                waveform = self._freq_shift(waveform, freq_shift_range)
            elif op == 'noise':
                waveform = self._add_noise(waveform, noise_snr_range)
            elif op == 'time_stretch':
                waveform = self._time_stretch(waveform)
            elif op == 'gain':
                waveform = self._random_gain(waveform)

        return waveform

    def _speed_perturb(self, waveform):
        """速度扰动 (Doppler 物理): 频率等比缩放 r, 谐波结构保持

        [spd 臂] macls 主增强对齐 (100% 概率 ±10%)。实现: 把波形声明为
        sr*r 采样率后重采样到 sr → 时长 ×1/r, 频率 ×r (等比, 谐波比不变;
        对比 freq_shift 的常量 Hz 平移会破坏谐波整数倍关系)。
        torchaudio.transforms.Resample 核在 __init__ 预计算 → 按档缓存;
        r 量化到 0.01 网格 (21 档) 限制核数量。
        裁剪/补零回原长 (r>1 → 随机裁, r<1 → 尾部补零, 与 _time_stretch 一致)。
        """
        r_q = round(np.random.uniform(*self.speed_range), 2)
        if abs(r_q - 1.0) < 1e-9:
            return waveform
        rs = self._speed_rs.get(r_q)
        if rs is None:
            rs = torchaudio.transforms.Resample(int(round(self.sr * r_q)), self.sr)
            self._speed_rs[r_q] = rs
        y = rs(waveform.unsqueeze(0)).squeeze(0)          # (T,) → (1,T) → (T,)
        orig_len = waveform.shape[-1]
        if y.shape[-1] > orig_len:
            start = torch.randint(0, y.shape[-1] - orig_len + 1, (1,)).item()
            y = y[start:start + orig_len]
        elif y.shape[-1] < orig_len:
            y = torch.nn.functional.pad(y, (0, orig_len - y.shape[-1]))
        return y

    def _freq_shift(self, waveform, freq_shift_range=None):
        """历史频域平移，或 E1 审计用恒等映射。

        ``legacy`` 保留原有 FFT-roll 行为，仅用于 E0 可复现对照。对实信号，
        该实现会破坏共轭对称并在取实部后产生双边带，不能解释为真实频移。

        ``off`` 仍抽取同一个 shift_hz、占据同一个随机增强槽位，但返回原波形，
        从而不提高其他增强被选中的概率，并尽量维持 E0/E1 的随机数轨迹。
        """
        fr = freq_shift_range if freq_shift_range is not None else self.freq_shift
        shift_hz = np.random.uniform(*fr)
        shift_bins = int(shift_hz * len(waveform) / self.sr)
        if self.freq_shift_mode == 'off':
            return waveform
        if shift_bins == 0:
            return waveform
        spec = torch.fft.fft(waveform)
        spec = torch.roll(spec, shift_bins)
        return torch.fft.ifft(spec).real

    def _add_noise(self, waveform, noise_snr_range=None):
        """高斯噪声注入，或 E6 审计用恒等映射。

        ``off`` 仍抽取 SNR 并生成同形状噪声，保持 NumPy/PyTorch 随机数
        消耗与 ``white`` 臂一致；只是不把噪声叠加到波形上。
        """
        sr = noise_snr_range if noise_snr_range is not None else self.noise_snr
        snr_db = np.random.uniform(*sr)
        signal_power = torch.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        if self.noise_mode == 'off':
            return waveform
        return waveform + noise

    def _time_stretch(self, waveform):
        """时间拉伸 (插值)"""
        rate = np.random.uniform(*self.time_stretch)
        orig_len = len(waveform)
        new_len = int(orig_len * rate)
        stretched = torch.nn.functional.interpolate(
            waveform.unsqueeze(0).unsqueeze(0), size=new_len,
            mode='linear', align_corners=False
        ).squeeze(0).squeeze(0)
        # 裁剪/填充回原始长度
        if len(stretched) > orig_len:
            start = torch.randint(0, len(stretched) - orig_len + 1, (1,)).item()
            stretched = stretched[start:start + orig_len]
        elif len(stretched) < orig_len:
            stretched = torch.nn.functional.pad(stretched, (0, orig_len - len(stretched)))
        return stretched

    def _random_gain(self, waveform):
        """随机增益"""
        gain = np.random.uniform(*self.gain_db)
        return waveform * (10 ** (gain / 20))


# ═══════════════════════════════════════════════════════════════
#  SpecAugment (LOFAR 版, 频率/时间掩码)
# ═══════════════════════════════════════════════════════════════

class SpecAugment:
    """SpecAugment 数据增强 (频率掩码 + 时间掩码)"""
    def __init__(self, n_freq_masks=2, n_time_masks=2,
                 freq_mask_param=30, time_mask_param=20):
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __call__(self, mel):
        """mel: (1, F, T) 或 (C, F, T)"""
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        c, n_freq, n_time = mel.shape

        for _ in range(self.n_freq_masks):
            f = min(self.freq_mask_param, n_freq)
            f_len = torch.randint(0, f, (1,)).item()
            f_start = torch.randint(0, n_freq - f_len + 1, (1,)).item()
            mel[:, f_start:f_start + f_len, :] = 0

        for _ in range(self.n_time_masks):
            t = min(self.time_mask_param, n_time)
            t_len = torch.randint(0, t, (1,)).item()
            t_start = torch.randint(0, n_time - t_len + 1, (1,)).item()
            mel[:, :, t_start:t_start + t_len] = 0

        return mel


# ═══════════════════════════════════════════════════════════════
#  音频段加载
# ═══════════════════════════════════════════════════════════════

def _load_audio_segment(wav_path, target_len, sample_rate, random_crop=False):
    """加载音频并裁剪/填充到固定长度"""
    audio, sr = safe_load_audio(wav_path)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    L = audio.shape[0]
    if L > target_len:
        if random_crop:
            start = torch.randint(0, L - target_len + 1, (1,)).item()
        else:
            start = (L - target_len) // 2
        audio = audio[start:start + target_len]
    elif L < target_len:
        audio = torch.nn.functional.pad(audio, (0, target_len - L))
    return audio


# ═══════════════════════════════════════════════════════════════
#  多目标标签映射 (main.m 坐标系 → config 坐标系)
# ═══════════════════════════════════════════════════════════════

def multitarget_type_to_label(type_name, config):
    """type_name → multi-hot label (config 坐标系 [Tanker, Passengership, Tug])

    main.m 标签顺序 [PassengerShip, Tanker, Tug] (位置 0,1,2)
    config 标签顺序 [Tanker, Passengership, Tug] (索引 0,1,2)

    映射:
      noise   → [0,0,0]  纯噪声
      0 (PS)  → [0,1,0]  单 Passengership
      1 (Tk)  → [1,0,0]  单 Tanker
      2 (Tug) → [0,0,1]  单 Tug
      0_1     → [1,1,0]  PS + Tanker
      0_2     → [0,1,1]  PS + Tug
      1_2     → [1,0,1]  Tanker + Tug
      0_1_2   → [1,1,1]  三目标
    """
    label = [0] * config.num_classes
    if type_name == 'noise':
        return label
    pos_map = config.multitarget_pos_to_label  # {0:1, 1:0, 2:2}
    for pos_str in type_name.split('_'):
        idx = pos_map[int(pos_str)]
        label[idx] = 1
    return label


def get_sample_type(type_name):
    """type_name → sample_type 分组名 (noise/single/double/triple)"""
    if type_name == 'noise':
        return 'noise'
    n_targets = len(type_name.split('_'))
    return {1: 'single', 2: 'double', 3: 'triple'}[n_targets]


# ═══════════════════════════════════════════════════
#  数据集: 多目标船舶 (8 种标签组合) — LOFAR + DEMON 双特征
# ═══════════════════════════════════════════════════

# sample_type 索引: 0=noise, 1=single, 2=double, 3=triple
SAMPLE_TYPE_MAP = {'noise': 0, 'single': 1, 'double': 2, 'triple': 3}
SAMPLE_TYPE_NAMES = ['noise', 'single', 'double', 'triple']


class MultiTargetDataset(Dataset):
    """多目标混合音频 → LOFAR + DEMON 双特征 + multi-hot 标签

    数据来源: MATLAB main.m concat 管线 (BELLHOP 多途 + 可选 Wenz 噪声)
      目录结构: {data_combined}/{type}/{split}/mix/combined_XXXXX.wav
      7 种 type: noise, 0, 1, 2, 0_1, 0_2, 1_2

    返回:
      lofar: (C, 513, 79) 主输入特征 (dual: C=2)
      demon: (1, demon_n_bins_out) DEMON 谱 (去DC后) 或 None
      label: (num_classes,) multi-hot float32
      sample_type_idx: int (0=noise, 1=single, 2=double, 3=triple)
    """
    def __init__(self, config, split='Train', lofar_transform=None,
                 demon_transform=None, waveform_augment=None, spec_augment=None,
                 include_types=None):
        assert split in ('Train', 'Val', 'Test')
        self.config = config
        self.split = split
        self.target_len = config.audio_len
        self.sample_rate = config.sample_rate
        self.lofar_transform = lofar_transform or LOFARTransform(config)
        self.demon_transform = demon_transform
        self.waveform_augment = waveform_augment
        self.spec_augment = spec_augment
        self.is_train = (split == 'Train')
        self.samples = []  # [(wav_path, label_list, sample_type_idx), ...]

        combined_root = config.data_combined
        if not os.path.isdir(combined_root):
            raise RuntimeError(
                f"未找到多目标数据目录: {combined_root}\n"
                f"请先运行 MATLAB data_gen/main.m 生成多目标混合数据。"
            )

        # 子集过滤 (solo 专家: 只用 noise + 单目标)
        types = config.multitarget_types if include_types is None else include_types

        type_counts = {}
        for type_name in types:
            mix_dir = os.path.join(combined_root, type_name, split, 'mix')
            if not os.path.isdir(mix_dir):
                print(f"  警告: 目录不存在 {mix_dir}")
                continue

            label_list = multitarget_type_to_label(type_name, config)
            stype = get_sample_type(type_name)
            stype_idx = SAMPLE_TYPE_MAP[stype]

            count = 0
            for fname in sorted(os.listdir(mix_dir)):
                if fname.endswith('.wav'):
                    wav_path = os.path.join(mix_dir, fname)
                    self.samples.append((wav_path, label_list, stype_idx))
                    count += 1

            type_counts[type_name] = count

        if not self.samples:
            raise RuntimeError(f"未找到 {split} 多目标数据, 请检查 {combined_root}")

        # 统计打印
        stype_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for _, _, st_idx in self.samples:
            stype_counts[st_idx] += 1
        print(f"  [{split}] 多目标数据集: {len(self.samples)} 样本 "
              f"[noise={stype_counts[0]} single={stype_counts[1]} "
              f"double={stype_counts[2]} triple={stype_counts[3]}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label_list, stype_idx = self.samples[idx]
        audio = _load_audio_segment(wav_path, self.target_len, self.sample_rate,
                                    random_crop=self.is_train)

        # 样本级 RMS 归一化 (消除能量捷径: 削峰归一化后 triple RMS > double > single > noise,
        # 模型会学到"能量高→全正/能量低→全负"的捷径, 归一化到 RMS=1 后强制关注频谱结构)
        rms = torch.sqrt(torch.mean(audio ** 2) + 1e-8)
        audio = audio / rms

        # 波形级增强 (仅训练, 分桶增强: 单目标温和, 双目标强增强)
        if self.waveform_augment is not None and self.is_train:
            audio = self.waveform_augment(audio, stype_idx=stype_idx)

        # LOFAR 特征
        lofar = self.lofar_transform(audio)

        if self.spec_augment is not None and self.is_train:
            lofar = self.spec_augment(lofar)

        # DEMON 特征 (可选)
        demon = None
        if self.demon_transform is not None:
            demon = self.demon_transform(audio)

        # multi-hot 标签
        label = torch.tensor(label_list, dtype=torch.float32)

        return lofar, demon, label, stype_idx


# ═══════════════════════════════════════════════════════════════
#  全局归一化统计量
# ═══════════════════════════════════════════════════════════════

def compute_lofar_stats(config, split='Train', max_samples=500):
    """计算主特征的 Train-only global 或 per-channel mean/std。"""
    if str(split).casefold() != 'train':
        raise ValueError('归一化统计只允许从 Train split 计算')
    mode = getattr(config, 'normalization_mode', 'global')
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f'未知 normalization_mode={mode!r}')
    print(f"  计算 {getattr(config, 'feature_type', 'lofar')} {mode} Train-only "
          f"统计量 (采样 {max_samples} 样本)...")
    tmp_transform = make_feature_transform(config)  # 无归一化, 用于统计
    ds = MultiTargetDataset(config, split, lofar_transform=tmp_transform)

    n = min(max_samples, len(ds))
    if n <= 0:
        raise RuntimeError('Train split 为空，无法计算归一化统计量')
    indices = np.random.RandomState(0).choice(len(ds), n, replace=False)

    if mode == 'global':
        # 保持历史累加顺序和 float32 张量求和口径，严格复现 E1。
        total = 0
        mean_sum = 0.0
        var_sum = 0.0
        for i in indices:
            feature = ds[int(i)][0]
            total += feature.numel()
            mean_sum += feature.sum().item()
            var_sum += (feature ** 2).sum().item()
        mean = mean_sum / total
        variance = var_sum / total - mean ** 2
        std = (max(variance, 0.0) + 1e-8) ** 0.5
        print(f"  global mean={mean:.6f}, std={std:.6f}")
        return mean, std

    channel_sum = None
    channel_sq_sum = None
    elements_per_channel = 0
    for i in indices:
        feature = ds[int(i)][0]
        if feature.dim() == 2:
            feature = feature.unsqueeze(0)
        values = feature.to(torch.float64)
        current_sum = values.sum(dim=tuple(range(1, values.dim())))
        current_sq_sum = (values ** 2).sum(dim=tuple(range(1, values.dim())))
        channel_sum = current_sum if channel_sum is None else channel_sum + current_sum
        channel_sq_sum = (current_sq_sum if channel_sq_sum is None
                          else channel_sq_sum + current_sq_sum)
        elements_per_channel += values[0].numel()

    means = channel_sum / elements_per_channel
    variances = channel_sq_sum / elements_per_channel - means ** 2
    stds = torch.sqrt(torch.clamp(variances, min=0.0) + 1e-8)
    if not torch.isfinite(means).all() or not torch.isfinite(stds).all():
        raise RuntimeError('per-channel 归一化统计包含 NaN 或 Inf')
    if torch.any(stds <= 0):
        raise RuntimeError('per-channel 归一化 std 必须为正数')
    mean_list = means.tolist()
    std_list = stds.tolist()
    for channel, (mean, std) in enumerate(zip(mean_list, std_list)):
        print(f"  ch{channel}: mean={mean:.6f}, std={std:.6f}")
    return mean_list, std_list


# ═══════════════════════════════════════════════════════════════
#  DataLoader 工厂
# ═══════════════════════════════════════════════════════════════

def collate_multi(batch):
    """多目标批次 collate — 兼容 demon=None"""
    items = list(zip(*batch))  # [(lofar...), (demon...), (label...), (stype...)]
    lofar = torch.stack(items[0])
    demon = torch.stack(items[1]) if items[1][0] is not None else None
    label = torch.stack(items[2])
    stype = torch.tensor(items[3], dtype=torch.long)
    return lofar, demon, label, stype

def create_loaders(config, include_types=None, include_test=True):
    """创建 Train/Val/Test DataLoader (多目标)

    Args:
        include_types: 类型子集过滤 (如 solo 专家 ['noise','0','1','2']);
                       None = 全量 7 类。归一化统计量始终从全量 Train 计算
                       (与主模型一致, 保证特征空间相同)。
        include_test: False 时不构造 Test Dataset/DataLoader，供 Val-only
                      消融选择使用；默认 True 保持历史生产行为。

    Returns:
        train_loader, val_loader, test_loader, info_dict
    """
    # Train-only 归一化统计量；E1=global，E2=per_channel。
    g_mean, g_std = compute_lofar_stats(config, 'Train')
    lofar_tf = make_feature_transform(config, global_mean=g_mean, global_std=g_std)
    ft = getattr(config, 'feature_type', 'lofar')
    if ft != 'lofar':
        print(f"  ★ 主特征切换: {ft}")

    # DEMON 变换 (可选)
    demon_tf = DEMONTransform(config) if config.use_demon else None

    # 波形级增强 (仅训练)
    waveform_aug = None
    if getattr(config, 'use_waveform_aug', False):
        waveform_aug = WaveformAugmentation(config)
        print(f"  启用波形级增强: freq_shift_mode={waveform_aug.freq_shift_mode}, "
              f"noise_mode={waveform_aug.noise_mode}, "
              f"freq_shift={config.aug_freq_shift_hz}, "
              f"noise_snr={config.aug_noise_snr_db}, "
              f"time_stretch={config.aug_time_stretch}, gain={config.aug_gain_db}")
        if waveform_aug.freq_shift_mode == 'legacy':
            print("  WARNING: legacy FFT-roll 频移仅用于 E0 历史对照，"
                  "不能解释为物理频移")
        else:
            print("  ★ E1: 历史 FFT-roll 频移已关闭；该增强槽位执行恒等映射")
        if waveform_aug.noise_mode == 'off':
            print("  ★ E6: 额外高斯白噪声已关闭；噪声增强槽位执行恒等映射")
        if getattr(config, 'use_speed_perturb', False):
            print(f"  ★ 速度扰动 (Doppler 对口): {config.aug_speed_perturb} @100% "
                  f"(频率等比缩放, 谐波结构保持)")

    # SpecAugment (仅训练)
    spec_aug = None
    if getattr(config, 'use_specaug', False):
        spec_aug = SpecAugment(
            n_freq_masks=config.specaug_n_freq_masks,
            n_time_masks=config.specaug_n_time_masks,
            freq_mask_param=config.specaug_freq_param,
            time_mask_param=config.specaug_time_param,
        )

    # 数据集
    train_ds = MultiTargetDataset(config, 'Train', lofar_transform=lofar_tf,
                                  demon_transform=demon_tf,
                                  waveform_augment=waveform_aug,
                                  spec_augment=spec_aug,
                                  include_types=include_types)
    val_ds = MultiTargetDataset(config, 'Val', lofar_transform=lofar_tf,
                                demon_transform=demon_tf,
                                include_types=include_types)
    test_ds = None
    if include_test:
        test_ds = MultiTargetDataset(config, 'Test', lofar_transform=lofar_tf,
                                     demon_transform=demon_tf,
                                     include_types=include_types)

    if getattr(config, 'use_spec_gate', False):
        print(f"  ★ 频谱条件门控启用: backbone 输出 → 频谱指纹通道调制")

    nw = config.num_workers
    kw = {'num_workers': nw, 'pin_memory': True}
    if nw > 0:
        kw['prefetch_factor'] = getattr(config, 'prefetch_factor', 4)
        kw['persistent_workers'] = True

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, drop_last=False,
                              collate_fn=collate_multi, **kw)

    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_multi, **kw)
    test_loader = None
    if include_test:
        test_loader = DataLoader(test_ds, batch_size=config.batch_size,
                                 shuffle=False, collate_fn=collate_multi, **kw)

    info = {
        'n_train': len(train_ds), 'n_val': len(val_ds),
        'n_test': len(test_ds) if test_ds is not None else None,
        'global_mean': g_mean, 'global_std': g_std,
        'normalization_mode': getattr(config, 'normalization_mode', 'global'),
        'use_demon': config.use_demon,
    }
    return train_loader, val_loader, test_loader, info


if __name__ == '__main__':
    from config import ConfigSmall
    cfg = ConfigSmall()
    print('=' * 70)
    print('M-UART v4.0 数据集测试 — LOFAR + DEMON 双特征 + 波形增强 (多目标)')
    print('=' * 70)
    train_loader, val_loader, test_loader, info = create_loaders(cfg)
    print(f"\nTrain: {info['n_train']}, Val: {info['n_val']}, Test: {info['n_test']}")
    print(f"use_demon: {info['use_demon']}")

    for lofar, demon, label, stype_idx in train_loader:
        print(f"\nBatch: lofar {lofar.shape}, demon {demon.shape if demon is not None else None}, "
              f"label {label.shape}, stype {stype_idx.tolist()}")
        print(f"LOFAR 范围: [{lofar.min():.3f}, {lofar.max():.3f}], mean={lofar.mean():.3f}")
        if demon is not None:
            print(f"DEMON 范围: [{demon.min():.3f}, {demon.max():.3f}], mean={demon.mean():.3f}")
        break
