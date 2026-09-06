%% generate_wenz_noise — Wenz 海洋环境噪声生成器 (原生版)
%
% 功能: 根据 Wenz (1962) 经验公式, 在频域合成三种噪声分量的功率谱,
%       叠加后配以随机相位, 通过 IFFT 生成具有物理上合理频谱形状的时域噪声。
%
% 三分量模型:
%   ① 湍流噪声 (Turbulence):
%      NL_turb = 17 - 30·log10(f_kHz)  [dB]
%      主导范围 1~100 Hz, 反映水体湍流引起的压力波动
%
%   ② 航运噪声 (Shipping):
%      NL_shipping = 40 + 20·(shipping_factor - 0.5)
%                    + 26·log10(f_kHz) - 60·log10(f_kHz + 0.03)  [dB]
%      主导范围 10~1000 Hz, 反映远处商船的累积噪声贡献
%
%   ③ 海面风噪 (Surface Wind):
%      NL_wind = 50 + 7.5·√(wind_speed)
%                + 20·log10(f_kHz) - 40·log10(f_kHz + 0.4)  [dB]
%      主导范围 50 Hz~20 kHz, 反映海面风浪破碎气泡的噪声
%
%   总功率谱: P_total(f) = P_turb(f) + P_shipping(f) + P_wind(f)
%            振幅响应: A(f) = √P_total(f)
%
% 原生版说明 (本版唯一边界):
%   湍流公式的实证适用域为 f ≥ 10Hz (Wenz 1962 发表曲线起点)。
%   原式若外推到 0.2Hz 会按 f^-3 发散 — 那是公式边界外的数学行为,
%   不是海况物理 (实测会使 99.5% 噪声功率落入 <10Hz 次声段, 且风速项
%   仅占发散后总功率的 1e-7, 海况旋钮完全失效)。因此:
%     f < 10Hz: 湍流分量不建模 (置零); 航运/风噪公式在该频段有界,
%               按原式继续使用 (原生)。
%     f ≥ 10Hz: 三分量全部按原式, 无任何带限, 直到奈奎斯特 (fs/2)。
%
% 输出为单位方差时序；生产调用方 gen_sl_dataset.m 再按 Wenz PSD 在
%   10Hz 至奈奎斯特的积分功率缩放到绝对 RMS（μPa）。
%
% 引用: Wenz, G.M. (1962) "Acoustic Ambient Noise in the Ocean:
%        Spectra and Sources", JASA, 34(12), 1936-1956.
%
% 输入:
%   N              - 输出信号长度 (采样点数), 通常为 5 × fs = 80000
%   fs             - 采样率 (Hz), 16000
%   wind_speed     - 海面风速 (m/s), 典型值 0~20 (海况旋钮)
%   shipping_factor- 航运强度因子 (0~1), 0=轻度航运, 1=重度航运
%
% 输出:
%   n_out          - 时域噪声信号 (N×1), 零均值, 单位标准差
%
% 依赖: 无 (纯 MATLAB 内置函数)

function n_out = generate_wenz_noise(N, fs, wind_speed, shipping_factor)

    half_N = floor(N/2) + 1;
    f_half = (0:half_N-1)' * (fs / N);               % 正半轴频率 (DC → Nyquist)

    % ===== 频率 → kHz (Wenz 公式单位) =====
    f_kHz = f_half / 1000;
    f_kHz(1) = 1e-6;                                 % DC 处取极小值避免 log10(0) = -Inf

    % ===== ① 湍流噪声功率谱 =====
    % 公式适用域 f ≥ 10Hz; 域外 (f < 10Hz) 不建模 (外推按 f^-3 发散,
    % 见函数头说明)。max(f,10Hz) 先防域内 log 越界, 掩码再置零域外。
    NL_turb = 17 - 30 * log10(max(f_kHz, 0.01));
    P_turb = 10 .^ (NL_turb / 10);                   % dB → 线性功率
    P_turb(f_half < 10) = 0;

    % ===== ② 航运噪声功率谱 =====
    % shipping_factor 调节航运密度: 0 → 轻度 (低10dB), 1 → 重度 (高10dB)
    % (公式低频有界, 全频段原样使用)
    NL_shipping = 40 + 20 * (shipping_factor - 0.5) ...
                  + 26 * log10(f_kHz) - 60 * log10(f_kHz + 0.03);
    P_shipping = 10 .^ (NL_shipping / 10);

    % ===== ③ 海面风噪功率谱 =====
    % 7.5·√(wind) 项: 风速每增加 4 m/s, 风噪升高 ≈15 dB (海况旋钮)
    NL_wind = 50 + 7.5 * sqrt(wind_speed) ...
              + 20 * log10(f_kHz) - 40 * log10(f_kHz + 0.4);
    P_wind = 10 .^ (NL_wind / 10);

    % ===== 三分量叠加 → 振幅响应 (无带限, 10Hz → 奈奎斯特) =====
    P_total = P_turb + P_shipping + P_wind;
    Amp_response = sqrt(P_total);                    % 振幅 = √功率

    % ===== 复高斯随机相位 (频域白化) =====
    % real + imag 各为标准高斯分布, 包络为 Rayleigh 分布
    phase_random = (randn(half_N, 1) + 1j * randn(half_N, 1)) / sqrt(2);

    % DC 和 Nyquist: 必须为实数 (共轭对称性约束)
    phase_random(1) = randn(1, 1);
    if mod(N, 2) == 0
        phase_random(end) = randn(1, 1);
    end

    % ===== 频谱成形: Z(f) = A(f) · e^{jφ_rand(f)} =====
    Z_half = Amp_response .* phase_random;

    % ===== 全谱重建 (共轭对称) → IFFT =====
    if mod(N, 2) == 0
        Z_full = [Z_half; conj(flipud(Z_half(2:end-1)))];    % 偶长度
    else
        Z_full = [Z_half; conj(flipud(Z_half(2:end)))];      % 奇长度
    end

    n_out = real(ifft(Z_full));

    % ===== 标准化: 零均值 + 单位标准差 =====
    n_out = n_out - mean(n_out);                     % 去直流偏置
    n_out = n_out / std(n_out);                      % 功率归一化, 等待外部 SNR 缩放
end
