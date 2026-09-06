function [mix_wav, signal_wav, noise_wav, refs_wav, audit] = ...
    scale_audio_with_audit(y_mix, y_signal, y_noise, y_channels, scale_c, peak_limit)
%% scale_audio_with_audit — 使用共同 scale_c 缩放并生成写盘审计量
%
% 输入 y_* 是写盘前物理声压 (μPa)。输出 *_wav 是乘以同一个 scale_c
% 后的数字幅值；它们本身不是 μPa。恢复关系为：
%   physical_pressure_uPa ≈ wav_amplitude / scale_c
% 近似误差来自浮点运算和实际 WAV 编码量化。

if nargin < 6 || isempty(peak_limit), peak_limit = 0.95; end
validate_signal(y_mix, 'y_mix');
validate_signal(y_signal, 'y_signal');
validate_signal(y_noise, 'y_noise');
validate_signal(y_channels, 'y_channels');

y_mix = y_mix(:);
y_signal = y_signal(:);
y_noise = y_noise(:);
if size(y_channels, 1) ~= numel(y_mix) || size(y_channels, 2) ~= 3 || ...
        numel(y_signal) ~= numel(y_mix) || numel(y_noise) ~= numel(y_mix)
    error('scale_audio_with_audit:SizeMismatch', ...
        'mix、signal、noise 必须等长，y_channels 必须为 N×3。');
end
if ~isnumeric(scale_c) || ~isscalar(scale_c) || ~isreal(scale_c) || ...
        ~isfinite(scale_c) || scale_c <= 0
    error('scale_audio_with_audit:InvalidScale', ...
        'scale_c 必须是有限正数。');
end
if ~isnumeric(peak_limit) || ~isscalar(peak_limit) || ~isreal(peak_limit) || ...
        ~isfinite(peak_limit) || peak_limit <= 0
    error('scale_audio_with_audit:InvalidPeakLimit', ...
        'peak_limit 必须是有限正数。');
end
if scale_c > 1
    warning('scale_audio_with_audit:ScaleAboveOne', ...
        'scale_c=%.12g > 1；按配置记录并放大写盘，未静默修改。', scale_c);
end

% 所有输出只在这里乘一次同一个 scale_c，禁止逐通道单独归一化。
mix_wav = y_mix * scale_c;
signal_wav = y_signal * scale_c;
noise_wav = y_noise * scale_c;
refs_wav = y_channels * scale_c;

audit = struct();
audit.scale_c = scale_c;
audit.pre_mix_peak = max(abs(y_mix));
audit.post_mix_peak = max(abs(mix_wav));
audit.pre_rms_mix = rms(y_mix);
audit.post_rms_mix = rms(mix_wav);
audit.pre_rms_signal = rms(y_signal);
audit.post_rms_signal = rms(signal_wav);
audit.pre_rms_noise = rms(y_noise);
audit.post_rms_noise = rms(noise_wav);
audit.pre_rms_channels = [rms(y_channels(:,1)), rms(y_channels(:,2)), rms(y_channels(:,3))];
audit.post_rms_channels = [rms(refs_wav(:,1)), rms(refs_wav(:,2)), rms(refs_wav(:,3))];

tolerance = 1e-12;
if audit.post_mix_peak > peak_limit + tolerance
    error('scale_audio_with_audit:PeakLimitExceeded', ...
        'post_mix_peak=%.12g 超过 %.12g + tolerance。', ...
        audit.post_mix_peak, peak_limit);
end
if max(abs(refs_wav(:))) > peak_limit + tolerance
    error('scale_audio_with_audit:ReferencePeakLimitExceeded', ...
        '参考通道写盘峰值超过共同削峰上限。');
end

assert_scaled(mix_wav, y_mix, scale_c, 'mix');
assert_scaled(signal_wav, y_signal, scale_c, 'signal');
assert_scaled(noise_wav, y_noise, scale_c, 'noise');
for channel_idx = 1:3
    assert_scaled(refs_wav(:,channel_idx), y_channels(:,channel_idx), ...
        scale_c, sprintf('s%d', channel_idx));
end

expected_rms = [audit.pre_rms_mix, audit.pre_rms_signal, audit.pre_rms_noise, ...
                audit.pre_rms_channels] * scale_c;
actual_rms = [audit.post_rms_mix, audit.post_rms_signal, audit.post_rms_noise, ...
              audit.post_rms_channels];
rms_tolerance = 1e-12 * max(1, max(abs(expected_rms)));
if any(abs(actual_rms - expected_rms) > rms_tolerance)
    error('scale_audio_with_audit:RMSRecoveryMismatch', ...
        '写盘前后 RMS 不满足 post_rms ≈ pre_rms * scale_c。');
end
end


function validate_signal(x, name)
if ~isnumeric(x) || ~isreal(x) || isempty(x) || any(~isfinite(x(:)))
    error('scale_audio_with_audit:InvalidSignal', ...
        '%s 必须是非空有限实数数组。', name);
end
end


function assert_scaled(post_value, pre_value, scale_c, name)
expected = pre_value * scale_c;
tolerance = 1e-12 * max(1, max(abs(expected(:))));
if any(abs(post_value(:) - expected(:)) > tolerance)
    error('scale_audio_with_audit:ScaleMismatch', ...
        '%s 未使用共同的 scale_c。', name);
end
end
