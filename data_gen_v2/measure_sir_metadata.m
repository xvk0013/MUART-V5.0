function audit = measure_sir_metadata(y_channels, envelope_channels, ...
    active_channels, fs, sir_mode, audit_band, emit_warning)
%% measure_sir_metadata — Bellhop 后接收端 SIR 的只读测量
%
% 本函数不接收 Wenz、不返回音频、不筛选候选样本，也不修改幅度。
% 输入必须是完成 Bellhop、去 DC 和时变包络后的 N×3 目标信号。

if nargin < 6 || isempty(audit_band), audit_band = [20 4000]; end
if nargin < 7, emit_warning = true; end
% 兼容第 5A 阶段的六参数调用，第六参数当时是 emit_warning。
if nargin == 6 && isscalar(audit_band) && ...
        (islogical(audit_band) || (isnumeric(audit_band) && any(audit_band == [0 1])))
    emit_warning = logical(audit_band);
    audit_band = [20 4000];
end
if ~(ischar(sir_mode) || (isstring(sir_mode) && isscalar(sir_mode)))
    error('measure_sir_metadata:InvalidMode', 'sir_mode 必须是字符串。');
end
mode = lower(strtrim(char(sir_mode)));
switch mode
    case {'measure_only', 'stratified'}
        % 始终只测量；stratified 的接受/重抽由 apply_sir_policy 决定。
    case 'reject_extreme'
        error('measure_sir_metadata:ModeNotImplemented', ...
            'sir_mode=''%s'' 未实现。', mode);
    otherwise
        error('measure_sir_metadata:InvalidMode', ...
            '未知 sir_mode=''%s''；仅支持 ''measure_only'' 或 ''stratified''。', mode);
end

if ~isnumeric(y_channels) || ~isreal(y_channels) || size(y_channels, 2) ~= 3 || ...
        isempty(y_channels) || any(~isfinite(y_channels(:)))
    error('measure_sir_metadata:InvalidSignal', ...
        'y_channels 必须是非空有限实数 N×3 数组。');
end
has_envelope = ~isempty(envelope_channels);
if has_envelope && (~isnumeric(envelope_channels) || ~isreal(envelope_channels) || ...
        ~isequal(size(envelope_channels), size(y_channels)) || ...
        any(~isfinite(envelope_channels(:))) || any(envelope_channels(:) < 0))
    error('measure_sir_metadata:InvalidEnvelope', ...
        '非空 envelope_channels 必须是与 y_channels 同尺寸的非负有限实数数组。');
end
if ~isnumeric(active_channels) || any(~isfinite(active_channels(:))) || ...
        any(active_channels(:) ~= fix(active_channels(:))) || ...
        any(active_channels(:) < 1 | active_channels(:) > 3) || ...
        numel(unique(active_channels(:))) ~= numel(active_channels)
    error('measure_sir_metadata:InvalidChannels', ...
        'active_channels 必须是 1..3 内不重复的整数索引。');
end
if ~isnumeric(fs) || ~isscalar(fs) || ~isreal(fs) || ~isfinite(fs) || fs <= 0
    error('measure_sir_metadata:InvalidFs', 'fs 必须是有限正数。');
end
if ~isnumeric(audit_band) || ~isreal(audit_band) || numel(audit_band) ~= 2 || ...
        any(~isfinite(audit_band)) || audit_band(1) < 0 || ...
        audit_band(1) >= audit_band(2) || audit_band(2) > fs / 2
    error('measure_sir_metadata:InvalidAuditBand', ...
        'audit_band 必须位于 [0, fs/2] 且严格递增。');
end
audit_band = reshape(audit_band, 1, 2);
if ~isscalar(emit_warning) || ...
        ~(islogical(emit_warning) || (isnumeric(emit_warning) && ...
          isfinite(emit_warning) && any(emit_warning == [0 1])))
    error('measure_sir_metadata:InvalidWarningFlag', ...
        'emit_warning 必须是逻辑标量。');
end
emit_warning = logical(emit_warning);

audit = struct( ...
    'overlap_start_s', 0, ...
    'overlap_end_s', 0, ...
    'overlap_samples', 0, ...
    'sir_rx_global_db', 0, ...
    'sir_rx_overlap_db', 0, ...
    'sir_rx_global_signed_db', 0, ...
    'sir_rx_overlap_signed_db', 0, ...
    'sir_rx_global_band_db', 0, ...
    'sir_rx_overlap_band_db', 0, ...
    'sir_rx_global_band_signed_db', 0, ...
    'sir_rx_overlap_band_signed_db', 0, ...
    'sir_global_valid', 0, ...
    'sir_overlap_valid', 0, ...
    'sir_global_band_valid', 0, ...
    'sir_overlap_band_valid', 0, ...
    'audit_band_hz', audit_band, ...
    'receive_rms', zeros(1, 3), ...
    'overlap_rms', zeros(1, 3), ...
    'receive_band_rms', zeros(1, 3), ...
    'overlap_band_rms', zeros(1, 3), ...
    'reject_candidate', false, ...
    'resample_candidate', false, ...
    'gain_adjustment_applied', false);

% noise 和单目标：SIR 不适用，用 valid=0 区分于有效的 0 dB。
if numel(active_channels) ~= 2
    return;
end

active_channels = active_channels(:).';
for idx = 1:2
    channel = active_channels(idx);
    audit.receive_rms(channel) = rms(y_channels(:, channel));
end
global_pair_rms = audit.receive_rms(active_channels);
if all(isfinite(global_pair_rms)) && all(global_pair_rms > 0)
    [audit.sir_rx_global_db, audit.sir_rx_global_signed_db] = ...
        sir_from_pair_rms(global_pair_rms);
    if ~isfinite(audit.sir_rx_global_db) || ...
            ~isfinite(audit.sir_rx_global_signed_db)
        error('measure_sir_metadata:NonfiniteGlobalSIR', ...
            '全局接收端 SIR 不是有限值。');
    end
    audit.sir_global_valid = 1;
elseif emit_warning
    warning('measure_sir_metadata:InvalidGlobalRMS', ...
        '双目标至少一路全局接收 RMS 无效或为零；全局 SIR 记为无效，不筛选样本。');
end

for idx = 1:2
    channel = active_channels(idx);
    audit.receive_band_rms(channel) = spectral_band_rms( ...
        y_channels(:, channel), fs, audit_band);
end
global_band_pair_rms = audit.receive_band_rms(active_channels);
if all(isfinite(global_band_pair_rms)) && all(global_band_pair_rms > 0)
    [audit.sir_rx_global_band_db, audit.sir_rx_global_band_signed_db] = ...
        sir_from_pair_rms(global_band_pair_rms);
    audit.sir_global_band_valid = 1;
elseif emit_warning
    warning('measure_sir_metadata:InvalidGlobalBandRMS', ...
        '双目标至少一路带内全局 RMS 无效或为零；带内全局 SIR 记为无效，不筛选样本。');
end

if ~has_envelope
    if emit_warning
        warning('measure_sir_metadata:MissingEnvelope', ...
            '双目标没有包络信息；重叠 SIR 记为无效，不筛选样本。');
    end
    return;
end

overlap_mask = all(envelope_channels(:, active_channels) > 0, 2);
overlap_indices = find(overlap_mask);
if isempty(overlap_indices)
    if emit_warning
        warning('measure_sir_metadata:NoOverlap', ...
            '双目标没有有效包络重叠区间；重叠 SIR 记为无效，不筛选样本。');
    end
    return;
end

audit.overlap_start_s = (overlap_indices(1) - 1) / fs;
audit.overlap_end_s = (overlap_indices(end) - 1) / fs;
audit.overlap_samples = numel(overlap_indices);
for idx = 1:2
    channel = active_channels(idx);
    audit.overlap_rms(channel) = rms(y_channels(overlap_mask, channel));
end
overlap_pair_rms = audit.overlap_rms(active_channels);
if all(isfinite(overlap_pair_rms)) && all(overlap_pair_rms > 0)
    [audit.sir_rx_overlap_db, audit.sir_rx_overlap_signed_db] = ...
        sir_from_pair_rms(overlap_pair_rms);
    if ~isfinite(audit.sir_rx_overlap_db) || ...
            ~isfinite(audit.sir_rx_overlap_signed_db)
        error('measure_sir_metadata:NonfiniteOverlapSIR', ...
            '重叠窗口接收端 SIR 不是有限值。');
    end
    audit.sir_overlap_valid = 1;
elseif emit_warning
    warning('measure_sir_metadata:InvalidOverlapRMS', ...
        '双目标至少一路重叠 RMS 无效或为零；重叠 SIR 记为无效，不筛选样本。');
end

for idx = 1:2
    channel = active_channels(idx);
    audit.overlap_band_rms(channel) = spectral_band_rms( ...
        y_channels(overlap_mask, channel), fs, audit_band);
end
overlap_band_pair_rms = audit.overlap_band_rms(active_channels);
if all(isfinite(overlap_band_pair_rms)) && all(overlap_band_pair_rms > 0)
    [audit.sir_rx_overlap_band_db, audit.sir_rx_overlap_band_signed_db] = ...
        sir_from_pair_rms(overlap_band_pair_rms);
    audit.sir_overlap_band_valid = 1;
elseif emit_warning
    warning('measure_sir_metadata:InvalidOverlapBandRMS', ...
        '双目标至少一路带内重叠 RMS 无效或为零；带内重叠 SIR 记为无效，不筛选样本。');
end
end


function [unsigned_db, signed_db] = sir_from_pair_rms(pair_rms)
signed_db = 20 * log10(pair_rms(1) / pair_rms(2));
unsigned_db = abs(signed_db);
end


function value = spectral_band_rms(x, fs, audit_band)
x = x(:);
n = numel(x);
spectrum = fft(x);
half_n = floor(n / 2) + 1;
frequency = (0:half_n-1)' * (fs / n);
weights = 2 * ones(half_n, 1);
weights(1) = 1;
if mod(n, 2) == 0, weights(end) = 1; end
in_band = frequency >= audit_band(1) & frequency <= audit_band(2);
if ~any(in_band)
    value = 0;
    return;
end
power_value = sum(weights(in_band) .* abs(spectrum(in_band)).^2) / n^2;
value = sqrt(max(real(power_value), 0));
end
