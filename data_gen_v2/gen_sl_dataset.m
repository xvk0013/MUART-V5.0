function pilot_records = gen_sl_dataset(target_indices, split_type, src_root, out_base, arr_path, ...
    num_samples, target_len_s, max_delay_s, class_ids, target_fs, save_refs, ...
    solo_range_s, fade_range_s, sl_cfg)
%% gen_sl_dataset — SL 声源级语义的拼接数据集合成
%
% 声呐方程范式: SNR = SL − TL − NL 全程物理涌现:
%     [1] SL: 模板 RMS 归一后缩放至 SL 压力 (10^(SL/20) μPa @1m);
%         SL 逐录音采样 (文献: Johnson 2025 / Veirs 2016, 表由驱动脚本预生成,
%         同录音同 SL 保持船身份一致)
%     [2] TL: BELLHOP 信道 H 物理内嵌 (单位源约定, 跨距离衰减实测符合柱面扩展)
%     [3] NL: Wenz 谱级 dB re 1μPa²/Hz 原样绝对定标 (湍流≥10Hz, 无带限),
%         电平 = ∫Wenz PSD df (10Hz→奈奎斯特), 只随海况 w 与航运 sf 变
%   ER 策略必须显式指定。bp12 主线使用 natural：逐录音 SL 原样进入声呐
%   方程，不做中值收拢；clip 仅保留作历史消融。
%   SIR 模式必须显式指定。bp12 主线使用 stratified：在 Bellhop、去 DC、
%   时变包络之后测量接收端重叠带内 SIR，只据此接受/重抽双目标候选；
%   不修改 SL、ER 或目标幅度。pilot 独立使用 measure_only。
%
% 噪声类 (0 目标): 纯 Wenz 绝对电平 (与目标样本背景谱逐 bin 一致, 无双噪声叠加)
%
% 削峰 0.95 仅是写盘数字缩放, mix 与 s1/s2/s3 同用一个 scale_c，SNR
% (比值) 不变。物理声压只能用 wav_amplitude / scale_c 近似恢复；WAV 数字
% 幅值本身不是 μPa。
%
% all_info.txt 保留全部旧字段并在末尾追加 scale/绝对声压/SL-ER 审计字段；
%   target 与 noise 使用统一表头。详见 all_info_schema.md。
%
% 配对设计支持: 本函数内全部随机消耗 (源选择/信道/包络/SL 查表外) 在 Wenz
%   生成之前; generate_wenz_noise 的 randn 数量与 w 无关 → 跨海况条件随机流对齐
%
% 输入:
%   sl_cfg 字段:
%     wind_speed     - 海面风速 m/s (海况旋钮)
%     shipping_factor- 航运因子 (0.2 = 轻度航运开阔海域)
%     sl_maps        - containers.Map, 键 '0'/'1'/'2' (标签位) → Map(文件名→SL dB)
%     recording_maps - Map(标签位→Map(文件名→录音键))，用于固定 SL 保护
%     recording_sl_maps - Map(标签位→Map(录音键→SL dB))，用于固定 SL 保护
%     source_sampling_mode - 可选，默认 template_uniform；D1 使用
%                     recording_uniform，且只允许 Train。先均匀选原始录音，
%                     再在该录音的 split 内模板中均匀选一个。
%     er_policy      - 必填: 'natural' (主线) 或 'clip' (显式历史消融)
%     er_clip_db     - 仅 er_policy='clip' 时必填；natural 完全忽略
%     metadata_only  - 可选，默认 false；仅 pilot 显式设 true，不写 WAV/最终日志
%     band           - SNR/SIR 审计频带 [lo hi] Hz (默认 [20 4000])
%     sir_bin_edges_db - stratified 分层边界，主线 [0 5 10 15]
%     sir_bin_weights  - stratified 精确配额比例，主线 [0.45 0.35 0.20]
%     sir_max_attempt_factor - 最大候选数/目标样本数，主线 20
%     r_kmax         - [E+A1 场景臂] 最大距离档索引 (默认 [] = 全部档; 3 → 1-3km)
%     z_kmin         - [E+A1 场景臂] 最小源深档索引 (默认 [] = 全部档; 2 → {10,15}m)
%                     采样方式 = randi 原范围 (档数不变) 后 clamp 到场景子集:
%                     随机流消耗与无约束严格一致 → 与基线数据集 all_info 逐列配对
%                     (同源文件/同噪声/同包络, 唯一差异 = 信道 r/z);
%                     clamp 语义为条件采样: P(3km)=1/2, P(10m)=2/3 (论文按
%                     "距离 1-3km / 源深 10-15m 场景约束" 披露, 分布细节见臂注释)

if nargin < 14
    solo_range_s = [0.3, 1.5];
    fade_range_s = [0.3, 0.8];
end
if ~isstruct(sl_cfg) || ~isfield(sl_cfg, 'er_policy')
    error('gen_sl_dataset:MissingERPolicy', ...
        'sl_cfg.er_policy 必须显式指定为 ''natural'' 或 ''clip''。');
end
if ~(ischar(sl_cfg.er_policy) || (isstring(sl_cfg.er_policy) && isscalar(sl_cfg.er_policy)))
    error('gen_sl_dataset:InvalidERPolicy', ...
        'sl_cfg.er_policy 必须是字符向量或字符串标量。');
end
er_policy = lower(strtrim(char(sl_cfg.er_policy)));
er_clip_db = [];
switch er_policy
    case 'natural'
        % 主线完全不读取也不使用兼容字段 er_clip_db。
    case 'clip'
        if ~isfield(sl_cfg, 'er_clip_db')
            error('gen_sl_dataset:MissingERClip', ...
                '显式 clip 策略必须提供 sl_cfg.er_clip_db。');
        end
        er_clip_db = sl_cfg.er_clip_db;
    otherwise
        error('gen_sl_dataset:InvalidERPolicy', ...
            '未知 er_policy=''%s''；仅支持 ''natural'' 或 ''clip''。', er_policy);
end
% 在任何输出目录创建前完成策略参数校验；clip 的人工约束警告也在此发出。
apply_er_policy([], er_policy, er_clip_db);
if ~isfield(sl_cfg, 'sir_mode') || ...
        ~(ischar(sl_cfg.sir_mode) || (isstring(sl_cfg.sir_mode) && isscalar(sl_cfg.sir_mode)))
    error('gen_sl_dataset:MissingSIRMode', ...
        'sl_cfg.sir_mode 必须显式指定为 ''measure_only'' 或 ''stratified''。');
end
sir_mode = lower(strtrim(char(sl_cfg.sir_mode)));
switch sir_mode
    case {'measure_only', 'stratified'}
        % measure_only 仅供 pilot；stratified 仅做候选接受/重抽。
    case 'reject_extreme'
        error('gen_sl_dataset:SIRModeNotImplemented', ...
            'sir_mode=''%s'' 未实现。', sir_mode);
    otherwise
        error('gen_sl_dataset:InvalidSIRMode', ...
            '未知 sir_mode=''%s''；仅支持 ''measure_only'' 或 ''stratified''。', sir_mode);
end
metadata_only = false;
if isfield(sl_cfg, 'metadata_only')
    metadata_value = sl_cfg.metadata_only;
    if ~isscalar(metadata_value) || ...
            ~(islogical(metadata_value) || (isnumeric(metadata_value) && isfinite(metadata_value) && ...
              any(metadata_value == [0 1])))
        error('gen_sl_dataset:InvalidMetadataOnly', ...
            'sl_cfg.metadata_only 必须是逻辑标量。');
    end
    metadata_only = logical(metadata_value);
end
source_sampling_mode = 'template_uniform';
if isfield(sl_cfg, 'source_sampling_mode')
    source_sampling_value = sl_cfg.source_sampling_mode;
    if ~(ischar(source_sampling_value) || ...
            (isstring(source_sampling_value) && isscalar(source_sampling_value)))
        error('gen_sl_dataset:InvalidSourceSamplingMode', ...
            'sl_cfg.source_sampling_mode 必须是字符向量或字符串标量。');
    end
    source_sampling_mode = lower(strtrim(char(source_sampling_value)));
end
if ~ismember(source_sampling_mode, {'template_uniform', 'recording_uniform'})
    error('gen_sl_dataset:InvalidSourceSamplingMode', ...
        '未知 source_sampling_mode=%s。', source_sampling_mode);
end
if strcmp(source_sampling_mode, 'recording_uniform') && ~strcmpi(split_type, 'Train')
    error('gen_sl_dataset:D1TrainOnly', ...
        'D1 recording_uniform 只允许用于 Train；Val 必须原样复用。');
end
pilot_records = struct([]);
audit_band = [20 4000];
if isfield(sl_cfg, 'band'), audit_band = sl_cfg.band; end
if ~isnumeric(audit_band) || ~isreal(audit_band) || numel(audit_band) ~= 2 || ...
        any(~isfinite(audit_band)) || audit_band(1) < 0 || ...
        audit_band(1) >= audit_band(2) || audit_band(2) > target_fs / 2
    error('gen_sl_dataset:InvalidAuditBand', ...
        'sl_cfg.band 必须位于 [0, target_fs/2] 且严格递增。');
end
audit_band = audit_band(:).';
is_stratified = strcmp(sir_mode, 'stratified');
sir_bin_edges_db = [];
sir_bin_weights = [];
sir_max_attempt_factor = 1;
if is_stratified
    required_sir_fields = {'sir_bin_edges_db', 'sir_bin_weights', 'sir_max_attempt_factor'};
    for field_idx = 1:numel(required_sir_fields)
        if ~isfield(sl_cfg, required_sir_fields{field_idx})
            error('gen_sl_dataset:MissingSIRPolicy', ...
                'stratified 模式缺少 sl_cfg.%s。', required_sir_fields{field_idx});
        end
    end
    sir_bin_edges_db = sl_cfg.sir_bin_edges_db(:).';
    sir_bin_weights = sl_cfg.sir_bin_weights(:).';
    sir_max_attempt_factor = sl_cfg.sir_max_attempt_factor;
    if numel(sir_bin_edges_db) ~= numel(sir_bin_weights) + 1 || ...
            any(~isfinite(sir_bin_edges_db)) || any(diff(sir_bin_edges_db) <= 0)
        error('gen_sl_dataset:InvalidSIRBins', ...
            'SIR 分层边界必须严格递增，且边界数等于权重数加一。');
    end
    allocate_sir_strata_counts(0, sir_bin_weights); % 校验权重。
    if ~isnumeric(sir_max_attempt_factor) || ~isscalar(sir_max_attempt_factor) || ...
            ~isfinite(sir_max_attempt_factor) || sir_max_attempt_factor < 1 || ...
            sir_max_attempt_factor ~= fix(sir_max_attempt_factor)
        error('gen_sl_dataset:InvalidSIRMaxAttempts', ...
            'sir_max_attempt_factor 必须是正整数。');
    end
end
r_kmax = [];                                              % [E+A1 场景臂] 距离约束档
if isfield(sl_cfg, 'r_kmax'), r_kmax = sl_cfg.r_kmax; end
z_kmin = [];                                              % [E+A1 场景臂] 源深约束档
if isfield(sl_cfg, 'z_kmin'), z_kmin = sl_cfg.z_kmin; end

n_targets = length(target_indices);
if metadata_only
    if is_stratified
        error('gen_sl_dataset:PilotMustMeasureOnly', ...
            'metadata_only pilot 必须使用 measure_only，不得执行分层筛选。');
    end
    if n_targets ~= 2
        error('gen_sl_dataset:PilotRequiresPair', ...
            'metadata_only pilot 只允许双目标类别对。');
    end
    if ~strcmpi(split_type, 'Train')
        error('gen_sl_dataset:PilotTrainOnly', ...
            'metadata_only pilot 只允许 Train split，禁止读取 Test 估计 SIR 区间。');
    end
end
stratified_enabled = is_stratified && n_targets == 2;
if stratified_enabled
    sir_target_counts = allocate_sir_strata_counts(num_samples, sir_bin_weights);
else
    sir_target_counts = zeros(1, numel(sir_bin_weights));
end
if n_targets > 0
    required_maps = {'sl_maps', 'recording_maps', 'recording_sl_maps'};
    for field_idx = 1:numel(required_maps)
        field_name = required_maps{field_idx};
        if ~isfield(sl_cfg, field_name) || ~isa(sl_cfg.(field_name), 'containers.Map')
            error('gen_sl_dataset:MissingSLGuardMap', ...
                '目标样本要求 sl_cfg.%s 为 containers.Map。', field_name);
        end
    end
end
if n_targets == 0
    type_name = 'noise';
else
    type_name = strjoin(arrayfun(@(x) num2str(x), target_indices, 'UniformOutput', false), '_');
end

% 旧字段顺序冻结；所有新审计字段只能追加在末尾。
all_info_fields = { ...
    'combIdx', 'fileA', 'fileB', 'fileC', ...
    'r1_km', 'r2_km', 'r3_km', 'z_src1', 'z_src2', 'z_src3', 'z_recv', ...
    'fs', 'wind', 'sf', 'n_rms_uPa', 'snr_band_db', ...
    'sl_1', 'sl_2', 'sl_3', 'er_1', 'er_2', 'er_3', ...
    'scale_c', 'pre_mix_peak', 'post_mix_peak', ...
    'pre_rms_mix', 'post_rms_mix', ...
    'pre_rms_signal', 'post_rms_signal', ...
    'pre_rms_noise', 'post_rms_noise', ...
    'pre_rms_1', 'pre_rms_2', 'pre_rms_3', ...
    'post_rms_1', 'post_rms_2', 'post_rms_3', ...
    'sl_raw_1', 'sl_raw_2', 'sl_raw_3', ...
    'sl_used_1', 'sl_used_2', 'sl_used_3', ...
    'er_raw_pair_db', 'er_used_pair_db', ...
    'overlap_start_s', 'overlap_end_s', 'overlap_samples', ...
    'sir_rx_global_db', 'sir_rx_overlap_db', ...
    'sir_global_valid', 'sir_overlap_valid', ...
    'sir_rx_global_signed_db', 'sir_rx_overlap_signed_db', ...
    'sir_rx_global_band_db', 'sir_rx_overlap_band_db', ...
    'sir_rx_global_band_signed_db', 'sir_rx_overlap_band_signed_db', ...
    'sir_global_band_valid', 'sir_overlap_band_valid'};

if ~metadata_only
    out_dir = fullfile(out_base, type_name, split_type);
    dir_mix = fullfile(out_dir, 'mix');
    dir_s1  = fullfile(out_dir, 's1');
    dir_s2  = fullfile(out_dir, 's2');
    dir_s3  = fullfile(out_dir, 's3');

    % 断点续跑: 该类型已完成则跳过
    completion_audit_ok = ~stratified_enabled || ...
        isfile(fullfile(out_dir, 'sir_sampling_info.txt'));
    source_audit_ok = strcmp(source_sampling_mode, 'template_uniform') || ...
        n_targets == 0 || isfile(fullfile(out_dir, 'source_sampling_info.tsv'));
    if exist(fullfile(out_dir, 'all_info.txt'), 'file') ...
            && numel(dir(fullfile(dir_mix, '*.wav'))) == num_samples ...
            && completion_audit_ok && source_audit_ok
        fprintf('  已存在且完整, 跳过: %s/%s (%d 样本)\n', type_name, split_type, num_samples);
        return;
    end

    if ~exist(dir_mix, 'dir'), mkdir(dir_mix); end
    if save_refs
        if ~exist(dir_s1, 'dir'), mkdir(dir_s1); end
        if ~exist(dir_s2, 'dir'), mkdir(dir_s2); end
        if ~exist(dir_s3, 'dir'), mkdir(dir_s3); end
    end
end

target_samples = round(target_len_s * target_fs);

% ===== 生产模式 Wenz 绝对电平；pilot metadata_only 完全不执行 Wenz =====
if metadata_only
    n_rms_abs = 0;
    fprintf('  pilot metadata_only: Train/%s，候选 %d；不执行 Wenz/scale_c/audiowrite\n', ...
        type_name, num_samples);
else
    n_rms_abs = sqrt(wenz_band_power(sl_cfg.wind_speed, sl_cfg.shipping_factor, ...
        10, target_fs / 2, target_fs));
    fprintf('  SL 语义: wind=%.1f sf=%.1f | Wenz 绝对 RMS = %.1f μPa (%.1f dB)\n', ...
        sl_cfg.wind_speed, sl_cfg.shipping_factor, n_rms_abs, 20 * log10(n_rms_abs));
end
if strcmp(er_policy, 'natural')
    fprintf('  ER 策略: natural | 逐录音固定 SL 原样使用，ER 截断关闭\n');
else
    fprintf('  ER 策略: clip | 显式历史消融，er_clip_db=%.3g dB\n', er_clip_db);
end

%% ===== 分支 A: 0 目标 (纯 Wenz) =====
if n_targets == 0
    fid_log = fopen(fullfile(out_dir, 'all_info.txt'), 'wt');
    if fid_log < 0, error('gen_sl_dataset:LogOpenFailed', '无法创建 all_info.txt: %s', out_dir); end
    write_all_info_header(fid_log, all_info_fields);
    for combIdx = 1:num_samples
        if mod(combIdx, 500) == 0
            fprintf('    %s/noise: %d/%d\n', split_type, combIdx, num_samples);
        end
        n_wenz = generate_wenz_noise(target_samples, target_fs, ...
            sl_cfg.wind_speed, sl_cfg.shipping_factor) * n_rms_abs;
        y_signal = zeros(target_samples, 1);
        y_channels = zeros(target_samples, 3);
        y_mix = n_wenz;
        max_val = max([abs(y_mix); abs(y_channels(:))]);
        if max_val > 0, scale_c = 0.95 / max_val; else, scale_c = 1.0; end

        [mix_wav, ~, ~, refs_wav, scale_audit] = scale_audio_with_audit( ...
            y_mix, y_signal, n_wenz, y_channels, scale_c, 0.95);
        audiowrite(fullfile(dir_mix, sprintf('combined_%05d.wav', combIdx)), ...
            mix_wav, target_fs);
        if save_refs
            audiowrite(fullfile(dir_s1, sprintf('combined_%05d.wav', combIdx)), refs_wav(:,1), target_fs);
            audiowrite(fullfile(dir_s2, sprintf('combined_%05d.wav', combIdx)), refs_wav(:,2), target_fs);
            audiowrite(fullfile(dir_s3, sprintf('combined_%05d.wav', combIdx)), refs_wav(:,3), target_fs);
        end
        row_values = { ...
            combIdx, 'none', 'none', 'none', ...
            0, 0, 0, 0, 0, 0, 0, ...
            target_fs, sl_cfg.wind_speed, sl_cfg.shipping_factor, n_rms_abs, -Inf, ...
            0, 0, 0, 0, 0, 0, ...
            scale_audit.scale_c, scale_audit.pre_mix_peak, scale_audit.post_mix_peak, ...
            scale_audit.pre_rms_mix, scale_audit.post_rms_mix, ...
            scale_audit.pre_rms_signal, scale_audit.post_rms_signal, ...
            scale_audit.pre_rms_noise, scale_audit.post_rms_noise, ...
            scale_audit.pre_rms_channels(1), scale_audit.pre_rms_channels(2), scale_audit.pre_rms_channels(3), ...
            scale_audit.post_rms_channels(1), scale_audit.post_rms_channels(2), scale_audit.post_rms_channels(3), ...
            0, 0, 0, 0, 0, 0, 0, 0, ...
            0, 0, 0, 0, 0, 0, 0, ...
            0, 0, 0, 0, 0, 0, 0, 0};
        write_all_info_row(fid_log, all_info_fields, row_values);
    end
    fclose(fid_log);
    fprintf('  完成: noise/%s (%d 样本)\n', split_type, num_samples);
    return;
end

%% ===== 分支 B: 1~2 目标 (主线无三目标) =====
[Arr_all, Pos, f_centers] = load_arr_cached(arr_path);
num_freqs = length(f_centers);
n_r_ranges = length(Pos.r.r);
n_s_depths = length(Pos.s.z);
nz_recv = 2;                                              % 接收深恒 1000m (主线语义)

% ===== 信道 H 预计算 (persistent, 跨条件/类型共享) =====
N_pad = target_samples + round(max_delay_s * target_fs);
persistent H_cached H_cached_key
H_cache_key = sprintf('%s|fs=%d|Npad=%d|md=%.3g', arr_path, target_fs, N_pad, max_delay_s);
if isequal(H_cached_key, H_cache_key) && ~isempty(H_cached)
    H_cache = H_cached;
    fprintf('  信道 H 预计算缓存命中 (%d 信道)\n', numel(H_cache));
else
    fprintf('  预计算 %d 信道 H (N_pad=%d)...\n', n_r_ranges * n_s_depths, N_pad);
    H_cache = cell(n_r_ranges, n_s_depths);
    for k_r = 1:n_r_ranges
        for n_d = 1:n_s_depths
            Amp_c = cell(1, num_freqs); tau_c = cell(1, num_freqs);
            for i = 1:num_freqs
                Amp_c{i} = double(Arr_all{i}(k_r, nz_recv, n_d).A);
                tau_c{i} = double(Arr_all{i}(k_r, nz_recv, n_d).delay);
            end
            H_cache{k_r, n_d} = build_channel_H(Amp_c, tau_c, f_centers, target_fs, N_pad, max_delay_s);
        end
    end
    H_cached = H_cache; H_cached_key = H_cache_key;
    fprintf('  H 预计算完成\n');
end

% ===== 源模板池 (data_sl_split_bp12, SL 按文件名查表) =====
src_lists = cell(1, n_targets);
src_pools = cell(1, n_targets);
for t = 1:n_targets
    label_idx = target_indices(t);
    class_folder = class_ids(label_idx + 1);
    src_dir = fullfile(src_root, split_type, num2str(class_folder));
    wavs = dir(fullfile(src_dir, '*.wav'));
    if isempty(wavs), error('未找到模板: %s', src_dir); end
    src_lists{t} = wavs;
    recording_map_t = sl_cfg.recording_maps(num2str(label_idx));
    src_pools{t} = build_source_sampling_pool(wavs, recording_map_t);
    fprintf('  源采样: 标签 %d | mode=%s | %d 录音 / %d 模板\n', ...
        label_idx, source_sampling_mode, src_pools{t}.n_recordings, ...
        src_pools{t}.n_templates);
end

if metadata_only
    fid_log = -1;
else
    fid_log = fopen(fullfile(out_dir, 'all_info.txt'), 'wt');
    if fid_log < 0, error('gen_sl_dataset:LogOpenFailed', '无法创建 all_info.txt: %s', out_dir); end
    write_all_info_header(fid_log, all_info_fields);
end

% 本次生成内的逐录音 SL 观测缓存：同一标签/录音再次出现时必须完全一致。
observed_recording_sl = containers.Map('KeyType', 'char', 'ValueType', 'double');
candidate_recording_counts = containers.Map('KeyType', 'char', 'ValueType', 'double');
accepted_recording_counts = containers.Map('KeyType', 'char', 'ValueType', 'double');

accepted_count = 0;
candidate_attempts = 0;
sir_accepted_counts = zeros(size(sir_target_counts));
sir_rejected_invalid = 0;
sir_rejected_outside = 0;
sir_rejected_full = 0;
if stratified_enabled
    max_candidate_attempts = sir_max_attempt_factor * num_samples;
    fprintf(['  SIR 分层: [%g,%g)/[%g,%g)/[%g,%g] dB，目标计数 %s；' ...
        '最大候选数 %d\n'], ...
        sir_bin_edges_db(1), sir_bin_edges_db(2), ...
        sir_bin_edges_db(2), sir_bin_edges_db(3), ...
        sir_bin_edges_db(3), sir_bin_edges_db(4), ...
        mat2str(sir_target_counts), max_candidate_attempts);
else
    max_candidate_attempts = num_samples;
end

while accepted_count < num_samples
    candidate_attempts = candidate_attempts + 1;
    if candidate_attempts > max_candidate_attempts
        error('gen_sl_dataset:SIRMaxAttempts', ...
            ['%s/%s 达到最大候选数 %d，仍只接受 %d/%d；分层目标=%s，' ...
             '已接受=%s，拒绝 invalid/outside/full=%d/%d/%d。'], ...
            split_type, type_name, max_candidate_attempts, accepted_count, num_samples, ...
            mat2str(sir_target_counts), mat2str(sir_accepted_counts), ...
            sir_rejected_invalid, sir_rejected_outside, sir_rejected_full);
    end

    k_ranges = randi(n_r_ranges, 1, n_targets);
    n_depths = randi(n_s_depths, 1, n_targets);

    % [E+A1 场景臂] 流保持约束: randi 原档数抽取后 clamp 到场景子集
    %   (距离 ≤ r_kmax 档 = 1-3km; 源深 ≥ z_kmin 档 = {10,15}m)
    %   随机流与无约束严格一致 → 跨臂配对语义; H_cache 为全 4×3 网格, 索引安全
    if ~isempty(r_kmax) && r_kmax < n_r_ranges
        k_ranges = min(k_ranges, r_kmax);
    end
    if ~isempty(z_kmin) && z_kmin > 1
        n_depths = max(n_depths, z_kmin);
    end

    % ----- Phase 1: 选模板 + 查 SL -----
    src_signals = cell(1, n_targets);
    sl_raw = zeros(1, n_targets);
    src_files = cell(1, n_targets);
    src_recordings = cell(1, n_targets);
    src_kr = zeros(1, n_targets); src_nd = zeros(1, n_targets);
    src_r_kms = zeros(1, n_targets); src_zs = zeros(1, n_targets);

    for t = 1:n_targets
        label_idx = target_indices(t);
        wavs_t = src_lists{t};
        [file_t, sampled_recording] = sample_source_from_pool( ...
            src_pools{t}, source_sampling_mode);
        [y_src, ~] = audioread(fullfile(file_t.folder, file_t.name));
        if size(y_src, 2) > 1, y_src = mean(y_src, 2); end
        y_src = extract_random_window(y_src, target_samples);

        label_key = num2str(label_idx);
        sl_map_t = sl_cfg.sl_maps(label_key);
        if ~sl_map_t.isKey(file_t.name)
            error('SL 表缺文件 %s (标签 %d) — D1 必须从冻结 Train 恢复逐录音 SL', file_t.name, label_idx);
        end
        sl_raw(t) = sl_map_t(file_t.name);

        recording_map_t = sl_cfg.recording_maps(label_key);
        recording_sl_map_t = sl_cfg.recording_sl_maps(label_key);
        if ~recording_map_t.isKey(file_t.name)
            error('gen_sl_dataset:RecordingMapMissingFile', ...
                '录音映射缺文件 %s (标签 %d)。', file_t.name, label_idx);
        end
        recording_key = recording_map_t(file_t.name);
        if ~strcmp(recording_key, sampled_recording)
            error('gen_sl_dataset:SampledRecordingMismatch', ...
                '两阶段采样返回的录音与文件映射不一致。');
        end
        if ~recording_sl_map_t.isKey(recording_key)
            error('gen_sl_dataset:RecordingSLMissing', ...
                '逐录音 SL 表缺录音 %s (标签 %d)。', recording_key, label_idx);
        end
        expected_recording_sl = recording_sl_map_t(recording_key);
        if abs(sl_raw(t) - expected_recording_sl) > 1e-9
            error('gen_sl_dataset:RecordingSLModified', ...
                ['检测到文件 SL 与逐录音固定 SL 不一致: 标签 %d, 录音 %s, ' ...
                 '文件值 %.12g dB, 录音值 %.12g dB。'], ...
                label_idx, recording_key, sl_raw(t), expected_recording_sl);
        end
        observation_key = sprintf('%d|%s', label_idx, recording_key);
        if observed_recording_sl.isKey(observation_key)
            if abs(observed_recording_sl(observation_key) - sl_raw(t)) > 1e-9
                error('gen_sl_dataset:RecordingSLChangedDuringRun', ...
                    '同一录音 %s 的 SL 在生成过程中发生变化。', observation_key);
            end
        else
            observed_recording_sl(observation_key) = sl_raw(t);
        end
        src_files{t} = file_t.name;
        src_recordings{t} = recording_key;
        candidate_recording_counts = increment_source_count( ...
            candidate_recording_counts, label_idx, recording_key);

        k_r = k_ranges(t); n_d = n_depths(t);
        src_kr(t) = k_r; src_nd(t) = n_d;
        src_r_kms(t) = Pos.r.r(k_r) / 1e3;
        src_zs(t) = Pos.s.z(n_d);
        src_signals{t} = y_src;
    end

    % ----- Phase 2: 显式 ER 策略；natural 不得改写逐录音 SL -----
    [sl_used, er_raw_db, er_used_db, er_raw_pair_db, er_used_pair_db] = ...
        apply_er_policy(sl_raw, er_policy, er_clip_db);
    if strcmp(er_policy, 'natural')
        if any(abs(sl_used - sl_raw) > 1e-9)
            error('gen_sl_dataset:NaturalSLModified', ...
                'natural 策略检测到 source SL 被修改。');
        end
        if any(abs(er_used_db - er_raw_db) > 1e-9) || ...
                (n_targets == 2 && abs(er_used_pair_db - er_raw_pair_db) > 1e-9)
            error('gen_sl_dataset:NaturalERModified', ...
                'natural 策略检测到实际 ER 与原始 ER 不一致。');
        end
    end
    if n_targets == 2
        er_raw_pair_log = er_raw_pair_db;
        er_used_pair_log = er_used_pair_db;
    else
        er_raw_pair_log = 0;
        er_used_pair_log = 0;
    end

    % ----- Phase 3: SL 压力定标 + BELLHOP 卷积 -----

    y_channels = zeros(target_samples, 3);
    sl_arr = zeros(1, 3); er_arr = zeros(1, 3);
    sl_raw_arr = zeros(1, 3); sl_used_arr = zeros(1, 3);
    file_names = {'none', 'none', 'none'};
    r_kms = [0, 0, 0]; z_srcs = [0, 0, 0];
    envelope_channels = zeros(target_samples, 3);

    for t = 1:n_targets
        x = src_signals{t};
        r_x = rms(x);
        if r_x < 1e-10, r_x = 1; end
        x = (x / r_x) * 10 ^ (sl_used(t) / 20);           % RMS = sl_used 压力 (μPa @1m)
        y_out = convolve_precomputed_H(x, H_cache{src_kr(t), src_nd(t)}, N_pad);
        y_out = y_out - mean(y_out);
        ch = target_indices(t) + 1;
        y_channels(:, ch) = y_out;
        sl_raw_arr(ch) = sl_raw(t);
        sl_used_arr(ch) = sl_used(t);
        sl_arr(ch) = sl_used_arr(ch);                       % 旧字段语义保持为 sl_used
        er_arr(ch) = er_used_db(t);
        file_names{ch} = src_files{t};
        r_kms(ch) = src_r_kms(t); z_srcs(ch) = src_zs(t);
        envelope_channels(:, ch) = 1;
    end
    % ----- 拼接时变包络 (仅 2 目标, 与主线一致) -----
    if n_targets == 2
        a_first = (rand() < 0.5);
        ch_indices = target_indices + 1;
        if a_first, ch_first = ch_indices(1); ch_second = ch_indices(2);
        else,       ch_first = ch_indices(2); ch_second = ch_indices(1); end
        solo_head_s = rand() * (solo_range_s(2) - solo_range_s(1)) + solo_range_s(1);
        solo_tail_s = rand() * (solo_range_s(2) - solo_range_s(1)) + solo_range_s(1);
        fade_A_s = rand() * (fade_range_s(2) - fade_range_s(1)) + fade_range_s(1);
        fade_B_s = rand() * (fade_range_s(2) - fade_range_s(1)) + fade_range_s(1);
        shape_A = randi(3); shape_B = randi(3);
        solo_head_n = round(solo_head_s * target_fs);
        solo_tail_n = round(solo_tail_s * target_fs);
        fade_A_n = max(round(fade_A_s * target_fs), 2);
        fade_B_n = max(round(fade_B_s * target_fs), 2);
        env_A = ones(target_samples, 1); env_B = ones(target_samples, 1);
        fade_out_end = target_samples - solo_tail_n;
        fade_out_start = max(fade_out_end - fade_A_n, 1);
        env_A(fade_out_start:fade_out_end) = 1.0 - fade_shape(shape_A, fade_out_end - fade_out_start + 1);
        env_A(fade_out_end:end) = 0;
        fade_in_start = solo_head_n;
        fade_in_end = min(fade_in_start + fade_B_n, target_samples);
        env_B(1:fade_in_start) = 0;
        env_B(fade_in_start:fade_in_end) = fade_shape(shape_B, fade_in_end - fade_in_start + 1);
        y_channels(:, ch_first) = y_channels(:, ch_first) .* env_A;
        y_channels(:, ch_second) = y_channels(:, ch_second) .* env_B;
        envelope_channels(:, ch_first) = env_A;
        envelope_channels(:, ch_second) = env_B;
    end

    % ----- Bellhop 后接收端 SIR：包络后、Wenz 叠加前、scale_c 前 -----
    y_channels_before_sir = y_channels;
    sir_audit = measure_sir_metadata(y_channels, envelope_channels, ...
        target_indices + 1, target_fs, sir_mode, audit_band, true);
    if ~isequaln(y_channels, y_channels_before_sir)
        error('gen_sl_dataset:SIRModifiedSignal', ...
            'SIR 测量改变了 y_channels，已停止生成。');
    end

    if stratified_enabled
        sir_decision = apply_sir_policy(sir_audit, sir_mode, sir_bin_edges_db, ...
            sir_accepted_counts, sir_target_counts);
        if sir_decision.reject_candidate
            switch sir_decision.reason
                case 'invalid_overlap_band_sir'
                    sir_rejected_invalid = sir_rejected_invalid + 1;
                case 'outside_range'
                    sir_rejected_outside = sir_rejected_outside + 1;
                case 'stratum_full'
                    sir_rejected_full = sir_rejected_full + 1;
                otherwise
                    error('gen_sl_dataset:SIRDecisionReason', ...
                        '未知 SIR 拒绝原因: %s', sir_decision.reason);
            end
            continue;
        end
        sir_accepted_counts(sir_decision.stratum) = ...
            sir_accepted_counts(sir_decision.stratum) + 1;
        if sir_decision.gain_adjustment_applied
            error('gen_sl_dataset:SIRGainAdjustment', ...
                'SIR 分层策略不得调整任何目标幅度。');
        end
    end

    for t = 1:n_targets
        accepted_recording_counts = increment_source_count( ...
            accepted_recording_counts, target_indices(t), src_recordings{t});
    end
    accepted_count = accepted_count + 1;
    combIdx = accepted_count;
    if mod(combIdx, 500) == 0 || combIdx == num_samples
        fprintf('    %s/%s: 已接受 %d/%d（候选 %d）\n', ...
            split_type, type_name, combIdx, num_samples, candidate_attempts);
    end

    if metadata_only
        pilot_record = struct( ...
            'sample_id', combIdx, ...
            'class_pair', type_name, ...
            'split', split_type, ...
            'recording_1', src_recordings{1}, ...
            'recording_2', src_recordings{2}, ...
            'file_1', src_files{1}, ...
            'file_2', src_files{2}, ...
            'sl_raw_1', sl_raw(1), ...
            'sl_raw_2', sl_raw(2), ...
            'er_raw_pair_db', er_raw_pair_db, ...
            'distance_1', src_r_kms(1), ...
            'distance_2', src_r_kms(2), ...
            'source_depth_1', src_zs(1), ...
            'source_depth_2', src_zs(2), ...
            'overlap_start_s', sir_audit.overlap_start_s, ...
            'overlap_end_s', sir_audit.overlap_end_s, ...
            'overlap_samples', sir_audit.overlap_samples, ...
            'overlap_duration_s', sir_audit.overlap_samples / target_fs, ...
            'sir_rx_global_db', sir_audit.sir_rx_global_db, ...
            'sir_rx_overlap_db', sir_audit.sir_rx_overlap_db, ...
            'sir_global_valid', sir_audit.sir_global_valid, ...
            'sir_overlap_valid', sir_audit.sir_overlap_valid, ...
            'sir_rx_global_signed_db', sir_audit.sir_rx_global_signed_db, ...
            'sir_rx_overlap_signed_db', sir_audit.sir_rx_overlap_signed_db, ...
            'sir_rx_global_band_db', sir_audit.sir_rx_global_band_db, ...
            'sir_rx_overlap_band_db', sir_audit.sir_rx_overlap_band_db, ...
            'sir_rx_global_band_signed_db', sir_audit.sir_rx_global_band_signed_db, ...
            'sir_rx_overlap_band_signed_db', sir_audit.sir_rx_overlap_band_signed_db, ...
            'sir_global_band_valid', sir_audit.sir_global_band_valid, ...
            'sir_overlap_band_valid', sir_audit.sir_overlap_band_valid, ...
            'sir_measurement_stage', 'after_envelope_before_wenz_and_scale', ...
            'reject_candidate', sir_audit.reject_candidate, ...
            'resample_candidate', sir_audit.resample_candidate, ...
            'gain_adjustment_applied', sir_audit.gain_adjustment_applied);
        if isempty(pilot_records)
            pilot_records = pilot_record;
        else
            pilot_records(end + 1) = pilot_record; %#ok<AGROW>
        end
    else
        % 候选通过物理 SIR 条件后才生成 Wenz；拒绝候选不会写盘或生成噪声。
        n_wenz = generate_wenz_noise(target_samples, target_fs, ...
            sl_cfg.wind_speed, sl_cfg.shipping_factor) * n_rms_abs;
        y_sig = y_channels(:, 1) + y_channels(:, 2) + y_channels(:, 3);
        y_mix = y_sig + n_wenz;

    if strcmp(er_policy, 'natural')
        active_channels = target_indices + 1;
        if any(abs(sl_raw_arr(active_channels) - sl_raw) > 1e-9) || ...
                any(abs(sl_used_arr(active_channels) - sl_used) > 1e-9) || ...
                any(abs(sl_arr - sl_used_arr) > 1e-9)
            error('gen_sl_dataset:NaturalSLAuditMismatch', ...
                'natural 策略下 all_info SL 审计值与内存中的 raw/used SL 不一致。');
        end
        if abs(er_used_pair_log - er_raw_pair_log) > 1e-9
            error('gen_sl_dataset:NaturalERAuditMismatch', ...
                'natural 策略下 all_info 原始/实际双目标 ER 不一致。');
        end
    end

    % 实测带内 SNR (audit, 削峰前)
    p_sig = band_power(y_sig, target_fs, audit_band(1), audit_band(2));
    p_noi = band_power(n_wenz, target_fs, audit_band(1), audit_band(2));
    snr_band = 10 * log10(p_sig / p_noi);

    max_val = max([abs(y_mix); abs(y_channels(:,1)); abs(y_channels(:,2)); abs(y_channels(:,3))]);
    if max_val > 0, scale_c = 0.95 / max_val; else, scale_c = 1.0; end

    [mix_wav, ~, ~, refs_wav, scale_audit] = scale_audio_with_audit( ...
        y_mix, y_sig, n_wenz, y_channels, scale_c, 0.95);

    % 共同缩放不应改变 SIR；此交叉验证使用内存缩放值，不参与筛选或增益调整。
    sir_scaled = measure_sir_metadata(refs_wav, envelope_channels, ...
        target_indices + 1, target_fs, sir_mode, audit_band, false);
    sir_before_scale = [sir_audit.sir_rx_global_db, sir_audit.sir_rx_overlap_db, ...
        sir_audit.sir_rx_global_signed_db, sir_audit.sir_rx_overlap_signed_db, ...
        sir_audit.sir_rx_global_band_db, sir_audit.sir_rx_overlap_band_db, ...
        sir_audit.sir_rx_global_band_signed_db, sir_audit.sir_rx_overlap_band_signed_db];
    sir_after_scale = [sir_scaled.sir_rx_global_db, sir_scaled.sir_rx_overlap_db, ...
        sir_scaled.sir_rx_global_signed_db, sir_scaled.sir_rx_overlap_signed_db, ...
        sir_scaled.sir_rx_global_band_db, sir_scaled.sir_rx_overlap_band_db, ...
        sir_scaled.sir_rx_global_band_signed_db, sir_scaled.sir_rx_overlap_band_signed_db];
    sir_valid_before = [sir_audit.sir_global_valid, sir_audit.sir_overlap_valid, ...
        sir_audit.sir_global_band_valid, sir_audit.sir_overlap_band_valid];
    sir_valid_after = [sir_scaled.sir_global_valid, sir_scaled.sir_overlap_valid, ...
        sir_scaled.sir_global_band_valid, sir_scaled.sir_overlap_band_valid];
    if ~isequal(sir_valid_after, sir_valid_before) || ...
            any(abs(sir_after_scale - sir_before_scale) > 1e-9)
        error('gen_sl_dataset:SIRScaleMismatch', ...
            '共同 scale_c 前后的接收端 SIR 不一致。');
    end

    out_name = sprintf('combined_%05d.wav', combIdx);
    audiowrite(fullfile(dir_mix, out_name), mix_wav, target_fs);
    if save_refs
        audiowrite(fullfile(dir_s1, out_name), refs_wav(:,1), target_fs);
        audiowrite(fullfile(dir_s2, out_name), refs_wav(:,2), target_fs);
        audiowrite(fullfile(dir_s3, out_name), refs_wav(:,3), target_fs);
    end

    row_values = { ...
        combIdx, file_names{1}, file_names{2}, file_names{3}, ...
        r_kms(1), r_kms(2), r_kms(3), z_srcs(1), z_srcs(2), z_srcs(3), ...
        Pos.r.z(nz_recv), target_fs, sl_cfg.wind_speed, sl_cfg.shipping_factor, ...
        n_rms_abs, snr_band, sl_arr(1), sl_arr(2), sl_arr(3), ...
        er_arr(1), er_arr(2), er_arr(3), ...
        scale_audit.scale_c, scale_audit.pre_mix_peak, scale_audit.post_mix_peak, ...
        scale_audit.pre_rms_mix, scale_audit.post_rms_mix, ...
        scale_audit.pre_rms_signal, scale_audit.post_rms_signal, ...
        scale_audit.pre_rms_noise, scale_audit.post_rms_noise, ...
        scale_audit.pre_rms_channels(1), scale_audit.pre_rms_channels(2), scale_audit.pre_rms_channels(3), ...
        scale_audit.post_rms_channels(1), scale_audit.post_rms_channels(2), scale_audit.post_rms_channels(3), ...
        sl_raw_arr(1), sl_raw_arr(2), sl_raw_arr(3), ...
        sl_used_arr(1), sl_used_arr(2), sl_used_arr(3), ...
        er_raw_pair_log, er_used_pair_log, ...
        sir_audit.overlap_start_s, sir_audit.overlap_end_s, sir_audit.overlap_samples, ...
        sir_audit.sir_rx_global_db, sir_audit.sir_rx_overlap_db, ...
        sir_audit.sir_global_valid, sir_audit.sir_overlap_valid, ...
        sir_audit.sir_rx_global_signed_db, sir_audit.sir_rx_overlap_signed_db, ...
        sir_audit.sir_rx_global_band_db, sir_audit.sir_rx_overlap_band_db, ...
        sir_audit.sir_rx_global_band_signed_db, sir_audit.sir_rx_overlap_band_signed_db, ...
        sir_audit.sir_global_band_valid, sir_audit.sir_overlap_band_valid};
        write_all_info_row(fid_log, all_info_fields, row_values);
    end
end

if metadata_only
    fprintf('  pilot 完成: Train/%s (%d 候选，仅元数据)\n', type_name, numel(pilot_records));
else
    fclose(fid_log);
    source_audit_path = fullfile(out_dir, 'source_sampling_info.tsv');
    write_source_sampling_audit(source_audit_path, source_sampling_mode, ...
        target_indices, class_ids, src_pools, candidate_recording_counts, ...
        accepted_recording_counts, candidate_attempts, accepted_count);
    if stratified_enabled
        if ~isequal(sir_accepted_counts, sir_target_counts)
            error('gen_sl_dataset:SIRQuotaMismatch', ...
                'SIR 实际分层计数 %s 与目标 %s 不一致。', ...
                mat2str(sir_accepted_counts), mat2str(sir_target_counts));
        end
        sampling_audit_path = fullfile(out_dir, 'sir_sampling_info.txt');
        write_sir_sampling_audit(sampling_audit_path, audit_band, ...
            sir_bin_edges_db, sir_bin_weights, sir_target_counts, ...
            sir_accepted_counts, candidate_attempts, sir_rejected_invalid, ...
            sir_rejected_outside, sir_rejected_full, sir_max_attempt_factor);
        fprintf(['  SIR 筛选完成: 接受=%s，候选=%d，拒绝 invalid/outside/full=' ...
            '%d/%d/%d\n'], mat2str(sir_accepted_counts), candidate_attempts, ...
            sir_rejected_invalid, sir_rejected_outside, sir_rejected_full);
    end
    fprintf('  完成: %s/%s (%d 样本)\n', type_name, split_type, num_samples);
end
end

%% ===== 辅助: D1 录音级候选/接受分布审计 =====
function count_map = increment_source_count(count_map, label_idx, recording_key)
key = sprintf('%d|%s', label_idx, recording_key);
if count_map.isKey(key)
    count_map(key) = count_map(key) + 1;
else
    count_map(key) = 1;
end
end

function value = source_count(count_map, label_idx, recording_key)
key = sprintf('%d|%s', label_idx, recording_key);
if count_map.isKey(key), value = count_map(key); else, value = 0; end
end

function write_source_sampling_audit(path, mode, target_indices, class_ids, ...
    pools, candidate_counts, accepted_counts, candidate_attempts, accepted_samples)
fid = fopen(path, 'wt');
if fid < 0
    error('gen_sl_dataset:SourceSamplingAuditOpenFailed', ...
        '无法创建源采样审计文件: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, ['source_sampling_mode\tlabel_idx\tclass_folder\trecording_key\t' ...
              'templates_in_split\tcandidate_uses\taccepted_uses\taccept_rate\t' ...
              'candidate_attempts\taccepted_samples\n']);
for t = 1:numel(target_indices)
    label_idx = target_indices(t);
    pool = pools{t};
    for r = 1:pool.n_recordings
        recording = pool.recording_keys{r};
        candidates = source_count(candidate_counts, label_idx, recording);
        accepted = source_count(accepted_counts, label_idx, recording);
        if candidates > 0, accept_rate = accepted / candidates; else, accept_rate = 0; end
        fprintf(fid, '%s\t%d\t%d\t%s\t%d\t%d\t%d\t%.17g\t%d\t%d\n', ...
            mode, label_idx, class_ids(label_idx + 1), recording, ...
            pool.template_counts(r), candidates, accepted, accept_rate, ...
            candidate_attempts, accepted_samples);
    end
end
clear cleanup
end

%% ===== 辅助: SIR 条件抽样审计 =====
function write_sir_sampling_audit(path, audit_band, bin_edges, bin_weights, ...
    target_counts, accepted_counts, candidate_attempts, rejected_invalid, ...
    rejected_outside, rejected_full, max_attempt_factor)
fid = fopen(path, 'wt');
if fid < 0
    error('gen_sl_dataset:SIRAuditOpenFailed', ...
        '无法创建 SIR 抽样审计文件: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'policy\tstratified\n');
fprintf(fid, 'metric\tsir_rx_overlap_band_db\n');
fprintf(fid, 'audit_band_hz\t%s\n', mat2str(audit_band));
fprintf(fid, 'bin_edges_db\t%s\n', mat2str(bin_edges));
fprintf(fid, 'bin_weights\t%s\n', mat2str(bin_weights));
fprintf(fid, 'target_counts\t%s\n', mat2str(target_counts));
fprintf(fid, 'accepted_counts\t%s\n', mat2str(accepted_counts));
fprintf(fid, 'candidate_attempts\t%d\n', candidate_attempts);
fprintf(fid, 'rejected_invalid\t%d\n', rejected_invalid);
fprintf(fid, 'rejected_outside_range\t%d\n', rejected_outside);
fprintf(fid, 'rejected_stratum_full\t%d\n', rejected_full);
fprintf(fid, 'max_attempt_factor\t%d\n', max_attempt_factor);
fprintf(fid, 'source_sl_modified\t0\n');
fprintf(fid, 'gain_adjustment_applied\t0\n');
clear cleanup
end

%% ===== 辅助: Wenz 带内功率积分 (μPa², 对数网格正确权重) =====
% dF = f·dlnf (log 网格的积分元); 曾用线性权重 (f_hi-f_lo)/n 导致 Wenz 低频段
% (谱值最大且网格最密) 被高估 ×20 (+13dB) — SS2 单目标实测 SNR 中位 4.6dB
% (应 ~17.5) 的根因, 逐行回归已验证 (残差中位 -1.2dB, r/z 无系统依赖)
function p = wenz_band_power(w, sf, f_lo, f_hi, fs_dummy) %#ok<INUSD>
    n = 20000;
    fgrid = f_lo * (f_hi / f_lo) .^ ((0:n-1)' / (n-1));
    dlnf = log(f_hi / f_lo) / (n - 1);
    fk = fgrid / 1000;
    p_turb = 10 .^ ((17 - 30 * log10(max(fk, 0.01))) / 10);
    p_turb(fgrid < 10) = 0;                               % 湍流适用域 (与生成器一致)
    p_ship = 10 .^ ((40 + 20 * (sf - 0.5) + 26 * log10(fk) - 60 * log10(fk + 0.03)) / 10);
    p_wind = 10 .^ ((50 + 7.5 * sqrt(w) + 20 * log10(fk) - 40 * log10(fk + 0.4)) / 10);
    p = sum((p_turb + p_ship + p_wind) .* fgrid) * dlnf;   % 权重 = f·dlnf
end

%% ===== 辅助: 随机窗口提取 =====
function y_seg = extract_random_window(y, target_samples)
    y = y(:);
    N = length(y);
    if N < target_samples
        n_repeats = ceil(target_samples / N);
        y_rep = repmat(y, n_repeats, 1);
        y_seg = y_rep(1:target_samples);
    elseif N == target_samples
        y_seg = y;
    else
        start = randi(N - target_samples + 1);
        y_seg = y(start:start + target_samples - 1);
    end
end

%% ===== 辅助: .arr 文件缓存加载 =====
function [Arr_all, Pos, f_centers] = load_arr_cached(arr_path)
    persistent cached_path cached_Arr cached_Pos cached_fc
    if isequal(cached_path, arr_path) && ~isempty(cached_Arr)
        Arr_all = cached_Arr; Pos = cached_Pos; f_centers = cached_fc;
        return;
    end
    f_centers = 50:100:7950;
    num_freqs = length(f_centers);
    Arr_all = cell(1, num_freqs);
    arr_prefix = 'Pos1Azi1';
    if exist(fullfile(arr_path, 'Pos1denseAzi1freq50Hz.arr'), 'file')
        arr_prefix = 'Pos1denseAzi1';
    end
    if exist(fullfile(arr_path, sprintf('%sfreq15950Hz.arr', arr_prefix)), 'file')
        f_centers = 50:100:15950;
        num_freqs = length(f_centers);
        Arr_all = cell(1, num_freqs);
    end
    fprintf('  加载 %d 个 .arr 文件 (前缀: %s, %d-%d Hz)...\n', ...
        num_freqs, arr_prefix, f_centers(1), f_centers(end));
    for i = 1:num_freqs
        fname = fullfile(arr_path, sprintf('%sfreq%dHz.arr', arr_prefix, f_centers(i)));
        [Arr_all{i}, Pos] = read_arrivals_asc(fname);
    end
    cached_path = arr_path; cached_Arr = Arr_all; cached_Pos = Pos; cached_fc = f_centers;
end

%% ===== 辅助: 信道 H 预计算 =====
function H = build_channel_H(cell_Amp, cell_tau, f_centers, fs, N_pad, max_delay_s)
    fk = (0:N_pad-1)' / N_pad * fs;
    half_N = floor(N_pad / 2) + 1;
    fk_half = fk(1:half_N);
    global_min_tau = inf;
    for i = 1:length(cell_tau)
        if ~isempty(cell_tau{i})
            global_min_tau = min(global_min_tau, min(real(cell_tau{i})));
        end
    end
    H = zeros(half_N, 1);
    num_freqs = length(f_centers);
    for i = 1:(num_freqs - 1)
        fc_left  = f_centers(i);
        fc_right = f_centers(i + 1);
        if i == 1
            idx = find(fk_half >= 0 & fk_half <= fc_right);
        elseif i == num_freqs - 1
            idx = find(fk_half > fc_left & fk_half <= fs / 2);
        else
            idx = find(fk_half > fc_left & fk_half <= fc_right);
        end
        if isempty(idx), continue; end
        f_eval = fk_half(idx);
        tau_L_all = real(cell_tau{i})     - global_min_tau;
        tau_R_all = real(cell_tau{i + 1}) - global_min_tau;
        mask_L = tau_L_all <= max_delay_s;
        mask_R = tau_R_all <= max_delay_s;
        tau_L = tau_L_all(mask_L); tau_R = tau_R_all(mask_R);
        Amp_L = cell_Amp{i}(mask_L);   Amp_R = cell_Amp{i + 1}(mask_R);
        if ~isempty(Amp_L)
            H_L = (Amp_L(:).') * exp(-1j * 2 * pi * tau_L(:) * (f_eval(:).' - fc_left));
        else
            H_L = zeros(1, length(idx));
        end
        if ~isempty(Amp_R)
            H_R = (Amp_R(:).') * exp(-1j * 2 * pi * tau_R(:) * (f_eval(:).' - fc_right));
        else
            H_R = zeros(1, length(idx));
        end
        w_R = (f_eval(:).' - fc_left) / (fc_right - fc_left);
        if i == 1,             w_R(f_eval(:).' < fc_left)  = 0; end
        if i == num_freqs - 1, w_R(f_eval(:).' > fc_right) = 1; end
        H(idx) = (1 - w_R) .* H_L + w_R .* H_R;
    end
end

%% ===== 辅助: 用预计算 H 做线性卷积 =====
function y = convolve_precomputed_H(x, H, N_pad)
    N = length(x);
    Fx = fft(x, N_pad);
    half_N = floor(N_pad / 2) + 1;
    Fy_half = Fx(1:half_N) .* H;
    Fy_half(1) = real(Fy_half(1));
    if mod(N_pad, 2) == 0
        Fy_half(end) = real(Fy_half(end));
        Fy = [Fy_half; conj(flipud(Fy_half(2:end-1)))];
    else
        Fy = [Fy_half; conj(flipud(Fy_half(2:end)))];
    end
    y = real(ifft(Fy));
    y = y(1:N);
end

%% ===== 辅助: 淡入淡出曲线 =====
function env = fade_shape(shape, n)
    t = (0:n-1)' / max(n - 1, 1);
    switch shape
        case 1,   env = t;
        case 2,   env = 0.5 - 0.5 * cos(pi * t);
        case 3,   env = (exp(2 * t) - 1) / (exp(2) - 1);
        otherwise, env = t;
    end
end

%% ===== 辅助: 带内功率 (单边谱积分, 比值口径一致即可) =====
function p = band_power(x, fs, f_lo, f_hi)
    N = length(x);
    X = fft(x);
    half_N = floor(N / 2) + 1;
    f_axis = (0:half_N-1)' * (fs / N);
    idx = f_axis >= f_lo & f_axis <= f_hi;
    p = 2 * sum(abs(X(idx)) .^ 2) / N ^ 2;
end

%% ===== 辅助: 统一 all_info 表头与行写入 =====
function write_all_info_header(fid, fields)
    fprintf(fid, '%s\n', strjoin(fields, '\t'));
end

function write_all_info_row(fid, fields, values)
    if numel(values) ~= numel(fields)
        error('gen_sl_dataset:AllInfoColumnMismatch', ...
            'all_info 行列数不一致: 表头 %d 列，数据 %d 列。', numel(fields), numel(values));
    end
    for field_idx = 1:numel(fields)
        if field_idx > 1, fprintf(fid, '\t'); end
        value = values{field_idx};
        if ischar(value) || (isstring(value) && isscalar(value))
            fprintf(fid, '%s', char(value));
        elseif isnumeric(value) && isscalar(value)
            if isnan(value)
                fprintf(fid, 'nan');
            elseif isinf(value)
                if value < 0, fprintf(fid, '-inf'); else, fprintf(fid, 'inf'); end
            else
                fprintf(fid, '%.17g', value);
            end
        else
            error('gen_sl_dataset:AllInfoValueType', ...
                'all_info 字段 %s 必须是字符串或数值标量。', fields{field_idx});
        end
    end
    fprintf(fid, '\n');
end
