function main_cargo_sl_d1(dry_run)
%% main_cargo_sl_d1 — D1：Train 录音均匀、Val 原样复用
%
% 唯一研究变量：候选源由“全部模板均匀”改为“先均匀选 Train 原始
% 录音，再在该录音的 Train 模板内均匀选一个”。SL/ER/SIR/Bellhop/
% Wenz/样本规模均继承冻结 bp12。正式输出目录非空时立即拒绝覆盖。
%
% 本入口只生成 Train；Val 从原 bp12 数据集复制并在后续审计中逐文件
% 做 SHA-256 内容核对。不枚举、不复制其他 split。

if nargin < 1, dry_run = false; end
if ~(islogical(dry_run) && isscalar(dry_run))
    error('main_cargo_sl_d1:InvalidDryRun', 'dry_run 必须是逻辑标量。');
end

proj = fileparts(mfilename('fullpath'));
addpath(proj);
cfg = d1_recording_uniform_config();
arr_path = fullfile(proj, '.arr_gen');
dataset_root = fullfile(cfg.output_root, cfg.cond_names{1}, 'dataset');

assert_d1_output_empty(cfg.output_root);
assert_reference_train_val(cfg.reference_dataset_root, cfg.combos(:,2));

fprintf('========== D1: 构建 Train-only 录音映射与冻结 SL ==========' );
fprintf('\n');
[recording_maps, recording_sl_maps, sl_maps] = ...
    build_train_maps_and_reference_sl(cfg);

if dry_run
    fprintf('D1 PREPARE CHECK PASS：逐录音 SL 已按原始确定性种子重建并由原 Train 交叉核验。\n');
    return;
end

if ~exist(cfg.output_root, 'dir'), mkdir(cfg.output_root); end
write_train_sl_table(fullfile(cfg.output_root, cfg.sl_table_name), ...
    cfg, recording_sl_maps);

combo_counts = d1_combo_counts(cfg.totals, cfg.props);
rng(42 + 3);
for c = 1:size(cfg.combos, 1)
    if combo_counts(c) <= 0, continue; end
    fprintf('  生成 D1 %s/Train (%d 样本)...\n', ...
        cfg.combos{c,2}, combo_counts(c));
    gen_sl_dataset(cfg.combos{c,1}, 'Train', cfg.split_root, dataset_root, ...
        arr_path, combo_counts(c), 5, 3.0, cfg.class_ids, 16000, true, ...
        [0.3, 1.5], [0.3, 0.8], ...
        struct('wind_speed', cfg.cond_winds(1), ...
               'shipping_factor', cfg.shipping_factor, ...
               'sl_maps', sl_maps, ...
               'recording_maps', recording_maps, ...
               'recording_sl_maps', recording_sl_maps, ...
               'er_policy', cfg.er_policy, ...
               'sir_mode', cfg.sir_mode, ...
               'band', cfg.audit_band, ...
               'sir_bin_edges_db', cfg.sir_bin_edges_db, ...
               'sir_bin_weights', cfg.sir_bin_weights, ...
               'sir_max_attempt_factor', cfg.sir_max_attempt_factor, ...
               'source_sampling_mode', cfg.source_sampling_mode));
end

if cfg.copy_frozen_val
    copy_frozen_val(cfg.reference_dataset_root, dataset_root, cfg.combos(:,2));
end
write_d1_protocol(fullfile(dataset_root, 'd1_protocol.txt'), cfg);

fprintf('\n========== D1 生成完成 ==========' );
fprintf('\nTrain: %s\n', dataset_root);
fprintf('Val: 从 %s 原样复制，下一步必须运行 audit_d1_dataset.py。\n', ...
    cfg.reference_dataset_root);
fprintf('不要在审计 PASS 前训练。\n');
end


function counts = d1_combo_counts(total, props)
n_noise = round(total * props.noise);
n_single = round(total * props.single / 3);
n_double = round(total * props.double / 3);
counts = [n_noise, n_single, n_single, n_single, ...
          n_double, n_double, n_double];
if sum(counts) ~= total
    error('main_cargo_sl_d1:CountMismatch', ...
        'D1 七集合计数 %d 与总数 %d 不一致。', sum(counts), total);
end
end


function assert_d1_output_empty(output_root)
if ~exist(output_root, 'dir'), return; end
items = dir(output_root);
items = items(~ismember({items.name}, {'.', '..'}));
if ~isempty(items)
    error('main_cargo_sl_d1:NonEmptyOutput', ...
        ['正式 D1 输出目录已经非空，拒绝删除或覆盖: %s\n' ...
         '请保留目录并先核查；若确需重跑，应显式指定新的输出根。'], ...
        output_root);
end
end


function assert_reference_train_val(reference_root, type_names)
if ~isfolder(reference_root)
    error('main_cargo_sl_d1:MissingReference', ...
        '原 bp12 数据集不存在: %s', reference_root);
end
for idx = 1:numel(type_names)
    for split = {'Train', 'Val'}
        path = fullfile(reference_root, type_names{idx}, split{1});
        if ~isfolder(path)
            error('main_cargo_sl_d1:MissingReferenceSplit', ...
                '原冻结目录不存在: %s', path);
        end
    end
end
end


function [recording_maps, recording_sl_maps, sl_maps] = ...
    build_train_maps_and_reference_sl(cfg)
recording_maps = containers.Map('KeyType', 'char', 'ValueType', 'any');
recording_sl_maps = containers.Map('KeyType', 'char', 'ValueType', 'any');
sl_maps = containers.Map('KeyType', 'char', 'ValueType', 'any');
all_recordings_by_label = cell(1, 3);

for label_idx = 0:2
    class_folder = cfg.class_ids(label_idx + 1);
    split_dir = fullfile(cfg.split_root, 'Train', num2str(class_folder));
    wavs = dir(fullfile(split_dir, '*.wav'));
    if isempty(wavs)
        error('main_cargo_sl_d1:EmptyTrainPool', 'Train 模板池为空: %s', split_dir);
    end
    allowed = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    for idx = 1:numel(wavs), allowed(wavs(idx).name) = true; end

    mapping_path = fullfile(cfg.template_root, num2str(class_folder), ...
        'segment_mapping.txt');
    fid = fopen(mapping_path, 'rt');
    if fid < 0
        error('main_cargo_sl_d1:MappingOpenFailed', '无法读取 %s', mapping_path);
    end
    cleanup = onCleanup(@() fclose(fid));
    fgetl(fid);
    file2rec = containers.Map('KeyType', 'char', 'ValueType', 'char');
    all_recordings = {};
    while ~feof(fid)
        line = fgetl(fid);
        if ~ischar(line) || isempty(strtrim(line)), continue; end
        parts = strsplit(strtrim(line), sprintf('\t'));
        if numel(parts) < 3, continue; end
        filename = strtrim(parts{1});
        recording = [lower(strtrim(parts{2})) '/' ...
                     lower(strtrim(parts{3}))];
        all_recordings{end+1} = recording; %#ok<AGROW>
        if allowed.isKey(filename)
            file2rec(filename) = recording;
        end
    end
    clear cleanup
    if file2rec.Count ~= numel(wavs)
        error('main_cargo_sl_d1:TrainMappingIncomplete', ...
            '类别 %d 的 Train 模板 %d 个，映射仅 %d 个。', ...
            class_folder, numel(wavs), file2rec.Count);
    end
    recording_maps(num2str(label_idx)) = file2rec;
    recording_sl_maps(num2str(label_idx)) = ...
        containers.Map('KeyType', 'char', 'ValueType', 'double');
    all_recordings_by_label{label_idx + 1} = sort(unique(all_recordings));
end

% 原主线先对完整 recording key 排序，再按标签独立种子生成逐录音 SL。
% 这里严格复现同一随机过程，然后只保留冻结 Train 中的录音。这样即使某条
% Train 录音碰巧从未出现在原 Train/all_info 中，也能恢复其未舍入的原始 SL。
for label_idx = 0:2
    key = num2str(label_idx);
    all_recordings = all_recordings_by_label{label_idx + 1};
    rng(cfg.rng_sl_seed + label_idx);
    full_rec2sl = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for idx = 1:numel(all_recordings)
        full_rec2sl(all_recordings{idx}) = ...
            cfg.sl_mean(label_idx + 1) + cfg.sl_std * randn();
    end
    file2rec = recording_maps(key);
    train_recordings = unique(values(file2rec));
    rec2sl = recording_sl_maps(key);
    for idx = 1:numel(train_recordings)
        recording = train_recordings{idx};
        if ~full_rec2sl.isKey(recording)
            error('main_cargo_sl_d1:TrainRecordingOutsideOriginalPool', ...
                'Train 录音不在原始完整录音池: 标签 %d, %s', label_idx, recording);
        end
        rec2sl(recording) = full_rec2sl(recording);
    end
    recording_sl_maps(key) = rec2sl;
end

% 用原 Train/all_info 对所有实际出现过的录音做精确交叉核验；不读取其他 split。
file_fields = {'fileA', 'fileB', 'fileC'};
sl_fields = {'sl_raw_1', 'sl_raw_2', 'sl_raw_3'};
for combo_idx = 2:size(cfg.combos, 1)
    active = cfg.combos{combo_idx, 1};
    info_path = fullfile(cfg.reference_dataset_root, ...
        cfg.combos{combo_idx,2}, 'Train', 'all_info.txt');
    [header, rows] = read_tsv(info_path);
    for active_idx = 1:numel(active)
        label_idx = active(active_idx);
        file_col = required_column(header, file_fields{label_idx + 1}, info_path);
        sl_col = required_column(header, sl_fields{label_idx + 1}, info_path);
        file2rec = recording_maps(num2str(label_idx));
        rec2sl = recording_sl_maps(num2str(label_idx));
        for row_idx = 1:numel(rows)
            parts = rows{row_idx};
            filename = parts{file_col};
            if strcmp(filename, 'none'), continue; end
            if ~file2rec.isKey(filename)
                error('main_cargo_sl_d1:ReferenceFileOutsideTrain', ...
                    '原 Train all_info 文件不在冻结 Train 池: %s', filename);
            end
            recording = file2rec(filename);
            sl_value = str2double(parts{sl_col});
            if ~isfinite(sl_value)
                error('main_cargo_sl_d1:InvalidReferenceSL', ...
                    '原 Train SL 非有限数: %s', parts{sl_col});
            end
            if ~rec2sl.isKey(recording)
                error('main_cargo_sl_d1:RebuiltSLMissing', ...
                    '重建 SL 表缺少 Train 录音: 标签 %d, %s', label_idx, recording);
            end
            if abs(rec2sl(recording)-sl_value) > 1e-9
                error('main_cargo_sl_d1:ReferenceSLChanged', ...
                    ['确定性重建 SL 与原 Train 不一致: 标签 %d, %s；' ...
                     '重建 %.17g，原记录 %.17g。'], ...
                    label_idx, recording, rec2sl(recording), sl_value);
            end
        end
        recording_sl_maps(num2str(label_idx)) = rec2sl;
    end
end

for label_idx = 0:2
    key = num2str(label_idx);
    file2rec = recording_maps(key);
    rec2sl = recording_sl_maps(key);
    recordings = unique(values(file2rec));
    missing = recordings(~cellfun(@(rec) rec2sl.isKey(rec), recordings));
    if ~isempty(missing)
        error('main_cargo_sl_d1:UnobservedTrainRecordingSL', ...
            '标签 %d 有 %d 条 Train 录音未在原 Train all_info 中观测到 SL。', ...
            label_idx, numel(missing));
    end
    f2sl = containers.Map('KeyType', 'char', 'ValueType', 'double');
    filenames = keys(file2rec);
    for idx = 1:numel(filenames)
        f2sl(filenames{idx}) = rec2sl(file2rec(filenames{idx}));
    end
    sl_maps(key) = f2sl;
    fprintf('  标签 %d: %d Train 录音 / %d 模板，SL 确定性重建并经原 Train 核验\n', ...
        label_idx, rec2sl.Count, f2sl.Count);
end
end


function [header, rows] = read_tsv(path)
fid = fopen(path, 'rt');
if fid < 0, error('main_cargo_sl_d1:TSVOpenFailed', '无法读取 %s', path); end
cleanup = onCleanup(@() fclose(fid));
first = fgetl(fid);
if ~ischar(first), error('main_cargo_sl_d1:EmptyTSV', '空文件: %s', path); end
header = strsplit(first, sprintf('\t'), 'CollapseDelimiters', false);
rows = {};
while ~feof(fid)
    line = fgetl(fid);
    if ~ischar(line) || isempty(line), continue; end
    parts = strsplit(line, sprintf('\t'), 'CollapseDelimiters', false);
    if numel(parts) ~= numel(header)
        error('main_cargo_sl_d1:TSVColumnMismatch', '列数不一致: %s', path);
    end
    rows{end+1} = parts; %#ok<AGROW>
end
clear cleanup
end


function idx = required_column(header, name, path)
idx = find(strcmp(header, name), 1);
if isempty(idx)
    error('main_cargo_sl_d1:MissingColumn', '%s 缺少列 %s', path, name);
end
end


function write_train_sl_table(path, cfg, recording_sl_maps)
fid = fopen(path, 'wt');
if fid < 0, error('main_cargo_sl_d1:SLOpenFailed', '无法写入 %s', path); end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'Label\tClass\tRecording\tSL_dB\n');
for label_idx = 0:2
    rec2sl = recording_sl_maps(num2str(label_idx));
    recordings = sort(keys(rec2sl));
    for idx = 1:numel(recordings)
        fprintf(fid, '%d\t%d\t%s\t%.17g\n', label_idx, ...
            cfg.class_ids(label_idx+1), recordings{idx}, rec2sl(recordings{idx}));
    end
end
clear cleanup
end


function copy_frozen_val(reference_root, destination_root, type_names)
fprintf('\n========== 复制冻结 Val（不重新生成） ==========\n');
for idx = 1:numel(type_names)
    source = fullfile(reference_root, type_names{idx}, 'Val');
    destination = fullfile(destination_root, type_names{idx}, 'Val');
    if exist(destination, 'dir') || exist(destination, 'file')
        error('main_cargo_sl_d1:ValDestinationExists', ...
            'D1 Val 目标已存在，拒绝覆盖: %s', destination);
    end
    [ok, message] = copyfile(source, destination);
    if ~ok
        error('main_cargo_sl_d1:ValCopyFailed', ...
            '复制冻结 Val 失败: %s', message);
    end
    fprintf('  已复制 %s/Val\n', type_names{idx});
end
end


function write_d1_protocol(path, cfg)
fid = fopen(path, 'wt');
if fid < 0, error('main_cargo_sl_d1:ProtocolOpenFailed', '无法写入 %s', path); end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'stage\tD1_recording_uniform\n');
fprintf(fid, 'source_sampling_mode\t%s\n', cfg.source_sampling_mode);
fprintf(fid, 'generated_splits\tTrain\n');
fprintf(fid, 'val_policy\tbyte_copy_from_frozen_reference\n');
fprintf(fid, 'reference_dataset_root\t%s\n', cfg.reference_dataset_root);
fprintf(fid, 'sir_edges_db\t%s\n', mat2str(cfg.sir_bin_edges_db));
fprintf(fid, 'sir_weights\t%s\n', mat2str(cfg.sir_bin_weights));
fprintf(fid, 'er_policy\t%s\n', cfg.er_policy);
fprintf(fid, 'test_loaded\t0\n');
clear cleanup
end
