function result = validate_d1_recording_uniform()
%% validate_d1_recording_uniform — D1 只读配置/输入检查

cfg = d1_recording_uniform_config();
fprintf('========== D1 recording-uniform 只读校验 ==========\n');

check(strcmp(cfg.source_sampling_mode, 'recording_uniform'), ...
    '采样模式', cfg.source_sampling_mode);
check(isequal(cfg.split_types, {'Train'}) && isscalar(cfg.totals) && cfg.totals == 11490, ...
    '只生成 Train', sprintf('%s / %d', strjoin(cfg.split_types, ','), cfg.totals));
check(cfg.copy_frozen_val, 'Val 策略', '只复制冻结 Val，不重新生成');
reference_output_root = fileparts(fileparts(fileparts(cfg.reference_dataset_root)));
check(~strcmpi(cfg.output_root, reference_output_root), ...
    '输出隔离', cfg.output_root);
check(isequal(cfg.sir_bin_edges_db, [0 5 10 15]) && ...
      isequal(cfg.sir_bin_weights, [0.45 0.35 0.20]), ...
    'SIR 保持', '0-15 dB, 45/35/20');
check(strcmp(cfg.er_policy, 'natural') && ...
      isequal(cfg.sl_mean, [183 183 180.5]) && cfg.sl_std == 3, ...
    'SL/ER 保持', 'natural, [183 183 180.5], std=3');

required = {'build_source_sampling_pool.m', 'sample_source_from_pool.m', ...
            'gen_sl_dataset.m', 'main_cargo_sl_d1.m'};
for idx = 1:numel(required)
    check(isfile(fullfile(cfg.data_gen_dir, required{idx})), ...
        required{idx}, '存在');
end
arr_files = dir(fullfile(cfg.data_gen_dir, '.arr_gen', '*.arr'));
check(numel(arr_files) == 80, 'Bellhop ARR 池', sprintf('%d 个', numel(arr_files)));

for label_idx = 0:2
    class_folder = cfg.class_ids(label_idx+1);
    source = fullfile(cfg.split_root, 'Train', num2str(class_folder));
    wavs = dir(fullfile(source, '*.wav'));
    check(~isempty(wavs), sprintf('Train/%d', class_folder), ...
        sprintf('%d 模板', numel(wavs)));
end
for idx = 1:size(cfg.combos,1)
    type_name = cfg.combos{idx,2};
    for split = {'Train','Val'}
        source = fullfile(cfg.reference_dataset_root, type_name, split{1});
        check(isfolder(source), sprintf('冻结 %s/%s', type_name, split{1}), source);
    end
end

generator = fileread(fullfile(cfg.data_gen_dir, 'gen_sl_dataset.m'));
entry = fileread(fullfile(cfg.data_gen_dir, 'main_cargo_sl_d1.m'));
check(contains(generator, 'gen_sl_dataset:D1TrainOnly') && ...
      contains(generator, 'source_sampling_info.tsv') && ...
      contains(generator, 'candidate_recording_counts') && ...
      contains(generator, 'accepted_recording_counts'), ...
    '生成器 D1 防护', 'Train-only + 候选/接受双重审计');
check(contains(entry, 'copy_frozen_val') && ...
      contains(entry, 'assert_d1_output_empty') && ...
      contains(entry, 'rng_sl_seed + label_idx') && ...
      contains(entry, '''Train'', ''all_info.txt'''), ...
    'D1 入口边界', '非空拒绝覆盖；SL 原种子重建并由 Train 核验；Val 仅复制');

main_cargo_sl_d1(true);

fprintf('校验结果: PASS（未生成数据）\n');
result = struct('passed', true, 'config', cfg);
end


function check(condition, label, value)
if ~condition
    error('validate_d1_recording_uniform:Failed', '%s 校验失败: %s', label, value);
end
fprintf('  [PASS] %-24s %s\n', label, value);
end
