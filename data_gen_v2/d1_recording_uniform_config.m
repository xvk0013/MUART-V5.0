function cfg = d1_recording_uniform_config()
%% d1_recording_uniform_config — D1 单变量数据消融配置
%
% 继承 bp12 的全部物理与规模常量，只覆盖：
%   1) 输出根目录；2) source_sampling_mode；3) 只生成 Train。
% Val 从原 bp12 数据集逐文件复制并由 Python 审计做内容哈希核对。

cfg = bp12_mainline_config();
cfg.reference_dataset_root = fullfile(cfg.output_root, cfg.cond_names{1}, 'dataset');
cfg.output_root = fullfile(cfg.data_root, ...
    'data_cargo_sl_sir15_d1_recording_uniform');
cfg.source_sampling_mode = 'recording_uniform';
cfg.split_types = {'Train'};
cfg.totals = cfg.totals(1);
cfg.copy_frozen_val = true;
cfg.sl_table_name = 'sl_tables_train_only.txt';
end
