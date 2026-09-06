function data_root = resolve_muart_data_root()
%% resolve_muart_data_root — 集中解析 MUART 外部大数据根目录
%
% 优先读取 MUART_DATA_ROOT。若已设置但目录或冻结 bp12 输入不存在，立即报错。
% 未设置时只检查迁移包内的 Data 子目录。有效根目录必须直接包含
% data_sl_templates_bp12 与 data_sl_split_bp12；不包含任何机器专属盘符。

env_name = 'MUART_DATA_ROOT';
env_root = strtrim(getenv(env_name));
if ~isempty(env_root)
    data_root = validate_data_root(env_root, env_name);
    fprintf('[MUART] 数据根目录（%s）: %s\n', env_name, data_root);
    return;
end

data_gen_dir = fileparts(mfilename('fullpath'));
package_root = fileparts(data_gen_dir);
package_data = fullfile(package_root, 'Data');
if ~is_valid_data_root(package_data)
    error('resolve_muart_data_root:NotFound', ...
        ['未找到包含 data_sl_templates_bp12 和 data_sl_split_bp12 的数据根目录。' ...
         '\n请设置环境变量 %s，或使用包内 Data 目录: %s'], ...
        env_name, package_data);
end
data_root = package_data;
fprintf('[MUART] 数据根目录（包内 Data）: %s\n', data_root);
end


function data_root = validate_data_root(candidate, source_name)
if ~isfolder(candidate)
    error('resolve_muart_data_root:InvalidEnvironmentPath', ...
        '%s 指向的目录不存在: %s', source_name, candidate);
end
if ~is_valid_data_root(candidate)
    error('resolve_muart_data_root:InvalidEnvironmentLayout', ...
        ['%s 必须直接指向同时包含 data_sl_templates_bp12 和 ' ...
         'data_sl_split_bp12 的目录: %s'], source_name, candidate);
end
data_root = candidate;
end


function tf = is_valid_data_root(candidate)
tf = isfolder(candidate) && ...
    isfolder(fullfile(candidate, 'data_sl_templates_bp12')) && ...
    isfolder(fullfile(candidate, 'data_sl_split_bp12'));
end
