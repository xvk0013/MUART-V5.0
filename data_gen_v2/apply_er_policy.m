function [sl_used, er_raw_db, er_used_db, er_raw_pair_db, er_used_pair_db] = ...
    apply_er_policy(sl_raw, er_policy, er_clip_db)
%% apply_er_policy — 纯 SL/ER 策略函数，不读取数据且不调用 Bellhop
%
% natural: sl_used 与逐录音表中的 sl_raw 完全一致。
% clip:    仅供显式历史消融；双目标超限时保持均值并向中值收拢。

if nargin < 2 || ~(ischar(er_policy) || (isstring(er_policy) && isscalar(er_policy)))
    error('apply_er_policy:InvalidPolicy', ...
        'er_policy 必须显式指定为 ''natural'' 或 ''clip''。');
end
if ~isnumeric(sl_raw) || ~isreal(sl_raw) || any(~isfinite(sl_raw(:)))
    error('apply_er_policy:InvalidSL', 'sl_raw 必须是有限实数数组。');
end

policy = lower(strtrim(char(er_policy)));
sl_raw = double(sl_raw);
sl_used = sl_raw;
persistent clip_warning_emitted

switch policy
    case 'natural'
        % 主线语义：完全忽略 er_clip_db，不允许改写源级 SL。
    case 'clip'
        if nargin < 3 || ~isscalar(er_clip_db) || ~isnumeric(er_clip_db) || ...
                ~isfinite(er_clip_db) || er_clip_db < 0
            error('apply_er_policy:InvalidClip', ...
                'clip 策略必须显式提供非负有限标量 er_clip_db。');
        end
        if isempty(clip_warning_emitted) || ~clip_warning_emitted
            warning('apply_er_policy:ArtificialERConstraint', ...
                ['er_policy=clip 是人为 ER 约束，仅用于历史消融；' ...
                 '它不是逐录音固定 SL 的物理主线。']);
            clip_warning_emitted = true;
        end
        if numel(sl_raw) == 2
            er_pair = sl_raw(1) - sl_raw(2);
            if abs(er_pair) > er_clip_db
                midpoint = (sl_raw(1) + sl_raw(2)) / 2;
                direction = sign(er_pair);
                sl_used(1) = midpoint + direction * er_clip_db / 2;
                sl_used(2) = midpoint - direction * er_clip_db / 2;
            end
        end
    otherwise
        error('apply_er_policy:InvalidPolicy', ...
            '未知 er_policy=''%s''；仅支持 ''natural'' 或 ''clip''。', policy);
end

if isempty(sl_raw)
    er_raw_db = sl_raw;
    er_used_db = sl_used;
else
    er_raw_db = sl_raw - min(sl_raw);
    er_used_db = sl_used - min(sl_used);
end

er_raw_pair_db = NaN;
er_used_pair_db = NaN;
if numel(sl_raw) == 2
    er_raw_pair_db = sl_raw(1) - sl_raw(2);
    er_used_pair_db = sl_used(1) - sl_used(2);
end

if strcmp(policy, 'natural')
    tol_db = 1e-9;
    if any(abs(sl_used(:) - sl_raw(:)) > tol_db)
        error('apply_er_policy:NaturalSLModified', ...
            'natural 策略检测到 sl_used 与 sl_raw 不一致。');
    end
    if any(abs(er_used_db(:) - er_raw_db(:)) > tol_db) || ...
            (numel(sl_raw) == 2 && abs(er_used_pair_db - er_raw_pair_db) > tol_db)
        error('apply_er_policy:NaturalERModified', ...
            'natural 策略检测到 er_used 与 er_raw 不一致。');
    end
end
end
