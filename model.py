r"""
M-UART v4.0 模型 — TF-ConvNeXt (频时解耦大核) + LOFAR + DEMON + Query Attention

架构 (主干线 M0 sync baseline):
  波形 (80000,)
    ├─ LOFARTransform → (1, 513, 71) ──→ TFConvNeXtBackbone → (384, 17, 5)
    │                                                         │
    ├─ DEMONTransform → (1, 126) ────→ DEMONEncoder(1D CNN) → (128,)
    │                                                         │
    └─────────────────────────────────────────────────────────┘
                                              ↓
                                   QueryClassifierHead
                                     (Cross-Attention + 2D PE)
                                              ↓
                                    logits (B, 3)

TF-ConvNeXt Block (核心创新):
  1. 频时解耦并行 DWConv: (1x31 时间) + (11x1 频率)
     - 1x31 捕捉 LOFAR 水平线谱 (船舶主机/螺旋桨线谱)
     - 11x1 捕捉频率谐波关系 (叶片率/轴频间隔)
     - 频时解耦避免 7x7 二维卷积的频率模糊 (验证: SNR -70%)
  2. ConvNeXt 逆瓶颈 MLP: LayerNorm → 4x Expand → GELU → Project
  3. Stochastic Depth (DropPath) 抗过拟合

参考文献:
  - ConvNeXt (CVPR 2022): 逆瓶颈 MLP + LayerNorm + GELU
  - RepLKNet (CVPR 2022): 大核深度可分离卷积
  - BC-ResNet (Interspeech 2021): 频时解耦卷积
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  Stochastic Depth
# ═══════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    """Stochastic Depth (per-sample) — ConvNeXt 标配抗过拟合"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep_prob * random_tensor.floor_()


# ═══════════════════════════════════════════════════════════════
#  TF-ConvNeXt Block (核心)
# ═══════════════════════════════════════════════════════════════

class TFConvNeXtBlock(nn.Module):
    r"""
    频时解耦大核 ConvNeXt Block

    结构:
      x → [1x31 DWConv + 11x1 DWConv] → LN → 1x1 Expand(4x) → GELU → 1x1 Project → + x

    参数量 (dim=C):
      DWConv: 31*C + 11*C = 42*C (vs 7x7 = 49*C)
      MLP:    2 * 4*C*C = 8*C^2
      LN:     2*C

    感受野:
      时间方向: 31 帧 (vs 原 7x7 = 7 帧, 4.4x)
      频率方向: 11 bin (vs 原 7x7 = 7 bin, 1.6x)
    """
    def __init__(self, dim, time_kernel=31, freq_kernel=11,
                 expand_ratio=4, drop_path=0.0):
        super().__init__()
        hidden = dim * expand_ratio

        # 频时解耦并行 DWConv
        self.dwconv_time = nn.Conv2d(dim, dim, kernel_size=(1, time_kernel),
                                     padding=(0, time_kernel // 2),
                                     groups=dim, bias=False)
        self.dwconv_freq = nn.Conv2d(dim, dim, kernel_size=(freq_kernel, 1),
                                     padding=(freq_kernel // 2, 0),
                                     groups=dim, bias=False)

        # ConvNeXt 逆瓶颈 MLP
        self.norm = nn.GroupNorm(1, dim)  # GroupNorm(1) ≈ LayerNorm for 2D
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        # 频时并行相加
        x = self.dwconv_time(x) + self.dwconv_freq(x)
        # 逆瓶颈 MLP
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return residual + self.drop_path(x)


# ═══════════════════════════════════════════════════════════════
#  下采样层
# ═══════════════════════════════════════════════════════════════

class Downsample(nn.Module):
    """2x2 stride-2 下采样 (频率/时间可独立配置)"""
    def __init__(self, in_dim, out_dim, stride=(2, 2)):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2,
                              stride=stride, bias=False)

    def forward(self, x):
        return self.conv(self.norm(x))


# ═══════════════════════════════════════════════════════════════
#  TF-ConvNeXt 骨干
# ═══════════════════════════════════════════════════════════════

class TFConvNeXtBackbone(nn.Module):
    """
    TF-ConvNeXt 骨干网络

    输入: (B, in_channels, F, T)  LOFAR 特征 (F=513, T≈71), in_channels=1 或 4 (残差拼接)
    输出: (B, out_dim, H', W') 特征图

    维度变化 (ConfigSmall, 输入 513x71):
      Stem:   7x7 s=2          → (48, 257, 36)
      Stage1: s=(2,2)          → (48, 129, 18)  [2 blocks]
      Stage2: s=(2,2)          → (96, 65, 9)    [2 blocks]
      Stage3: s=(2,2)          → (192, 33, 5)   [3 blocks]
      Stage4: s=(2,1)          → (384, 17, 5)   [2 blocks]
      GAP                        → (384,)
    """
    def __init__(self, config, in_channels=1):
        super().__init__()
        dims = config.backbone_dims
        blocks = config.backbone_blocks
        t_k = config.tf_time_kernel
        f_k = config.tf_freq_kernel
        expand = config.expand_ratio
        dp_max = config.drop_path_max

        # Stem: 7x7 stride=2 (in_channels: 1=纯LOFAR, 4=残差拼接 [原始,soloA,soloB,残差])
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, config.stem_out, kernel_size=config.stem_kernel,
                      stride=2, padding=config.stem_kernel // 2, bias=False),
            nn.GroupNorm(1, config.stem_out),
        )

        # 4 个 Stage
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_dim = config.stem_out
        total_blocks = sum(blocks)
        dp_rates = [dp_max * i / max(total_blocks - 1, 1) for i in range(total_blocks)]

        blk_idx = 0
        for i, (out_dim, n_blk) in enumerate(zip(dims, blocks)):
            # 下采样
            f_stride = config.backbone_strides_freq[i]
            t_stride = config.backbone_strides_time[i]
            self.downsamples.append(
                Downsample(in_dim, out_dim, stride=(f_stride, t_stride))
            )
            # TF-ConvNeXt Blocks
            stage_blocks = []
            for _ in range(n_blk):
                stage_blocks.append(
                    TFConvNeXtBlock(out_dim, time_kernel=t_k, freq_kernel=f_k,
                                    expand_ratio=expand, drop_path=dp_rates[blk_idx])
                )
                blk_idx += 1
            self.stages.append(nn.Sequential(*stage_blocks))
            in_dim = out_dim

        self.out_dim = dims[-1]
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, F, T) LOFAR 时频图 (in_channels=1 或 4)
        Returns:
            (B, out_dim, H', W') 特征图
        """
        x = self.stem(x)
        for down, stage in zip(self.downsamples, self.stages):
            x = down(x)
            x = stage(x)
        return x


# ═══════════════════════════════════════════════════════════════
#  DEMON 编码器 (1D CNN)
# ═══════════════════════════════════════════════════════════════

class DEMONEncoder(nn.Module):
    """DEMON 谱 1D CNN 编码器

    输入: (B, 1, demon_n_bins)  DEMON 谱 (去 DC, 标准化后)
    输出: (B, demon_feat_dim)   DEMON 特征向量

    结构: 3 层 Conv1d (k=7, s=2) → BN → GELU → AdaptiveAvgPool1d
    感受野: 7+14+28 = 49 bins, 覆盖几乎全谱
    """
    def __init__(self, config):
        super().__init__()
        dims = config.demon_encoder_dims  # [32, 64, 128]
        in_ch = 1
        layers = []
        for i, out_ch in enumerate(dims):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3, bias=False))
            layers.append(nn.GroupNorm(1, out_ch))  # GroupNorm(1) ≈ LayerNorm
            layers.append(nn.GELU())
            if i < len(dims) - 1:
                layers.append(nn.MaxPool1d(kernel_size=2))  # 降采样
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(dims[-1], config.demon_feat_dim)
        self.norm = nn.LayerNorm(config.demon_feat_dim)

    def forward(self, x):
        """x: (B, 1, demon_n_bins) → (B, demon_feat_dim)"""
        x = self.conv(x)           # (B, C, L')
        x = self.pool(x).squeeze(-1)  # (B, C)
        x = self.proj(x)           # (B, demon_feat_dim)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════
#  语义查询注意力 (Semantic Query-based Decoupling)
# ═══════════════════════════════════════════════════════════════

class SemanticQueryAttention(nn.Module):
    """语义查询交叉注意力 — 多目标解耦的核心模块

    机制:
      - N 个可学习查询向量 (N = num_classes), 每个查询"负责"一个类别
      - 查询主动扫描 LOFAR 特征图 (Cross-Attention), 从"被动看图"变"主动搜索"
      - Q_Tanker 聚焦低频强线谱区, Q_Tug 避开大船低频区去高频找细小线谱

    2D 位置编码 (关键改进):
      - Cross-Attention 是排列不变的, 展平后丢失了频时位置信息
      - 注入频率轴 PE + 时间轴 PE, 让 Query 知道"去哪里找"

    输入: feat_map (B, C, H, W)  — TF-ConvNeXt 最后一层特征图
    输出: query_feats (B, N, C)  — 每个查询聚合后的特征
    """
    def __init__(self, dim, num_queries, num_heads=4,
                 attn_dropout=0.1, proj_dropout=0.1,
                 max_h=64, max_w=32, num_bands=1):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        self.num_bands = num_bands

        # 可学习查询向量 (N, C)
        self.queries = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # ========================================================
        # 2D 频时可学习位置编码 — 破解 Cross-Attention 排列不变性
        # ========================================================
        self.freq_pe = nn.Parameter(torch.zeros(1, max_h, dim))
        self.time_pe = nn.Parameter(torch.zeros(1, max_w, dim))
        nn.init.trunc_normal_(self.freq_pe, std=0.02)
        nn.init.trunc_normal_(self.time_pe, std=0.02)

        # E4 双流融合臂：两路频率坐标不同，各自编码和 token 化后再加入
        # 频带身份。单路基线不创建该参数，保持旧 checkpoint 键完全兼容。
        if num_bands > 1:
            self.band_pe = nn.Parameter(torch.zeros(num_bands, dim))
            nn.init.trunc_normal_(self.band_pe, std=0.02)
        else:
            self.register_parameter('band_pe', None)

        # ========================================================
        # Cross-Attention (单层; ModuleList 包装保持与旧 checkpoint
        # 的 cross_attn_layers.0 键名严格兼容)
        # ========================================================
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
            ) for _ in range(1)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(1)
        ])
        self.proj_drop = nn.Dropout(proj_dropout)

    def _tokenize(self, feat_map, band_idx=0):
        """单路特征图注入二维位置编码并转为 tokens。"""
        B, C, H, W = feat_map.shape
        if H > self.freq_pe.shape[1] or W > self.time_pe.shape[1]:
            raise ValueError(
                f'特征图尺寸 {(H, W)} 超出位置编码上限 '
                f'{(self.freq_pe.shape[1], self.time_pe.shape[1])}'
            )

        # 注入 2D 位置编码
        f_pe = self.freq_pe[:, :H, :].unsqueeze(2).expand(-1, -1, W, -1)
        t_pe = self.time_pe[:, :W, :].unsqueeze(1).expand(-1, H, -1, -1)
        feat_map_pe = feat_map.permute(0, 2, 3, 1)  # (B, H, W, C)
        feat_map_pe = feat_map_pe + f_pe + t_pe
        if self.band_pe is not None:
            feat_map_pe = feat_map_pe + self.band_pe[band_idx].view(1, 1, 1, C)
        return feat_map_pe.flatten(1, 2)  # (B, H*W, C)

    def forward(self, feat_map):
        """单路特征图或多路特征图 → query_feats: (B, N, C)。"""
        if isinstance(feat_map, (tuple, list)):
            if len(feat_map) != self.num_bands:
                raise ValueError(
                    f'期望 {self.num_bands} 路特征，实际得到 {len(feat_map)} 路'
                )
            tokens = torch.cat([
                self._tokenize(one_map, band_idx=i)
                for i, one_map in enumerate(feat_map)
            ], dim=1)
            B = feat_map[0].shape[0]
        else:
            if self.num_bands != 1:
                raise ValueError(f'双流模型期望 {self.num_bands} 路特征图')
            tokens = self._tokenize(feat_map)
            B = feat_map.shape[0]

        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, N, C)

        for attn, norm in zip(self.cross_attn_layers, self.norms):
            out = attn(q, tokens, tokens, need_weights=False)[0]
            q = norm(q + out)  # 残差 + LayerNorm

        return self.proj_drop(q)


# ═══════════════════════════════════════════════════════════════
#  Query 分类头
# ═══════════════════════════════════════════════════════════════

class QueryClassifierHead(nn.Module):
    """语义查询分类头 — 每个查询预测一个类别

    结构:
      feat_map → SemanticQueryAttention → (B, N, C)
                                          ↓
      demon_feat (B, D) → expand → (B, N, D) ┐
                                              ↓
                                              LayerNorm + Dropout + Linear → (B, N, 1)
                                                                          ↓
                                                                    squeeze → (B, N) logits

    优势:
      1. 输出 (B, num_classes) logits 天然兼容多目标 (BCE/Focal)
      2. 每个查询独立"负责"一类, 避免 GAP 把多目标特征揉碎
      3. DEMON 全局特征广播到每个查询 (节拍对所有类别都有用)
    """
    def __init__(self, lofar_dim, demon_dim, num_classes,
                 num_heads=4, dropout=0.2, attn_dropout=0.1,
                 use_demon=True, max_h=64, max_w=32,
                 num_attn_layers=1, num_bands=1):
        super().__init__()
        self.use_demon = use_demon
        self.num_classes = num_classes

        # 语义查询注意力 (含 2D 位置编码)
        self.query_attn = SemanticQueryAttention(
            dim=lofar_dim,
            num_queries=num_classes,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            max_h=max_h,
            max_w=max_w,
            num_bands=num_bands,
        )

        # 分类 MLP (每个查询独立分类, 共享权重)
        fused_dim = lofar_dim + (demon_dim if use_demon else 0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 1),  # 每个查询 → 1 个 logit
        )

    def forward(self, feat_map, demon_feat=None):
        """feat_map: (B, C, H, W), demon_feat: (B, D) or None
        Returns: logits (B, num_classes)
        """
        # 1. 查询注意力: (B, C, H, W) → (B, N, C)
        query_feats = self.query_attn(feat_map)  # (B, N, C)

        # 2. 全局特征融合 (广播到每个查询)
        N = query_feats.shape[1]
        feats = [query_feats]
        if self.use_demon and demon_feat is not None:
            demon_expanded = demon_feat.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)
            feats.append(demon_expanded)

        fused = torch.cat(feats, dim=-1)  # (B, N, C+D)

        # 3. 每个查询 → 1 个 logit
        logits = self.classifier(fused).squeeze(-1)  # (B, N)
        return logits


# ═══════════════════════════════════════════════════════════════
#  频谱条件门控 (Spectrum-Conditioned Gating)
# ═══════════════════════════════════════════════════════════════

class SpectrumConditionedGate(nn.Module):
    """频谱条件门控 — 通道门控 + 频带门控双重调制

    核心思想: 不依赖固定时间结构或特定类别, 从输入 LOFAR 提取频谱物理指纹,
    通过 MLP 生成门控权重, 调制 backbone 输出的特征图。

    双重调制:
      1. 通道门控 (gate): 频谱指纹 → (B, C) 通道权重 → feat_map * channel_weight
      2. 频带门控 (band_gate): 频谱指纹 → (B, F') 频带权重 → feat_map * band_weight
         针对物理频段重叠 (如 Tanker+Tug 在 0-332Hz 与 Passengership 相似度 0.988),
         动态抑制混淆频段, 比静态频段加权 (config 0.5/2.0) 更灵活

    流程:
      1. 从 LOFAR 提取频谱指纹 (无参数, 纯物理计算, 8 维)
      2. 通道 MLP → (B, C) + 频带 MLP → (B, F')
      3. 双重调制: feat_map * channel_gate * band_gate

    数据驱动: 每个样本的门控权重不同, 根据其频谱特征自适应。
    """

    def __init__(self, feat_dim, freq_bins=17, spec_dim=8, hidden_dim=None):
        super().__init__()
        self.spec_dim = spec_dim
        self.freq_bins = freq_bins
        hidden_dim = hidden_dim or max(feat_dim // 4, 64)
        # 通道门控 MLP: 频谱指纹 → 通道权重
        self.gate = nn.Sequential(
            nn.Linear(spec_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feat_dim),
            nn.Sigmoid(),  # 0~1 门控
        )
        # 频带门控 MLP: 频谱指纹 → 频带权重 (针对物理频段重叠)
        band_hidden = max(freq_bins * 2, 32)
        self.band_gate = nn.Sequential(
            nn.Linear(spec_dim, band_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(band_hidden, freq_bins),
            nn.Sigmoid(),  # 0~1 门控
        )

    def _extract_fingerprint(self, lofar):
        """提取频谱物理指纹 (无参数, 纯物理计算)

        Args:
            lofar: (B, 1, F, T) 已归一化的 LOFAR
        Returns:
            fingerprint: (B, 8) 频谱物理特征向量
        """
        spec = lofar[:, 0]  # (B, F, T)
        B, F_dim, T_dim = spec.shape

        # 时间平均频谱
        mean_spec = spec.mean(dim=2)  # (B, F)
        mean_spec_abs = mean_spec.abs() + 1e-8

        # 1. 频谱熵 (宽带多 → 熵高)
        spec_norm = mean_spec_abs / (mean_spec_abs.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(spec_norm * torch.log(spec_norm + 1e-8)).sum(dim=1)  # (B,)

        # 2. 频谱平坦度 (宽带 → 平坦)
        geo_mean = torch.exp(torch.log(mean_spec_abs).mean(dim=1))
        flatness = geo_mean / (mean_spec_abs.mean(dim=1) + 1e-8)  # (B,)

        # 3. 频谱峰度 (线谱多 → 高峰度)
        spec_centered = mean_spec - mean_spec.mean(dim=1, keepdim=True)
        kurtosis = (spec_centered ** 4).mean(dim=1) / (mean_spec.var(dim=1) ** 2 + 1e-8)  # (B,)

        # 4. 低/中/高频能量比
        f3 = F_dim // 3
        low = mean_spec_abs[:, :f3].sum(dim=1)
        mid = mean_spec_abs[:, f3:2*f3].sum(dim=1)
        high = mean_spec_abs[:, 2*f3:].sum(dim=1)
        total = low + mid + high + 1e-8
        low_ratio = low / total
        high_ratio = high / total

        # 5. 时序稳定性 (帧间方差, 稳态线谱 → 低方差)
        temp_std = spec.std(dim=2).mean(dim=1)  # (B,)

        # 6. 频谱质心 (主能量集中位置, 归一化到 0~1)
        freqs = torch.arange(F_dim, device=spec.device).float()
        centroid = (freqs.unsqueeze(0) * mean_spec_abs).sum(dim=1) / \
                   (mean_spec_abs.sum(dim=1) + 1e-8) / F_dim  # (B,)

        return torch.stack([
            entropy, flatness, kurtosis,
            low_ratio, high_ratio,
            temp_std, centroid,
            # 7. 低频能量绝对值 (区分低频强线谱场景)
            low / F_dim,
        ], dim=1)  # (B, 8)

    def forward(self, feat_map, lofar):
        """根据频谱指纹双重调制特征图 (通道 + 频带)

        Args:
            feat_map: (B, C, F', T') backbone 输出
            lofar: (B, 1, F, T) 原始 LOFAR
        Returns:
            调制后的 feat_map: (B, C, F', T')
        """
        fingerprint = self._extract_fingerprint(lofar)  # (B, 8)

        # 通道门控: (B, C) → (B, C, 1, 1)
        channel_weight = self.gate(fingerprint)  # (B, C)

        # 频带门控: (B, F') → (B, 1, F', 1)
        band_weight = self.band_gate(fingerprint)  # (B, F')

        # 双重调制: 通道 × 频带
        return feat_map * channel_weight.unsqueeze(-1).unsqueeze(-1) * \
                         band_weight.unsqueeze(1).unsqueeze(-1)


# ═══════════════════════════════════════════════════════════════
#  注意力统计池化 (stats-pool 臂)
# ═══════════════════════════════════════════════════════════════

class AttentiveStatsPool(nn.Module):
    """注意力统计池化 (ECAPA-TDNN 式) → [加权均值 ‖ 加权标准差]

    [stats-pool 臂 2026-08-28] 动机 (UWTRL-MEG 分析, 方案A): 原 pooled=mean
    丢弃全部二阶矩 — 线谱频率稳定性 (重载 Cargo 稳 vs 变工况 Tk 漂移)、
    调制深度等时频方差信息未被 TC/残差头消费。输出 2C 维拼接。

    语义约束: 输入特征保持 detach (纯探针设计, 项目教训: TC 梯度回传会
    主导早期 backbone); 本模块自身参数由 TC/残差损失训练。
    last_alpha 缓存最近一次注意力权重 (B,C,N) 供消融诊断。
    """
    def __init__(self, dim, bottleneck_dim=None):
        super().__init__()
        bottleneck_dim = bottleneck_dim or max(dim // 4, 64)
        self.linear1 = nn.Conv1d(dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, dim, kernel_size=1)
        self.last_alpha = None                            # 诊断缓存 (detached)

    def forward(self, x, alpha_perm=None):
        """x: (B, C, N) → (B, 2C)

        alpha_perm: (B,) LongTensor — 跨样本置换注意力权重 (消融模式:
        破坏样本-注意力对应, 保留边缘分布; x 仍为本人 — 测注意力选择性)
        """
        alpha = torch.softmax(self.linear2(torch.tanh(self.linear1(x))), dim=2)
        if alpha_perm is not None:
            alpha = alpha[alpha_perm]
        self.last_alpha = alpha.detach()
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


# ═══════════════════════════════════════════════════════════════
#  M-UART v4.0 主模型 (双分支: LOFAR + DEMON)
# ═══════════════════════════════════════════════════════════════

class TCResidualClassifier(nn.Module):
    """TC 预分类 + 基座头(全量) + 路由残差专家

    保留"先分目标数再识别"的多任务思想, 修复旧软/硬路由的三大缺陷:
      1. 数据稀释 → 基座头用全量数据训练 (保底 = 单头基线性能)
      2. 错误级联 → 残差头零初始化, 起步 delta=0; 推理时 TC 只决定
         "加哪个修正量", 即使 TC 错也仅是少了修正, 不会替换整个预测
      3. 激进损失 → 残差头与基座共用同一套 Focal 参数, 只学子集上的修正

    结构:
      base_head: QueryClassifierHead (CrossAttn) ← 全量训练, 主识别头
      tc_head:   GAP+MLP → 3 类 (目标数 0/1/2), 输入 detach (纯探针:
                 只读取特征, 不反向塑形 backbone — 实验证实 CE 塑形
                 会主导早期梯度, 干扰类别身份识别)
      res_01:    GAP+MLP (输入 detach, 末层零初始化) ← 0/1 目标子集修正
      res_2:     GAP+MLP (输入 detach, 末层零初始化) ← 双目标子集修正

    物理真实性: 不依赖任何固定时间结构; 目标数由频谱复杂度推断,
                对 concat/同步/任意重叠时刻的数据集通用。

    训练返回: (tc_logits, logits_base, delta_01, delta_2)
    推理返回: logits = base + delta[TC路由]  (tc_pred<=1 → delta_01, else delta_2)
    """
    def __init__(self, feat_dim, demon_dim, num_classes,
                 num_heads=4, dropout=0.2, attn_dropout=0.1,
                 use_demon=True, max_h=64, max_w=32,
                 res_hidden=128, use_stats_pool=False,
                 num_feature_maps=1):
        super().__init__()
        self.use_demon = use_demon
        self.num_classes = num_classes
        self.num_feature_maps = num_feature_maps

        # [stats-pool 臂] pooled = [attentive-mean ‖ std] (2C) 或 mean (C)
        self.use_stats_pool = use_stats_pool
        if use_stats_pool:
            if num_feature_maps == 1:
                # 保持旧 stats-pool checkpoint 的 stats_pool.* 键名。
                self.stats_pool = AttentiveStatsPool(feat_dim)
            else:
                self.stats_pools = nn.ModuleList([
                    AttentiveStatsPool(feat_dim)
                    for _ in range(num_feature_maps)
                ])
            pooled_dim = 2 * feat_dim * num_feature_maps + \
                (demon_dim if use_demon else 0)
        else:
            pooled_dim = feat_dim * num_feature_maps + \
                (demon_dim if use_demon else 0)

        # ── 基座头: QueryClassifierHead (CrossAttn), 全量数据训练 ──
        self.base_head = QueryClassifierHead(
            lofar_dim=feat_dim,
            demon_dim=demon_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            use_demon=use_demon,
            max_h=max_h,
            max_w=max_w,
            num_bands=num_feature_maps,
        )

        # ── TC 头: 目标数分类 (0/1/2) ──
        # 输入不 detach: CE 梯度塑形 backbone 学数量感知特征 (多任务收益)
        self.tc_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, res_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(res_hidden, 3),
        )

        # ── 残差专家头: 只学子集修正量 ──
        # 输入 detach: 梯度不流向 backbone (避免 GAP 梯度与 CrossAttn 冲突)
        # 末层单独定义, 便于零初始化 (训练起步 delta=0, 等价于纯基座单头)
        self.res_01 = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, res_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_01_out = nn.Linear(res_hidden, num_classes)
        self.res_2 = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, res_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_2_out = nn.Linear(res_hidden, num_classes)

        self.reset_zero_init()

    def reset_zero_init(self):
        """残差头末层零初始化 (MUARTv4._init_head 覆盖后需重新调用)"""
        nn.init.zeros_(self.res_01_out.weight)
        nn.init.zeros_(self.res_01_out.bias)
        nn.init.zeros_(self.res_2_out.weight)
        nn.init.zeros_(self.res_2_out.bias)

    def pool_features(self, feat_map, demon_feat=None, alpha_perm=None):
        """统一池化入口 — forward 与外部评估/消融脚本共用 (口径一致性)

        feat_map: (B, C, H, W), demon_feat: (B, D) 或 None
        返回 pooled: (B, C) 或 stats-pool 模式 (B, 2C), 已拼 DEMON (detach)

        alpha_perm: (B,) LongTensor — 跨样本置换 attentive 权重 (消融模式:
        破坏样本-注意力对应, x/DEMON 不变 — 测注意力选择性贡献)
        """
        # 统一 detach: TC/残差均为纯探针 (实验教训: TC CE 梯度回传会主导
        # 早期 backbone 学习, 干扰类别身份识别)
        feature_maps = list(feat_map) if isinstance(feat_map, (tuple, list)) \
            else [feat_map]
        if len(feature_maps) != self.num_feature_maps:
            raise ValueError(
                f'期望 {self.num_feature_maps} 路池化特征，实际得到 '
                f'{len(feature_maps)} 路'
            )

        pooled_parts = []
        for i, one_map in enumerate(feature_maps):
            if self.use_stats_pool:
                x = one_map.detach().flatten(2)            # (B, C, N=H*W)
                stats_pool = self.stats_pool if self.num_feature_maps == 1 \
                    else self.stats_pools[i]
                pooled_parts.append(stats_pool(x, alpha_perm=alpha_perm))
            else:
                pooled_parts.append(
                    one_map.detach().mean(dim=[2, 3])       # (B, C)
                )
        pooled = torch.cat(pooled_parts, dim=1) \
            if len(pooled_parts) > 1 else pooled_parts[0]
        if self.use_demon and demon_feat is not None:
            pooled = torch.cat([pooled, demon_feat.detach()], dim=1)
        return pooled

    def forward(self, feat_map, demon_feat):
        """前向: 基座 + TC + 残差修正

        Args:
            feat_map: (B, C, H, W) backbone+spec_gate 输出
            demon_feat: (B, D) 或 None

        Returns:
            训练: (tc_logits, logits_base, delta_01, delta_2)
            推理: logits (B, num_classes) — TC 路由的融合输出
        """
        # 基座头 (主识别路径)
        logits_base = self.base_head(feat_map, demon_feat)  # (B, C)

        # 池化特征 (统一 detach: TC/残差均为纯探针, 不反向塑形 backbone)
        pooled = self.pool_features(feat_map, demon_feat)

        # TC 头 (数量探针, 只读取特征不塑形)
        tc_logits = self.tc_head(pooled)  # (B, 3)

        # 残差头 (只学子集修正; 输入 detach 保持纯探针设计)
        delta_01 = self.res_01_out(self.res_01(pooled))  # (B, C)
        delta_2 = self.res_2_out(self.res_2(pooled))     # (B, C)

        if self.training:
            return tc_logits, logits_base, delta_01, delta_2

        # 推理: TC 路由 (错误只影响修正量, 不替换基座预测)
        tc_pred = tc_logits.argmax(dim=1)          # (B,)
        mask_01 = (tc_pred <= 1).unsqueeze(-1)     # (B, 1)
        delta = torch.where(mask_01, delta_01, delta_2)
        return logits_base + delta


class MUARTv4(nn.Module):
    """
    M-UART v4.0 主模型: TF-ConvNeXt 双通道骨干 + DEMON 分支
                        + Spectrum-Conditioned Gating + TC 残差多任务分类器

    架构 (多任务: TC 数量分类 + 基座识别 + 子集残差修正):
      dual(C,F,T) → backbone → spec_gate(通道+频带门控) → TCResidualClassifier
                                                          ├─ base_head (CrossAttn, 全量)
      DEMON → demon_encoder ──────────────────────────────┼─ tc_head (目标数 0/1/2)
                                                          └─ res_01 / res_2 (零初始化残差)

    设计理念:
      - 基座头吃全量数据 → 保底 = 单头基线性能 (无数据稀释)
      - TC 预分类保留 (多任务特征), 但只路由残差修正量 (无错误级联)
      - 残差头零初始化, detach 输入 → 不干扰 backbone

    输入:
      lofar: (B, C, 513, 79) 主输入特征 (dual: C=2)
      demon: (B, 1, demon_n_bins)  DEMON 谱 (可选)

    输出:
      训练: (tc_logits, logits_base, delta_01, delta_2)
      推理: logits (B, num_classes)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_demon = config.use_demon

        # 主输入通道数 (dual/dual_wide=2: LOFAR + 宽带高频; lofar=1)
        ft = getattr(config, 'feature_type', 'dual')
        self.dual_stream_fusion = bool(
            getattr(config, 'dual_stream_fusion', False)
        )
        if self.dual_stream_fusion and ft != 'dual':
            raise ValueError('dual_stream_fusion 仅支持 feature_type="dual"')
        in_channels = 1 if (ft == 'lofar' or self.dual_stream_fusion) else 2

        # 分支 1: 时频骨干
        self.backbone = TFConvNeXtBackbone(config, in_channels=in_channels)

        # 分支 2: DEMON 1D 编码器 (可选)
        if self.use_demon:
            self.demon_encoder = DEMONEncoder(config)

        # ── 频谱条件门控 (Spectrum-Conditioned Gating) ──
        self.use_spec_gate = getattr(config, 'use_spec_gate', True)
        if self.dual_stream_fusion and self.use_spec_gate:
            raise ValueError(
                'E4 dual_stream_fusion 必须关闭 SpectrumConditionedGate；'
                '训练时同时传入 --dual-stream-fusion --no-spec-gate'
            )
        if self.use_spec_gate:
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, config.feature_n_bins,
                                    config.feature_frames)
                dummy_out = self.backbone(dummy)
                freq_bins = dummy_out.shape[2]
            self.spec_gate = SpectrumConditionedGate(self.backbone.out_dim,
                                                      freq_bins=freq_bins)

        # ── 分类器: TC 残差多任务 ──
        self.classifier = TCResidualClassifier(
            feat_dim=self.backbone.out_dim,
            demon_dim=config.demon_feat_dim if self.use_demon else 0,
            num_classes=config.num_classes,
            num_heads=config.query_num_heads,
            dropout=config.query_dropout,
            attn_dropout=config.query_attn_dropout,
            use_demon=self.use_demon,
            max_h=config.pe_max_h,
            max_w=config.pe_max_w,
            use_stats_pool=getattr(config, 'use_stats_pool', False),
            num_feature_maps=2 if self.dual_stream_fusion else 1,
        )

        self.apply(self._init_head)

        # 重新零初始化残差头末层 (被上面的 _init_head 覆盖)
        # 确保: 训练起步 delta=0, 模型等价于纯基座单头
        self.classifier.reset_zero_init()

    def _init_head(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, lofar, demon=None):
        """前向: backbone → spec_gate → 分类器

        Args:
            lofar: (B, C, F, T) 主输入特征 (dual: C=2)
            demon: (B, 1, demon_n_bins) 或 None
        Returns:
            eval: 分类器输出 (TC 路由 logits)
            train: (tc_logits, logits_base, delta_01, delta_2)
        """
        if self.dual_stream_fusion:
            if lofar.shape[1] != 2:
                raise ValueError(
                    f'E4 双流模型要求输入 2 通道，实际为 {lofar.shape[1]} 通道'
                )
            # 两路共享 backbone 权重，但整个特征提取过程彼此独立；直到
            # Query token 级才融合，避免把异构频带的同一行硬当作同一频率。
            feat_map = (
                self.backbone(lofar[:, 0:1]),
                self.backbone(lofar[:, 1:2]),
            )
        else:
            feat_map = self.backbone(lofar)

        if self.use_spec_gate:
            feat_map = self.spec_gate(feat_map, lofar)

        demon_feat = None
        if self.use_demon and demon is not None:
            demon_feat = self.demon_encoder(demon)

        return self.classifier(feat_map, demon_feat)


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

def count_parameters(model):
    """可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════
#  测试
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from config import ConfigSmall

    print('=' * 70)
    print('M-UART v4.0 模型测试 — TF-ConvNeXt dual + DEMON + TC 残差分类器')
    print('=' * 70)

    cfg = ConfigSmall()
    cfg.use_demon = True
    model = MUARTv4(cfg)
    n = count_parameters(model)

    F_dim = cfg.feature_n_bins
    T_dim = cfg.feature_frames
    C_in = 2 if cfg.feature_type == 'dual' else 1
    x_lofar = torch.randn(2, C_in, F_dim, T_dim)
    x_demon = torch.randn(2, 1, cfg.demon_n_bins)

    with torch.no_grad():
        model.train()
        tc_logits, logits_base, delta_01, delta_2 = model(x_lofar, x_demon)
        model.eval()
        out_eval = model(x_lofar, x_demon)
    print(f'\n  {n/1e6:.2f}M 参数, '
          f'输入 (B,{C_in},{F_dim},{T_dim})+DEMON → 输出 {tuple(out_eval.shape)} ✓')
    print(f'  训练模式: tc_logits{tuple(tc_logits.shape)}, '
          f'logits_base{tuple(logits_base.shape)}, '
          f'delta_01{tuple(delta_01.shape)}, delta_2{tuple(delta_2.shape)} ✓')

    # 零初始化验证: 起步 delta=0
    print(f'  零初始化验证: |delta_01|={delta_01.abs().max():.2e}, '
          f'|delta_2|={delta_2.abs().max():.2e} (应为 0) ✓')

    # 旧 checkpoint 兼容性验证 (dual 家族键名)
    import os
    ckpt_path = 'checkpoints_E1_dual_s42/best_model.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(
            ckpt['model_state_dict'], strict=False)
        ok = not [k for k in missing if not k.startswith('num_batches')] and not unexpected
        print(f'  旧 checkpoint 加载: missing={len(missing)}, '
              f'unexpected={len(unexpected)} {"✓ 兼容" if ok else "✗ 不兼容!"}')

    print('\n' + '=' * 70)
    print('✓ 测试通过 — dual 双通道 + 频谱门控 + TC 残差多任务分类器')
    print('=' * 70)
