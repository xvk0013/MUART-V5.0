"""MUART V6-A: conditional classification over the seven legal target sets.

The TF-ConvNeXt and DEMON encoders are reused from V5.  The old BCE base head
and routed residual experts are replaced by three heads trained through one
normalized legal-set likelihood:

    target count:       noise / single / double
    single identity:    Tanker / Cargo / Tug
    double identity:    Tanker+Cargo / Tanker+Tug / Cargo+Tug

For example P(Tanker+Cargo) = P(double) * P(Tanker+Cargo | double).
Consequently every prediction is one of the seven legal sets and the seven
probabilities sum to one without threshold search.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    DEMONEncoder,
    SemanticQueryAttention,
    SpectrumConditionedGate,
    TFConvNeXtBackbone,
)


# Set order is part of the checkpoint/evaluation contract.
LEGAL_SET_NAMES: Tuple[str, ...] = (
    "noise",
    "Tanker",
    "Cargo",
    "Tug",
    "Tanker+Cargo",
    "Tanker+Tug",
    "Cargo+Tug",
)

LEGAL_SET_MULTI_HOT = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=torch.float32,
)

PAIR_CLASS_INDICES: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))


def multihot_to_set_targets(labels: torch.Tensor) -> torch.Tensor:
    """Convert legal [Tanker, Cargo, Tug] multi-hot rows to set indices.

    Illegal soft, negative, or three-target rows are rejected rather than
    silently mapped into a legal class.
    """

    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError(f"labels must have shape (B, 3), got {tuple(labels.shape)}")
    if not torch.isfinite(labels).all():
        raise ValueError("labels contain NaN or Inf")
    binary = (labels == 0) | (labels == 1)
    if not bool(binary.all()):
        raise ValueError("labels must be exact binary multi-hot values")
    counts = labels.sum(dim=1)
    if bool((counts > 2).any()):
        raise ValueError("V6-A rejects triple-target labels")

    # bit code in class order: Tanker=1, Cargo=2, Tug=4.
    bit_weights = labels.new_tensor([1, 2, 4])
    codes = (labels * bit_weights).sum(dim=1).long()
    lookup = torch.full((8,), -1, dtype=torch.long, device=labels.device)
    lookup[labels.new_tensor([0, 1, 2, 4, 3, 5, 6], dtype=torch.long)] = \
        labels.new_tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)
    targets = lookup[codes]
    if bool((targets < 0).any()):
        raise ValueError("labels contain an unsupported target set")
    return targets


def set_targets_to_multihot(targets: torch.Tensor) -> torch.Tensor:
    """Convert legal-set indices to [Tanker, Cargo, Tug] multi-hot rows."""

    if targets.ndim != 1:
        raise ValueError(f"targets must have shape (B,), got {tuple(targets.shape)}")
    if targets.numel() and (bool((targets < 0).any()) or bool((targets >= 7).any())):
        raise ValueError("legal-set target index must be in [0, 6]")
    table = LEGAL_SET_MULTI_HOT.to(device=targets.device)
    return table[targets.long()]


def conditional_joint_log_probs(
    tc_logits: torch.Tensor,
    single_logits: torch.Tensor,
    pair_logits: torch.Tensor,
) -> torch.Tensor:
    """Compose normalized log probabilities for all seven legal sets."""

    if tc_logits.ndim != 2 or tc_logits.shape[1] != 3:
        raise ValueError("tc_logits must have shape (B, 3)")
    if single_logits.shape != tc_logits.shape:
        raise ValueError("single_logits must have shape (B, 3)")
    if pair_logits.shape != tc_logits.shape:
        raise ValueError("pair_logits must have shape (B, 3)")

    log_count = F.log_softmax(tc_logits, dim=1)
    log_single = F.log_softmax(single_logits, dim=1)
    log_pair = F.log_softmax(pair_logits, dim=1)
    return torch.cat(
        [
            log_count[:, 0:1],
            log_count[:, 1:2] + log_single,
            log_count[:, 2:3] + log_pair,
        ],
        dim=1,
    )


class ConditionalSetHead(nn.Module):
    """Shared semantic representation with count/single/pair conditionals."""

    def __init__(
        self,
        feat_dim: int,
        demon_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        attn_dropout: float = 0.1,
        use_demon: bool = True,
        max_h: int = 64,
        max_w: int = 32,
        tc_hidden: int = 128,
        pair_hidden: int = 256,
        num_feature_maps: int = 1,
    ) -> None:
        super().__init__()
        self.use_demon = use_demon
        self.num_feature_maps = num_feature_maps
        self.query_attn = SemanticQueryAttention(
            dim=feat_dim,
            num_queries=3,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            max_h=max_h,
            max_w=max_w,
            num_bands=num_feature_maps,
        )

        query_dim = feat_dim + (demon_dim if use_demon else 0)
        pooled_dim = feat_dim * num_feature_maps + \
            (demon_dim if use_demon else 0)

        # One shared scorer applied to the three class-specific query tokens.
        self.single_scorer = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim, 1),
        )

        # A symmetric pair representation [sum, |difference|, product] keeps
        # pair scores independent of the arbitrary order of the two classes.
        self.pair_scorer = nn.Sequential(
            nn.LayerNorm(3 * query_dim),
            nn.Linear(3 * query_dim, pair_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden, 1),
        )

        self.tc_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, tc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tc_hidden, 3),
        )

    def _pool_feature_maps(self, feat_map) -> torch.Tensor:
        maps = list(feat_map) if isinstance(feat_map, (tuple, list)) else [feat_map]
        if len(maps) != self.num_feature_maps:
            raise ValueError(
                f"expected {self.num_feature_maps} feature maps, got {len(maps)}"
            )
        return torch.cat([one.mean(dim=(2, 3)) for one in maps], dim=1)

    def forward(self, feat_map, demon_feat=None) -> Dict[str, torch.Tensor]:
        query_features = self.query_attn(feat_map)  # (B, 3, C)
        pooled = self._pool_feature_maps(feat_map)

        if self.use_demon:
            if demon_feat is None:
                raise ValueError("DEMON is enabled but demon input/features are missing")
            expanded = demon_feat.unsqueeze(1).expand(-1, 3, -1)
            query_features = torch.cat([query_features, expanded], dim=2)
            pooled = torch.cat([pooled, demon_feat], dim=1)

        tc_logits = self.tc_head(pooled)
        single_logits = self.single_scorer(query_features).squeeze(-1)

        pair_features = []
        for left, right in PAIR_CLASS_INDICES:
            a = query_features[:, left]
            b = query_features[:, right]
            pair_features.append(torch.cat([a + b, (a - b).abs(), a * b], dim=1))
        pair_tensor = torch.stack(pair_features, dim=1)
        pair_logits = self.pair_scorer(pair_tensor).squeeze(-1)

        set_log_probs = conditional_joint_log_probs(
            tc_logits, single_logits, pair_logits
        )
        return {
            "tc_logits": tc_logits,
            "single_logits": single_logits,
            "pair_logits": pair_logits,
            "set_log_probs": set_log_probs,
            "query_features": query_features,
        }


class MUARTV6A(nn.Module):
    """E8 shared encoder plus the V6-A conditional legal-set head."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.use_demon = bool(config.use_demon)

        feature_type = getattr(config, "feature_type", "dual")
        self.dual_stream_fusion = bool(
            getattr(config, "dual_stream_fusion", False)
        )
        if self.dual_stream_fusion and feature_type != "dual":
            raise ValueError("dual_stream_fusion only supports feature_type='dual'")
        in_channels = 1 if (
            feature_type == "lofar" or self.dual_stream_fusion
        ) else 2

        self.backbone = TFConvNeXtBackbone(config, in_channels=in_channels)
        if self.use_demon:
            self.demon_encoder = DEMONEncoder(config)

        self.use_spec_gate = bool(getattr(config, "use_spec_gate", False))
        if self.dual_stream_fusion and self.use_spec_gate:
            raise ValueError("dual-stream fusion and spectrum gate cannot be combined")
        if self.use_spec_gate:
            with torch.no_grad():
                dummy = torch.zeros(
                    1, in_channels, config.feature_n_bins, config.feature_frames
                )
                freq_bins = self.backbone(dummy).shape[2]
            self.spec_gate = SpectrumConditionedGate(
                self.backbone.out_dim, freq_bins=freq_bins
            )

        self.classifier = ConditionalSetHead(
            feat_dim=self.backbone.out_dim,
            demon_dim=config.demon_feat_dim if self.use_demon else 0,
            num_heads=config.query_num_heads,
            dropout=config.query_dropout,
            attn_dropout=config.query_attn_dropout,
            use_demon=self.use_demon,
            max_h=config.pe_max_h,
            max_w=config.pe_max_w,
            tc_hidden=getattr(config, "v6a_tc_hidden", 128),
            pair_hidden=getattr(config, "v6a_pair_hidden", 256),
            num_feature_maps=2 if self.dual_stream_fusion else 1,
        )
        self.classifier.apply(self._init_linear)

        self.register_buffer(
            "legal_set_multihot",
            LEGAL_SET_MULTI_HOT.clone(),
            persistent=True,
        )

    @staticmethod
    def _init_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, lofar: torch.Tensor, demon=None) -> Dict[str, torch.Tensor]:
        if self.dual_stream_fusion:
            if lofar.ndim != 4 or lofar.shape[1] != 2:
                raise ValueError("dual-stream V6-A expects a two-channel input")
            feat_map = (
                self.backbone(lofar[:, 0:1]),
                self.backbone(lofar[:, 1:2]),
            )
        else:
            feat_map = self.backbone(lofar)

        if self.use_spec_gate:
            feat_map = self.spec_gate(feat_map, lofar)

        demon_feat = None
        if self.use_demon:
            if demon is None:
                raise ValueError("V6-A configuration requires DEMON input")
            demon_feat = self.demon_encoder(demon)

        return self.classifier(feat_map, demon_feat)

    @torch.no_grad()
    def predict_set_indices(self, lofar: torch.Tensor, demon=None) -> torch.Tensor:
        return self(lofar, demon)["set_log_probs"].argmax(dim=1)

    @torch.no_grad()
    def predict_multihot(self, lofar: torch.Tensor, demon=None) -> torch.Tensor:
        indices = self.predict_set_indices(lofar, demon)
        return self.legal_set_multihot[indices]


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

