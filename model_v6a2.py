"""Shared single-class evidence across noise/single/double supervision.

presence_logits = e(x)                          [three shared class queries]
single_logits = e(x)
pair_logits(i,j) = e_i(x) + e_j(x) + 0.5*tanh(r_ij(x))

The interaction MLP is shared over the three symmetric pairs, hidden=32,
zero-initialized output. It cannot become an unconstrained independent pair
classifier. The auxiliary BCE supervises e on ALL mixtures, including noise.
"""
import torch
from torch import nn
from model_v6a import (
    MUARTV6A, ConditionalSetHead, PAIR_CLASS_INDICES,
    conditional_joint_log_probs,
)


class SharedEvidenceHead(ConditionalSetHead):
    def __init__(self, old_head, hidden=32, interaction_bound=0.5):
        nn.Module.__init__(self)
        # Keep the A1 shared-query/single/TC initialization, replacing only the
        # old pair MLP. No pretrained A1/E8 checkpoint is loaded.
        self.use_demon = old_head.use_demon
        self.num_feature_maps = old_head.num_feature_maps
        self.query_attn = old_head.query_attn
        self.single_scorer = old_head.single_scorer
        self.tc_head = old_head.tc_head
        self.interaction_bound = float(interaction_bound)
        dim = self.single_scorer[-1].in_features
        dropout = self.single_scorer[2].p
        self.pair_scorer = nn.Sequential(
            nn.LayerNorm(3 * dim), nn.Linear(3 * dim, hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self.pair_scorer.apply(MUARTV6A._init_linear)
        nn.init.zeros_(self.pair_scorer[-1].weight)
        nn.init.zeros_(self.pair_scorer[-1].bias)

    def forward(self, feat_map, demon_feat=None):
        query_features = self.query_attn(feat_map)
        pooled = self._pool_feature_maps(feat_map)
        if self.use_demon:
            if demon_feat is None:
                raise ValueError('DEMON input is required')
            query_features = torch.cat(
                [query_features, demon_feat[:, None, :].expand(-1, 3, -1)], dim=2)
            pooled = torch.cat([pooled, demon_feat], dim=1)
        evidence = self.single_scorer(query_features).squeeze(-1)
        pair_features = []
        additive = []
        for i, j in PAIR_CLASS_INDICES:
            a, b = query_features[:, i], query_features[:, j]
            pair_features.append(torch.cat([a+b, (a-b).abs(), a*b], dim=1))
            additive.append(evidence[:, i] + evidence[:, j])
        interaction = self.interaction_bound * torch.tanh(
            self.pair_scorer(torch.stack(pair_features, dim=1)).squeeze(-1))
        pair_logits = torch.stack(additive, dim=1) + interaction
        tc_logits = self.tc_head(pooled)
        return {
            'tc_logits': tc_logits, 'presence_logits': evidence,
            'single_logits': evidence, 'pair_logits': pair_logits,
            'pair_interaction': interaction,
            # Float32 composition even under mixed precision.
            'set_log_probs': conditional_joint_log_probs(
                tc_logits.float(), evidence.float(), pair_logits.float()),
        }


class MUARTV6A2(MUARTV6A):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = SharedEvidenceHead(
            self.classifier, config.pair_interaction_hidden,
            config.pair_interaction_bound)
