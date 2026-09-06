"""V6-A2: restore shared identity evidence; same Train and s42 budget as A1."""
from config_v6a import ConfigV6A, validate_v6a_config


class ConfigV6A2(ConfigV6A):
    experiment_stage = 'V6-A2'
    architecture_name = 'MUART-V6A2-SharedEvidence'
    v6a_loss = 'set_nll_plus_presence_bce'
    # Fixed engineering choices, not fitted to Val. Both heads use these same
    # presence logits; pair interaction starts at zero and stays bounded.
    presence_loss_weight = 0.5
    pair_interaction_hidden = 32
    pair_interaction_bound = 0.5
    monitor_per_set = 32
    monitor_seed = 6002
    # Preserve A1 early stopping / optimization. NLL alert is diagnostic only.
    monitor_nll_rise = 0.05
    monitor_nll_patience = 5


def validate_v6a2_config(config):
    validate_v6a_config(config)
    if config.presence_loss_weight != 0.5:
        raise ValueError('This A2 arm fixes presence_loss_weight=0.5')
    if config.pair_interaction_hidden != 32 or config.pair_interaction_bound != 0.5:
        raise ValueError('This A2 arm fixes interaction hidden=32, bound=0.5')
    if config.monitor_per_set < 1:
        raise ValueError('monitor_per_set must be positive')

