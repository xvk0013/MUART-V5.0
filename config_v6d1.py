"""V6-D1: unchanged V6-A2 model trained on recording-uniform Train data."""

from config_v6a2 import ConfigV6A2, validate_v6a2_config


class ConfigV6D1(ConfigV6A2):
    experiment_stage = "V6-D1"
    # Architecture string deliberately stays identical: D1 is a data-only arm.
    architecture_name = ConfigV6A2.architecture_name
    data_ablation = "Train recording-uniform then template-uniform; frozen Val"


def validate_v6d1_config(config):
    validate_v6a2_config(config)
    if config.architecture_name != ConfigV6A2.architecture_name:
        raise ValueError("D1 must not change the V6-A2 architecture")
    if config.experiment_stage != "V6-D1":
        raise ValueError("D1 experiment_stage must be V6-D1")
