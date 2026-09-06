"""V6-A conditional legal-set classifier configuration.

This module deliberately inherits the frozen Cargo V5 data/feature contract and
only changes the classifier/training objective.  Test is never part of V6-A
model selection; ``train_v6a.py`` always constructs Train and Val only.
"""

from config import ConfigSmallCargo


class ConfigV6A(ConfigSmallCargo):
    """E8-matched shared encoder with the V6-A conditional set head."""

    experiment_stage = "V6-A"
    architecture_name = "MUART-V6A-ConditionalSet"
    checkpoint_schema_version = 1

    # Frozen E8 input/encoder arm.  These values are explicit so a later change
    # to ConfigSmallCargo cannot silently change the V6-A experiment.
    feature_type = "dual"
    dual_stream_fusion = False
    normalization_mode = "per_channel"
    use_demon = True
    demon_mode = "demon"
    use_spec_gate = False
    freq_shift_mode = "off"
    waveform_noise_mode = "off"

    # The seven legal sets in model class order [Tanker, Cargo, Tug].
    legal_set_names = [
        "noise",
        "Tanker",
        "Cargo",
        "Tug",
        "Tanker+Cargo",
        "Tanker+Tug",
        "Cargo+Tug",
    ]
    num_legal_sets = 7

    # Conditional head.  There are no residual experts and no distillation.
    v6a_tc_hidden = 128
    v6a_pair_hidden = 256
    v6a_use_distillation = False
    v6a_loss = "joint_legal_set_nll"

    # Keep the existing semantic-query regularizer as part of the frozen E8
    # representation recipe.  This is not a prototype/contrastive loss.
    ortho_loss_weight = 0.1

    # Frozen comparison budget.
    batch_size = 32
    num_epochs = 60
    early_stop_patience = 20
    val_only = True


def validate_v6a_config(config):
    """Fail closed if a command accidentally drifts outside the V6-A arm."""

    errors = []
    if list(config.class_names) != ["Tanker", "Cargo", "Tug"]:
        errors.append("class_names must be [Tanker, Cargo, Tug]")
    if int(config.num_classes) != 3:
        errors.append("num_classes must be 3")
    if list(config.multitarget_types) != [
        "noise", "0", "1", "2", "0_1", "0_2", "1_2"
    ]:
        errors.append("dataset must contain exactly the seven frozen legal types")
    if getattr(config, "feature_type", None) != "dual":
        errors.append("feature_type must remain dual")
    if getattr(config, "dual_stream_fusion", None) is not False:
        errors.append("dual_stream_fusion must remain False")
    if getattr(config, "normalization_mode", None) != "per_channel":
        errors.append("normalization_mode must remain per_channel")
    if getattr(config, "use_demon", None) is not True:
        errors.append("DEMON must remain enabled")
    if getattr(config, "use_spec_gate", None) is not False:
        errors.append("spectrum gate must remain disabled")
    if getattr(config, "freq_shift_mode", None) != "off":
        errors.append("legacy FFT-roll frequency shift must remain disabled")
    if getattr(config, "waveform_noise_mode", None) != "off":
        errors.append("extra white-noise augmentation must remain disabled")
    if getattr(config, "v6a_use_distillation", None) is not False:
        errors.append("distillation is forbidden in V6-A")
    if getattr(config, "val_only", None) is not True:
        errors.append("V6-A model selection must be Train/Val only")
    if int(getattr(config, "num_legal_sets", -1)) != 7:
        errors.append("num_legal_sets must be 7")
    if errors:
        raise ValueError("V6-A configuration rejected: " + "; ".join(errors))

