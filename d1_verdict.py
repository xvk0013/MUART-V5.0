"""Pure, pre-registered D1 frozen-Val decision rule."""

STRONG_RECOVERY_TARGETS = {
    "cargo_correct": 155,
    "single_cargo_tanker_swaps": 76,
    "tug_pair_cargo_tanker_swaps": 205,
}


def _confusion_value(metrics: dict, true_name: str, predicted_name: str) -> int:
    order = metrics["confusion_order"]
    return int(metrics["confusion_matrix"][order.index(true_name)][order.index(predicted_name)])


def d1_diagnostic_verdict(current: dict, previous: dict) -> dict:
    def summary(metrics: dict) -> dict:
        matrix = metrics["confusion_matrix"]
        return {
            "overall_correct": sum(int(matrix[i][i]) for i in range(len(matrix))),
            "cargo_correct": int(metrics["by_legal_set"]["Cargo"]["correct"]),
            "single_cargo_tanker_swaps": (
                _confusion_value(metrics, "Cargo", "Tanker") +
                _confusion_value(metrics, "Tanker", "Cargo")),
            "tug_pair_cargo_tanker_swaps": (
                _confusion_value(metrics, "Cargo+Tug", "Tanker+Tug") +
                _confusion_value(metrics, "Tanker+Tug", "Cargo+Tug")),
        }

    old, new = summary(previous), summary(current)
    delta = {key: new[key] - old[key] for key in old}
    diagnostic_success = (
        delta["cargo_correct"] > 0 and
        delta["single_cargo_tanker_swaps"] < 0 and
        delta["tug_pair_cargo_tanker_swaps"] < 0 and
        delta["overall_correct"] >= 0
    )
    strong_breakthrough = (
        new["cargo_correct"] >= STRONG_RECOVERY_TARGETS["cargo_correct"] and
        new["single_cargo_tanker_swaps"] <=
        STRONG_RECOVERY_TARGETS["single_cargo_tanker_swaps"] and
        new["tug_pair_cargo_tanker_swaps"] <=
        STRONG_RECOVERY_TARGETS["tug_pair_cargo_tanker_swaps"] and
        delta["overall_correct"] >= 0
    )
    if strong_breakthrough:
        label = "strong_breakthrough"
    elif diagnostic_success:
        label = "recording_bias_supported"
    elif delta["cargo_correct"] > 0:
        label = "partial_cargo_gain_with_tradeoff"
    else:
        label = "recording_bias_not_supported"
    return {"label": label, "strong_breakthrough": strong_breakthrough,
            "diagnostic_success": diagnostic_success, "v6a2": old, "d1": new,
            "delta_d1_minus_v6a2": delta,
            "strong_recovery_targets": STRONG_RECOVERY_TARGETS}
