# SPDX-License-Identifier: Apache-2.0
"""Target-platform defaults without changing baseline serve arguments."""

_PRESETS = {
    "metax": {
        "ffn_w8a8": True,
        "ffn_w8a8_interleaved": True,
        "linear_swiglu_fusion": True,
        "ffn_down_w8a8": True,
        "attention_o_w8a8": True,
        "qkv_w8a8": True,
        "tiled_sampling": True,
        "prefill_tail": True,
    },
    "iluvatar": {
        "ffn_w8a8": True,
        "ffn_w8a8_interleaved": False,
        "linear_swiglu_fusion": False,
        "ffn_down_w8a8": False,
        "attention_o_w8a8": False,
        "qkv_w8a8": True,
        "tiled_sampling": True,
        "prefill_tail": True,
        "bf16_linear_tuning": True,
        "triton_attention_tuning": True,
    },
}


def prepare_optimizations(config, vendor):
    """Apply target-platform defaults while retaining explicit feature overrides.

    The submission works with the original serve command. ``disabled`` stops
    default expansion for development; it does not revert backend/kernel fixes.
    """
    extra = config.additional_config
    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        return
    if vendor not in _PRESETS:
        return
    profile = extra.get("optimization_profile", "competition")
    if profile in (None, "disabled"):
        return
    if profile != "competition":
        raise ValueError("optimization_profile must be 'competition' or 'disabled'")
    settings = dict(extra)
    defaults = dict(_PRESETS.get(vendor, {}))
    if settings.get("ffn_w8a8") is False:
        # An explicit opt-out must not inherit a dependent layout/fusion opt-in.
        # Explicitly conflicting settings remain visible to installer validation.
        defaults["ffn_w8a8_interleaved"] = False
        defaults["linear_swiglu_fusion"] = False
    for name, value in defaults.items():
        settings.setdefault(name, value)
    config.additional_config = settings


def feature_enabled(config, name):
    extra = config.additional_config
    value = extra.get(name, False) if isinstance(extra, dict) else False
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value
