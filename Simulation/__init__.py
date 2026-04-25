"""Simulation helpers for the Steane-encoded Shor-15 workstream."""

from .shor_simulation import (
    DEFAULT_NOISE_RATE,
    SHOR_15_A,
    SHOR_15_N,
    STEANE_DISTANCE,
    SteaneColorCodeBlock,
    build_encoded_precompiled_shor_15_circuit,
    build_unencoded_precompiled_shor_15_circuit,
    decode_logical_observable_predictions,
    factor_candidates_from_phase_bit,
    generate_steane_code_instances,
    logical_block_summaries,
    run_precompiled_shor_15,
    sample_logical_observable,
    shor_15_success_explanation,
)

__all__ = [
    "DEFAULT_NOISE_RATE",
    "SHOR_15_A",
    "SHOR_15_N",
    "STEANE_DISTANCE",
    "SteaneColorCodeBlock",
    "build_encoded_precompiled_shor_15_circuit",
    "build_unencoded_precompiled_shor_15_circuit",
    "decode_logical_observable_predictions",
    "factor_candidates_from_phase_bit",
    "generate_steane_code_instances",
    "logical_block_summaries",
    "run_precompiled_shor_15",
    "sample_logical_observable",
    "shor_15_success_explanation",
]
