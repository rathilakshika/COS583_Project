"""Stim/color-code simulation helpers for the simplified Shor-15 experiment.

This module starts Workstream A from ``PROJECT_CONTEXT.md``.  It uses
``color_code_stim`` to instantiate three distance-3 triangular color-code
patches, which are the Steane-code blocks used to encode the three qubits in
the precompiled ``a = 11, N = 15`` Hadamard-test circuit.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from math import gcd
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import stim
from color_code_stim import ColorCode, NoiseModel
from color_code_stim.decoders.concat_matching_decoder import ConcatMatchingDecoder
from color_code_stim.dem_utils.dem_manager import DemManager

SHOR_15_A = 11
SHOR_15_N = 15
STEANE_DISTANCE = 3
STEANE_DATA_QUBITS = 7
DEFAULT_NOISE_RATE = 1e-3
IDLE_BLOCK_SYNDROME_POLICY = "idle_blocks"
NO_SYNDROME_POLICY = "none"

# Same Steane |0_L> preparation convention used in Compilation/shor_steane_encoding.py.
STEANE_ZERO_H_INDICES = (0, 5, 6)
STEANE_ZERO_CNOTS = (
    (5, 4),
    (0, 1),
    (6, 3),
    (5, 2),
    (6, 4),
    (0, 3),
    (5, 1),
    (3, 2),
)

__all__ = [
    "SHOR_15_A",
    "SHOR_15_N",
    "STEANE_DISTANCE",
    "STEANE_DATA_QUBITS",
    "DEFAULT_NOISE_RATE",
    "SteaneColorCodeBlock",
    "generate_steane_code_instances",
    "logical_block_summaries",
    "build_unencoded_precompiled_shor_15_circuit",
    "build_encoded_precompiled_shor_15_circuit",
    "decode_logical_observable_predictions",
    "sample_logical_observable",
    "factor_candidates_from_phase_bit",
    "shor_15_success_explanation",
    "run_precompiled_shor_15",
]

_COLOR_TO_VALUE = {"r": 0, "g": 1, "b": 2}
_PAULI_TO_VALUE = {"X": 0, "Z": 2}


@dataclass
class _MeasurementRecordTracker:
    """Track absolute measurement indices so DETECTOR rec offsets stay valid."""

    next_measurement_index: int = 0

    def record_many(self, keys: Sequence[tuple]) -> dict[tuple, int]:
        records = {}
        for key in keys:
            records[key] = self.next_measurement_index
            self.next_measurement_index += 1
        return records

    def rec_target(self, absolute_measurement_index: int):
        return stim.target_rec(absolute_measurement_index - self.next_measurement_index)


@dataclass(frozen=True)
class SteaneColorCodeBlock:
    """One logical qubit represented by a d=3 color-code/Steane patch."""

    logical_index: int
    code: ColorCode
    qid_offset: int
    label: str

    @property
    def patch_qubit_count(self) -> int:
        return int(self.code.tanner_graph.vcount())

    @property
    def data_qids(self) -> tuple[int, ...]:
        return tuple(int(qid) for qid in self.code.qubit_groups["data"]["qid"])

    @property
    def ancilla_qids(self) -> tuple[int, ...]:
        return tuple(int(qid) for qid in self.code.qubit_groups["anc"]["qid"])

    @property
    def observable_qids(self) -> tuple[int, ...]:
        return tuple(
            int(vertex["qid"])
            for vertex in self.code.qubit_groups["data"]
            if bool(vertex["obs"])
        )

    @property
    def global_data_qids(self) -> tuple[int, ...]:
        return tuple(self.qid_offset + qid for qid in self.data_qids)

    @property
    def global_ancilla_qids(self) -> tuple[int, ...]:
        return tuple(self.qid_offset + qid for qid in self.ancilla_qids)

    @property
    def global_qids(self) -> tuple[int, ...]:
        return tuple(self.qid_offset + qid for qid in range(self.patch_qubit_count))

    @property
    def global_z_ancilla_qids(self) -> tuple[int, ...]:
        return tuple(
            self.qid_offset + int(qid)
            for qid in self.code.qubit_groups["anc_Z"]["qid"]
        )

    @property
    def global_x_ancilla_qids(self) -> tuple[int, ...]:
        return tuple(
            self.qid_offset + int(qid)
            for qid in self.code.qubit_groups["anc_X"]["qid"]
        )

    @property
    def global_observable_qids(self) -> tuple[int, ...]:
        return tuple(self.qid_offset + qid for qid in self.observable_qids)

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "logical_index": self.logical_index,
            "distance": self.code.d,
            "circuit_type": self.code.circuit_type,
            "patch_qubits": self.patch_qubit_count,
            "data_qubits": len(self.data_qids),
            "ancilla_qubits": len(self.ancilla_qids),
            "local_data_qids": self.data_qids,
            "local_logical_observable_qids": self.observable_qids,
            "global_data_qids": self.global_data_qids,
            "global_logical_observable_qids": self.global_observable_qids,
        }


def generate_steane_code_instances(
    *,
    num_logical_qubits: int = 3,
    distance: int = STEANE_DISTANCE,
    rounds: int = 1,
    noise_rate: float = DEFAULT_NOISE_RATE,
    noise_model: NoiseModel | None = None,
    temp_bdry_type: str = "Z",
    generate_dem: bool = False,
) -> tuple[SteaneColorCodeBlock, ...]:
    """Generate distance-3 color-code instances for Steane logical qubits."""
    if distance != STEANE_DISTANCE:
        raise ValueError("This simulation track treats only d=3 as the Steane code.")
    if num_logical_qubits < 1:
        raise ValueError("num_logical_qubits must be at least 1.")
    if rounds < 1:
        raise ValueError("rounds must be at least 1.")
    if noise_rate < 0:
        raise ValueError("noise_rate must be non-negative.")

    resolved_noise_model = noise_model or NoiseModel.uniform_circuit_noise(noise_rate)
    blocks = []
    patch_qubit_count = None

    for logical_index in range(num_logical_qubits):
        code = ColorCode(
            d=distance,
            rounds=rounds,
            circuit_type="tri",
            temp_bdry_type=temp_bdry_type,
            noise_model=resolved_noise_model,
            _generate_dem=generate_dem,
            _decompose_dem=generate_dem,
        )
        if patch_qubit_count is None:
            patch_qubit_count = int(code.tanner_graph.vcount())

        blocks.append(
            SteaneColorCodeBlock(
                logical_index=logical_index,
                code=code,
                qid_offset=logical_index * patch_qubit_count,
                label=f"logical_{logical_index}",
            )
        )

    return tuple(blocks)


def logical_block_summaries(
    blocks: Sequence[SteaneColorCodeBlock],
) -> list[dict[str, Any]]:
    """Return notebook-friendly summaries of the generated logical blocks."""
    return [block.summary() for block in blocks]


def _validate_steane_data_block(data_qids: Sequence[int]) -> tuple[int, ...]:
    data = tuple(int(qid) for qid in data_qids)
    if len(data) != STEANE_DATA_QUBITS:
        raise ValueError(f"Expected {STEANE_DATA_QUBITS} Steane data qubits, got {len(data)}.")
    return data


def _noise_arg(noise_rate: float) -> float | None:
    return noise_rate if noise_rate > 0 else None


def _append_depolarize1(
    circuit: stim.Circuit,
    qids: Sequence[int],
    noise_rate: float,
) -> None:
    targets = tuple(int(qid) for qid in qids)
    if targets and noise_rate > 0:
        circuit.append("DEPOLARIZE1", targets, noise_rate)


def _append_depolarize2(
    circuit: stim.Circuit,
    cnot_targets: Sequence[int],
    noise_rate: float,
) -> None:
    targets = tuple(int(qid) for qid in cnot_targets)
    if targets and noise_rate > 0:
        circuit.append("DEPOLARIZE2", targets, noise_rate)


def _append_reset_error(
    circuit: stim.Circuit,
    instruction_name: str,
    qids: Sequence[int],
    noise_rate: float,
) -> None:
    targets = tuple(int(qid) for qid in qids)
    if targets and noise_rate > 0:
        circuit.append(instruction_name, targets, noise_rate)


def _append_h(
    circuit: stim.Circuit,
    qids: Sequence[int],
    noise_rate: float,
) -> None:
    targets = tuple(int(qid) for qid in qids)
    if not targets:
        return
    circuit.append("H", targets)
    _append_depolarize1(circuit, targets, noise_rate)


def _append_cx(
    circuit: stim.Circuit,
    cnot_targets: Sequence[int],
    noise_rate: float,
) -> None:
    targets = tuple(int(qid) for qid in cnot_targets)
    if not targets:
        return
    circuit.append("CX", targets)
    _append_depolarize2(circuit, targets, noise_rate)


def _append_steane_zero_encoder(
    circuit: stim.Circuit,
    data_qids: Sequence[int],
    noise_rate: float,
) -> None:
    data = _validate_steane_data_block(data_qids)
    _append_h(circuit, [data[index] for index in STEANE_ZERO_H_INDICES], noise_rate)

    for control_index, target_index in STEANE_ZERO_CNOTS:
        _append_cx(circuit, [data[control_index], data[target_index]], noise_rate)


def _append_transversal_h(
    circuit: stim.Circuit,
    data_qids: Sequence[int],
    noise_rate: float,
) -> None:
    _append_h(circuit, _validate_steane_data_block(data_qids), noise_rate)


def _append_transversal_cnot(
    circuit: stim.Circuit,
    control_data_qids: Sequence[int],
    target_data_qids: Sequence[int],
    noise_rate: float,
) -> None:
    controls = _validate_steane_data_block(control_data_qids)
    targets = _validate_steane_data_block(target_data_qids)
    cnot_targets = []
    for control, target in zip(controls, targets, strict=True):
        cnot_targets.extend((control, target))
    _append_cx(circuit, cnot_targets, noise_rate)


def _append_logical_z_observable_measurement(
    circuit: stim.Circuit,
    block: SteaneColorCodeBlock,
    tracker: _MeasurementRecordTracker,
    latest_syndrome_records: dict[tuple[int, str, int], int],
    *,
    observable_index: int = 0,
    noise_rate: float = 0.0,
    detector_time: int = 0,
) -> None:
    support = tuple(block.global_observable_qids)
    if not support:
        raise ValueError("A logical observable measurement needs at least one physical qubit.")

    data_qids = tuple(block.global_data_qids)
    measurement_arg = _noise_arg(noise_rate)
    if measurement_arg is None:
        circuit.append("M", data_qids)
    else:
        circuit.append("M", data_qids, measurement_arg)

    data_record_keys = [
        ("final_data", block.logical_index, local_qid)
        for local_qid in block.data_qids
    ]
    data_records = tracker.record_many(data_record_keys)
    local_qid_to_record = {
        local_qid: data_records[("final_data", block.logical_index, local_qid)]
        for local_qid in block.data_qids
    }

    for ancilla in block.code.qubit_groups["anc_Z"]:
        ancilla_qid = int(ancilla["qid"])
        previous_record = latest_syndrome_records.get(
            (block.logical_index, "Z", ancilla_qid)
        )
        if previous_record is None:
            continue
        neighbor_targets = [
            tracker.rec_target(local_qid_to_record[int(data_qubit["qid"])])
            for data_qubit in ancilla.neighbors()
        ]
        targets = neighbor_targets + [tracker.rec_target(previous_record)]
        circuit.append(
            "DETECTOR",
            targets,
            _detector_coords(block, ancilla, "Z", detector_time),
        )

    record_targets = [
        tracker.rec_target(local_qid_to_record[local_qid])
        for local_qid in block.observable_qids
    ]
    circuit.append("OBSERVABLE_INCLUDE", record_targets, observable_index)


def _append_color_code_coordinates(
    circuit: stim.Circuit,
    blocks: Sequence[SteaneColorCodeBlock],
    *,
    block_x_spacing: float = 20.0,
) -> None:
    for block in blocks:
        for vertex in block.code.tanner_graph.vs:
            qid = block.qid_offset + int(vertex["qid"])
            x = float(vertex["x"]) + block_x_spacing * block.logical_index
            y = float(vertex["y"])
            circuit.append("QUBIT_COORDS", [qid], [x, y, float(block.logical_index)])


def _detector_coords(
    block: SteaneColorCodeBlock,
    ancilla,
    pauli: str,
    detector_time: int,
) -> tuple[float, float, float, float, float]:
    return (
        float(ancilla["x"]),
        float(ancilla["y"]),
        float(detector_time),
        float(_PAULI_TO_VALUE[pauli]),
        float(_COLOR_TO_VALUE[ancilla["color"]]),
    )


def _append_syndrome_detectors(
    circuit: stim.Circuit,
    block: SteaneColorCodeBlock,
    tracker: _MeasurementRecordTracker,
    latest_syndrome_records: dict[tuple[int, str, int], int],
    measured_records: dict[tuple[int, str, int], int],
    *,
    detector_time: int,
) -> None:
    for pauli, ancilla_group in (
        ("Z", block.code.qubit_groups["anc_Z"]),
        ("X", block.code.qubit_groups["anc_X"]),
    ):
        for ancilla in ancilla_group:
            ancilla_qid = int(ancilla["qid"])
            record_key = (block.logical_index, pauli, ancilla_qid)
            current_record = measured_records[record_key]
            previous_record = latest_syndrome_records.get(record_key)
            targets = [tracker.rec_target(current_record)]
            if previous_record is not None:
                targets.append(tracker.rec_target(previous_record))

            circuit.append(
                "DETECTOR",
                targets,
                _detector_coords(block, ancilla, pauli, detector_time),
            )
            latest_syndrome_records[record_key] = current_record


def _syndrome_cx_targets_for_timeslice(
    block: SteaneColorCodeBlock,
    timeslice: int,
) -> tuple[list[int], set[int]]:
    cnot_targets = []
    operated_local_qids = set()
    offsets = {
        0: (-2, 1),
        1: (2, 1),
        2: (4, 0),
        3: (2, -1),
        4: (-2, -1),
        5: (-4, 0),
        6: (-2, 1),
        7: (2, 1),
        8: (4, 0),
        9: (2, -1),
        10: (-2, -1),
        11: (-4, 0),
    }

    targets = [
        index
        for index, scheduled_timeslice in enumerate(block.code.cnot_schedule)
        if scheduled_timeslice == timeslice
    ]
    for target in targets:
        offset = offsets[target]
        target_ancillas = (
            block.code.qubit_groups["anc_Z"]
            if target < 6
            else block.code.qubit_groups["anc_X"]
        )
        for ancilla in target_ancillas:
            data_qubit_name = f"{ancilla['face_x'] + offset[0]}-{ancilla['face_y'] + offset[1]}"
            try:
                data_qubit = block.code.tanner_graph.vs.find(name=data_qubit_name)
            except ValueError:
                continue

            ancilla_qid = int(ancilla["qid"])
            data_qid = int(data_qubit["qid"])
            operated_local_qids.update({ancilla_qid, data_qid})

            if target < 6:
                cnot_targets.extend(
                    [block.qid_offset + data_qid, block.qid_offset + ancilla_qid]
                )
            else:
                cnot_targets.extend(
                    [block.qid_offset + ancilla_qid, block.qid_offset + data_qid]
                )

    return cnot_targets, operated_local_qids


def _unit_syndrome_correction_qids(
    block: SteaneColorCodeBlock,
    pauli: str,
) -> tuple[int, ...]:
    ancillas = (
        block.code.qubit_groups["anc_Z"]
        if pauli == "Z"
        else block.code.qubit_groups["anc_X"]
    )
    support_by_ancilla = [
        {int(data_qubit["qid"]) for data_qubit in ancilla.neighbors()}
        for ancilla in ancillas
    ]
    correction_qids = []
    for check_index in range(len(support_by_ancilla)):
        unit_syndrome = tuple(int(index == check_index) for index in range(len(support_by_ancilla)))
        for data_qid in block.data_qids:
            syndrome = tuple(
                int(data_qid in support)
                for support in support_by_ancilla
            )
            if syndrome == unit_syndrome:
                correction_qids.append(block.qid_offset + data_qid)
                break
        else:
            raise ValueError(f"No unit-syndrome correction qubit found for {pauli} check {check_index}.")
    return tuple(correction_qids)


def _append_feedback_corrections(
    circuit: stim.Circuit,
    blocks: Sequence[SteaneColorCodeBlock],
    tracker: _MeasurementRecordTracker,
    measured_records: dict[tuple[int, str, int], int],
    *,
    noise_rate: float,
) -> None:
    correction_targets = []
    for block in blocks:
        x_correction_qids = _unit_syndrome_correction_qids(block, "Z")
        z_correction_qids = _unit_syndrome_correction_qids(block, "X")

        for ancilla, correction_qid in zip(
            block.code.qubit_groups["anc_Z"],
            x_correction_qids,
            strict=True,
        ):
            record = measured_records[(block.logical_index, "Z", int(ancilla["qid"]))]
            circuit.append("CX", [tracker.rec_target(record), correction_qid])
            correction_targets.append(correction_qid)

        for ancilla, correction_qid in zip(
            block.code.qubit_groups["anc_X"],
            z_correction_qids,
            strict=True,
        ):
            record = measured_records[(block.logical_index, "X", int(ancilla["qid"]))]
            circuit.append("CZ", [tracker.rec_target(record), correction_qid])
            correction_targets.append(correction_qid)

    _append_depolarize1(circuit, sorted(set(correction_targets)), noise_rate)
    circuit.append("TICK")


def _append_syndrome_round(
    circuit: stim.Circuit,
    blocks: Sequence[SteaneColorCodeBlock],
    tracker: _MeasurementRecordTracker,
    latest_syndrome_records: dict[tuple[int, str, int], int],
    *,
    detector_time: int,
    noise_rate: float,
    additional_idle_qids: Sequence[int] = (),
    emit_detectors: bool = True,
    apply_feedback_corrections: bool = False,
) -> None:
    if not blocks:
        return

    max_timeslice = max(max(block.code.cnot_schedule) for block in blocks)
    for timeslice in range(1, max_timeslice + 1):
        cnot_targets = []
        idle_qids = list(additional_idle_qids)
        for block in blocks:
            block_targets, operated_local_qids = _syndrome_cx_targets_for_timeslice(
                block,
                timeslice,
            )
            cnot_targets.extend(block_targets)
            idle_qids.extend(
                block.qid_offset + qid
                for qid in range(block.patch_qubit_count)
                if qid not in operated_local_qids
            )

        _append_cx(circuit, cnot_targets, noise_rate)
        _append_depolarize1(circuit, sorted(set(idle_qids)), noise_rate)
        circuit.append("TICK")

    measured_records = {}
    measurement_arg = _noise_arg(noise_rate)
    z_targets = [qid for block in blocks for qid in block.global_z_ancilla_qids]
    if measurement_arg is None:
        circuit.append("MRZ", z_targets)
    else:
        circuit.append("MRZ", z_targets, measurement_arg)
    measured_records.update(
        tracker.record_many(
            [
                (block.logical_index, "Z", int(qid))
                for block in blocks
                for qid in block.code.qubit_groups["anc_Z"]["qid"]
            ]
        )
    )

    x_targets = [qid for block in blocks for qid in block.global_x_ancilla_qids]
    if measurement_arg is None:
        circuit.append("MRX", x_targets)
    else:
        circuit.append("MRX", x_targets, measurement_arg)
    measured_records.update(
        tracker.record_many(
            [
                (block.logical_index, "X", int(qid))
                for block in blocks
                for qid in block.code.qubit_groups["anc_X"]["qid"]
            ]
        )
    )

    if emit_detectors:
        for block in blocks:
            _append_syndrome_detectors(
                circuit,
                block,
                tracker,
                latest_syndrome_records,
                measured_records,
                detector_time=detector_time,
            )

    _append_depolarize1(
        circuit,
        [qid for block in blocks for qid in block.global_data_qids],
        noise_rate,
    )
    for block in blocks:
        _append_reset_error(circuit, "X_ERROR", block.global_z_ancilla_qids, noise_rate)
        _append_reset_error(circuit, "Z_ERROR", block.global_x_ancilla_qids, noise_rate)
    circuit.append("TICK")

    if apply_feedback_corrections:
        _append_feedback_corrections(
            circuit,
            blocks,
            tracker,
            measured_records,
            noise_rate=noise_rate,
        )


def build_unencoded_precompiled_shor_15_circuit(
    *,
    include_measurement: bool = True,
) -> stim.Circuit:
    """Build the bare three-qubit circuit from the simplified Shor-15 figure."""
    circuit = stim.Circuit()
    for qid in range(3):
        circuit.append("QUBIT_COORDS", [qid], [float(qid), 0.0])

    circuit.append("R", [0, 1, 2])
    circuit.append("TICK")
    circuit.append("H", [0])
    circuit.append("TICK")
    circuit.append("CX", [0, 1, 0, 2])
    circuit.append("TICK")
    circuit.append("H", [0])

    if include_measurement:
        circuit.append("TICK")
        circuit.append("M", [0])
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)

    return circuit


def build_encoded_precompiled_shor_15_circuit(
    blocks: Sequence[SteaneColorCodeBlock] | None = None,
    *,
    include_coordinate_annotations: bool = True,
    include_measurement: bool = True,
    noise_rate: float = DEFAULT_NOISE_RATE,
    include_syndrome_extraction: bool = True,
    syndrome_policy: str = IDLE_BLOCK_SYNDROME_POLICY,
    _decoder_reference_identity: bool = False,
) -> stim.Circuit:
    """Build the three-logical-qubit Steane-encoded precompiled Shor-15 circuit."""
    if noise_rate < 0:
        raise ValueError("noise_rate must be non-negative.")
    if syndrome_policy not in {IDLE_BLOCK_SYNDROME_POLICY, NO_SYNDROME_POLICY}:
        raise ValueError(
            f"syndrome_policy must be '{IDLE_BLOCK_SYNDROME_POLICY}' or "
            f"'{NO_SYNDROME_POLICY}'."
        )
    if not include_syndrome_extraction:
        syndrome_policy = NO_SYNDROME_POLICY

    resolved_blocks = tuple(
        blocks or generate_steane_code_instances(noise_rate=noise_rate)
    )
    if len(resolved_blocks) != 3:
        raise ValueError("The simplified Shor-15 circuit needs exactly three logical blocks.")

    control, target_0, target_1 = resolved_blocks
    circuit = stim.Circuit()
    tracker = _MeasurementRecordTracker()
    latest_syndrome_records: dict[tuple[int, str, int], int] = {}
    detector_time = 0

    if include_coordinate_annotations:
        _append_color_code_coordinates(circuit, resolved_blocks)

    for block in resolved_blocks:
        circuit.append("R", block.global_data_qids)
        circuit.append("RZ", block.global_z_ancilla_qids)
        circuit.append("RX", block.global_x_ancilla_qids)
        _append_reset_error(circuit, "X_ERROR", block.global_data_qids, noise_rate)
        _append_reset_error(circuit, "X_ERROR", block.global_z_ancilla_qids, noise_rate)
        _append_reset_error(circuit, "Z_ERROR", block.global_x_ancilla_qids, noise_rate)
    circuit.append("TICK")

    if not _decoder_reference_identity:
        for block in resolved_blocks:
            _append_steane_zero_encoder(circuit, block.global_data_qids, noise_rate)
            circuit.append("TICK")

    if syndrome_policy == IDLE_BLOCK_SYNDROME_POLICY:
        _append_syndrome_round(
            circuit,
            resolved_blocks,
            tracker,
            latest_syndrome_records,
            detector_time=detector_time,
            noise_rate=noise_rate,
            emit_detectors=False,
            apply_feedback_corrections=True,
        )

    if _decoder_reference_identity:
        _append_depolarize1(
            circuit,
            [qid for block in resolved_blocks for qid in block.global_qids],
            noise_rate,
        )
    else:
        _append_transversal_h(circuit, control.global_data_qids, noise_rate)
        _append_depolarize1(circuit, control.global_ancilla_qids, noise_rate)
    circuit.append("TICK")
    if syndrome_policy == IDLE_BLOCK_SYNDROME_POLICY:
        _append_syndrome_round(
            circuit,
            (target_0, target_1),
            tracker,
            latest_syndrome_records,
            detector_time=detector_time,
            noise_rate=noise_rate,
            additional_idle_qids=control.global_qids,
        )
        detector_time += 1

    if _decoder_reference_identity:
        _append_depolarize1(
            circuit,
            [qid for block in resolved_blocks for qid in block.global_qids],
            noise_rate,
        )
    else:
        _append_transversal_cnot(
            circuit,
            control.global_data_qids,
            target_0.global_data_qids,
            noise_rate,
        )
        _append_transversal_cnot(
            circuit,
            control.global_data_qids,
            target_1.global_data_qids,
            noise_rate,
        )
    _append_depolarize1(
        circuit,
        [qid for block in resolved_blocks for qid in block.global_ancilla_qids],
        noise_rate,
    )
    circuit.append("TICK")

    if _decoder_reference_identity:
        _append_depolarize1(
            circuit,
            [qid for block in resolved_blocks for qid in block.global_qids],
            noise_rate,
        )
    else:
        _append_transversal_h(circuit, control.global_data_qids, noise_rate)
        _append_depolarize1(circuit, control.global_ancilla_qids, noise_rate)
    circuit.append("TICK")
    if syndrome_policy == IDLE_BLOCK_SYNDROME_POLICY:
        _append_syndrome_round(
            circuit,
            (target_0, target_1),
            tracker,
            latest_syndrome_records,
            detector_time=detector_time,
            noise_rate=noise_rate,
            additional_idle_qids=control.global_qids,
        )
        detector_time += 1

        _append_syndrome_round(
            circuit,
            resolved_blocks,
            tracker,
            latest_syndrome_records,
            detector_time=detector_time,
            noise_rate=noise_rate,
        )
        detector_time += 1

    if include_measurement:
        circuit.append("TICK")
        _append_logical_z_observable_measurement(
            circuit,
            control,
            tracker,
            latest_syndrome_records,
            observable_index=0,
            noise_rate=noise_rate,
            detector_time=detector_time,
        )

    return circuit


def sample_logical_observable(
    circuit: stim.Circuit,
    *,
    shots: int = 1024,
    seed: int | None = None,
    observable_index: int = 0,
) -> dict[int, int]:
    """Sample a Stim circuit logical observable and return bit counts."""
    if shots < 1:
        raise ValueError("shots must be at least 1.")

    sampler = circuit.compile_detector_sampler(seed=seed)
    _, observables = sampler.sample(shots=shots, separate_observables=True)
    if observable_index >= observables.shape[1]:
        raise ValueError(
            f"Circuit has only {observables.shape[1]} observables; "
            f"observable_index={observable_index} is unavailable."
        )

    counts = Counter(int(row[observable_index]) for row in observables)
    return {0: counts.get(0, 0), 1: counts.get(1, 0)}


def _bit_counts(bits: Sequence[bool] | np.ndarray) -> dict[int, int]:
    counts = Counter(int(bit) for bit in np.asarray(bits, dtype=bool).ravel())
    return {0: counts.get(0, 0), 1: counts.get(1, 0)}


def decode_logical_observable_predictions(
    circuit: stim.Circuit,
    detector_outcomes: np.ndarray,
    *,
    blocks: Sequence[SteaneColorCodeBlock] | None = None,
    decoder_circuit: stim.Circuit | None = None,
) -> np.ndarray:
    """Decode detector outcomes into predicted logical observable flips."""
    outcomes = np.asarray(detector_outcomes, dtype=bool)
    if outcomes.ndim == 1:
        outcomes = outcomes.reshape(1, -1)

    if circuit.num_detectors == 0:
        return np.zeros(outcomes.shape[0], dtype=bool)

    resolved_blocks = tuple(blocks or generate_steane_code_instances(noise_rate=0.0))
    dem_circuit = decoder_circuit or circuit
    dem_manager = DemManager(
        circuit=dem_circuit,
        tanner_graph=resolved_blocks[0].code.tanner_graph,
        circuit_type="tri",
        comparative_decoding=False,
        remove_non_edge_like_errors=True,
    )
    decoder = ConcatMatchingDecoder(dem_manager)
    predictions = decoder.decode(outcomes, colors="all")
    return np.asarray(predictions, dtype=bool).reshape(-1)


def factor_candidates_from_phase_bit(
    phase_bit: int,
    *,
    a: int = SHOR_15_A,
    N: int = SHOR_15_N,
) -> tuple[int, int] | None:
    """Map the useful one-bit phase outcome to Shor factor candidates."""
    if phase_bit not in {0, 1}:
        raise ValueError("The simplified circuit produces a single phase bit.")
    if phase_bit == 0:
        return None

    candidate_order = 2
    midpoint_power = pow(a, candidate_order // 2, N)
    if midpoint_power in {1, N - 1}:
        return None

    factors = tuple(
        sorted(
            {
                gcd(midpoint_power - 1, N),
                gcd(midpoint_power + 1, N),
            }
        )
    )
    if len(factors) != 2 or any(factor in {1, N} for factor in factors):
        return None
    return factors


def shor_15_success_explanation(
    *,
    a: int = SHOR_15_A,
    N: int = SHOR_15_N,
) -> str:
    """Return the notebook-facing explanation for the useful phase-bit branch."""
    candidate_order = 2
    midpoint_power = pow(a, candidate_order // 2, N)
    lower_factor = gcd(midpoint_power - 1, N)
    upper_factor = gcd(midpoint_power + 1, N)
    return (
        f"For the simplified one-bit Hadamard-test circuit, decoded phase bit 1 "
        f"is the useful branch. It corresponds to the candidate order r = "
        f"{candidate_order}. With a = {a} and N = {N}, "
        f"{a}^(r/2) = {midpoint_power} mod {N}. The classical Shor "
        f"post-processing then gives gcd({midpoint_power} - 1, {N}) = "
        f"gcd({midpoint_power - 1}, {N}) = {lower_factor} and "
        f"gcd({midpoint_power} + 1, {N}) = gcd({midpoint_power + 1}, {N}) = "
        f"{upper_factor}. Thus phase bit 1 yields the nontrivial factors "
        f"{tuple(sorted((lower_factor, upper_factor)))}. Phase bit 0 is "
        "inconclusive, not a wrong factor."
    )


def run_precompiled_shor_15(
    *,
    shots: int = 1024,
    seed: int | None = None,
    blocks: Sequence[SteaneColorCodeBlock] | None = None,
    decode: bool = True,
    noise_rate: float = DEFAULT_NOISE_RATE,
) -> dict[str, Any]:
    """Run the current encoded Shor-15 simulation path and summarize the outcomes."""
    if shots < 1:
        raise ValueError("shots must be at least 1.")
    if noise_rate < 0:
        raise ValueError("noise_rate must be non-negative.")

    resolved_blocks = tuple(
        blocks or generate_steane_code_instances(noise_rate=noise_rate)
    )
    circuit = build_encoded_precompiled_shor_15_circuit(
        resolved_blocks,
        noise_rate=noise_rate,
        include_syndrome_extraction=True,
        syndrome_policy=IDLE_BLOCK_SYNDROME_POLICY,
    )
    sampler = circuit.compile_detector_sampler(seed=seed)
    detector_outcomes, observables = sampler.sample(
        shots=shots,
        separate_observables=True,
    )
    raw_phase_bits = observables[:, 0].astype(bool)

    if decode and noise_rate > 0 and circuit.num_detectors:
        # Stim detector error models require deterministic logical observables.
        # The actual Shor phase bit is intentionally random, so decoding uses a
        # reference circuit with the same detector/readout layout but a fixed
        # logical-|0> branch to learn noise-induced logical flips.
        decoder_circuit = build_encoded_precompiled_shor_15_circuit(
            resolved_blocks,
            noise_rate=noise_rate,
            include_syndrome_extraction=True,
            syndrome_policy=IDLE_BLOCK_SYNDROME_POLICY,
            _decoder_reference_identity=True,
        )
        decoder_predictions = decode_logical_observable_predictions(
            circuit,
            detector_outcomes,
            blocks=resolved_blocks,
            decoder_circuit=decoder_circuit,
        )
    else:
        decoder_predictions = np.zeros(shots, dtype=bool)

    decoded_phase_bits = np.logical_xor(raw_phase_bits, decoder_predictions)
    raw_phase_bit_counts = _bit_counts(raw_phase_bits)
    decoded_phase_bit_counts = _bit_counts(decoded_phase_bits)
    decoder_prediction_counts = _bit_counts(decoder_predictions)

    factor_candidates_by_phase_bit = {
        phase_bit: factor_candidates_from_phase_bit(phase_bit)
        for phase_bit in sorted(decoded_phase_bit_counts)
    }
    factor_counts = Counter(
        factor_candidates_from_phase_bit(int(phase_bit))
        for phase_bit in decoded_phase_bits
    )
    successful_factor_shots = sum(
        count
        for phase_bit, count in decoded_phase_bit_counts.items()
        if factor_candidates_by_phase_bit[phase_bit] is not None
    )

    return {
        "a": SHOR_15_A,
        "N": SHOR_15_N,
        "noise_rate": noise_rate,
        "decode": decode,
        "shots": shots,
        "raw_phase_bit_counts": raw_phase_bit_counts,
        "decoded_phase_bit_counts": decoded_phase_bit_counts,
        "decoder_prediction_counts": decoder_prediction_counts,
        "phase_bit_counts": decoded_phase_bit_counts,
        "factor_candidates_by_phase_bit": factor_candidates_by_phase_bit,
        "factor_counts": dict(factor_counts),
        "successful_factor_shots": successful_factor_shots,
        "estimated_success_probability": successful_factor_shots / shots,
        "detectors": circuit.num_detectors,
        "observables": circuit.num_observables,
    }
