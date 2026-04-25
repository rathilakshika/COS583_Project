"""Reusable compilation helpers for the Shor-15 demonstration notebook."""

from collections import Counter, defaultdict
from math import isclose
from pathlib import Path

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate

__all__ = [
    "generic_space_optimized_budget",
    "generic_round_schedule",
    "retained_round_schedule",
    "controlled_modular_multiply_k0_schedule",
    "asap_schedule_circuit",
    "build_controlled_modular_multiply_k0_block",
    "build_controlled_modular_multiply_k0_gate_only_block",
    "build_original_10q_shor_15_layout",
    "build_first_pass_reduced_10q_shor_15_layout",
    "build_first_pass_reduced_10q_shor_15_decomposed_k0_layout",
    "build_first_pass_reduced_10q_shor_15_gate_only_k0_layout",
    "first_pass_compilation_summary",
    "DEFAULT_QUANTINUUM_DEVICE_NAME",
    "DEFAULT_QUANTINUUM_OPTIMISATION_LEVEL",
    "REQUIRED_TKET_MODULES",
    "CONTROL_FLOW_LIKE_OPS",
    "TIMING_MODEL_SOURCE_URLS",
    "NATIVE_OPERATION_DURATIONS_US",
    "NATIVE_OPERATION_DURATION_NOTES",
    "QUANTINUUM_NATIVE_OPS",
    "ALLOWED_METADATA_OPS",
    "ARTIFACT_DIR",
    "SCHEDULED_PHYSICAL_K0_QPY_PATH",
    "FULL_STEANE_PHYSICAL_K0_QPY_PATH",
    "save_qpy_circuit",
    "load_qpy_circuit",
    "save_scheduled_physical_k0_circuit",
    "load_scheduled_physical_k0_circuit",
    "require_scheduled_physical_k0_circuit",
    "save_full_steane_physical_k0_circuit",
    "load_full_steane_physical_k0_circuit",
    "show_table",
    "require_full_steane_physical_k0_circuit",
    "source_circuit_summary",
    "require_tket_quantinuum_dependencies",
    "source_control_flow_like_counts",
    "make_quantinuum_backend",
    "compile_quantinuum_native",
    "duration_table",
    "op_type_name",
    "underlying_conditional_op",
    "command_family",
    "command_resources",
    "build_command_records",
    "assert_quantinuum_native_command_records",
    "native_gate_count_tables",
    "asap_schedule_records",
    "build_moment_records",
    "analyze_compiled_tket_circuit",
    "resource_summary",
]


def generic_space_optimized_budget(n):
    """Return the generic qubit budget for a space-optimized n-bit Shor layout."""
    return {
        "n_bits": n,
        "phase_estimation_rounds": 2 * n,
        "recycled_phase_qubits": 1,
        "modular_value_register_qubits": n,
        "arithmetic_register_qubits": n,
        "clean_ancillas": {
            "Takahashi_Kunihiro_2n_plus_2": 1,
            "Beauregard_2n_plus_3": 2,
        },
        "total_qubits": {
            "Takahashi_Kunihiro_2n_plus_2": 2 * n + 2,
            "Beauregard_2n_plus_3": 2 * n + 3,
        },
    }


def generic_round_schedule(a=11, N=15, n=4):
    """List each semiclassical phase-estimation round before any compilation pruning."""
    schedule = []

    for k in reversed(range(2 * n)):
        multiplier = pow(a, 2**k, N)
        schedule.append(
            {
                "phase_bit": k,
                "a^(2^k) mod N": multiplier,
                "controlled_operation": f"multiply by {multiplier} mod {N}",
                "compiled_away_in_first_pass": multiplier == 1,
            }
        )

    return schedule


def retained_round_schedule(a=11, N=15, n=4):
    """Keep only the rounds whose controlled modular multiplication is nontrivial."""
    return [
        round_info
        for round_info in generic_round_schedule(a=a, N=N, n=n)
        if not round_info["compiled_away_in_first_pass"]
    ]


def controlled_modular_multiply_k0_schedule(a=11, N=15, n=4):
    """Return the arithmetic schedule for the surviving k = 0 modular multiply.

    This is the generic Shor-style out-of-place construction, not the small
    N = 15 permutation shortcut.  The value-register index convention matches
    the notebook: ``value[-1]`` is the least-significant bit.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")

    try:
        inverse_multiplier = pow(a, -1, N)
    except ValueError as exc:
        raise ValueError(f"a={a} is not invertible modulo N={N}.") from exc

    compute_adds = []
    uncompute_subs = []
    for bit_power in range(n):
        value_index = n - 1 - bit_power
        compute_adds.append(
            {
                "value_index": value_index,
                "bit_power": bit_power,
                "constant": (a * 2**bit_power) % N,
                "operation": "add",
            }
        )
        uncompute_subs.append(
            {
                "value_index": value_index,
                "bit_power": bit_power,
                "constant": (inverse_multiplier * 2**bit_power) % N,
                "operation": "subtract",
            }
        )

    return {
        "phase_bit": 0,
        "multiplier": a % N,
        "inverse_multiplier": inverse_multiplier,
        "modulus": N,
        "register_bits": n,
        "compute_adds": compute_adds,
        "controlled_register_swap": list(range(n)),
        "uncompute_subs": list(reversed(uncompute_subs)),
    }


def _instruction_resource_keys(circuit, instruction):
    return [
        ("q", circuit.find_bit(qubit).index)
        for qubit in instruction.qubits
    ] + [
        ("c", circuit.find_bit(clbit).index)
        for clbit in instruction.clbits
    ]


def _append_mapped_instruction(source, target, instruction):
    qargs = [
        target.qubits[source.find_bit(qubit).index]
        for qubit in instruction.qubits
    ]
    cargs = [
        target.clbits[source.find_bit(clbit).index]
        for clbit in instruction.clbits
    ]
    target.append(instruction.operation, qargs, cargs)


def asap_schedule_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of ``circuit`` ordered into ASAP dependency layers.

    Within each barrier-delimited segment, every instruction is placed in the
    earliest abstract layer that does not conflict with earlier instructions on
    any shared qubit or classical bit.  Barriers are retained as hard boundaries.
    """
    scheduled = circuit.copy_empty_like()
    scheduled.metadata = dict(circuit.metadata or {})

    original_index_to_layer = [None] * len(circuit.data)
    instruction_layers = []
    next_resource_layer = {}
    pending_instructions = []
    segment_start_layer = 0
    barrier_indices = []

    def append_scheduled_instruction(original_index, instruction, layer):
        scheduled_index = len(scheduled.data)
        _append_mapped_instruction(circuit, scheduled, instruction)
        instruction_layers.append(
            {
                "original_index": original_index,
                "scheduled_index": scheduled_index,
                "operation": instruction.operation.name,
                "scheduled_layer": layer,
            }
        )

    def flush_segment():
        nonlocal pending_instructions
        if not pending_instructions:
            return segment_start_layer

        segment_end_layer = max(layer for layer, _, _ in pending_instructions) + 1
        for layer, original_index, instruction in sorted(
            pending_instructions,
            key=lambda scheduled_instruction: (
                scheduled_instruction[0],
                scheduled_instruction[1],
            ),
        ):
            append_scheduled_instruction(original_index, instruction, layer)
        pending_instructions = []
        return segment_end_layer

    for original_index, instruction in enumerate(circuit.data):
        if instruction.operation.name == "barrier":
            barrier_layer = flush_segment()
            original_index_to_layer[original_index] = barrier_layer
            barrier_indices.append(original_index)
            append_scheduled_instruction(original_index, instruction, barrier_layer)
            segment_start_layer = barrier_layer + 1
            next_resource_layer = {}
            continue

        resources = _instruction_resource_keys(circuit, instruction)
        layer = max(
            (next_resource_layer.get(resource, segment_start_layer) for resource in resources),
            default=segment_start_layer,
        )
        for resource in resources:
            next_resource_layer[resource] = layer + 1

        original_index_to_layer[original_index] = layer
        pending_instructions.append((layer, original_index, instruction))

    flush_segment()

    scheduled_layers = [
        layer
        for layer in original_index_to_layer
        if layer is not None
    ]
    scheduled_layer_count = (
        max(scheduled_layers) + 1
        if scheduled_layers
        else 0
    )
    scheduled.metadata["asap_schedule"] = {
        "barrier_policy": "preserved_as_segment_boundaries",
        "barrier_indices": barrier_indices,
        "scheduled_layer_count": scheduled_layer_count,
        "original_index_to_layer": original_index_to_layer,
        "instruction_layers": instruction_layers,
    }
    return scheduled


def _controlled_modular_add_gate(operation, constant, modulus, n):
    verb = "add" if operation == "add" else "sub"
    sign = "+" if operation == "add" else "-"
    return Gate(
        name=f"cc_mod_{verb}_{constant}_mod_{modulus}",
        num_qubits=2 + n + 1,
        params=[],
        label=f"{sign}{constant} mod {modulus}",
    )


def _basis_bits(value, width):
    return [(value >> bit_index) & 1 for bit_index in reversed(range(width))]


def _append_pattern_controlled_x(
    circuit,
    external_controls,
    target_register,
    target_index,
    pattern,
):
    controls = list(external_controls)
    control_state = (1 << len(controls)) - 1

    for bit_index, qubit in enumerate(target_register):
        if bit_index == target_index:
            continue
        if pattern[bit_index]:
            control_state |= 1 << len(controls)
        controls.append(qubit)

    target = target_register[target_index]
    if not controls:
        circuit.x(target)
    elif len(controls) == 1:
        if control_state:
            circuit.cx(controls[0], target)
        else:
            circuit.x(controls[0])
            circuit.cx(controls[0], target)
            circuit.x(controls[0])
    elif len(controls) == 2 and control_state == 0b11:
        circuit.ccx(controls[0], controls[1], target)
    else:
        circuit.mcx(controls, target, ctrl_state=control_state)


def _append_controlled_adjacent_basis_swap(
    circuit,
    external_controls,
    target_register,
    state_bits,
    flip_index,
):
    _append_pattern_controlled_x(
        circuit,
        external_controls,
        target_register,
        flip_index,
        state_bits,
    )


def _append_controlled_basis_state_swap(
    circuit,
    external_controls,
    target_register,
    left,
    right,
):
    width = len(target_register)
    left_bits = _basis_bits(left, width)
    right_bits = _basis_bits(right, width)
    differing_indices = [
        bit_index
        for bit_index, (left_bit, right_bit) in enumerate(zip(left_bits, right_bits, strict=True))
        if left_bit != right_bit
    ]

    if not differing_indices:
        return

    current_bits = left_bits[:]
    prefix_steps = []
    for flip_index in differing_indices[:-1]:
        _append_controlled_adjacent_basis_swap(
            circuit,
            external_controls,
            target_register,
            current_bits,
            flip_index,
        )
        prefix_steps.append((current_bits[:], flip_index))
        current_bits[flip_index] ^= 1

    _append_controlled_adjacent_basis_swap(
        circuit,
        external_controls,
        target_register,
        current_bits,
        differing_indices[-1],
    )

    for state_bits, flip_index in reversed(prefix_steps):
        _append_controlled_adjacent_basis_swap(
            circuit,
            external_controls,
            target_register,
            state_bits,
            flip_index,
        )


def _append_phase_controlled_0000_1111_fixup_with_clean_workspace(
    circuit,
    phase_qubit,
    value_register,
    scratch_register,
    flag_qubit,
):
    """Swap ``|0000>`` and ``|1111>`` using clean scratch qubits.

    The scratch and flag qubits must enter as ``|0>``.  They are returned to
    ``|0>`` before this helper exits.
    """
    if len(value_register) != 4 or len(scratch_register) < 3:
        raise ValueError("The clean-workspace fix-up expects four value qubits and three scratch qubits.")

    # A 4-bit string is 0000 or 1111 exactly when all adjacent bit parities
    # are zero.  Store those parities in clean scratch, conditionally flip all
    # value bits, then uncompute the scratch.
    circuit.cx(value_register[0], scratch_register[0])
    circuit.cx(value_register[1], scratch_register[0])
    circuit.cx(value_register[1], scratch_register[1])
    circuit.cx(value_register[2], scratch_register[1])
    circuit.cx(value_register[2], scratch_register[2])
    circuit.cx(value_register[3], scratch_register[2])

    controls = [phase_qubit, scratch_register[0], scratch_register[1], scratch_register[2]]
    circuit.mcx(controls, flag_qubit, ctrl_state=0b0001)
    for qubit in value_register:
        circuit.cx(flag_qubit, qubit)
    circuit.mcx(controls, flag_qubit, ctrl_state=0b0001)

    circuit.cx(value_register[2], scratch_register[2])
    circuit.cx(value_register[3], scratch_register[2])
    circuit.cx(value_register[1], scratch_register[1])
    circuit.cx(value_register[2], scratch_register[1])
    circuit.cx(value_register[0], scratch_register[0])
    circuit.cx(value_register[1], scratch_register[0])


def build_controlled_modular_multiply_k0_block(a=11, N=15, n=4):
    """Build the generic arithmetic block for ``ctrl-U^(2^0)``.

    The block implements the standard reversible modular-multiplication shape:
    controlled out-of-place multiply into the arithmetic register, controlled
    register swap, then inverse multiply to clean the arithmetic register.
    """
    schedule = controlled_modular_multiply_k0_schedule(a=a, N=N, n=n)

    phase = QuantumRegister(1, "phase")
    value = QuantumRegister(n, "value")
    arith = QuantumRegister(n, "arith")
    anc = QuantumRegister(1, "anc")
    qc = QuantumCircuit(
        phase,
        value,
        arith,
        anc,
        name=f"ctrl_u_0_modmul_{a}_mod_{N}",
    )

    for step in schedule["compute_adds"]:
        qc.append(
            _controlled_modular_add_gate(step["operation"], step["constant"], N, n),
            [phase[0], value[step["value_index"]], *arith, anc[0]],
        )

    qc.barrier()

    for value_index in schedule["controlled_register_swap"]:
        qc.cswap(phase[0], value[value_index], arith[value_index])

    qc.barrier()

    for step in schedule["uncompute_subs"]:
        qc.append(
            _controlled_modular_add_gate(step["operation"], step["constant"], N, n),
            [phase[0], value[step["value_index"]], *arith, anc[0]],
        )

    qc.metadata = {
        "phase_bit": schedule["phase_bit"],
        "multiplier": schedule["multiplier"],
        "inverse_multiplier": schedule["inverse_multiplier"],
        "modulus": schedule["modulus"],
        "register_bits": schedule["register_bits"],
        "construction": "generic_out_of_place_controlled_modular_multiply",
        "schedule": schedule,
    }
    return qc


def build_controlled_modular_multiply_k0_gate_only_block(a=11, N=15, n=4):
    """Build a compact gate-only compilation of ``ctrl-U^(2^0)``.

    This compiles the public modular-multiplication map for ``a = 11`` and
    ``N = 15`` directly, while keeping the full 4-bit unitary by fixing the
    out-of-range states ``|0000>`` and ``|1111>``.
    """
    if (a, N, n) != (11, 15, 4):
        raise ValueError("The compact gate-only k=0 block is implemented for a=11, N=15, n=4.")

    schedule = controlled_modular_multiply_k0_schedule(a=a, N=N, n=n)

    phase = QuantumRegister(1, "phase")
    value = QuantumRegister(n, "value")
    arith = QuantumRegister(n, "arith")
    anc = QuantumRegister(1, "anc")
    qc = QuantumCircuit(
        phase,
        value,
        arith,
        anc,
        name=f"ctrl_u_0_gate_only_{a}_mod_{N}",
    )

    # The affine shortcut for multiply-by-11 swaps 0 and 15.  This fix-up makes
    # the full 4-bit modular-multiplication map fix those out-of-range states on
    # the clean-workspace subspace used by this reduced circuit.
    _append_phase_controlled_0000_1111_fixup_with_clean_workspace(
        qc,
        phase[0],
        value,
        arith,
        anc[0],
    )

    qc.barrier()

    # Controlled swap value[1] <-> value[3].
    qc.cx(value[3], value[1])
    qc.ccx(phase[0], value[1], value[3])
    qc.cx(value[3], value[1])

    # Controlled swap value[0] <-> value[2].
    qc.cx(value[2], value[0])
    qc.ccx(phase[0], value[0], value[2])
    qc.cx(value[2], value[0])

    for qubit in value:
        qc.cx(phase[0], qubit)

    qc.metadata = {
        "phase_bit": schedule["phase_bit"],
        "multiplier": schedule["multiplier"],
        "inverse_multiplier": schedule["inverse_multiplier"],
        "modulus": schedule["modulus"],
        "register_bits": schedule["register_bits"],
        "construction": "direct_compiled_gate_only_controlled_modular_multiply",
        "invalid_state_fixup": "phase-controlled |0000><->|1111| fix-up using clean arith/anc workspace",
        "workspace_requirement": "arith and anc enter as |0> and are uncomputed",
        "schedule": schedule,
    }
    return qc


def build_original_10q_shor_15_layout(a=11, N=15, n=4):
    """Build the full pre-compilation 10-qubit order-finding layout for N = 15."""
    phase = QuantumRegister(1, "phase")
    value = QuantumRegister(n, "value")
    arith = QuantumRegister(n, "arith")
    anc = QuantumRegister(1, "anc")
    bits = ClassicalRegister(2 * n, "c")
    qc = QuantumCircuit(phase, value, arith, anc, bits)

    # Start the modular register in |1> for order finding.
    qc.x(value[-1])

    total_rounds = 2 * n
    block_width = 1 + n + n + 1
    block_qubits = [phase[0], *value, *arith, anc[0]]

    for round_power in reversed(range(total_rounds)):
        # Each round prepares the recycled phase qubit, applies the controlled
        # modular block for that power, and then measures the resulting bit.
        qc.h(phase[0])
        qc.append(
            Gate(
                name=f"ctrl_u_{round_power}",
                num_qubits=block_width,
                params=[],
                label=f"ctrl-U^(2^{round_power})",
            ),
            block_qubits,
        )

        if round_power != total_rounds - 1:
            qc.append(
                Gate(
                    name=f"phase_fix_{round_power}",
                    num_qubits=1,
                    params=[],
                    label="phase fix",
                ),
                [phase[0]],
            )

        qc.h(phase[0])
        qc.measure(phase[0], bits[round_power])

        if round_power:
            qc.reset(phase[0])
            qc.barrier()

    return qc


def build_first_pass_reduced_10q_shor_15_layout(a=11, N=15, n=4):
    """Build the same layout after removing identity rounds from the first pass."""
    phase = QuantumRegister(1, "phase")
    value = QuantumRegister(n, "value")
    arith = QuantumRegister(n, "arith")
    anc = QuantumRegister(1, "anc")
    retained_rounds = retained_round_schedule(a=a, N=N, n=n)
    # Size the classical register to the highest surviving phase bit.
    reduced_bits = max((round_info["phase_bit"] for round_info in retained_rounds), default=0) + 1
    bits = ClassicalRegister(reduced_bits, "c")
    qc = QuantumCircuit(phase, value, arith, anc, bits)

    # Start the modular register in |1> for order finding.
    qc.x(value[-1])

    block_width = 1 + n + n + 1
    block_qubits = [phase[0], *value, *arith, anc[0]]

    for retained_index, round_info in enumerate(retained_rounds):
        # Rebuild only the rounds that remain after identity elimination.
        round_power = round_info["phase_bit"]
        qc.h(phase[0])
        qc.append(
            Gate(
                name=f"ctrl_u_{round_power}",
                num_qubits=block_width,
                params=[],
                label=f"ctrl-U^(2^{round_power})",
            ),
            block_qubits,
        )

        if retained_index:
            qc.append(
                Gate(
                    name=f"phase_fix_{round_power}",
                    num_qubits=1,
                    params=[],
                    label="phase fix",
                ),
                [phase[0]],
            )

        qc.h(phase[0])
        qc.measure(phase[0], bits[round_power])

        if retained_index != len(retained_rounds) - 1:
            qc.reset(phase[0])
            qc.barrier()

    return qc


def build_first_pass_reduced_10q_shor_15_decomposed_k0_layout(a=11, N=15, n=4):
    """Build the reduced circuit with the surviving k = 0 block expanded.

    The expansion is arithmetic-level: controlled modular add/subtract blocks are
    still named gates, but the opaque ``ctrl-U^(2^0)`` box is replaced by the
    generic out-of-place modular-multiplication structure.
    """
    retained_rounds = retained_round_schedule(a=a, N=N, n=n)
    if len(retained_rounds) != 1 or retained_rounds[0]["phase_bit"] != 0:
        raise ValueError(
            "The decomposed k=0 layout expects exactly one retained round: phase_bit 0."
        )

    phase = QuantumRegister(1, "phase")
    value = QuantumRegister(n, "value")
    arith = QuantumRegister(n, "arith")
    anc = QuantumRegister(1, "anc")
    bits = ClassicalRegister(1, "c")
    qc = QuantumCircuit(
        phase,
        value,
        arith,
        anc,
        bits,
        name="first_pass_reduced_k0_decomposed",
    )

    qc.x(value[-1])
    qc.h(phase[0])

    k0_block = build_controlled_modular_multiply_k0_block(a=a, N=N, n=n)
    qc.compose(k0_block, [phase[0], *value, *arith, anc[0]], inplace=True)

    qc.h(phase[0])
    qc.measure(phase[0], bits[0])
    qc.metadata = {
        "source_circuit": "build_first_pass_reduced_10q_shor_15_layout",
        "expanded_block": "ctrl-U^(2^0)",
        "expansion_level": "arithmetic",
        "k0_block_metadata": k0_block.metadata,
    }
    return qc


def build_first_pass_reduced_10q_shor_15_gate_only_k0_layout(a=11, N=15, n=4):
    """Build the reduced circuit with the k = 0 block expanded to gates."""
    retained_rounds = retained_round_schedule(a=a, N=N, n=n)
    if len(retained_rounds) != 1 or retained_rounds[0]["phase_bit"] != 0:
        raise ValueError(
            "The gate-only k=0 layout expects exactly one retained round: phase_bit 0."
        )

    phase = QuantumRegister(1, "phase")
    value = QuantumRegister(n, "value")
    arith = QuantumRegister(n, "arith")
    anc = QuantumRegister(1, "anc")
    bits = ClassicalRegister(1, "c")
    qc = QuantumCircuit(
        phase,
        value,
        arith,
        anc,
        bits,
        name="first_pass_reduced_k0_gate_only",
    )

    qc.x(value[-1])
    qc.h(phase[0])

    k0_block = build_controlled_modular_multiply_k0_gate_only_block(a=a, N=N, n=n)
    qc.compose(k0_block, [phase[0], *value, *arith, anc[0]], inplace=True)

    qc.h(phase[0])
    qc.measure(phase[0], bits[0])
    qc.metadata = {
        "source_circuit": "build_first_pass_reduced_10q_shor_15_layout",
        "expanded_block": "ctrl-U^(2^0)",
        "expansion_level": "gate_only",
        "k0_block_metadata": k0_block.metadata,
    }
    return qc


def first_pass_compilation_summary(a=11, N=15, n=4):
    """Split the full round schedule into compiled-away and retained blocks."""
    full_schedule = generic_round_schedule(a=a, N=N, n=n)
    return {
        "compiled_away_as_identity": [
            round_info
            for round_info in full_schedule
            if round_info["compiled_away_in_first_pass"]
        ],
        "retained_after_first_pass": [
            round_info
            for round_info in full_schedule
            if not round_info["compiled_away_in_first_pass"]
        ],
    }


# --- Second half: post compilation helpers ---

DEFAULT_QUANTINUUM_DEVICE_NAME = "H2-1E"
DEFAULT_QUANTINUUM_OPTIMISATION_LEVEL = 0

REQUIRED_TKET_MODULES = (
    "pytket",
    "pytket.extensions.qiskit",
    "pytket.extensions.quantinuum",
)

CONTROL_FLOW_LIKE_OPS = {
    "while_loop",
    "for_loop",
    "switch_case",
    "break_loop",
    "continue_loop",
    "store",
}

TIMING_MODEL_SOURCE_URLS = {
    "h2_native_gates": "https://docs.quantinuum.com/systems/user_guide/hardware_user_guide/h2.html",
    "h2_product_data_sheet": "https://docs.quantinuum.com/h-series/data_sheets/Quantinuum%20H2%20Product%20Data%20Sheet.pdf",
    "quantinuum_timing_faq": "https://docs.quantinuum.com/systems/support/faqs.html",
}

NATIVE_OPERATION_DURATIONS_US = {
    "Rz": 0.0,
    "PhasedX": 10.0,
    "ZZMax": 250.0,
    "ZZPhase": 250.0,
    "Measure": 400.0,
    "Reset": 400.0,
    "Barrier": 0.0,
    "Conditional": 0.0,
    "SetBits": 0.0,
    "CopyBits": 0.0,
    "RangePredicate": 0.0,
    "ExplicitPredicate": 0.0,
    "ClassicalTransform": 0.0,
    "ClExpr": 0.0,
    "WASM": 0.0,
    "Label": 0.0,
    "Goto": 0.0,
    "Branch": 0.0,
    "Stop": 0.0,
    "NoOp": 0.0,
}

NATIVE_OPERATION_DURATION_NOTES = {
    "Rz": "Virtual software-frame operation documented by Quantinuum.",
    "PhasedX": "Explicit lower-bound model input; replace with calibrated H2 timing if available.",
    "ZZMax": "Explicit lower-bound model input; transport/cooling excluded.",
    "ZZPhase": "Explicit lower-bound model input; transport/cooling excluded.",
    "Measure": "Explicit lower-bound model input; full SPAM/shot overhead excluded.",
    "Reset": "Explicit lower-bound model input; full SPAM/shot overhead excluded.",
}

QUANTINUUM_NATIVE_OPS = {"Rz", "PhasedX", "ZZMax", "ZZPhase", "Measure", "Reset"}

ALLOWED_METADATA_OPS = {
    "Barrier",
    "Conditional",
    "SetBits",
    "CopyBits",
    "RangePredicate",
    "ExplicitPredicate",
    "ClassicalTransform",
    "ClExpr",
    "WASM",
    "Label",
    "Goto",
    "Branch",
    "Stop",
    "NoOp",
}

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
SCHEDULED_PHYSICAL_K0_QPY_PATH = ARTIFACT_DIR / "scheduled_physical_k0_qc.qpy"
FULL_STEANE_PHYSICAL_K0_QPY_PATH = ARTIFACT_DIR / "full_steane_physical_k0_circuit.qpy"


def save_qpy_circuit(circuit, path):
    """Save a Qiskit circuit as a QPY artifact and return the artifact path."""
    from qiskit import qpy

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as handle:
        qpy.dump([circuit], handle)
    return artifact_path


def load_qpy_circuit(path):
    """Load the first Qiskit circuit from a QPY artifact."""
    from qiskit import qpy

    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)
    with artifact_path.open("rb") as handle:
        circuits = qpy.load(handle)
    if not circuits:
        raise ValueError(f"No circuits were found in QPY artifact: {artifact_path}")
    return circuits[0]


def save_scheduled_physical_k0_circuit(circuit):
    """Persist the bare ASAP-scheduled K0 handoff circuit."""
    return save_qpy_circuit(circuit, SCHEDULED_PHYSICAL_K0_QPY_PATH)


def load_scheduled_physical_k0_circuit():
    """Load the bare ASAP-scheduled K0 handoff circuit."""
    return load_qpy_circuit(SCHEDULED_PHYSICAL_K0_QPY_PATH)


def require_scheduled_physical_k0_circuit(namespace):
    """Fetch the scheduled handoff circuit from memory or its QPY artifact."""
    if "scheduled_physical_k0_qc" in namespace:
        return namespace["scheduled_physical_k0_qc"]
    try:
        return load_scheduled_physical_k0_circuit()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "scheduled_physical_k0_qc is not defined and no saved QPY artifact was "
            f"found at {SCHEDULED_PHYSICAL_K0_QPY_PATH}. Run "
            "Compilation/shor_15_initial_compilation.ipynb first."
        ) from exc


def save_full_steane_physical_k0_circuit(circuit):
    """Persist the full Steane physical K0 circuit."""
    return save_qpy_circuit(circuit, FULL_STEANE_PHYSICAL_K0_QPY_PATH)


def load_full_steane_physical_k0_circuit():
    """Load the full Steane physical K0 circuit."""
    return load_qpy_circuit(FULL_STEANE_PHYSICAL_K0_QPY_PATH)


def show_table(records, *, index=None):
    """Return a pandas table when pandas is installed, otherwise return records."""
    try:
        import pandas as pd
    except ImportError:
        return records

    table = pd.DataFrame(records)
    if index is None:
        return table
    if index not in table.columns:
        return pd.DataFrame(columns=[index]).set_index(index)
    return table.set_index(index)


def require_full_steane_physical_k0_circuit(namespace):
    """Fetch the final Steane circuit from memory or its QPY artifact."""
    if "full_steane_physical_k0_circuit" in namespace:
        return namespace["full_steane_physical_k0_circuit"]
    try:
        return load_full_steane_physical_k0_circuit()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "full_steane_physical_k0_circuit is not defined and no saved QPY artifact "
            f"was found at {FULL_STEANE_PHYSICAL_K0_QPY_PATH}. Run "
            "Compilation/shor_15_steane_encoding.ipynb first."
        ) from exc


def source_circuit_summary(source_circuit):
    """Summarize the final Steane source circuit before Quantinuum compilation."""
    source_metadata = dict(source_circuit.metadata or {})
    return {
        "name": source_circuit.name,
        "qubits": source_circuit.num_qubits,
        "classical_bits": source_circuit.num_clbits,
        "top_level_operation_counts": dict(source_circuit.count_ops()),
        "logical_qubits": source_metadata.get("logical_qubits"),
        "source_asap_layers": source_metadata.get("source_asap_layers"),
        "source_circuit": source_metadata.get("source_circuit"),
        "k0_parameters": source_metadata.get("k0_parameters"),
        "magic_refresh_counts": source_metadata.get("magic_factory", {}).get("refresh_counts"),
        "maintenance_syndrome_rounds": source_metadata.get("syndrome_policy", {}).get(
            "total_maintenance_rounds"
        ),
    }


def require_tket_quantinuum_dependencies():
    """Import the TKET/Quantinuum dependencies or raise an install message."""
    from importlib import import_module

    missing_tket_modules = []
    for module_name in REQUIRED_TKET_MODULES:
        try:
            import_module(module_name)
        except ImportError:
            missing_tket_modules.append(module_name)

    if missing_tket_modules:
        raise ImportError(
            "Missing TKET/Quantinuum dependencies: "
            f"{missing_tket_modules}. Install them with: "
            "pip install pytket pytket-qiskit pytket-quantinuum"
        )

    qiskit_extension = import_module("pytket.extensions.qiskit")
    quantinuum_extension = import_module("pytket.extensions.quantinuum")
    return {
        "qiskit_to_tk": qiskit_extension.qiskit_to_tk,
        "tk_to_qiskit": qiskit_extension.tk_to_qiskit,
        "H2": quantinuum_extension.H2,
        "QuantinuumBackend": quantinuum_extension.QuantinuumBackend,
    }


def source_control_flow_like_counts(circuit):
    """Count source operations that are likely to block Qiskit-to-TKET conversion."""
    op_counts = dict(circuit.count_ops())
    return {
        op_name: op_counts[op_name]
        for op_name in sorted(CONTROL_FLOW_LIKE_OPS)
        if op_name in op_counts
    }


def make_quantinuum_backend(device_name=DEFAULT_QUANTINUUM_DEVICE_NAME):
    """Build a Quantinuum backend, using packaged H2 data for offline H2 compilation."""
    tket_deps = require_tket_quantinuum_dependencies()
    H2 = tket_deps["H2"]
    QuantinuumBackend = tket_deps["QuantinuumBackend"]

    direct_backend = QuantinuumBackend(device_name)
    try:
        _ = direct_backend.backend_info
        return direct_backend, "Quantinuum device list"
    except Exception as exc:
        if not device_name.startswith("H2-"):
            raise
        print(
            f"{device_name} was not available from the local Quantinuum device list "
            f"({type(exc).__name__}: {exc}). Using packaged H2 target data for "
            "offline compilation."
        )
        return QuantinuumBackend(device_name, data=H2), "packaged pytket-quantinuum H2 data"


def compile_quantinuum_native(
    source_circuit,
    *,
    device_name=DEFAULT_QUANTINUUM_DEVICE_NAME,
    optimisation_level=DEFAULT_QUANTINUUM_OPTIMISATION_LEVEL,
):
    """Compile a Qiskit circuit through the official Qiskit-to-TKET Quantinuum path."""
    tket_deps = require_tket_quantinuum_dependencies()
    qiskit_to_tk = tket_deps["qiskit_to_tk"]
    tk_to_qiskit = tket_deps["tk_to_qiskit"]
    H2 = tket_deps["H2"]

    try:
        tk_circuit = qiskit_to_tk(source_circuit)
    except Exception as exc:
        raise RuntimeError(
            "Qiskit-to-TKET conversion failed for full_steane_physical_k0_circuit. "
            "Control-flow/classical-expression-like source operations detected: "
            f"{source_control_flow_like_counts(source_circuit) or 'none'}. Exact converter "
            f"exception: {type(exc).__name__}: {exc}"
        ) from exc

    backend, backend_data_source = make_quantinuum_backend(device_name)

    try:
        compiled_tk_circuit = backend.get_compiled_circuit(
            tk_circuit,
            optimisation_level=optimisation_level,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Quantinuum TKET compilation failed for {device_name} at "
            f"optimisation_level={optimisation_level}. The exact compiler "
            "exception is attached as the cause."
        ) from exc

    try:
        compiled_qiskit_circuit = tk_to_qiskit(compiled_tk_circuit)
        qiskit_conversion_error = None
    except Exception as exc:
        compiled_qiskit_circuit = None
        qiskit_conversion_error = exc

    compiled_summary = {
        "target": device_name,
        "optimisation_level": optimisation_level,
        "backend_data_source": backend_data_source,
        "target_nominal_qubits": H2.n_qubits,
        "target_nominal_classical_registers": H2.n_cl_reg,
        "source_exceeds_target_nominal_qubits": source_circuit.num_qubits > H2.n_qubits,
        "source_tket_commands": len(tk_circuit.get_commands()),
        "compiled_tket_commands": len(compiled_tk_circuit.get_commands()),
        "compiled_qubits": compiled_tk_circuit.n_qubits,
        "compiled_bits": compiled_tk_circuit.n_bits,
        "converted_back_to_qiskit": compiled_qiskit_circuit is not None,
    }

    return {
        "backend": backend,
        "backend_data_source": backend_data_source,
        "compiled_qiskit_circuit": compiled_qiskit_circuit,
        "compiled_summary": compiled_summary,
        "compiled_tk_circuit": compiled_tk_circuit,
        "qiskit_conversion_error": qiskit_conversion_error,
        "target_data": H2,
        "tk_circuit": tk_circuit,
    }


def duration_table(
    durations_us=NATIVE_OPERATION_DURATIONS_US,
    duration_notes=NATIVE_OPERATION_DURATION_NOTES,
):
    """Build the visible native-operation timing table."""
    return [
        {
            "operation_family": op_name,
            "duration_us": duration_us,
            "note": duration_notes.get(
                op_name,
                "Scheduling/control metadata; zero-time bookkeeping.",
            ),
        }
        for op_name, duration_us in durations_us.items()
    ]


def op_type_name(op) -> str:
    """Return a stable operation type name for TKET operations."""
    op_type = getattr(op, "type", op)
    name = getattr(op_type, "name", None)
    if name is not None:
        return name
    return str(op_type).split(".")[-1]


def underlying_conditional_op(op):
    """Best-effort extraction of a TKET conditional operation's child op."""
    for attr_name in ("op", "op_to_apply", "underlying_op"):
        if hasattr(op, attr_name):
            candidate = getattr(op, attr_name)
            if candidate is not None:
                return candidate
    for method_name in ("get_op", "get_op_to_apply"):
        method = getattr(op, method_name, None)
        if method is not None:
            try:
                candidate = method()
            except TypeError:
                candidate = None
            if candidate is not None:
                return candidate
    return None


def command_family(command) -> tuple[str, str, bool]:
    """Classify a TKET command by raw op, native family, and conditional status."""
    raw_name = op_type_name(command.op)
    if raw_name != "Conditional":
        return raw_name, raw_name, False

    child_op = underlying_conditional_op(command.op)
    if child_op is None:
        return raw_name, raw_name, True
    return raw_name, op_type_name(child_op), True


def command_resources(command) -> tuple[str, ...]:
    """Return qubit/classical resources touched by a TKET command."""
    resources = []

    for qubit in getattr(command, "qubits", ()):
        resources.append(f"q:{qubit}")
    for bit in getattr(command, "bits", ()):
        resources.append(f"c:{bit}")

    if resources:
        return tuple(resources)

    for arg in getattr(command, "args", ()):
        resources.append(str(arg))
    return tuple(resources)


def build_command_records(
    circuit,
    *,
    native_ops=QUANTINUUM_NATIVE_OPS,
    metadata_ops=ALLOWED_METADATA_OPS,
    durations_us=NATIVE_OPERATION_DURATIONS_US,
):
    """Convert TKET commands into resource-analysis records."""
    records = []
    for command_index, command in enumerate(circuit.get_commands()):
        raw_name, family, conditioned = command_family(command)

        if family in native_ops:
            category = "native"
            duration_key = family
        elif family in metadata_ops:
            category = "metadata"
            duration_key = family
        else:
            category = "non_native"
            duration_key = family

        records.append(
            {
                "command_index": command_index,
                "raw_op_type": raw_name,
                "operation_family": family,
                "conditioned": conditioned,
                "category": category,
                "duration_us": durations_us.get(duration_key),
                "resources": command_resources(command),
                "command": str(command),
            }
        )
    return records


def assert_quantinuum_native_command_records(command_records):
    """Assert that all post-compile commands are native or allowed metadata."""
    non_native_records = [
        record for record in command_records if record["category"] == "non_native"
    ]
    if non_native_records:
        raise AssertionError(
            "Quantinuum compilation left non-native operations in the circuit. "
            f"First unresolved operations: {non_native_records[:25]}"
        )

    missing_duration_records = [
        record for record in command_records if record["duration_us"] is None
    ]
    if missing_duration_records:
        raise AssertionError(
            "At least one post-compile operation has no declared duration. "
            f"Add it to NATIVE_OPERATION_DURATIONS_US. First missing operations: "
            f"{missing_duration_records[:25]}"
        )


def native_gate_count_tables(
    command_records,
    *,
    durations_us=NATIVE_OPERATION_DURATIONS_US,
):
    """Count native gate families and scheduling/control metadata families."""
    native_gate_counts = Counter(
        record["operation_family"]
        for record in command_records
        if record["category"] == "native"
    )
    metadata_counts = Counter(
        record["operation_family"]
        for record in command_records
        if record["category"] == "metadata"
    )
    raw_op_counts = Counter(record["raw_op_type"] for record in command_records)

    native_gate_count_table = [
        {
            "operation_family": operation_family,
            "count": count,
            "duration_us": durations_us[operation_family],
            "serial_time_us": count * durations_us[operation_family],
        }
        for operation_family, count in sorted(native_gate_counts.items())
    ]

    metadata_count_table = [
        {"operation_family": operation_family, "count": count}
        for operation_family, count in sorted(metadata_counts.items())
    ]

    return {
        "metadata_count_table": metadata_count_table,
        "metadata_counts": metadata_counts,
        "native_gate_count_table": native_gate_count_table,
        "native_gate_counts": native_gate_counts,
        "raw_op_counts": raw_op_counts,
    }


def asap_schedule_records(records):
    """Schedule command records into earliest non-conflicting moments."""
    next_resource_moment = {}
    scheduled_records = []

    for record in records:
        resources = tuple(record["resources"])
        moment = max(
            (next_resource_moment.get(resource, 0) for resource in resources),
            default=0,
        )

        scheduled_record = dict(record)
        scheduled_record["moment"] = moment
        scheduled_records.append(scheduled_record)

        for resource in resources:
            next_resource_moment[resource] = moment + 1

    return scheduled_records


def build_moment_records(command_records):
    """Build ASAP moment rows and total lower-bound gate-operation time."""
    scheduled_command_records = asap_schedule_records(command_records)

    records_by_moment = defaultdict(list)
    for record in scheduled_command_records:
        records_by_moment[record["moment"]].append(record)

    moment_records = []
    for moment, moment_commands in sorted(records_by_moment.items()):
        duration_us = max(record["duration_us"] for record in moment_commands)
        slowest_ops = sorted(
            {
                record["operation_family"]
                for record in moment_commands
                if record["duration_us"] == duration_us
            }
        )
        moment_op_counts = Counter(
            record["operation_family"] for record in moment_commands
        )
        moment_records.append(
            {
                "moment": moment,
                "duration_us": duration_us,
                "slowest_ops": ", ".join(slowest_ops),
                "command_count": len(moment_commands),
                "operation_counts": dict(sorted(moment_op_counts.items())),
            }
        )

    total_time_us = sum(record["duration_us"] for record in moment_records)
    assert isclose(total_time_us, sum(record["duration_us"] for record in moment_records))

    return scheduled_command_records, moment_records, total_time_us


def analyze_compiled_tket_circuit(compiled_tk_circuit):
    """Run native checks, count gates, and schedule ASAP moments."""
    command_records = build_command_records(compiled_tk_circuit)
    assert_quantinuum_native_command_records(command_records)
    count_tables = native_gate_count_tables(command_records)
    scheduled_command_records, moment_records, total_time_us = build_moment_records(
        command_records
    )

    return {
        **count_tables,
        "command_records": command_records,
        "moment_records": moment_records,
        "scheduled_command_records": scheduled_command_records,
        "total_time_us": total_time_us,
    }


def resource_summary(
    source_circuit,
    compiled_tk_circuit,
    analysis_result,
    compilation_result,
    *,
    device_name=DEFAULT_QUANTINUUM_DEVICE_NAME,
    optimisation_level=DEFAULT_QUANTINUUM_OPTIMISATION_LEVEL,
):
    """Build the final report-ready resource summary."""
    target_data = compilation_result["target_data"]
    command_records = analysis_result["command_records"]
    native_gate_counts = analysis_result["native_gate_counts"]
    metadata_counts = analysis_result["metadata_counts"]
    raw_op_counts = analysis_result["raw_op_counts"]
    moment_records = analysis_result["moment_records"]
    total_time_us = analysis_result["total_time_us"]

    return {
        "input_source": "full_steane_physical_k0_circuit from shor_15_steane_encoding.ipynb",
        "target": device_name,
        "optimisation_level": optimisation_level,
        "backend_data_source": compilation_result["backend_data_source"],
        "target_nominal_qubits": target_data.n_qubits,
        "target_nominal_classical_registers": target_data.n_cl_reg,
        "compiled_exceeds_target_nominal_qubits": compiled_tk_circuit.n_qubits
        > target_data.n_qubits,
        "source_qubits": source_circuit.num_qubits,
        "source_classical_bits": source_circuit.num_clbits,
        "compiled_qubits": compiled_tk_circuit.n_qubits,
        "compiled_classical_bits": compiled_tk_circuit.n_bits,
        "compiled_commands": len(command_records),
        "native_gate_counts": dict(sorted(native_gate_counts.items())),
        "metadata_counts": dict(sorted(metadata_counts.items())),
        "raw_operation_counts": dict(sorted(raw_op_counts.items())),
        "native_moments": len(moment_records),
        "total_gate_operation_time_us": total_time_us,
        "total_gate_operation_time_ms": total_time_us / 1_000,
        "total_gate_operation_time_s": total_time_us / 1_000_000,
        "timing_scope": (
            "Native-gate lower bound; excludes transport, cooling, queueing, Nexus overhead, "
            "and complete hardware shot overhead."
        ),
        "timing_model_sources": TIMING_MODEL_SOURCE_URLS,
        "faithfulness_rule": (
            "Uses public a=11, N=15 arithmetic simplifications only; does not use factors "
            "or a pre-known measured period."
        ),
    }
