"""Steane-code encoding and syndrome helpers for the Shor-15 compilation pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from math import pi

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from . import shor_compilation

STEANE_BLOCK_SIZE = 7
STEANE_SYNDROME_ANCILLA_COUNT = 3
STEANE_MAGIC_INPUT_INDEX = 6
STEANE_MAGIC_PLUS_INDICES = (1, 2, 3)
STEANE_MAGIC_DATA_CNOTS = (
    (6, 5),
    (1, 0),
    (2, 4),
    (3, 5),
    (2, 0),
    (6, 4),
    (2, 6),
    (3, 4),
    (1, 5),
    (1, 6),
    (3, 0),
)
PAPER_STEANE_H_INDICES = (0, 5, 6)
PAPER_STEANE_DATA_CNOTS = (
    (5, 4),
    (0, 1),
    (6, 3),
    (5, 2),
    (6, 4),
    (0, 3),
    (5, 1),
    (3, 2),
)
PAPER_STEANE_VERIFY_CONTROLS = (4, 1, 3)
STEANE_SYNDROME_DATA_QUBITS = (
    (1, 3, 4, 6),
    (0, 3, 5, 6),
    (2, 4, 5, 6),
)

__all__ = [
    "STEANE_BLOCK_SIZE",
    "append_steane_logical_cnot",
    "append_steane_logical_h",
    "append_steane_logical_s",
    "append_steane_logical_sdg",
    "append_steane_logical_t",
    "append_steane_logical_tdg",
    "append_steane_syndrome_extraction",
    "append_steane_magic_factory",
    "append_steane_magic_injection",
    "append_steane_logical_x",
    "append_steane_logical_z",
    "append_steane_zero_prep",
    "build_first_pass_explicit_steane_zero_initialization",
    "build_first_pass_steane_physical_k0_circuit",
    "build_steane_physical_circuit_from_logical",
    "build_steane_magic_factory",
    "build_steane_zero_prep",
    "build_first_pass_steane_zero_initialization",
]


def _validated_block(data_block: Sequence) -> tuple:
    data = tuple(data_block)
    if len(data) != STEANE_BLOCK_SIZE:
        raise ValueError(
            f"Steane preparation expects exactly {STEANE_BLOCK_SIZE} data qubits, "
            f"received {len(data)}."
        )
    return data


def _validated_attempt_resources(resources: Sequence, expected_size: int, resource_name: str) -> tuple:
    validated = tuple(resources)
    if len(validated) != expected_size:
        raise ValueError(
            f"Steane preparation expects exactly {expected_size} {resource_name}, "
            f"received {len(validated)}."
        )
    return validated


def _validated_magic_output_kind(output_kind: str) -> str:
    normalized = output_kind.lower()
    if normalized not in {"h", "t", "tdg"}:
        raise ValueError("output_kind must be one of 'h', 't', or 'tdg'.")
    return normalized


def _validated_magic_gate(gate: str) -> str:
    normalized = gate.lower()
    if normalized not in {"t", "tdg"}:
        raise ValueError("gate must be one of 't' or 'tdg'.")
    return normalized


def _source_bit_label(circuit: QuantumCircuit, bit) -> str:
    location = circuit.find_bit(bit)
    if location.registers:
        register, index = location.registers[0]
        return f"{register.name}[{index}]"
    return f"bit[{location.index}]"


def _single_classical_register_locations(
    circuit: QuantumCircuit,
    bits: Sequence,
) -> tuple[ClassicalRegister, tuple[int, ...]]:
    locations = []
    for bit in bits:
        bit_locations = [
            (register, index)
            for register, index in circuit.find_bit(bit).registers
            if isinstance(register, ClassicalRegister)
        ]
        if not bit_locations:
            raise ValueError("Parity-conditioned correction bits must belong to a classical register.")
        locations.append(bit_locations[0])

    register = locations[0][0]
    if any(location_register != register for location_register, _ in locations):
        raise ValueError(
            "Parity-conditioned corrections require all parity bits to come from "
            "one classical register so Qiskit-to-TKET conversion can use register "
            "conditions instead of unsupported runtime Store expressions."
        )

    return register, tuple(index for _, index in locations)


def _odd_parity_register_values(register_size: int, parity_indices: Sequence[int]) -> list[int]:
    return [
        register_value
        for register_value in range(1 << register_size)
        if sum((register_value >> bit_index) & 1 for bit_index in parity_indices) % 2
    ]


def _append_parity_conditioned_logical_correction(
    circuit: QuantumCircuit,
    data_block: Sequence,
    parity_bits: Sequence,
    *,
    correction: str,
) -> list[int]:
    """Append a logical correction controlled by odd parity of classical bits.

    The Qiskit-to-TKET converter used by the final notebook cannot convert
    Qiskit runtime ``Store`` instructions or expression-valued ``if_test``
    conditions.  Register-equality conditions do convert, so this helper emits
    one supported branch for each register value whose selected bits have odd
    parity.
    """
    data = _validated_block(data_block)
    register, parity_indices = _single_classical_register_locations(circuit, parity_bits)
    correction_values = _odd_parity_register_values(len(register), parity_indices)

    for register_value in correction_values:
        with circuit.if_test((register, register_value)):
            if correction == "s":
                append_steane_logical_s(circuit, data)
            elif correction == "sdg":
                append_steane_logical_sdg(circuit, data)
            else:
                raise ValueError("correction must be one of 's' or 'sdg'.")

    return correction_values


def _append_steane_logical_z_measurement(
    circuit: QuantumCircuit,
    data_block: Sequence,
    measure_bits: Sequence,
) -> None:
    data = _validated_block(data_block)
    bits = _validated_attempt_resources(measure_bits, STEANE_BLOCK_SIZE, "logical measurement bits")

    for qubit, bit in zip(data, bits, strict=True):
        circuit.measure(qubit, bit)


def _apply_paper_single_attempt(
    circuit: QuantumCircuit,
    data_block: Sequence,
    ancilla,
    verify_bit=None,
) -> None:
    """Apply one paper-faithful Steane |0_L> preparation attempt."""
    data = _validated_block(data_block)

    circuit.barrier(*data, ancilla)
    for qubit in data:
        circuit.reset(qubit)
    circuit.reset(ancilla)
    circuit.barrier(*data, ancilla)

    for qubit_index in PAPER_STEANE_H_INDICES:
        circuit.h(data[qubit_index])

    for control_index, target_index in PAPER_STEANE_DATA_CNOTS:
        circuit.cx(data[control_index], data[target_index])

    circuit.barrier(ancilla, data[1], data[3], data[5])
    for control_index in PAPER_STEANE_VERIFY_CONTROLS:
        circuit.cx(data[control_index], ancilla)

    if verify_bit is not None:
        circuit.measure(ancilla, verify_bit)


def _append_transversal_single_qubit_gate(
    circuit: QuantumCircuit,
    data_block: Sequence,
    gate_name: str,
) -> QuantumCircuit:
    data = _validated_block(data_block)
    gate = getattr(circuit, gate_name)
    for qubit in data:
        gate(qubit)
    return circuit


def append_steane_logical_h(
    circuit: QuantumCircuit,
    data_block: Sequence,
) -> QuantumCircuit:
    """Append a transversal logical H to one Steane data block."""
    return _append_transversal_single_qubit_gate(circuit, data_block, "h")


def append_steane_logical_x(
    circuit: QuantumCircuit,
    data_block: Sequence,
) -> QuantumCircuit:
    """Append a transversal logical X to one Steane data block."""
    return _append_transversal_single_qubit_gate(circuit, data_block, "x")


def append_steane_logical_z(
    circuit: QuantumCircuit,
    data_block: Sequence,
) -> QuantumCircuit:
    """Append a transversal logical Z to one Steane data block."""
    return _append_transversal_single_qubit_gate(circuit, data_block, "z")


def append_steane_logical_s(
    circuit: QuantumCircuit,
    data_block: Sequence,
) -> QuantumCircuit:
    """Append a transversal logical S to one Steane data block.

    The Steane code convention used here has the physical phase gate conjugated:
    logical S is implemented by applying physical Sdg to every data qubit.
    """
    return _append_transversal_single_qubit_gate(circuit, data_block, "sdg")


def append_steane_logical_sdg(
    circuit: QuantumCircuit,
    data_block: Sequence,
) -> QuantumCircuit:
    """Append a transversal logical Sdg to one Steane data block.

    The Steane code convention used here has the physical phase gate conjugated:
    logical Sdg is implemented by applying physical S to every data qubit.
    """
    return _append_transversal_single_qubit_gate(circuit, data_block, "s")


def append_steane_logical_cnot(
    circuit: QuantumCircuit,
    control_data_block: Sequence,
    target_data_block: Sequence,
) -> QuantumCircuit:
    """Append a transversal logical CNOT between two Steane data blocks."""
    controls = _validated_block(control_data_block)
    targets = _validated_block(target_data_block)

    for control, target in zip(controls, targets, strict=True):
        circuit.cx(control, target)

    return circuit


def _append_steane_encoded_h_magic_state(
    circuit: QuantumCircuit,
    data_block: Sequence,
) -> None:
    """Prepare the paper's H-eigenstate magic state encoded in one Steane block."""
    data = _validated_block(data_block)

    circuit.barrier(*data)
    for qubit in data:
        circuit.reset(qubit)
    circuit.barrier(*data)

    circuit.ry(pi / 4, data[STEANE_MAGIC_INPUT_INDEX])
    for qubit_index in STEANE_MAGIC_PLUS_INDICES:
        circuit.h(data[qubit_index])

    for control_index, target_index in STEANE_MAGIC_DATA_CNOTS:
        circuit.cx(data[control_index], data[target_index])


def _append_cat_state(
    circuit: QuantumCircuit,
    cat_block: Sequence,
) -> None:
    cat = _validated_block(cat_block)
    for qubit in cat:
        circuit.reset(qubit)

    circuit.h(cat[0])
    for qubit in cat[1:]:
        circuit.cx(cat[0], qubit)


def append_steane_magic_factory(
    circuit: QuantumCircuit,
    magic_block: Sequence,
    cat_block: Sequence,
    h_test_bits: Sequence,
    *,
    syndrome_ancilla_bank: Sequence | None = None,
    z_syndrome_bits: Sequence | None = None,
    x_syndrome_bits: Sequence | None = None,
    output_kind: str = "t",
) -> QuantumCircuit:
    """Append a Steane-native magic-state factory.

    The factory creates the paper's H-eigenstate magic state, verifies it by a
    transversal logical-H Hadamard test, and optionally records one Steane
    syndrome-extraction round. The output remains a Steane block.
    """
    output_kind = _validated_magic_output_kind(output_kind)
    magic = _validated_block(magic_block)
    cat = _validated_block(cat_block)
    h_bits = _validated_attempt_resources(h_test_bits, STEANE_BLOCK_SIZE, "H-test bits")

    syndrome_resources = (syndrome_ancilla_bank, z_syndrome_bits, x_syndrome_bits)
    include_syndrome = any(resource is not None for resource in syndrome_resources)
    if include_syndrome and any(resource is None for resource in syndrome_resources):
        raise ValueError(
            "syndrome_ancilla_bank, z_syndrome_bits, and x_syndrome_bits must "
            "all be provided together."
        )

    _append_steane_encoded_h_magic_state(circuit, magic)
    _append_cat_state(circuit, cat)

    circuit.barrier(*magic, *cat)
    for control, target in zip(cat, magic, strict=True):
        circuit.ch(control, target)

    for qubit, bit in zip(cat, h_bits, strict=True):
        circuit.h(qubit)
        circuit.measure(qubit, bit)

    if include_syndrome:
        append_steane_syndrome_extraction(
            circuit,
            magic,
            syndrome_ancilla_bank,
            z_syndrome_bits,
            x_syndrome_bits,
        )

    if output_kind == "t":
        append_steane_logical_sdg(circuit, magic)
        append_steane_logical_h(circuit, magic)
    elif output_kind == "tdg":
        append_steane_logical_s(circuit, magic)
        append_steane_logical_h(circuit, magic)

    return circuit


def build_steane_magic_factory(
    output_kind: str = "t",
    *,
    include_syndrome: bool = True,
) -> QuantumCircuit:
    """Build a standalone Steane-native magic-state factory circuit."""
    output_kind = _validated_magic_output_kind(output_kind)

    magic = QuantumRegister(STEANE_BLOCK_SIZE, "magic")
    cat = QuantumRegister(STEANE_BLOCK_SIZE, "magic_cat")
    h_test = ClassicalRegister(STEANE_BLOCK_SIZE, "magic_h_test")

    registers = [magic, cat]
    syndrome = None
    z_syndrome = None
    x_syndrome = None
    if include_syndrome:
        syndrome = QuantumRegister(STEANE_SYNDROME_ANCILLA_COUNT, "magic_syndrome")
        z_syndrome = ClassicalRegister(STEANE_SYNDROME_ANCILLA_COUNT, "magic_z_syndrome")
        x_syndrome = ClassicalRegister(STEANE_SYNDROME_ANCILLA_COUNT, "magic_x_syndrome")
        registers.append(syndrome)
    registers.append(h_test)
    if include_syndrome:
        registers.extend((z_syndrome, x_syndrome))

    circuit = QuantumCircuit(*registers, name=f"steane_magic_factory_{output_kind}")
    append_steane_magic_factory(
        circuit,
        magic,
        cat,
        h_test,
        syndrome_ancilla_bank=syndrome,
        z_syndrome_bits=z_syndrome,
        x_syndrome_bits=x_syndrome,
        output_kind=output_kind,
    )

    circuit.metadata = {
        "factory": "steane_zero_level_magic",
        "output_kind": output_kind,
        "steane_block_size": STEANE_BLOCK_SIZE,
        "h_magic_state": "plus-one eigenstate of logical H verified by transversal H test",
        "h_test_acceptance": "accept when parity of magic_h_test[0:7] is even",
        "syndrome_acceptance": (
            "accept when magic_z_syndrome and magic_x_syndrome are all zero"
            if include_syndrome
            else None
        ),
        "surface_code_transfer": False,
        "include_syndrome": include_syndrome,
    }
    return circuit


def append_steane_magic_injection(
    circuit: QuantumCircuit,
    data_block: Sequence,
    magic_block: Sequence,
    measure_bits: Sequence,
    *,
    gate: str = "t",
) -> QuantumCircuit:
    """Inject a Steane-encoded magic state into a Steane data block."""
    gate = _validated_magic_gate(gate)
    data = _validated_block(data_block)
    magic = _validated_block(magic_block)
    bits = _validated_attempt_resources(measure_bits, STEANE_BLOCK_SIZE, "magic measurement bits")

    append_steane_logical_cnot(circuit, data, magic)
    for qubit, bit in zip(magic, bits, strict=True):
        circuit.measure(qubit, bit)

    _append_parity_conditioned_logical_correction(
        circuit,
        data,
        bits[4:7],
        correction="s" if gate == "t" else "sdg",
    )

    return circuit


def append_steane_logical_t(
    circuit: QuantumCircuit,
    data_block: Sequence,
    magic_block: Sequence,
    measure_bits: Sequence,
) -> QuantumCircuit:
    """Append logical T through Steane magic-state injection."""
    return append_steane_magic_injection(
        circuit,
        data_block,
        magic_block,
        measure_bits,
        gate="t",
    )


def append_steane_logical_tdg(
    circuit: QuantumCircuit,
    data_block: Sequence,
    magic_block: Sequence,
    measure_bits: Sequence,
) -> QuantumCircuit:
    """Append logical Tdg through Steane magic-state injection."""
    return append_steane_magic_injection(
        circuit,
        data_block,
        magic_block,
        measure_bits,
        gate="tdg",
    )


def append_steane_syndrome_extraction(
    circuit: QuantumCircuit,
    data_block: Sequence,
    ancilla_bank: Sequence,
    z_syndrome_bits: Sequence,
    x_syndrome_bits: Sequence,
) -> QuantumCircuit:
    """Append one Steane syndrome-extraction round with a reused 3-ancilla bank."""
    data = _validated_block(data_block)
    ancillas = _validated_attempt_resources(
        ancilla_bank,
        STEANE_SYNDROME_ANCILLA_COUNT,
        "syndrome ancillas",
    )
    z_bits = _validated_attempt_resources(
        z_syndrome_bits,
        STEANE_SYNDROME_ANCILLA_COUNT,
        "Z syndrome bits",
    )
    x_bits = _validated_attempt_resources(
        x_syndrome_bits,
        STEANE_SYNDROME_ANCILLA_COUNT,
        "X syndrome bits",
    )

    for ancilla in ancillas:
        circuit.reset(ancilla)

    for ancilla in ancillas:
        circuit.h(ancilla)

    for ancilla, data_indices in zip(ancillas, STEANE_SYNDROME_DATA_QUBITS, strict=True):
        for data_index in data_indices:
            circuit.cx(ancilla, data[data_index])

    for ancilla in ancillas:
        circuit.h(ancilla)

    for ancilla, syndrome_bit in zip(ancillas, z_bits, strict=True):
        circuit.measure(ancilla, syndrome_bit)

    for ancilla in ancillas:
        circuit.reset(ancilla)

    for ancilla, data_indices in zip(ancillas, STEANE_SYNDROME_DATA_QUBITS, strict=True):
        for data_index in data_indices:
            circuit.cx(data[data_index], ancilla)

    for ancilla, syndrome_bit in zip(ancillas, x_bits, strict=True):
        circuit.measure(ancilla, syndrome_bit)

    for ancilla in ancillas:
        circuit.reset(ancilla)

    return circuit


def append_steane_zero_prep(
    circuit: QuantumCircuit,
    data_block: Sequence,
    ancilla,
    verify_bit=None,
    *,
    max_attempts: int = 3,
) -> QuantumCircuit:
    """Append the Quantinuum paper's verified Steane |0_L> preparation gadget.

    The implementation follows the local paper QASM assets and supports the same
    repeat-until-success structure used there: the first attempt is unconditional,
    and later attempts only execute if the previous ancilla measurement was `1`.

    Args:
        circuit: Circuit to mutate in place.
        data_block: Seven physical qubits that store one logical Steane block.
        ancilla: Verification ancilla used by the paper circuit.
        verify_bit: Classical bit that stores the verification outcome. Required
            when `max_attempts > 1`.
        max_attempts: Maximum number of preparation attempts. The paper uses `3`.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    if max_attempts > 1 and verify_bit is None:
        raise ValueError("verify_bit is required when max_attempts > 1.")

    _apply_paper_single_attempt(circuit, data_block, ancilla, verify_bit)

    if verify_bit is None:
        return circuit

    for _ in range(max_attempts - 1):
        with circuit.if_test((verify_bit, 1)):
            _apply_paper_single_attempt(circuit, data_block, ancilla, verify_bit)

    return circuit


def _append_explicit_steane_zero_prep(
    circuit: QuantumCircuit,
    data_block: Sequence,
    ancilla_bank: Sequence,
    verify_bits: Sequence,
) -> QuantumCircuit:
    data = _validated_block(data_block)
    ancillas = _validated_attempt_resources(ancilla_bank, len(verify_bits), "attempt ancillas")

    _apply_paper_single_attempt(circuit, data, ancillas[0], verify_bits[0])

    for attempt_index in range(1, len(ancillas)):
        with circuit.if_test((verify_bits[attempt_index - 1], 1)):
            _apply_paper_single_attempt(circuit, data, ancillas[attempt_index], verify_bits[attempt_index])

    return circuit


def build_steane_zero_prep(
    num_logical_qubits: int,
    *,
    max_attempts: int = 3,
    measure_ancilla: bool = True,
) -> QuantumCircuit:
    """Build a generic logical-zero Steane initializer for `num_logical_qubits`.

    Each logical qubit gets its own 7-qubit Steane data block. A single ancilla is
    reused across blocks, since the preparation is done during initialization.
    """
    if num_logical_qubits < 1:
        raise ValueError("num_logical_qubits must be at least 1.")
    if max_attempts > 1 and not measure_ancilla:
        raise ValueError("measure_ancilla must be True when max_attempts > 1.")

    logical_blocks = [
        QuantumRegister(STEANE_BLOCK_SIZE, f"logical_{block_index}")
        for block_index in range(num_logical_qubits)
    ]
    prep_ancilla = QuantumRegister(1, "prep_ancilla")

    registers = [*logical_blocks, prep_ancilla]
    init_flag = None
    if measure_ancilla:
        init_flag = ClassicalRegister(num_logical_qubits, "init_flag")
        registers.append(init_flag)

    circuit = QuantumCircuit(*registers, name="steane_zero_init")

    for block_index, logical_block in enumerate(logical_blocks):
        verify_bit = None if init_flag is None else init_flag[block_index]
        append_steane_zero_prep(
            circuit,
            logical_block,
            prep_ancilla[0],
            verify_bit,
            max_attempts=max_attempts,
        )

    circuit.metadata = {
        "logical_qubits": num_logical_qubits,
        "steane_block_size": STEANE_BLOCK_SIZE,
        "shared_prep_ancilla": True,
        "max_attempts": max_attempts,
        "measure_ancilla": measure_ancilla,
    }
    return circuit


def build_first_pass_explicit_steane_zero_initialization(
    a: int = 11,
    N: int = 15,
    n: int = 4,
    *,
    max_attempts: int = 3,
) -> QuantumCircuit:
    """Show every first-pass Steane |0_L> preparation with explicit ancilla banks.

    Unlike `build_first_pass_steane_zero_initialization`, this helper does not reuse
    a single verification ancilla across the whole circuit. Each logical block gets
    one ancilla per allowed preparation attempt, making the full state-preparation
    footprint explicit in the resulting circuit diagram.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    first_pass_qc = shor_compilation.build_first_pass_reduced_10q_shor_15_layout(a=a, N=N, n=n)
    logical_blocks = []
    prep_ancilla_banks = []
    init_flag_registers = []
    block_map = []
    circuit_registers = []

    for qreg in first_pass_qc.qregs:
        for block_index in range(len(qreg)):
            logical_reg = QuantumRegister(STEANE_BLOCK_SIZE, f"{qreg.name}_{block_index}")
            prep_reg = QuantumRegister(max_attempts, f"{qreg.name}_{block_index}_prep")
            init_reg = ClassicalRegister(max_attempts, f"{qreg.name}_{block_index}_init")
            logical_blocks.append(logical_reg)
            prep_ancilla_banks.append(prep_reg)
            init_flag_registers.append(init_reg)
            circuit_registers.extend((logical_reg, prep_reg, init_reg))
            block_map.append(
                {
                    "source_qubit": f"{qreg.name}[{block_index}]",
                    "logical_block_register": logical_reg.name,
                    "prep_ancilla_register": prep_reg.name,
                    "verification_flag_register": init_reg.name,
                }
            )

    circuit = QuantumCircuit(*circuit_registers, name="first_pass_steane_zero_explicit_init")

    for logical_block, prep_ancillas, init_flags in zip(
        logical_blocks,
        prep_ancilla_banks,
        init_flag_registers,
        strict=True,
    ):
        _append_explicit_steane_zero_prep(circuit, logical_block, prep_ancillas, init_flags)

    circuit.metadata = {
        "source_circuit": "build_first_pass_reduced_10q_shor_15_layout",
        "source_qubits": first_pass_qc.num_qubits,
        "source_register_layout": {qreg.name: len(qreg) for qreg in first_pass_qc.qregs},
        "logical_qubits": len(logical_blocks),
        "steane_block_size": STEANE_BLOCK_SIZE,
        "prep_ancillas_per_logical_qubit": max_attempts,
        "physical_qubits_per_logical_qubit": STEANE_BLOCK_SIZE + max_attempts,
        "max_attempts": max_attempts,
        "block_map": block_map,
    }
    return circuit


def build_first_pass_steane_zero_initialization(
    a: int = 11,
    N: int = 15,
    n: int = 4,
    *,
    max_attempts: int = 3,
) -> QuantumCircuit:
    """Prepare every qubit of `first_pass_qc` in Steane logical |0_L>.

    This is the initialization layer only. The original reduced Shor circuit later
    applies nontrivial logical operations on top of these logical-zero blocks.
    """
    first_pass_qc = shor_compilation.build_first_pass_reduced_10q_shor_15_layout(a=a, N=N, n=n)
    logical_blocks = []
    block_map = []

    for qreg in first_pass_qc.qregs:
        for block_index in range(len(qreg)):
            logical_reg = QuantumRegister(STEANE_BLOCK_SIZE, f"{qreg.name}_{block_index}")
            logical_blocks.append(logical_reg)
            block_map.append(
                {
                    "source_qubit": f"{qreg.name}[{block_index}]",
                    "logical_block_register": logical_reg.name,
                }
            )

    prep_ancilla = QuantumRegister(1, "prep_ancilla")
    init_flag = ClassicalRegister(len(logical_blocks), "init_flag")
    circuit = QuantumCircuit(*logical_blocks, prep_ancilla, init_flag, name="first_pass_steane_zero_init")

    for block_index, logical_block in enumerate(logical_blocks):
        append_steane_zero_prep(
            circuit,
            logical_block,
            prep_ancilla[0],
            init_flag[block_index],
            max_attempts=max_attempts,
        )

    circuit.metadata = {
        "source_circuit": "build_first_pass_reduced_10q_shor_15_layout",
        "source_qubits": first_pass_qc.num_qubits,
        "source_register_layout": {qreg.name: len(qreg) for qreg in first_pass_qc.qregs},
        "logical_qubits": len(logical_blocks),
        "steane_block_size": STEANE_BLOCK_SIZE,
        "shared_prep_ancilla": True,
        "max_attempts": max_attempts,
        "block_map": block_map,
    }
    return circuit


def _source_register_block_name(qreg_name: str, qreg_index: int) -> str:
    return f"{qreg_name}_{qreg_index}"


def _safe_identifier(label: str) -> str:
    identifier = "".join(character if character.isalnum() else "_" for character in label).strip("_")
    return identifier or "bit"


def _source_clbit_register_name(circuit: QuantumCircuit, clbit, suffix: str) -> str:
    location = circuit.find_bit(clbit)
    if location.registers:
        register, index = location.registers[0]
        return f"{register.name}_{index}_{suffix}"
    return f"clbit_{location.index}_{suffix}"


def _build_explicit_steane_zero_initialization_for_logical_circuit(
    logical_circuit: QuantumCircuit,
    *,
    max_attempts: int,
) -> tuple[QuantumCircuit, list[QuantumRegister], list[QuantumRegister], list[dict]]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    logical_blocks = [None] * logical_circuit.num_qubits
    prep_ancilla_banks = [None] * logical_circuit.num_qubits
    init_flag_registers = [None] * logical_circuit.num_qubits
    block_map = []
    circuit_registers = []

    for qreg in logical_circuit.qregs:
        for qreg_index, qubit in enumerate(qreg):
            source_index = logical_circuit.find_bit(qubit).index
            block_name = _source_register_block_name(qreg.name, qreg_index)
            logical_reg = QuantumRegister(STEANE_BLOCK_SIZE, block_name)
            prep_reg = QuantumRegister(max_attempts, f"{block_name}_prep")
            init_reg = ClassicalRegister(max_attempts, f"{block_name}_init")

            logical_blocks[source_index] = logical_reg
            prep_ancilla_banks[source_index] = prep_reg
            init_flag_registers[source_index] = init_reg
            circuit_registers.extend((logical_reg, prep_reg, init_reg))
            block_map.append(
                {
                    "source_qubit": _source_bit_label(logical_circuit, qubit),
                    "source_qubit_index": source_index,
                    "logical_block_register": logical_reg.name,
                    "prep_ancilla_register": prep_reg.name,
                    "verification_flag_register": init_reg.name,
                }
            )

    if any(block is None for block in logical_blocks):
        raise ValueError("Every source qubit must belong to a named quantum register.")

    circuit = QuantumCircuit(*circuit_registers, name="steane_physical_from_logical")
    for logical_block, prep_ancillas, init_flags in zip(
        logical_blocks,
        prep_ancilla_banks,
        init_flag_registers,
        strict=True,
    ):
        _append_explicit_steane_zero_prep(circuit, logical_block, prep_ancillas, init_flags)

    return circuit, logical_blocks, prep_ancilla_banks, block_map


def _append_initial_logical_gate(
    circuit: QuantumCircuit,
    logical_blocks: Sequence,
    operation_name: str,
    source_qubit_index: int,
) -> None:
    if source_qubit_index < 0 or source_qubit_index >= len(logical_blocks):
        raise ValueError(f"Initial logical gate targets unknown source qubit {source_qubit_index}.")

    block = logical_blocks[source_qubit_index]
    if operation_name == "h":
        append_steane_logical_h(circuit, block)
    elif operation_name == "x":
        append_steane_logical_x(circuit, block)
    elif operation_name == "z":
        append_steane_logical_z(circuit, block)
    elif operation_name == "s":
        append_steane_logical_s(circuit, block)
    elif operation_name == "sdg":
        append_steane_logical_sdg(circuit, block)
    else:
        raise ValueError(f"Unsupported initial logical operation '{operation_name}'.")


def _append_encoded_barrier(
    circuit: QuantumCircuit,
    source_circuit: QuantumCircuit,
    logical_blocks: Sequence,
    instruction,
) -> None:
    source_indices = [
        source_circuit.find_bit(qubit).index
        for qubit in instruction.qubits
    ]
    if not source_indices:
        source_indices = list(range(len(logical_blocks)))

    barrier_qubits = []
    for source_index in source_indices:
        barrier_qubits.extend(logical_blocks[source_index])

    if barrier_qubits:
        circuit.barrier(*barrier_qubits)


def _group_instructions_by_asap_layer(source_circuit: QuantumCircuit) -> dict[int, list]:
    schedule_metadata = (source_circuit.metadata or {}).get("asap_schedule")
    if schedule_metadata is None:
        raise ValueError("The logical source circuit must include asap_schedule metadata.")

    instruction_layers = schedule_metadata.get("instruction_layers", [])
    if len(instruction_layers) != len(source_circuit.data):
        raise ValueError("ASAP schedule metadata does not match the logical source circuit data.")

    layers = {}
    for layer_metadata, instruction in zip(instruction_layers, source_circuit.data, strict=True):
        layer = layer_metadata["scheduled_layer"]
        layers.setdefault(layer, []).append(instruction)
    return layers


def build_steane_physical_circuit_from_logical(
    logical_circuit: QuantumCircuit,
    *,
    max_attempts: int = 3,
    initial_logical_gates: Sequence[tuple[str, int]] = (),
    skip_source_gates: Sequence[tuple[str, int]] = (),
) -> QuantumCircuit:
    """Expand an ASAP-scheduled logical circuit into Steane physical gadgets.

    The source circuit is interpreted as one logical qubit per source qubit. Each
    source qubit is initialized as an explicit Steane block with three reusable
    ancillas, then scheduled logical operations are expanded layer by layer. In
    each non-barrier ASAP layer, inactive logical blocks receive one syndrome
    extraction round using their reused prep ancillas.
    """
    if max_attempts < STEANE_SYNDROME_ANCILLA_COUNT:
        raise ValueError(
            "The full physical builder reuses prep ancillas for syndrome extraction "
            f"and therefore requires at least {STEANE_SYNDROME_ANCILLA_COUNT} attempts."
        )

    source_circuit = logical_circuit
    if (source_circuit.metadata or {}).get("asap_schedule") is None:
        source_circuit = shor_compilation.asap_schedule_circuit(source_circuit)

    circuit, logical_blocks, prep_ancilla_banks, block_map = (
        _build_explicit_steane_zero_initialization_for_logical_circuit(
            source_circuit,
            max_attempts=max_attempts,
        )
    )

    initial_gate_metadata = []
    for operation_name, source_qubit_index in initial_logical_gates:
        normalized_operation = operation_name.lower()
        _append_initial_logical_gate(
            circuit,
            logical_blocks,
            normalized_operation,
            source_qubit_index,
        )
        initial_gate_metadata.append(
            {
                "operation": normalized_operation,
                "source_qubit_index": source_qubit_index,
                "source_qubit": _source_bit_label(source_circuit, source_circuit.qubits[source_qubit_index]),
            }
        )

    data_z_syndrome_bits = []
    data_x_syndrome_bits = []
    for block_entry in block_map:
        block_name = block_entry["logical_block_register"]
        z_register = ClassicalRegister(STEANE_SYNDROME_ANCILLA_COUNT, f"{block_name}_z_syndrome")
        x_register = ClassicalRegister(STEANE_SYNDROME_ANCILLA_COUNT, f"{block_name}_x_syndrome")
        circuit.add_register(z_register)
        circuit.add_register(x_register)
        data_z_syndrome_bits.append(z_register)
        data_x_syndrome_bits.append(x_register)
        block_entry["z_syndrome_register"] = z_register.name
        block_entry["x_syndrome_register"] = x_register.name

    magic = QuantumRegister(STEANE_BLOCK_SIZE, "magic")
    magic_cat = QuantumRegister(STEANE_BLOCK_SIZE, "magic_cat")
    magic_syndrome = QuantumRegister(STEANE_SYNDROME_ANCILLA_COUNT, "magic_syndrome")
    magic_h_test = ClassicalRegister(STEANE_BLOCK_SIZE, "magic_h_test")
    magic_z_syndrome = ClassicalRegister(STEANE_SYNDROME_ANCILLA_COUNT, "magic_z_syndrome")
    magic_x_syndrome = ClassicalRegister(STEANE_SYNDROME_ANCILLA_COUNT, "magic_x_syndrome")
    magic_measure_rest = ClassicalRegister(STEANE_BLOCK_SIZE - 3, "magic_measure_rest")
    magic_measure = ClassicalRegister(3, "magic_measure")
    for register in (
        magic,
        magic_cat,
        magic_syndrome,
        magic_h_test,
        magic_z_syndrome,
        magic_x_syndrome,
        magic_measure_rest,
        magic_measure,
    ):
        circuit.add_register(register)
    magic_measure_bits = [*magic_measure_rest, *magic_measure]

    readout_bits_by_clbit = {}
    readout_metadata = []
    for clbit in source_circuit.clbits:
        source_clbit_index = source_circuit.find_bit(clbit).index
        readout_register = ClassicalRegister(
            STEANE_BLOCK_SIZE,
            _source_clbit_register_name(source_circuit, clbit, "readout"),
        )
        circuit.add_register(readout_register)
        readout_bits_by_clbit[source_clbit_index] = readout_register
        readout_metadata.append(
            {
                "source_clbit": _source_bit_label(source_circuit, clbit),
                "source_clbit_index": source_clbit_index,
                "readout_register": readout_register.name,
            }
        )

    skip_counts = {}
    for operation_name, source_qubit_index in skip_source_gates:
        skip_key = (operation_name.lower(), source_qubit_index)
        skip_counts[skip_key] = skip_counts.get(skip_key, 0) + 1

    layers = _group_instructions_by_asap_layer(source_circuit)
    layer_summaries = []
    skipped_source_gates = []
    expanded_operation_counts = {}
    magic_refresh_counts = {"t": 0, "tdg": 0}
    total_maintenance_rounds = 0

    def source_qubit_indices(instruction) -> list[int]:
        return [
            source_circuit.find_bit(qubit).index
            for qubit in instruction.qubits
        ]

    def append_single_block_gate(operation_name: str, source_index: int) -> None:
        if operation_name == "h":
            append_steane_logical_h(circuit, logical_blocks[source_index])
        elif operation_name == "x":
            append_steane_logical_x(circuit, logical_blocks[source_index])
        elif operation_name == "z":
            append_steane_logical_z(circuit, logical_blocks[source_index])
        elif operation_name == "s":
            append_steane_logical_s(circuit, logical_blocks[source_index])
        elif operation_name == "sdg":
            append_steane_logical_sdg(circuit, logical_blocks[source_index])
        else:
            raise ValueError(f"Unsupported logical operation '{operation_name}' in source circuit.")

    for layer in sorted(layers):
        layer_instructions = layers[layer]
        operation_names = [instruction.operation.name for instruction in layer_instructions]

        if all(operation_name == "barrier" for operation_name in operation_names):
            for instruction in layer_instructions:
                _append_encoded_barrier(circuit, source_circuit, logical_blocks, instruction)
            layer_summaries.append(
                {
                    "scheduled_layer": layer,
                    "operations": operation_names,
                    "active_source_qubit_indices": [],
                    "maintenance_rounds": 0,
                    "barrier": True,
                }
            )
            continue

        active_source_indices = set()
        for instruction in layer_instructions:
            if instruction.operation.name != "barrier":
                active_source_indices.update(source_qubit_indices(instruction))

        for instruction in layer_instructions:
            operation_name = instruction.operation.name
            qindices = source_qubit_indices(instruction)

            if operation_name == "barrier":
                _append_encoded_barrier(circuit, source_circuit, logical_blocks, instruction)
                continue

            if len(qindices) == 1:
                skip_key = (operation_name, qindices[0])
                if skip_counts.get(skip_key, 0):
                    skip_counts[skip_key] -= 1
                    skipped_source_gates.append(
                        {
                            "operation": operation_name,
                            "source_qubit_index": qindices[0],
                            "scheduled_layer": layer,
                        }
                    )
                    continue

            if operation_name in {"h", "x", "z", "s", "sdg"}:
                if len(qindices) != 1:
                    raise ValueError(f"Logical operation '{operation_name}' expects one source qubit.")
                append_single_block_gate(operation_name, qindices[0])
            elif operation_name == "cx":
                if len(qindices) != 2:
                    raise ValueError("Logical operation 'cx' expects two source qubits.")
                append_steane_logical_cnot(
                    circuit,
                    logical_blocks[qindices[0]],
                    logical_blocks[qindices[1]],
                )
            elif operation_name in {"t", "tdg"}:
                if len(qindices) != 1:
                    raise ValueError(f"Logical operation '{operation_name}' expects one source qubit.")
                append_steane_magic_factory(
                    circuit,
                    magic,
                    magic_cat,
                    magic_h_test,
                    syndrome_ancilla_bank=magic_syndrome,
                    z_syndrome_bits=magic_z_syndrome,
                    x_syndrome_bits=magic_x_syndrome,
                    output_kind=operation_name,
                )
                if operation_name == "t":
                    append_steane_logical_t(circuit, logical_blocks[qindices[0]], magic, magic_measure_bits)
                else:
                    append_steane_logical_tdg(circuit, logical_blocks[qindices[0]], magic, magic_measure_bits)
                magic_refresh_counts[operation_name] += 1
            elif operation_name == "measure":
                if len(qindices) != 1 or len(instruction.clbits) != 1:
                    raise ValueError("Logical operation 'measure' expects one source qubit and one clbit.")
                source_clbit_index = source_circuit.find_bit(instruction.clbits[0]).index
                readout_bits = readout_bits_by_clbit[source_clbit_index]
                _append_steane_logical_z_measurement(circuit, logical_blocks[qindices[0]], readout_bits)
                readout_metadata[source_clbit_index]["logical_measurement_parity_bits"] = [
                    _source_bit_label(circuit, bit)
                    for bit in readout_bits[4:7]
                ]
            else:
                raise ValueError(f"Unsupported logical operation '{operation_name}' in source circuit.")

            expanded_operation_counts[operation_name] = expanded_operation_counts.get(operation_name, 0) + 1

        inactive_source_indices = [
            source_index
            for source_index in range(source_circuit.num_qubits)
            if source_index not in active_source_indices
        ]
        for source_index in inactive_source_indices:
            append_steane_syndrome_extraction(
                circuit,
                logical_blocks[source_index],
                prep_ancilla_banks[source_index][:STEANE_SYNDROME_ANCILLA_COUNT],
                data_z_syndrome_bits[source_index],
                data_x_syndrome_bits[source_index],
            )

        total_maintenance_rounds += len(inactive_source_indices)
        layer_summaries.append(
            {
                "scheduled_layer": layer,
                "operations": operation_names,
                "active_source_qubit_indices": sorted(active_source_indices),
                "maintenance_rounds": len(inactive_source_indices),
                "barrier": False,
            }
        )

    unapplied_skips = {
        f"{operation_name}:{source_qubit_index}": count
        for (operation_name, source_qubit_index), count in skip_counts.items()
        if count
    }
    if unapplied_skips:
        raise ValueError(f"Requested source gates were not found for skipping: {unapplied_skips}.")

    source_schedule_metadata = (source_circuit.metadata or {}).get("asap_schedule", {})
    circuit.metadata = {
        "source_circuit": source_circuit.name,
        "source_qubits": source_circuit.num_qubits,
        "source_clbits": source_circuit.num_clbits,
        "source_register_layout": {qreg.name: len(qreg) for qreg in source_circuit.qregs},
        "source_operation_counts": dict(source_circuit.count_ops()),
        "source_asap_layers": source_schedule_metadata.get("scheduled_layer_count"),
        "logical_qubits": source_circuit.num_qubits,
        "steane_block_size": STEANE_BLOCK_SIZE,
        "max_attempts": max_attempts,
        "prep_ancillas_reused_for_syndrome": True,
        "syndrome_policy": {
            "granularity": "per_asap_layer",
            "reuse_classical_bits": True,
            "total_maintenance_rounds": total_maintenance_rounds,
            "non_barrier_layers": sum(1 for summary in layer_summaries if not summary["barrier"]),
        },
        "magic_factory": {
            "shared_factory": True,
            "refresh_before_each_injection": True,
            "refresh_counts": magic_refresh_counts,
            "magic_register": magic.name,
            "cat_register": magic_cat.name,
            "syndrome_register": magic_syndrome.name,
            "measure_registers": {
                "physical_measurement_remainder": magic_measure_rest.name,
                "logical_parity_bits": magic_measure.name,
            },
        },
        "initial_logical_gates": initial_gate_metadata,
        "skipped_source_gates": skipped_source_gates,
        "expanded_operation_counts": expanded_operation_counts,
        "readout_map": readout_metadata,
        "block_map": block_map,
        "layer_summaries": layer_summaries,
    }
    return circuit


def build_first_pass_steane_physical_k0_circuit(
    a: int = 11,
    N: int = 15,
    n: int = 4,
    *,
    max_attempts: int = 3,
) -> QuantumCircuit:
    """Build the full Steane physical circuit for the ASAP-scheduled K0 layout."""
    cx_basis_gates = ["cx", "h", "t", "tdg", "s", "sdg", "x"]
    gate_only_k0 = shor_compilation.build_first_pass_reduced_10q_shor_15_gate_only_k0_layout(
        a=a,
        N=N,
        n=n,
    )
    cx_transpiled_k0 = transpile(
        gate_only_k0,
        basis_gates=cx_basis_gates,
        optimization_level=3,
        seed_transpiler=0,
    )
    scheduled_k0 = shor_compilation.asap_schedule_circuit(cx_transpiled_k0)
    circuit = build_steane_physical_circuit_from_logical(
        scheduled_k0,
        max_attempts=max_attempts,
        initial_logical_gates=(("h", 0), ("x", n)),
        skip_source_gates=(("h", 0), ("x", n)),
    )
    circuit.name = "first_pass_steane_physical_k0"
    circuit.metadata["source_circuit"] = "build_first_pass_reduced_10q_shor_15_gate_only_k0_layout"
    circuit.metadata["transpile_basis_gates"] = cx_basis_gates
    circuit.metadata["transpile_optimization_level"] = 3
    circuit.metadata["transpile_seed"] = 0
    circuit.metadata["k0_parameters"] = {"a": a, "N": N, "n": n}
    return circuit
