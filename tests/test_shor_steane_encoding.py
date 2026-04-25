import unittest

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from Compilation.shor_steane_encoding import (
    STEANE_BLOCK_SIZE,
    append_steane_logical_cnot,
    append_steane_logical_h,
    append_steane_logical_s,
    append_steane_logical_sdg,
    append_steane_logical_t,
    append_steane_logical_tdg,
    append_steane_magic_factory,
    append_steane_magic_injection,
    append_steane_syndrome_extraction,
    append_steane_logical_x,
    append_steane_logical_z,
    build_first_pass_explicit_steane_zero_initialization,
    build_first_pass_steane_physical_k0_circuit,
    build_first_pass_steane_zero_initialization,
    build_steane_physical_circuit_from_logical,
    build_steane_magic_factory,
    build_steane_zero_prep,
)


class ShorSteaneEncodingTest(unittest.TestCase):
    def _count_operations_recursive(self, circuit: QuantumCircuit, operation_name: str) -> int:
        total = 0
        for instruction in circuit.data:
            operation = instruction.operation
            if operation.name == operation_name:
                total += 1
            for block in getattr(operation, "blocks", ()):
                total += self._count_operations_recursive(block, operation_name)
        return total

    def test_transversal_logical_h_applies_h_to_every_physical_qubit(self) -> None:
        data = QuantumRegister(STEANE_BLOCK_SIZE, "logical")
        circuit = QuantumCircuit(data)

        returned_circuit = append_steane_logical_h(circuit, data)

        self.assertIs(returned_circuit, circuit)
        self.assertEqual(circuit.count_ops(), {"h": STEANE_BLOCK_SIZE})
        self.assertEqual([instruction.operation.name for instruction in circuit.data], ["h"] * STEANE_BLOCK_SIZE)
        self.assertEqual(
            [circuit.find_bit(instruction.qubits[0]).index for instruction in circuit.data],
            list(range(STEANE_BLOCK_SIZE)),
        )

    def test_transversal_logical_x_applies_x_to_every_physical_qubit(self) -> None:
        data = QuantumRegister(STEANE_BLOCK_SIZE, "logical")
        circuit = QuantumCircuit(data)

        returned_circuit = append_steane_logical_x(circuit, data)

        self.assertIs(returned_circuit, circuit)
        self.assertEqual(circuit.count_ops(), {"x": STEANE_BLOCK_SIZE})
        self.assertEqual([instruction.operation.name for instruction in circuit.data], ["x"] * STEANE_BLOCK_SIZE)
        self.assertEqual(
            [circuit.find_bit(instruction.qubits[0]).index for instruction in circuit.data],
            list(range(STEANE_BLOCK_SIZE)),
        )

    def test_transversal_logical_z_applies_z_to_every_physical_qubit(self) -> None:
        data = QuantumRegister(STEANE_BLOCK_SIZE, "logical")
        circuit = QuantumCircuit(data)

        returned_circuit = append_steane_logical_z(circuit, data)

        self.assertIs(returned_circuit, circuit)
        self.assertEqual(circuit.count_ops(), {"z": STEANE_BLOCK_SIZE})
        self.assertEqual([instruction.operation.name for instruction in circuit.data], ["z"] * STEANE_BLOCK_SIZE)
        self.assertEqual(
            [circuit.find_bit(instruction.qubits[0]).index for instruction in circuit.data],
            list(range(STEANE_BLOCK_SIZE)),
        )

    def test_transversal_logical_s_applies_physical_sdg_to_every_physical_qubit(self) -> None:
        data = QuantumRegister(STEANE_BLOCK_SIZE, "logical")
        circuit = QuantumCircuit(data)

        returned_circuit = append_steane_logical_s(circuit, data)

        self.assertIs(returned_circuit, circuit)
        self.assertEqual(circuit.count_ops(), {"sdg": STEANE_BLOCK_SIZE})
        self.assertEqual([instruction.operation.name for instruction in circuit.data], ["sdg"] * STEANE_BLOCK_SIZE)
        self.assertEqual(
            [circuit.find_bit(instruction.qubits[0]).index for instruction in circuit.data],
            list(range(STEANE_BLOCK_SIZE)),
        )

    def test_transversal_logical_sdg_applies_physical_s_to_every_physical_qubit(self) -> None:
        data = QuantumRegister(STEANE_BLOCK_SIZE, "logical")
        circuit = QuantumCircuit(data)

        returned_circuit = append_steane_logical_sdg(circuit, data)

        self.assertIs(returned_circuit, circuit)
        self.assertEqual(circuit.count_ops(), {"s": STEANE_BLOCK_SIZE})
        self.assertEqual([instruction.operation.name for instruction in circuit.data], ["s"] * STEANE_BLOCK_SIZE)
        self.assertEqual(
            [circuit.find_bit(instruction.qubits[0]).index for instruction in circuit.data],
            list(range(STEANE_BLOCK_SIZE)),
        )

    def test_transversal_logical_cnot_applies_pairwise_cx_between_blocks(self) -> None:
        control = QuantumRegister(STEANE_BLOCK_SIZE, "control")
        target = QuantumRegister(STEANE_BLOCK_SIZE, "target")
        circuit = QuantumCircuit(control, target)

        returned_circuit = append_steane_logical_cnot(circuit, control, target)

        self.assertIs(returned_circuit, circuit)
        self.assertEqual(circuit.count_ops(), {"cx": STEANE_BLOCK_SIZE})
        self.assertEqual([instruction.operation.name for instruction in circuit.data], ["cx"] * STEANE_BLOCK_SIZE)
        self.assertEqual(
            [
                (
                    circuit.find_bit(instruction.qubits[0]).index,
                    circuit.find_bit(instruction.qubits[1]).index,
                )
                for instruction in circuit.data
            ],
            [(index, STEANE_BLOCK_SIZE + index) for index in range(STEANE_BLOCK_SIZE)],
        )

    def test_syndrome_extraction_reuses_ancillas_for_z_and_x_stabilizers(self) -> None:
        data = QuantumRegister(STEANE_BLOCK_SIZE, "logical")
        ancillas = QuantumRegister(3, "ancilla")
        z_syndrome = ClassicalRegister(3, "z_syndrome")
        x_syndrome = ClassicalRegister(3, "x_syndrome")
        circuit = QuantumCircuit(data, ancillas, z_syndrome, x_syndrome)

        returned_circuit = append_steane_syndrome_extraction(
            circuit,
            data,
            ancillas,
            z_syndrome,
            x_syndrome,
        )

        self.assertIs(returned_circuit, circuit)
        self.assertEqual(circuit.count_ops()["h"], 6)
        self.assertEqual(circuit.count_ops()["cx"], 24)
        self.assertEqual(circuit.count_ops()["measure"], 6)
        self.assertEqual(circuit.count_ops()["reset"], 9)

        hadamard_targets = [
            circuit.find_bit(instruction.qubits[0]).index
            for instruction in circuit.data
            if instruction.operation.name == "h"
        ]
        reset_targets = [
            circuit.find_bit(instruction.qubits[0]).index
            for instruction in circuit.data
            if instruction.operation.name == "reset"
        ]
        cx_pairs = [
            (
                circuit.find_bit(instruction.qubits[0]).index,
                circuit.find_bit(instruction.qubits[1]).index,
            )
            for instruction in circuit.data
            if instruction.operation.name == "cx"
        ]
        measure_pairs = [
            (
                circuit.find_bit(instruction.qubits[0]).index,
                circuit.find_bit(instruction.clbits[0]).index,
            )
            for instruction in circuit.data
            if instruction.operation.name == "measure"
        ]

        self.assertEqual(hadamard_targets, [7, 8, 9, 7, 8, 9])
        self.assertEqual(reset_targets, [7, 8, 9, 7, 8, 9, 7, 8, 9])
        self.assertEqual(
            cx_pairs[:12],
            [
                (7, 1),
                (7, 3),
                (7, 4),
                (7, 6),
                (8, 0),
                (8, 3),
                (8, 5),
                (8, 6),
                (9, 2),
                (9, 4),
                (9, 5),
                (9, 6),
            ],
        )
        self.assertEqual(
            cx_pairs[12:],
            [
                (1, 7),
                (3, 7),
                (4, 7),
                (6, 7),
                (0, 8),
                (3, 8),
                (5, 8),
                (6, 8),
                (2, 9),
                (4, 9),
                (5, 9),
                (6, 9),
            ],
        )
        self.assertEqual(
            measure_pairs,
            [
                (7, 0),
                (8, 1),
                (9, 2),
                (7, 3),
                (8, 4),
                (9, 5),
            ],
        )

    def test_syndrome_extraction_validates_ancilla_and_syndrome_sizes(self) -> None:
        data = QuantumRegister(STEANE_BLOCK_SIZE, "logical")
        ancillas = QuantumRegister(3, "ancilla")
        z_syndrome = ClassicalRegister(3, "z_syndrome")
        x_syndrome = ClassicalRegister(3, "x_syndrome")

        with self.subTest("wrong ancilla count"):
            circuit = QuantumCircuit(data, ancillas, z_syndrome, x_syndrome)
            with self.assertRaises(ValueError):
                append_steane_syndrome_extraction(circuit, data, ancillas[:2], z_syndrome, x_syndrome)

        with self.subTest("wrong z syndrome count"):
            circuit = QuantumCircuit(data, ancillas, z_syndrome, x_syndrome)
            with self.assertRaises(ValueError):
                append_steane_syndrome_extraction(circuit, data, ancillas, z_syndrome[:2], x_syndrome)

        with self.subTest("wrong x syndrome count"):
            circuit = QuantumCircuit(data, ancillas, z_syndrome, x_syndrome)
            with self.assertRaises(ValueError):
                append_steane_syndrome_extraction(circuit, data, ancillas, z_syndrome, x_syndrome[:2])

    def test_single_block_prep_matches_paper_zero_state_support(self) -> None:
        circuit = build_steane_zero_prep(1, max_attempts=1, measure_ancilla=False)
        state = Statevector.from_instruction(circuit)
        support = {
            format(index, "08b")[::-1]
            for index, amplitude in enumerate(state.data)
            if abs(amplitude) > 1e-9
        }

        self.assertEqual(
            support,
            {
                "00000000",
                "11110000",
                "01101100",
                "10011100",
                "10100110",
                "01010110",
                "11001010",
                "00111010",
            },
        )

    def test_single_block_prep_uses_qubit_five_for_fanout_and_qubit_four_for_verification(self) -> None:
        circuit = build_steane_zero_prep(1, max_attempts=1, measure_ancilla=False)

        hadamard_targets = [
            circuit.find_bit(instruction.qubits[0]).index
            for instruction in circuit.data
            if instruction.operation.name == "h"
        ]
        cx_pairs = [
            (
                circuit.find_bit(instruction.qubits[0]).index,
                circuit.find_bit(instruction.qubits[1]).index,
            )
            for instruction in circuit.data
            if instruction.operation.name == "cx"
        ]

        self.assertEqual(hadamard_targets, [0, 5, 6])
        self.assertEqual(
            cx_pairs,
            [
                (5, 4),
                (0, 1),
                (6, 3),
                (5, 2),
                (6, 4),
                (0, 3),
                (5, 1),
                (3, 2),
                (4, 7),
                (1, 7),
                (3, 7),
            ],
        )

    def test_retry_path_requires_measurement(self) -> None:
        with self.assertRaises(ValueError):
            build_steane_zero_prep(1, max_attempts=2, measure_ancilla=False)

    def test_magic_factory_contains_transversal_h_test_and_x_basis_cat_measurements(self) -> None:
        circuit = build_steane_magic_factory(output_kind="h", include_syndrome=False)

        self.assertEqual(circuit.count_ops()["ch"], STEANE_BLOCK_SIZE)
        self.assertEqual(circuit.count_ops()["measure"], STEANE_BLOCK_SIZE)

        measure_targets = [
            circuit.find_bit(instruction.qubits[0]).index
            for instruction in circuit.data
            if instruction.operation.name == "measure"
        ]
        self.assertEqual(measure_targets, list(range(STEANE_BLOCK_SIZE, 2 * STEANE_BLOCK_SIZE)))

        for instruction_index, instruction in enumerate(circuit.data):
            if instruction.operation.name != "measure":
                continue
            previous_instruction = circuit.data[instruction_index - 1]
            self.assertEqual(previous_instruction.operation.name, "h")
            self.assertEqual(previous_instruction.qubits, instruction.qubits)

        self.assertFalse(circuit.metadata["surface_code_transfer"])
        self.assertEqual(circuit.metadata["h_test_acceptance"], "accept when parity of magic_h_test[0:7] is even")

    def test_magic_factory_t_and_tdg_outputs_apply_expected_logical_clifford_conversions(self) -> None:
        t_factory = build_steane_magic_factory(output_kind="t", include_syndrome=False)
        tdg_factory = build_steane_magic_factory(output_kind="tdg", include_syndrome=False)

        self.assertEqual(t_factory.count_ops()["s"], STEANE_BLOCK_SIZE)
        self.assertNotIn("sdg", t_factory.count_ops())
        self.assertEqual(t_factory.metadata["output_kind"], "t")

        self.assertEqual(tdg_factory.count_ops()["sdg"], STEANE_BLOCK_SIZE)
        self.assertNotIn("s", tdg_factory.count_ops())
        self.assertEqual(tdg_factory.metadata["output_kind"], "tdg")

    def test_magic_factory_validates_blocks_syndrome_resources_and_output_kind(self) -> None:
        magic = QuantumRegister(STEANE_BLOCK_SIZE, "magic")
        cat = QuantumRegister(STEANE_BLOCK_SIZE, "cat")
        syndrome = QuantumRegister(3, "syndrome")
        h_test = ClassicalRegister(STEANE_BLOCK_SIZE, "h_test")
        z_syndrome = ClassicalRegister(3, "z_syndrome")
        x_syndrome = ClassicalRegister(3, "x_syndrome")

        with self.subTest("wrong cat size"):
            circuit = QuantumCircuit(magic, cat, h_test)
            with self.assertRaises(ValueError):
                append_steane_magic_factory(circuit, magic, cat[:6], h_test)

        with self.subTest("partial syndrome resources"):
            circuit = QuantumCircuit(magic, cat, syndrome, h_test, z_syndrome, x_syndrome)
            with self.assertRaises(ValueError):
                append_steane_magic_factory(
                    circuit,
                    magic,
                    cat,
                    h_test,
                    syndrome_ancilla_bank=syndrome,
                    z_syndrome_bits=z_syndrome,
                )

        with self.subTest("wrong syndrome size"):
            circuit = QuantumCircuit(magic, cat, syndrome, h_test, z_syndrome, x_syndrome)
            with self.assertRaises(ValueError):
                append_steane_magic_factory(
                    circuit,
                    magic,
                    cat,
                    h_test,
                    syndrome_ancilla_bank=syndrome[:2],
                    z_syndrome_bits=z_syndrome,
                    x_syndrome_bits=x_syndrome,
                )

        with self.subTest("invalid output kind"):
            with self.assertRaises(ValueError):
                build_steane_magic_factory(output_kind="surface")

    def test_magic_injection_measures_magic_block_and_conditionally_corrects_t(self) -> None:
        data = QuantumRegister(STEANE_BLOCK_SIZE, "data")
        magic = QuantumRegister(STEANE_BLOCK_SIZE, "magic")
        magic_measure_rest = ClassicalRegister(STEANE_BLOCK_SIZE - 3, "magic_measure_rest")
        magic_measure = ClassicalRegister(3, "magic_measure")
        measure_bits = [*magic_measure_rest, *magic_measure]
        circuit = QuantumCircuit(data, magic, magic_measure_rest, magic_measure)

        returned_circuit = append_steane_magic_injection(circuit, data, magic, measure_bits, gate="t")

        self.assertIs(returned_circuit, circuit)
        self.assertEqual(circuit.count_ops()["cx"], STEANE_BLOCK_SIZE)
        self.assertEqual(circuit.count_ops()["measure"], STEANE_BLOCK_SIZE)
        self.assertNotIn("store", circuit.count_ops())
        self.assertEqual(circuit.count_ops()["if_else"], 4)
        self.assertEqual(self._count_operations_recursive(circuit, "sdg"), 4 * STEANE_BLOCK_SIZE)

        cx_pairs = [
            (
                circuit.find_bit(instruction.qubits[0]).index,
                circuit.find_bit(instruction.qubits[1]).index,
            )
            for instruction in circuit.data
            if instruction.operation.name == "cx"
        ]
        self.assertEqual(
            cx_pairs,
            [(index, STEANE_BLOCK_SIZE + index) for index in range(STEANE_BLOCK_SIZE)],
        )

        measure_pairs = [
            (
                circuit.find_bit(instruction.qubits[0]).index,
                circuit.find_bit(instruction.clbits[0]).index,
            )
            for instruction in circuit.data
            if instruction.operation.name == "measure"
        ]
        self.assertEqual(
            measure_pairs,
            [(STEANE_BLOCK_SIZE + index, index) for index in range(STEANE_BLOCK_SIZE)],
        )

    def test_magic_injection_tdg_uses_s_correction_and_wrappers_delegate(self) -> None:
        data = QuantumRegister(STEANE_BLOCK_SIZE, "data")
        magic = QuantumRegister(STEANE_BLOCK_SIZE, "magic")
        magic_measure_rest = ClassicalRegister(STEANE_BLOCK_SIZE - 3, "magic_measure_rest")
        magic_measure = ClassicalRegister(3, "magic_measure")
        measure_bits = [*magic_measure_rest, *magic_measure]
        circuit = QuantumCircuit(data, magic, magic_measure_rest, magic_measure)

        append_steane_logical_t(circuit, data, magic, measure_bits)
        append_steane_logical_tdg(circuit, data, magic, measure_bits)

        self.assertNotIn("store", circuit.count_ops())
        self.assertEqual(circuit.count_ops()["if_else"], 8)
        self.assertEqual(self._count_operations_recursive(circuit, "sdg"), 4 * STEANE_BLOCK_SIZE)
        self.assertEqual(self._count_operations_recursive(circuit, "s"), 4 * STEANE_BLOCK_SIZE)

    def test_magic_injection_validates_gate_and_resource_sizes(self) -> None:
        data = QuantumRegister(STEANE_BLOCK_SIZE, "data")
        magic = QuantumRegister(STEANE_BLOCK_SIZE, "magic")
        magic_measure = ClassicalRegister(STEANE_BLOCK_SIZE, "magic_measure")

        with self.subTest("invalid gate"):
            circuit = QuantumCircuit(data, magic, magic_measure)
            with self.assertRaises(ValueError):
                append_steane_magic_injection(circuit, data, magic, magic_measure, gate="a")

        with self.subTest("wrong magic block size"):
            circuit = QuantumCircuit(data, magic, magic_measure)
            with self.assertRaises(ValueError):
                append_steane_magic_injection(circuit, data, magic[:6], magic_measure)

        with self.subTest("wrong measurement size"):
            circuit = QuantumCircuit(data, magic, magic_measure)
            with self.assertRaises(ValueError):
                append_steane_magic_injection(circuit, data, magic, magic_measure[:6])

    def test_first_pass_initializer_allocates_one_steane_block_per_source_qubit(self) -> None:
        circuit = build_first_pass_steane_zero_initialization(max_attempts=3)

        self.assertEqual(circuit.num_qubits, 10 * STEANE_BLOCK_SIZE + 1)
        self.assertEqual(circuit.num_clbits, 10)
        self.assertEqual(circuit.count_ops()["measure"], 10)
        self.assertEqual(circuit.count_ops()["if_else"], 20)
        self.assertEqual(circuit.metadata["source_register_layout"], {"phase": 1, "value": 4, "arith": 4, "anc": 1})
        self.assertEqual(len(circuit.metadata["block_map"]), 10)
        self.assertEqual(circuit.metadata["block_map"][0]["source_qubit"], "phase[0]")
        self.assertEqual(circuit.metadata["block_map"][-1]["source_qubit"], "anc[0]")

    def test_explicit_first_pass_initializer_uses_100_qubits_for_three_attempts(self) -> None:
        circuit = build_first_pass_explicit_steane_zero_initialization(max_attempts=3)

        self.assertEqual(circuit.num_qubits, 10 * (STEANE_BLOCK_SIZE + 3))
        self.assertEqual(circuit.num_clbits, 30)
        self.assertEqual(self._count_operations_recursive(circuit, "measure"), 30)
        self.assertEqual(circuit.count_ops()["if_else"], 20)
        self.assertEqual(circuit.metadata["source_register_layout"], {"phase": 1, "value": 4, "arith": 4, "anc": 1})
        self.assertEqual(circuit.metadata["prep_ancillas_per_logical_qubit"], 3)
        self.assertEqual(circuit.metadata["physical_qubits_per_logical_qubit"], 10)
        self.assertEqual(len(circuit.metadata["block_map"]), 10)
        self.assertEqual(circuit.metadata["block_map"][0]["prep_ancilla_register"], "phase_0_prep")
        self.assertEqual(circuit.metadata["block_map"][-1]["verification_flag_register"], "anc_0_init")

    def test_star_import_exposes_full_physical_builders(self) -> None:
        namespace = {}

        exec("from Compilation.shor_steane_encoding import *", namespace)

        self.assertIs(namespace["build_first_pass_steane_physical_k0_circuit"], build_first_pass_steane_physical_k0_circuit)
        self.assertIs(namespace["build_steane_physical_circuit_from_logical"], build_steane_physical_circuit_from_logical)

    def test_full_physical_k0_builder_uses_expected_shared_resources_and_metadata(self) -> None:
        circuit = build_first_pass_steane_physical_k0_circuit()
        metadata = circuit.metadata

        self.assertEqual(circuit.num_qubits, 117)
        self.assertEqual(circuit.num_clbits, 117)
        self.assertEqual(metadata["source_asap_layers"], 86)
        self.assertEqual(metadata["source_operation_counts"]["t"], 24)
        self.assertEqual(metadata["source_operation_counts"]["tdg"], 20)
        self.assertEqual(metadata["syndrome_policy"]["total_maintenance_rounds"], 664)
        self.assertEqual(metadata["syndrome_policy"]["non_barrier_layers"], 85)
        self.assertEqual(metadata["magic_factory"]["refresh_counts"], {"t": 24, "tdg": 20})
        self.assertEqual(metadata["expanded_operation_counts"]["t"], 24)
        self.assertEqual(metadata["expanded_operation_counts"]["tdg"], 20)
        self.assertEqual(len(metadata["block_map"]), 10)
        self.assertEqual(metadata["readout_map"][0]["readout_register"], "c_0_readout")

        qreg_names = [qreg.name for qreg in circuit.qregs]
        self.assertEqual(qreg_names.count("magic"), 1)
        self.assertEqual(qreg_names.count("magic_cat"), 1)
        self.assertEqual(qreg_names.count("magic_syndrome"), 1)

        self.assertNotIn("t", circuit.count_ops())
        self.assertNotIn("tdg", circuit.count_ops())
        self.assertEqual(self._count_operations_recursive(circuit, "t"), 0)
        self.assertEqual(self._count_operations_recursive(circuit, "tdg"), 0)

    def test_full_physical_k0_builder_applies_and_skips_initial_h_and_x_once(self) -> None:
        circuit = build_first_pass_steane_physical_k0_circuit()
        metadata = circuit.metadata

        self.assertEqual(
            metadata["initial_logical_gates"],
            [
                {"operation": "h", "source_qubit_index": 0, "source_qubit": "phase[0]"},
                {"operation": "x", "source_qubit_index": 4, "source_qubit": "value[3]"},
            ],
        )
        self.assertEqual(
            metadata["skipped_source_gates"],
            [
                {"operation": "h", "source_qubit_index": 0, "scheduled_layer": 0},
                {"operation": "x", "source_qubit_index": 4, "scheduled_layer": 0},
            ],
        )
        self.assertEqual(
            metadata["expanded_operation_counts"]["h"],
            metadata["source_operation_counts"]["h"] - 1,
        )
        self.assertEqual(
            metadata["expanded_operation_counts"]["x"],
            metadata["source_operation_counts"]["x"] - 1,
        )
        first_layer = metadata["layer_summaries"][0]
        self.assertEqual(first_layer["scheduled_layer"], 0)
        self.assertIn(0, first_layer["active_source_qubit_indices"])
        self.assertIn(4, first_layer["active_source_qubit_indices"])

    def test_full_physical_builder_rejects_unsupported_source_operations(self) -> None:
        logical = QuantumCircuit(1)
        logical.y(0)

        with self.assertRaisesRegex(ValueError, "Unsupported logical operation 'y'"):
            build_steane_physical_circuit_from_logical(logical)


if __name__ == "__main__":
    unittest.main()
