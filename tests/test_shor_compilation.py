import unittest

from qiskit import QuantumCircuit

from Compilation.shor_compilation import (
    asap_schedule_circuit,
    build_controlled_modular_multiply_k0_block,
    build_controlled_modular_multiply_k0_gate_only_block,
    build_first_pass_reduced_10q_shor_15_decomposed_k0_layout,
    build_first_pass_reduced_10q_shor_15_gate_only_k0_layout,
    controlled_modular_multiply_k0_schedule,
)


class ShorCompilationTest(unittest.TestCase):
    def _basis_index(self, phase: int, value: int, arith: int, anc: int = 0) -> int:
        bits = [0] * 10
        bits[0] = phase
        for offset, bit_power in enumerate((3, 2, 1, 0), start=1):
            bits[offset] = (value >> bit_power) & 1
        for offset, bit_power in enumerate((3, 2, 1, 0), start=5):
            bits[offset] = (arith >> bit_power) & 1
        bits[9] = anc
        return sum(bit << bit_index for bit_index, bit in enumerate(bits))

    def _decode_basis_index(self, basis_index: int) -> tuple[int, int, int, int]:
        bits = [(basis_index >> bit_index) & 1 for bit_index in range(10)]
        value = sum(
            bits[offset] << bit_power
            for offset, bit_power in zip(range(1, 5), (3, 2, 1, 0), strict=True)
        )
        arith = sum(
            bits[offset] << bit_power
            for offset, bit_power in zip(range(5, 9), (3, 2, 1, 0), strict=True)
        )
        return bits[0], value, arith, bits[9]

    def _evolve_classical_reversible_circuit(self, circuit, basis_index: int) -> int:
        bits = [(basis_index >> bit_index) & 1 for bit_index in range(circuit.num_qubits)]

        for instruction in circuit.data:
            operation = instruction.operation
            qubit_indices = [
                circuit.find_bit(qubit).index
                for qubit in instruction.qubits
            ]

            if operation.name in {"barrier", "delay"}:
                continue
            if operation.name == "x":
                bits[qubit_indices[0]] ^= 1
                continue
            if operation.name == "cx":
                if bits[qubit_indices[0]]:
                    bits[qubit_indices[1]] ^= 1
                continue
            if operation.name == "ccx":
                if bits[qubit_indices[0]] and bits[qubit_indices[1]]:
                    bits[qubit_indices[2]] ^= 1
                continue
            if operation.name.startswith("mcx"):
                controls = qubit_indices[:-1]
                target = qubit_indices[-1]
                control_state = getattr(operation, "ctrl_state", 2 ** len(controls) - 1)
                actual_state = sum(bits[qubit] << index for index, qubit in enumerate(controls))
                if actual_state == control_state:
                    bits[target] ^= 1
                continue

            raise AssertionError(f"Unsupported classical reversible operation: {operation.name}")

        return sum(bit << bit_index for bit_index, bit in enumerate(bits))

    def _operation_names(self, circuit) -> list[str]:
        return [instruction.operation.name for instruction in circuit.data]

    def test_asap_scheduler_moves_independent_gates_to_earliest_layer(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.metadata = {"source": "unit-test"}
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.x(2)

        scheduled = asap_schedule_circuit(circuit)

        self.assertIsNot(scheduled, circuit)
        self.assertEqual(self._operation_names(circuit), ["h", "cx", "x"])
        self.assertEqual(self._operation_names(scheduled), ["h", "x", "cx"])
        self.assertEqual(dict(scheduled.count_ops()), dict(circuit.count_ops()))
        self.assertEqual(scheduled.metadata["source"], "unit-test")
        self.assertNotIn("asap_schedule", circuit.metadata)
        self.assertEqual(
            scheduled.metadata["asap_schedule"]["original_index_to_layer"],
            [0, 1, 0],
        )
        self.assertEqual(scheduled.metadata["asap_schedule"]["scheduled_layer_count"], 2)

    def test_asap_scheduler_left_aligns_within_barrier_boundaries(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.x(2)
        circuit.barrier()
        circuit.h(2)

        scheduled = asap_schedule_circuit(circuit)

        self.assertEqual(self._operation_names(scheduled), ["h", "x", "cx", "barrier", "h"])
        self.assertEqual(
            scheduled.metadata["asap_schedule"]["original_index_to_layer"],
            [0, 1, 0, 2, 3],
        )
        self.assertEqual(
            scheduled.metadata["asap_schedule"]["barrier_policy"],
            "preserved_as_segment_boundaries",
        )
        self.assertEqual(
            scheduled.metadata["asap_schedule"]["barrier_indices"],
            [3],
        )
        self.assertEqual(scheduled.metadata["asap_schedule"]["scheduled_layer_count"], 4)

    def test_asap_scheduler_preserves_gate_only_k0_block_behavior(self) -> None:
        circuit = build_controlled_modular_multiply_k0_gate_only_block()
        scheduled = asap_schedule_circuit(circuit)

        self.assertEqual(scheduled.num_qubits, circuit.num_qubits)
        self.assertEqual(scheduled.num_clbits, circuit.num_clbits)
        self.assertEqual(dict(scheduled.count_ops()), dict(circuit.count_ops()))
        self.assertEqual(scheduled.metadata["construction"], circuit.metadata["construction"])

        for value in range(16):
            expected_value = (11 * value) % 15 if value < 15 else 15
            with self.subTest(phase=1, value=value, expected_value=expected_value):
                output_basis = self._evolve_classical_reversible_circuit(
                    scheduled,
                    self._basis_index(phase=1, value=value, arith=0),
                )
                self.assertEqual(
                    self._decode_basis_index(output_basis),
                    (1, expected_value, 0, 0),
                )

        for value in range(16):
            with self.subTest(phase=0, value=value):
                output_basis = self._evolve_classical_reversible_circuit(
                    scheduled,
                    self._basis_index(phase=0, value=value, arith=0),
                )
                self.assertEqual(self._decode_basis_index(output_basis), (0, value, 0, 0))

    def test_k0_schedule_uses_generic_modular_multiply_constants(self) -> None:
        schedule = controlled_modular_multiply_k0_schedule(a=11, N=15, n=4)

        self.assertEqual(schedule["phase_bit"], 0)
        self.assertEqual(schedule["multiplier"], 11)
        self.assertEqual(schedule["inverse_multiplier"], 11)
        self.assertEqual(
            [(step["value_index"], step["bit_power"], step["constant"]) for step in schedule["compute_adds"]],
            [
                (3, 0, 11),
                (2, 1, 7),
                (1, 2, 14),
                (0, 3, 13),
            ],
        )
        self.assertEqual(
            [(step["value_index"], step["bit_power"], step["constant"]) for step in schedule["uncompute_subs"]],
            [
                (0, 3, 13),
                (1, 2, 14),
                (2, 1, 7),
                (3, 0, 11),
            ],
        )

    def test_k0_block_has_expected_arithmetic_shape(self) -> None:
        circuit = build_controlled_modular_multiply_k0_block(a=11, N=15, n=4)

        self.assertEqual(circuit.num_qubits, 10)
        self.assertEqual(
            circuit.count_ops(),
            {
                "cc_mod_add_11_mod_15": 1,
                "cc_mod_add_7_mod_15": 1,
                "cc_mod_add_14_mod_15": 1,
                "cc_mod_add_13_mod_15": 1,
                "barrier": 2,
                "cswap": 4,
                "cc_mod_sub_13_mod_15": 1,
                "cc_mod_sub_14_mod_15": 1,
                "cc_mod_sub_7_mod_15": 1,
                "cc_mod_sub_11_mod_15": 1,
            },
        )
        self.assertEqual(circuit.metadata["construction"], "generic_out_of_place_controlled_modular_multiply")

    def test_k0_block_controls_first_add_with_lsb_value_qubit(self) -> None:
        circuit = build_controlled_modular_multiply_k0_block(a=11, N=15, n=4)
        first_instruction = circuit.data[0]
        first_instruction_qubits = [
            circuit.find_bit(qubit).index
            for qubit in first_instruction.qubits
        ]

        self.assertEqual(first_instruction.operation.name, "cc_mod_add_11_mod_15")
        self.assertEqual(first_instruction_qubits, [0, 4, 5, 6, 7, 8, 9])

    def test_k0_block_swaps_value_and_arithmetic_registers_under_phase_control(self) -> None:
        circuit = build_controlled_modular_multiply_k0_block(a=11, N=15, n=4)
        cswap_qubits = [
            tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
            for instruction in circuit.data
            if instruction.operation.name == "cswap"
        ]

        self.assertEqual(
            cswap_qubits,
            [
                (0, 1, 5),
                (0, 2, 6),
                (0, 3, 7),
                (0, 4, 8),
            ],
        )

    def test_k0_schedule_rejects_noninvertible_multiplier(self) -> None:
        with self.assertRaises(ValueError):
            controlled_modular_multiply_k0_schedule(a=5, N=15, n=4)

    def test_gate_only_k0_block_removes_modular_add_placeholders(self) -> None:
        circuit = build_controlled_modular_multiply_k0_gate_only_block()
        operation_names = set(circuit.count_ops())

        self.assertNotIn("cswap", operation_names)
        self.assertFalse(any(operation_name.startswith("cc_mod_") for operation_name in operation_names))
        self.assertEqual(circuit.count_ops()["cx"], 24)
        self.assertEqual(circuit.count_ops()["ccx"], 2)
        self.assertEqual(sum(count for name, count in circuit.count_ops().items() if name.startswith("mcx")), 2)
        self.assertTrue(any(operation_name.startswith("mcx") for operation_name in operation_names))
        self.assertEqual(circuit.metadata["workspace_requirement"], "arith and anc enter as |0> and are uncomputed")

    def test_gate_only_k0_block_matches_clean_workspace_controlled_multiply_map(self) -> None:
        circuit = build_controlled_modular_multiply_k0_gate_only_block()

        for value in range(16):
            expected_value = (11 * value) % 15 if value < 15 else 15
            with self.subTest(phase=1, value=value, expected_value=expected_value):
                output_basis = self._evolve_classical_reversible_circuit(
                    circuit,
                    self._basis_index(phase=1, value=value, arith=0),
                )
                self.assertEqual(
                    self._decode_basis_index(output_basis),
                    (1, expected_value, 0, 0),
                )

        for value in range(16):
            with self.subTest(phase=0, value=value):
                output_basis = self._evolve_classical_reversible_circuit(
                    circuit,
                    self._basis_index(phase=0, value=value, arith=0),
                )
                self.assertEqual(self._decode_basis_index(output_basis), (0, value, 0, 0))

    def test_reduced_layout_replaces_opaque_k0_gate_with_arithmetic_block(self) -> None:
        circuit = build_first_pass_reduced_10q_shor_15_decomposed_k0_layout()

        self.assertEqual(circuit.num_qubits, 10)
        self.assertEqual(circuit.num_clbits, 1)
        self.assertNotIn("ctrl_u_0", circuit.count_ops())
        self.assertEqual(circuit.count_ops()["h"], 2)
        self.assertEqual(circuit.count_ops()["x"], 1)
        self.assertEqual(circuit.count_ops()["measure"], 1)
        self.assertEqual(circuit.count_ops()["cswap"], 4)
        self.assertEqual(circuit.count_ops()["cc_mod_add_11_mod_15"], 1)
        self.assertEqual(circuit.count_ops()["cc_mod_sub_11_mod_15"], 1)
        self.assertEqual(circuit.metadata["expanded_block"], "ctrl-U^(2^0)")

    def test_reduced_gate_only_layout_has_no_opaque_modular_blocks(self) -> None:
        circuit = build_first_pass_reduced_10q_shor_15_gate_only_k0_layout()
        operation_names = set(circuit.count_ops())

        self.assertEqual(circuit.num_qubits, 10)
        self.assertEqual(circuit.num_clbits, 1)
        self.assertNotIn("ctrl_u_0", operation_names)
        self.assertNotIn("cswap", operation_names)
        self.assertFalse(any(operation_name.startswith("cc_mod_") for operation_name in operation_names))
        self.assertEqual(circuit.count_ops()["h"], 2)
        self.assertEqual(circuit.count_ops()["x"], 1)
        self.assertEqual(circuit.count_ops()["measure"], 1)
        self.assertEqual(circuit.metadata["expansion_level"], "gate_only")


if __name__ == "__main__":
    unittest.main()
