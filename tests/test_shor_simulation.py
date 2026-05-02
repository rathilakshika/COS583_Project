import unittest

from Simulation import shor_simulation as sim


class ShorSimulationTest(unittest.TestCase):
    def test_generate_three_d3_color_code_instances(self) -> None:
        blocks = sim.generate_steane_code_instances()

        self.assertEqual(len(blocks), 3)
        for logical_index, block in enumerate(blocks):
            self.assertEqual(block.logical_index, logical_index)
            self.assertEqual(block.code.d, 3)
            self.assertEqual(block.code.circuit_type, "tri")
            self.assertEqual(block.patch_qubit_count, 13)
            self.assertEqual(len(block.data_qids), sim.STEANE_DATA_QUBITS)
            self.assertEqual(block.observable_qids, (0, 1, 4))
            self.assertEqual(
                block.global_observable_qids,
                tuple(block.qid_offset + qid for qid in block.observable_qids),
            )

    def test_encoded_precompiled_shor_circuit_has_expected_shape(self) -> None:
        blocks = sim.generate_steane_code_instances()
        circuit = sim.build_encoded_precompiled_shor_15_circuit(blocks)

        self.assertEqual(circuit.num_qubits, 39)
        self.assertEqual(circuit.num_measurements, 67)
        self.assertEqual(circuit.num_observables, 1)
        self.assertEqual(circuit.num_detectors, 45)

    def test_noisy_encoded_circuit_contains_expected_noise_rate(self) -> None:
        circuit = sim.build_encoded_precompiled_shor_15_circuit(
            sim.generate_steane_code_instances(),
            noise_rate=sim.DEFAULT_NOISE_RATE,
        )
        circuit_text = str(circuit)

        self.assertIn("DEPOLARIZE1(0.001)", circuit_text)
        self.assertIn("DEPOLARIZE2(0.001)", circuit_text)
        self.assertIn("MR(0.001)", circuit_text)
        self.assertIn("MRX(0.001)", circuit_text)

    def test_idle_block_syndrome_schedule_detector_count(self) -> None:
        blocks = sim.generate_steane_code_instances()
        circuit = sim.build_encoded_precompiled_shor_15_circuit(blocks)
        no_syndrome_circuit = sim.build_encoded_precompiled_shor_15_circuit(
            blocks,
            include_syndrome_extraction=False,
            noise_rate=0.0,
        )

        self.assertEqual(circuit.num_detectors, 45)
        self.assertEqual(no_syndrome_circuit.num_detectors, 0)

    def test_sampling_and_factor_mapping(self) -> None:
        circuit = sim.build_encoded_precompiled_shor_15_circuit(
            sim.generate_steane_code_instances(),
            include_syndrome_extraction=False,
            noise_rate=0.0,
        )

        counts = sim.sample_logical_observable(circuit, shots=32, seed=7)

        self.assertEqual(sum(counts.values()), 32)
        self.assertEqual(set(counts), {0, 1})
        self.assertIsNone(sim.factor_candidates_from_phase_bit(0))
        self.assertEqual(sim.factor_candidates_from_phase_bit(1), (3, 5))

    def test_noisy_decoded_run_returns_corrected_counts(self) -> None:
        blocks = sim.generate_steane_code_instances()

        result = sim.run_precompiled_shor_15(
            shots=8,
            seed=11,
            blocks=blocks,
            noise_rate=sim.DEFAULT_NOISE_RATE,
            decode=True,
        )

        self.assertEqual(sum(result["raw_phase_bit_counts"].values()), 8)
        self.assertEqual(sum(result["decoded_phase_bit_counts"].values()), 8)
        self.assertEqual(sum(result["decoder_prediction_counts"].values()), 8)
        self.assertEqual(result["noise_rate"], sim.DEFAULT_NOISE_RATE)
        self.assertEqual(result["factor_candidates_by_phase_bit"][1], (3, 5))

    def test_success_explanation_includes_factor_math(self) -> None:
        explanation = sim.shor_15_success_explanation()

        self.assertIn("11", explanation)
        self.assertIn("gcd(10, 15) = 5", explanation)
        self.assertIn("gcd(12, 15) = 3", explanation)
        self.assertIn("Phase bit 0 is inconclusive", explanation)


if __name__ == "__main__":
    unittest.main()
