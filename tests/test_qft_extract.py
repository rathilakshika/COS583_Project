import unittest

from steane_qft.artifacts import load_paper_qft_artifacts
from steane_qft.qasm_ir import CommentStmt, GateStmt, ResetStmt
from steane_qft.qft_extract import (
    build_shor_semiclassical_inverse_qft_round,
    build_shor_terminal_inverse_qft,
    extract_forward_qft_template,
)


class QftExtractionTest(unittest.TestCase):
    def test_forward_template_captures_primary_structure(self) -> None:
        template = extract_forward_qft_template()
        self.assertEqual(template.variant, "ancilla_assisted")
        self.assertEqual(template.direction, "forward")
        self.assertEqual(template.logical_data_blocks, 3)
        self.assertEqual(template.workspace_blocks, 1)
        self.assertTrue(template.terminal_measurement)
        self.assertEqual(template.t_gadget_style, "paper_native")
        self.assertEqual(template.physical_qubits, 28)
        self.assertEqual(template.t_gadget_count, 13)
        self.assertEqual(template.output_bits, 3)
        self.assertIn("c0", template.creg_layout)
        self.assertTrue(template.core_segments)
        self.assertTrue(template.measurement_postprocess_segments)

    def test_forward_template_tracks_primary_coverage(self) -> None:
        template = extract_forward_qft_template()
        expected = {(family, f"{index:03b}") for family in ("comp", "fourier") for index in range(8)}
        self.assertEqual(set(template.parameter_coverage), expected)
        self.assertEqual(set(template.benchmark_cases), expected)

    def test_secondary_fourier_cases_are_supported_by_primary_template(self) -> None:
        template = extract_forward_qft_template()
        artifacts = load_paper_qft_artifacts()
        secondary_overlap = {
            (artifact.basis_family, artifact.basis_label)
            for artifact in artifacts
            if artifact.source_file.endswith("QFT_ancilla_assisted1.json")
        }
        self.assertEqual(secondary_overlap, set(template.validation_coverage))
        self.assertTrue(secondary_overlap.issubset(set(template.parameter_coverage)))

    def test_shor_inverse_template_interface(self) -> None:
        inverse = build_shor_terminal_inverse_qft()
        self.assertEqual(inverse.variant, "ancilla_assisted")
        self.assertEqual(inverse.direction, "inverse")
        self.assertEqual(inverse.logical_data_blocks, 3)
        self.assertEqual(inverse.workspace_blocks, 1)
        self.assertTrue(inverse.terminal_measurement)

    def test_semiclassical_shor_round_template_interface(self) -> None:
        round_template = build_shor_semiclassical_inverse_qft_round(total_phase_bits=8)

        self.assertEqual(round_template.variant, "ancilla_assisted")
        self.assertEqual(round_template.direction, "inverse_semiclassical_round")
        self.assertEqual(round_template.logical_data_blocks, 1)
        self.assertEqual(round_template.workspace_blocks, 0)
        self.assertEqual(round_template.physical_qubits, 7)
        self.assertEqual(round_template.output_bits, 1)
        self.assertEqual(round_template.qreg_layout, {"phase": 7})
        self.assertEqual(round_template.creg_layout["phase_out"], 1)
        self.assertEqual(round_template.creg_layout["phase_history_6"], 1)
        self.assertEqual(len(round_template.core_segments), 2)
        self.assertTrue(round_template.measurement_postprocess_segments)

        phase_corrections = round_template.core_segments[0]
        self.assertFalse(phase_corrections.metadata["includes_state_preparation"])
        self.assertEqual(phase_corrections.metadata["required_phase_denominators"], (2, 4, 8, 16, 32, 64, 128))
        self.assertEqual(phase_corrections.metadata["directly_supported_phase_denominators"], (2,))
        self.assertEqual(phase_corrections.metadata["unsupported_phase_denominators"], (4, 8, 16, 32, 64, 128))

        correction_angles = [
            statement.params[0]
            for statement in phase_corrections.statements
            if isinstance(statement, GateStmt)
        ]
        self.assertEqual(
            correction_angles,
            ["-pi/2", "-pi/4", "-pi/8", "-pi/16", "-pi/32", "-pi/64", "-pi/128"],
        )

        all_statements = [
            statement
            for segment in (round_template.core_segments + round_template.measurement_postprocess_segments)
            for statement in segment.statements
        ]
        self.assertFalse(any(isinstance(statement, ResetStmt) for statement in all_statements))
        self.assertFalse(
            any(
                isinstance(statement, CommentStmt) and statement.text == "Initialize logical |T> = T|+>"
                for statement in all_statements
            )
        )


if __name__ == "__main__":
    unittest.main()
