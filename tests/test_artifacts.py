import unittest
from pathlib import Path

from steane_qft.artifacts import load_paper_qft_artifacts


class ArtifactLoadingTest(unittest.TestCase):
    def test_qft_sources_have_expected_case_coverage(self) -> None:
        artifacts = load_paper_qft_artifacts()
        by_source: dict[str, list] = {}
        for artifact in artifacts:
            by_source.setdefault(Path(artifact.source_file).name, []).append(artifact)

        primary = by_source["QFT_ancilla_assisted2.json"]
        alternate = by_source["QFT_recursive_teleportation.json"]
        secondary = by_source["QFT_ancilla_assisted1.json"]

        expected_labels = {(family, f"{index:03b}") for family in ("comp", "fourier") for index in range(8)}
        self.assertEqual(len(primary), 16)
        self.assertEqual({(artifact.basis_family, artifact.basis_label) for artifact in primary}, expected_labels)

        self.assertEqual(len(alternate), 16)
        self.assertEqual({(artifact.basis_family, artifact.basis_label) for artifact in alternate}, expected_labels)

        self.assertEqual(len(secondary), 8)
        self.assertEqual({artifact.basis_family for artifact in secondary}, {"fourier"})
        self.assertEqual({artifact.basis_label for artifact in secondary}, {f"{index:03b}" for index in range(8)})

    def test_primary_artifacts_capture_feedforward_and_layout(self) -> None:
        artifacts = load_paper_qft_artifacts()
        primary = [artifact for artifact in artifacts if Path(artifact.source_file).name == "QFT_ancilla_assisted2.json"]
        self.assertTrue(all(artifact.uses_feedforward for artifact in primary))
        self.assertTrue(all(artifact.qreg_layout == {"q0": 7, "q1": 7, "q2": 7, "q3": 7} for artifact in primary))
        self.assertTrue(all(artifact.t_gadget_count == 13 for artifact in primary))


if __name__ == "__main__":
    unittest.main()
