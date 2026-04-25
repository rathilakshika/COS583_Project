import unittest

from steane_qft.qasm_ir import (
    AssignmentStmt,
    BarrierStmt,
    GateStmt,
    MeasureStmt,
    ResetStmt,
    parse_program,
    parse_statement,
)


class QasmIrParsingTest(unittest.TestCase):
    def test_parse_whole_register_gate(self) -> None:
        statement = parse_statement("h q0;")
        self.assertIsInstance(statement, GateStmt)
        self.assertEqual(statement.op, "h")
        self.assertEqual(statement.targets[0].name, "q0")
        self.assertTrue(statement.targets[0].is_whole_register)

    def test_parse_conditional_reset(self) -> None:
        statement = parse_statement("if(init[0] == 1) reset q0;")
        self.assertIsInstance(statement, ResetStmt)
        self.assertIsNotNone(statement.condition)
        self.assertEqual(statement.condition.lhs.name, "init")
        self.assertEqual(statement.condition.lhs.index, 0)
        self.assertEqual(statement.target.name, "q0")
        self.assertTrue(statement.target.is_whole_register)

    def test_parse_measurement_barrier_and_assignment(self) -> None:
        measure = parse_statement("measure q3[0] -> cT12[0];")
        barrier = parse_statement("barrier q0, q3[0];")
        assign = parse_statement("c2_log = c2_log ^ pf[0];")

        self.assertIsInstance(measure, MeasureStmt)
        self.assertEqual(measure.source.name, "q3")
        self.assertEqual(measure.target.name, "cT12")

        self.assertIsInstance(barrier, BarrierStmt)
        self.assertEqual(len(barrier.targets), 2)
        self.assertEqual(barrier.targets[0].name, "q0")
        self.assertTrue(barrier.targets[0].is_whole_register)

        self.assertIsInstance(assign, AssignmentStmt)
        self.assertEqual(assign.target.name, "c2_log")
        self.assertEqual(assign.expr, "c2_log ^ pf[0]")

    def test_parse_program_preserves_hqslib_include_without_resolution(self) -> None:
        qasm = "\n".join(
            [
                "OPENQASM 2.0;",
                'include "hqslib1.inc";',
                "qreg q0[7];",
                "creg c0[7];",
                "rz(-pi/2) q0;",
            ]
        )
        program = parse_program(qasm)
        self.assertEqual(program.version, "2.0")
        self.assertEqual(program.includes, ["hqslib1.inc"])
        self.assertEqual(program.qregs, {"q0": 7})
        self.assertEqual(program.cregs, {"c0": 7})
        self.assertEqual(program.statements[-1].op, "rz")


if __name__ == "__main__":
    unittest.main()
