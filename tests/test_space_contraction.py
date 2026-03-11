import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "qiskit_from_scratch.ipynb"
DEFINITION_CELLS = (3, 5, 7, 9, 11, 13, 15)


def ensure_tntorch_available():
    if importlib.util.find_spec("tntorch") is not None:
        return

    module = types.ModuleType("tntorch")

    class Tensor:
        def __init__(self, cores):
            self.cores = list(cores)

        def round_tt(self, eps=0):
            return None

    module.Tensor = Tensor
    sys.modules["tntorch"] = module


def load_notebook_namespace():
    ensure_tntorch_available()
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {}
    for cell_index in DEFINITION_CELLS:
        exec("".join(notebook["cells"][cell_index]["source"]), namespace)
    return namespace


def dense_amplitude(tensor, state: int, n_qudits: int):
    return tensor.reshape(-1)[state].item()


class SpaceContractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = load_notebook_namespace()
        cls.QuantumCircuit = namespace["QuantumCircuit"]

    def test_space_statevector_matches_basis_amplitudes_for_single_qubit_gate(self):
        qc = self.QuantumCircuit(3)
        qc.h(1)

        result = qc.contract(scheme="space", representation="statevector", eps=0)

        for state in range(2 ** qc.n_qudits):
            amplitude = dense_amplitude(result.tensor, state, qc.n_qudits)
            self.assertAlmostEqual(amplitude.real, qc.check_state(state).real)
            self.assertAlmostEqual(amplitude.imag, qc.check_state(state).imag)

    def assert_space_matches_check_state(self, qc):
        result = qc.contract(scheme="space", representation="statevector", eps=0)
        for state in range(2 ** qc.n_qudits):
            amplitude = dense_amplitude(result.tensor, state, qc.n_qudits)
            expected = qc.check_state(state)
            self.assertAlmostEqual(amplitude.real, expected.real)
            self.assertAlmostEqual(amplitude.imag, expected.imag)

    def test_space_statevector_matches_basis_amplitudes_for_single_qubit_gates_on_different_rows(self):
        for qudit in range(3):
            with self.subTest(qudit=qudit):
                qc = self.QuantumCircuit(3)
                qc.x(qudit)
                self.assert_space_matches_check_state(qc)

    def test_space_statevector_matches_basis_amplitudes_for_cx_control_above_target(self):
        qc = self.QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 3)
        self.assert_space_matches_check_state(qc)

    def test_space_statevector_matches_basis_amplitudes_for_cx_control_below_target(self):
        qc = self.QuantumCircuit(4)
        qc.h(3)
        qc.cx(3, 0)
        self.assert_space_matches_check_state(qc)

    def test_space_statevector_handles_vertical_boundaries_that_change_between_rows(self):
        qc = self.QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 3)
        qc.h(1)
        qc.cx(1, 2)
        self.assert_space_matches_check_state(qc)

    def test_space_statevector_handles_ccx_with_target_between_controls(self):
        qc = self.QuantumCircuit(4)
        qc.h(0)
        qc.h(3)
        qc.ccx(0, 3, 1)
        self.assert_space_matches_check_state(qc)

    def test_space_statevector_allows_rows_with_no_shared_vertical_bond(self):
        qc = self.QuantumCircuit(5)
        qc.h([0, 2])
        qc.cx(0, 4)
        qc.cx(2, 3)
        qc.ccx(0, 2, 1)
        self.assert_space_matches_check_state(qc)

    def test_space_mps_is_rejected_with_clear_error(self):
        qc = self.QuantumCircuit(3)
        qc.h(0)
        with self.assertRaisesRegex(
            NotImplementedError,
            "Space contraction is only implemented for the statevector representation.",
        ):
            qc.contract(scheme="space", representation="mps", eps=0)


if __name__ == "__main__":
    unittest.main()
