import json
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "qiskit_from_scratch_tensorkrowch.ipynb"
DEFINITION_CELLS = (3, 5, 7, 9, 11, 13, 15)


def load_notebook_namespace():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {}
    for cell_index in DEFINITION_CELLS:
        exec("".join(notebook["cells"][cell_index]["source"]), namespace)
    return namespace


def dense_amplitude(tensor, state: int):
    return tensor.reshape(-1)[state].item()


class TensorKrowchNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = load_notebook_namespace()
        cls.QuantumCircuit = namespace["QuantumCircuit"]
        cls.device = namespace["device"]
        cls.mps_nodes_to_dense_tensor = staticmethod(namespace["mps_nodes_to_dense_tensor"])

    def assert_state_matches_basis_queries(self, qc, *, representation="statevector", eps=0):
        state_vector = qc.state_vector()
        for state, amplitude in enumerate(state_vector):
            expected = qc.check_state(state, representation=representation, eps=eps)
            self.assertAlmostEqual(amplitude.real, expected.real)
            self.assertAlmostEqual(amplitude.imag, expected.imag)

    def test_notebook_definitions_load_and_create_tensors_on_runtime_device(self):
        qc = self.QuantumCircuit(2)
        self.assertEqual(qc.tensors[0][0].tensor.device.type, self.device.type)

    def test_statevector_matches_check_state_without_mutating_depth(self):
        qc = self.QuantumCircuit(3)
        qc.h([0, 1, 2])
        qc.cx(0, 2)
        qc.rz(0.7, 1)

        initial_depth = qc.depth
        self.assert_state_matches_basis_queries(qc)
        qc.check_state(5)
        self.assertEqual(qc.depth, initial_depth)

    def test_expected_is_repeatable_and_non_destructive(self):
        qc = self.QuantumCircuit(3)
        qc.h([0, 1])
        qc.cx(0, 2)

        operator = self.QuantumCircuit(3, name="Operator")
        operator.x(2)
        operator.rz(0.3, 1)

        initial_depth = qc.depth
        expected_once = qc.expected(operator.to_gate())
        expected_twice = qc.expected(operator.to_gate())

        self.assertAlmostEqual(expected_once.real, expected_twice.real)
        self.assertAlmostEqual(expected_once.imag, expected_twice.imag)
        self.assertEqual(qc.depth, initial_depth)

    def test_space_statevector_matches_basis_amplitudes(self):
        qc = self.QuantumCircuit(4)
        qc.h([0, 3])
        qc.cx(0, 3)
        qc.ccx(0, 3, 1)

        result = qc.contract(scheme="space", representation="statevector", eps=0)
        for state in range(2 ** qc.n_qudits):
            amplitude = dense_amplitude(result.tensor, state)
            expected = qc.check_state(state)
            self.assertAlmostEqual(amplitude.real, expected.real)
            self.assertAlmostEqual(amplitude.imag, expected.imag)

    def test_mps_eps_zero_matches_statevector(self):
        qc = self.QuantumCircuit(4)
        qc.h([0, 1, 2, 3])
        qc.cx(0, 3)
        qc.cz(1, 2)
        qc.ccx(0, 2, 1)

        mps_result = qc.contract(representation="mps", eps=0)
        mps_dense = (
            self.mps_nodes_to_dense_tensor(mps_result).reshape(-1).detach().cpu().numpy()
        )
        expected_dense = qc.state_vector()

        self.assertEqual(len(mps_result), qc.n_qudits)
        np.testing.assert_allclose(mps_dense, expected_dense)

    def test_mps_with_truncation_stays_close_to_statevector(self):
        qc = self.QuantumCircuit(5)
        qc.h([0, 1, 2, 3, 4])
        qc.cx(0, 4)
        qc.cx(1, 3)
        qc.ccx(0, 2, 1)
        qc.rz(0.25, 4)

        approx_mps = qc.contract(representation="mps", eps=1e-6)
        approx_dense = (
            self.mps_nodes_to_dense_tensor(approx_mps).reshape(-1).detach().cpu().numpy()
        )
        exact_dense = qc.state_vector()

        self.assertEqual(approx_dense.shape, exact_dense.shape)
        np.testing.assert_allclose(approx_dense, exact_dense, atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_execution_keeps_tensors_on_gpu(self):
        qc = self.QuantumCircuit(2)
        qc.h(0)
        result = qc.contract(representation="statevector", eps=0)

        self.assertEqual(qc.tensors[0][0].tensor.device.type, "cuda")
        self.assertEqual(result.tensor.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
