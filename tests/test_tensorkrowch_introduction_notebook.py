import json
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "tensorkrowch_introduction.ipynb"


def executable_python_source(source: str) -> str:
    return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))


class DummyAxes:
    def set_title(self, title: str) -> None:
        return None


def show_tensor_network_stub(*args: Any, **kwargs: Any) -> tuple[None, DummyAxes]:
    return None, DummyAxes()


class TensorKrowchIntroductionNotebookTests(unittest.TestCase):
    def load_notebook(self):
        self.assertTrue(
            NOTEBOOK_PATH.exists(),
            f"Expected notebook at {NOTEBOOK_PATH}, but it does not exist.",
        )
        return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    def test_notebook_has_parallel_tutorial_structure(self):
        notebook = self.load_notebook()

        headings = []
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "markdown":
                continue
            source = "".join(cell.get("source", [])).strip()
            if not source:
                continue
            headings.append(source.splitlines()[0])

        self.assertEqual(
            headings,
            [
                "# Tutorial: Complex Tensor Networks by Hand with TensorKrowch",
                "## Outline",
                "## Shared Setup",
                "## Family 1 - Layered Feed-Forward Network",
                "## Family 2 - Loopy Lattice Patch",
                "## Family 3 - Hierarchical Tree Network",
                "## Topology Comparison",
            ],
        )

    def test_notebook_uses_tensorkrowch_and_not_project_helpers(self):
        notebook = self.load_notebook()
        combined_source = "\n\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn("import tensorkrowch as tk", combined_source)
        self.assertIn('engine="tensorkrowch"', combined_source)
        self.assertNotIn("import tensornetwork as tn", combined_source)
        self.assertNotIn("auxiliary_functions_tensorkrowch", combined_source)
        self.assertNotIn("qiskit_from_scratch_tensorkrowch", combined_source)

    def test_notebook_executes_top_to_bottom_and_exposes_examples(self):
        notebook = self.load_notebook()
        namespace = {}

        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if not source.strip():
                continue
            exec(executable_python_source(source), namespace)
            namespace["show_tensor_network"] = show_tensor_network_stub

        expected_symbols = {
            "normalized_tensor",
            "render_2d_3d",
            "shape_report",
            "build_layered_network",
            "build_lattice_patch",
            "build_tree_network",
            "layer05",
            "lattice_result",
            "tree_result",
        }
        self.assertTrue(expected_symbols.issubset(namespace.keys()))
        self.assertEqual(tuple(namespace["layer05"].tensor.shape), (2,))
        self.assertEqual(tuple(namespace["lattice_result"].tensor.shape), (2,))
        self.assertEqual(tuple(namespace["tree_result"].tensor.shape), ())


if __name__ == "__main__":
    unittest.main()
