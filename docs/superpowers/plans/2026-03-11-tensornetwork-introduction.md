# TensorNetwork Introduction Notebook Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `tensornetwork_introduction.ipynb` into a stronger tutorial with three manually constructed complex tensor-network families, each rendered in 2D and 3D and contracted in a readable way.

**Architecture:** Keep the notebook self-contained: one shared setup section plus three independent example families. Each family should build nodes manually with `tn.Node`, connect edges with `^`, render the connected component with `tensor_network_viz`, and demonstrate at least one contraction path using direct TensorNetwork APIs.

**Tech Stack:** Python 3.12, Jupyter Notebook, `tensornetwork`, `numpy`, `matplotlib`, `tensor_network_viz`

---

## File Structure

- Modify: `c:\Users\aleja\Documents\qiskit_tn\Mini-Qiskit_with_Tensor_Networks\tensornetwork_introduction.ipynb`
  - Replace the current minimal tutorial with the new three-family tutorial.
- Reference: `c:\Users\aleja\Documents\qiskit_tn\Mini-Qiskit_with_Tensor_Networks\docs\superpowers\specs\2026-03-11-tensornetwork-introduction-design.md`
  - Use as the source of scope and acceptance criteria.
- Reference: `c:\Users\aleja\Documents\qiskit_tn\Mini-Qiskit_with_Tensor_Networks\qiskit_from_scratch.ipynb`
  - Keep the notebook style aligned with the repo’s actual TensorNetwork usage.

## Chunk 1: Define Notebook Skeleton and Shared Helpers

### Task 1: Replace the current notebook outline

**Files:**
- Modify: `c:\Users\aleja\Documents\qiskit_tn\Mini-Qiskit_with_Tensor_Networks\tensornetwork_introduction.ipynb`

- [ ] **Step 1: Read the current notebook and list its existing section headings**

Run:

```powershell
python -c "import json; from pathlib import Path; nb=json.loads(Path('tensornetwork_introduction.ipynb').read_text(encoding='utf-8')); print([(''.join(c.get('source', [])).splitlines()[0] if c.get('cell_type')=='markdown' and ''.join(c.get('source', [])).strip() else None) for c in nb['cells']])"
```

Expected: a short list of current markdown headings from the old notebook.

- [ ] **Step 2: Write the new markdown skeleton in the notebook**

Include these top-level sections in English:

```text
# Tutorial: Complex Tensor Networks by Hand
## Outline
## Shared setup
## Family 1 - Layered feed-forward network
## Family 2 - Loopy lattice patch
## Family 3 - Hierarchical tree network
## Topology comparison
```

Expected: the notebook now has the approved tutorial structure before detailed code is added.

- [ ] **Step 3: Verify the notebook JSON still loads**

Run:

```powershell
python -c "import json; from pathlib import Path; json.loads(Path('tensornetwork_introduction.ipynb').read_text(encoding='utf-8')); print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add tensornetwork_introduction.ipynb
git commit -m "refactor: replace TensorNetwork notebook structure"
```

### Task 2: Add the shared helper cell

**Files:**
- Modify: `c:\Users\aleja\Documents\qiskit_tn\Mini-Qiskit_with_Tensor_Networks\tensornetwork_introduction.ipynb`

- [ ] **Step 1: Write the helper code cell**

Add one shared setup code cell containing:

```python
from __future__ import annotations

import numpy as np
import tensornetwork as tn
import matplotlib.pyplot as plt

from tensor_network_viz import PlotConfig, show_tensor_network

rng = np.random.default_rng(7)

def normalized_tensor(shape):
    data = rng.normal(size=shape)
    norm = np.linalg.norm(data.reshape(-1))
    return data / norm if norm else data

def reachable_nodes(seed):
    return sorted(tn.reachable(seed), key=lambda node: node.name)

def render_2d_3d(seed, title, *, figsize_2d=(8, 5), figsize_3d=(8, 6), layout_iterations=300):
    nodes = reachable_nodes(seed)
    fig2d, ax2d = show_tensor_network(
        nodes,
        engine="tensornetwork",
        view="2d",
        config=PlotConfig(figsize=figsize_2d, layout_iterations=layout_iterations),
        show=False,
    )
    ax2d.set_title(f"{title} (2D)")
    plt.show()

    fig3d, ax3d = show_tensor_network(
        nodes,
        engine="tensornetwork",
        view="3d",
        config=PlotConfig(figsize=figsize_3d, layout_iterations=layout_iterations),
        show=False,
    )
    ax3d.set_title(f"{title} (3D)")
    plt.show()
    return nodes

def shape_report(label, tensor):
    print(f"{label}: shape={tensor.shape}")
```

- [ ] **Step 2: Run only the setup cell in isolation**

Run:

```powershell
python -c "import json; from pathlib import Path; ns={}; nb=json.loads(Path('tensornetwork_introduction.ipynb').read_text(encoding='utf-8')); code_cells=[c for c in nb['cells'] if c.get('cell_type')=='code']; exec(''.join(code_cells[0]['source']), ns); print(sorted(k for k in ns if k in {'normalized_tensor','reachable_nodes','render_2d_3d','shape_report'}))"
```

Expected: all four helper names are printed.

- [ ] **Step 3: Commit**

```bash
git add tensornetwork_introduction.ipynb
git commit -m "feat: add shared TensorNetwork notebook helpers"
```

## Chunk 2: Add the Three Network Families

### Task 3: Add the layered feed-forward network section

**Files:**
- Modify: `c:\Users\aleja\Documents\qiskit_tn\Mini-Qiskit_with_Tensor_Networks\tensornetwork_introduction.ipynb`

- [ ] **Step 1: Add explanatory markdown for the layered family**

The markdown should explain:

- why the topology is acyclic,
- what the input, hidden, and readout tensors represent structurally,
- why staged contraction is easy to follow.

- [ ] **Step 2: Add the code cell that builds and renders the layered network**

Use a small hand-built network such as:

```python
x0 = tn.Node(normalized_tensor((3,)), name="x0", axis_names=["f0"])
x1 = tn.Node(normalized_tensor((3,)), name="x1", axis_names=["f1"])
x2 = tn.Node(normalized_tensor((3,)), name="x2", axis_names=["f2"])

h0 = tn.Node(normalized_tensor((3, 3, 4)), name="h0", axis_names=["f0", "f1", "h0"])
h1 = tn.Node(normalized_tensor((3, 4, 4)), name="h1", axis_names=["f2", "h0_in", "h1"])
readout = tn.Node(normalized_tensor((4, 2)), name="readout", axis_names=["h1_in", "class"])

x0["f0"] ^ h0["f0"]
x1["f1"] ^ h0["f1"]
h0["h0"] ^ h1["h0_in"]
x2["f2"] ^ h1["f2"]
h1["h1"] ^ readout["h1_in"]

render_2d_3d(x0, "Layered feed-forward network")
```

- [ ] **Step 3: Add the contraction cell for the layered family**

Contract in readable stages:

```python
layer01 = tn.contract_between(x0, h0, name="x0_h0", output_edge_order=[x1["f1"], h0["h0"]], axis_names=["f1", "h0"])
layer02 = tn.contract_between(x1, layer01, name="hidden0", output_edge_order=[layer01["h0"]], axis_names=["h0"])
layer03 = tn.contract_between(layer02, h1, name="hidden1", output_edge_order=[x2["f2"], h1["h1"]], axis_names=["f2", "h1"])
layer04 = tn.contract_between(x2, layer03, name="pre_readout", output_edge_order=[layer03["h1"]], axis_names=["h1"])
layer05 = tn.contract_between(layer04, readout, name="output", output_edge_order=[readout["class"]], axis_names=["class"])
shape_report("Layered output", layer05.tensor)
print(layer05.tensor)
```

Expected: a small output vector with shape `(2,)`.

- [ ] **Step 4: Run the layered family cells**

Run:

```powershell
python -c "import json; from pathlib import Path; import matplotlib; matplotlib.use('Agg'); nb=json.loads(Path('tensornetwork_introduction.ipynb').read_text(encoding='utf-8')); ns={}; [exec(''.join(cell.get('source', [])), ns) for cell in nb['cells'] if cell.get('cell_type')=='code']; print('executed_through_layered_family')"
```

Expected: no exceptions; the final printed output includes shape `(2,)`.

- [ ] **Step 5: Commit**

```bash
git add tensornetwork_introduction.ipynb
git commit -m "feat: add layered TensorNetwork example"
```

### Task 4: Add the loopy lattice patch section

**Files:**
- Modify: `c:\Users\aleja\Documents\qiskit_tn\Mini-Qiskit_with_Tensor_Networks\tensornetwork_introduction.ipynb`

- [ ] **Step 1: Add explanatory markdown for the lattice family**

The markdown should explain:

- why this is a loopy topology,
- how boundary legs differ from internal bonds,
- why the 2D and 3D renders are especially useful here.

- [ ] **Step 2: Add the code cell that builds and renders the lattice patch**

Use a small patch such as `2x3` with rank-4 site tensors:

```python
sites = {}
for row in range(2):
    for col in range(3):
        sites[row, col] = tn.Node(
            normalized_tensor((2, 2, 2, 2)),
            name=f"A{row}{col}",
            axis_names=["up", "right", "down", "left"],
        )

for row in range(2):
    for col in range(3):
        if col < 2:
            sites[row, col]["right"] ^ sites[row, col + 1]["left"]
        if row < 1:
            sites[row, col]["down"] ^ sites[row + 1, col]["up"]

boundary = []
for col in range(3):
    if col != 1:
        probe = tn.Node(normalized_tensor((2,)), name=f"top{col}", axis_names=[f"top{col}"])
        probe[f"top{col}"] ^ sites[0, col]["up"]
        boundary.append(probe)

for col in range(3):
    probe = tn.Node(normalized_tensor((2,)), name=f"bottom{col}", axis_names=[f"bottom{col}"])
    probe[f"bottom{col}"] ^ sites[1, col]["down"]
    boundary.append(probe)

for row in range(2):
    probe = tn.Node(normalized_tensor((2,)), name=f"left{row}", axis_names=[f"left{row}"])
    probe[f"left{row}"] ^ sites[row, 0]["left"]
    boundary.append(probe)

for row in range(2):
    probe = tn.Node(normalized_tensor((2,)), name=f"right{row}", axis_names=[f"right{row}"])
    probe[f"right{row}"] ^ sites[row, 2]["right"]
    boundary.append(probe)

all_lattice_nodes = list(sites.values()) + boundary
render_2d_3d(sites[0, 0], "Loopy lattice patch", figsize_2d=(9, 5), figsize_3d=(9, 7))
```

- [ ] **Step 3: Add the contraction cell for the lattice family**

Contract the partially closed patch to one remaining probe leg:

```python
lattice_result = tn.contractors.greedy(
    all_lattice_nodes,
    output_edge_order=[sites[0, 1]["up"]],
)
shape_report("Lattice result", lattice_result.tensor)
print(lattice_result.tensor)
```

Expected: a small output vector with shape `(2,)`.

- [ ] **Step 4: Run the lattice family cells**

Run:

```powershell
python -c "import json; from pathlib import Path; import matplotlib; matplotlib.use('Agg'); nb=json.loads(Path('tensornetwork_introduction.ipynb').read_text(encoding='utf-8')); ns={}; [exec(''.join(cell.get('source', [])), ns) for cell in nb['cells'] if cell.get('cell_type')=='code']; print('executed_through_lattice_family')"
```

Expected: no exceptions; the final printed output includes shape `(2,)`.

- [ ] **Step 5: Commit**

```bash
git add tensornetwork_introduction.ipynb
git commit -m "feat: add loopy lattice TensorNetwork example"
```

### Task 5: Add the hierarchical tree section

**Files:**
- Modify: `c:\Users\aleja\Documents\qiskit_tn\Mini-Qiskit_with_Tensor_Networks\tensornetwork_introduction.ipynb`

- [ ] **Step 1: Add explanatory markdown for the tree family**

The markdown should explain:

- why trees are hierarchical but not loopy,
- why bottom-up contraction is natural,
- how this differs from the layered and lattice examples.

- [ ] **Step 2: Add the code cell that builds and renders the tree network**

Use a binary tree with eight leaves:

```python
leaves = [tn.Node(normalized_tensor((3,)), name=f"leaf{i}", axis_names=[f"x{i}"]) for i in range(8)]

merge_a = [tn.Node(normalized_tensor((3, 3, 4)), name=f"merge_a{i}", axis_names=[f"l{i}", f"r{i}", f"p{i}"]) for i in range(4)]
merge_b = [tn.Node(normalized_tensor((4, 4, 5)), name=f"merge_b{i}", axis_names=[f"l2_{i}", f"r2_{i}", f"p2_{i}"]) for i in range(2)]
root = tn.Node(normalized_tensor((5, 5)), name="root", axis_names=["left_root", "right_root"])

for i in range(4):
    leaves[2 * i][f"x{2 * i}"] ^ merge_a[i][f"l{i}"]
    leaves[2 * i + 1][f"x{2 * i + 1}"] ^ merge_a[i][f"r{i}"]

for i in range(2):
    merge_a[2 * i][f"p{2 * i}"] ^ merge_b[i][f"l2_{i}"]
    merge_a[2 * i + 1][f"p{2 * i + 1}"] ^ merge_b[i][f"r2_{i}"]

merge_b[0]["p2_0"] ^ root["left_root"]
merge_b[1]["p2_1"] ^ root["right_root"]

render_2d_3d(leaves[0], "Hierarchical tree network", figsize_2d=(9, 5), figsize_3d=(9, 7))
```

- [ ] **Step 3: Add the contraction cell for the tree family**

Contract level by level:

```python
level1 = []
for i in range(4):
    left = tn.contract_between(leaves[2 * i], merge_a[i], name=f"leaf_merge_{i}_left")
    merged = tn.contract_between(leaves[2 * i + 1], left, name=f"level1_{i}")
    level1.append(merged)

level2 = []
for i in range(2):
    left = tn.contract_between(level1[2 * i], merge_b[i], name=f"level2_left_{i}")
    merged = tn.contract_between(level1[2 * i + 1], left, name=f"level2_{i}")
    level2.append(merged)

top = tn.contract_between(level2[0], root, name="top")
tree_result = tn.contract_between(level2[1], top, name="tree_result")
shape_report("Tree result", tree_result.tensor)
print(tree_result.tensor)
```

Expected: a scalar or a very small tensor.

- [ ] **Step 4: Run the tree family cells**

Run:

```powershell
python -c "import json; from pathlib import Path; import matplotlib; matplotlib.use('Agg'); nb=json.loads(Path('tensornetwork_introduction.ipynb').read_text(encoding='utf-8')); ns={}; [exec(''.join(cell.get('source', [])), ns) for cell in nb['cells'] if cell.get('cell_type')=='code']; print('executed_through_tree_family')"
```

Expected: no exceptions; the final output shape is empty `()` or very small.

- [ ] **Step 5: Commit**

```bash
git add tensornetwork_introduction.ipynb
git commit -m "feat: add hierarchical tree TensorNetwork example"
```

## Chunk 3: Finish Tutorial Quality and Verification

### Task 6: Add the closing comparison section

**Files:**
- Modify: `c:\Users\aleja\Documents\qiskit_tn\Mini-Qiskit_with_Tensor_Networks\tensornetwork_introduction.ipynb`

- [ ] **Step 1: Add final markdown comparing the three topologies**

Include one short comparison table or bullet list covering:

- acyclic layered networks,
- loopy lattice networks,
- hierarchical tree networks.

- [ ] **Step 2: Re-read the notebook text for clarity**

Check manually that:

- all markdown is in English,
- each section explains why the topology is interesting,
- the notebook reads as a tutorial, not as a code dump.

- [ ] **Step 3: Commit**

```bash
git add tensornetwork_introduction.ipynb
git commit -m "docs: improve TensorNetwork tutorial explanations"
```

### Task 7: Run full notebook verification

**Files:**
- Modify: `c:\Users\aleja\Documents\qiskit_tn\Mini-Qiskit_with_Tensor_Networks\tensornetwork_introduction.ipynb`

- [ ] **Step 1: Execute all notebook code cells headlessly**

Run:

```powershell
python -c "import json; from pathlib import Path; import matplotlib; matplotlib.use('Agg'); nb=json.loads(Path('tensornetwork_introduction.ipynb').read_text(encoding='utf-8')); ns={}; [exec(''.join(cell.get('source', [])), ns) for cell in nb['cells'] if cell.get('cell_type')=='code']; print('executed_all_code_cells')"
```

Expected: `executed_all_code_cells`

- [ ] **Step 2: Sanity-check the notebook structure**

Run:

```powershell
python -c "import json; from pathlib import Path; nb=json.loads(Path('tensornetwork_introduction.ipynb').read_text(encoding='utf-8')); print('cells=', len(nb['cells'])); print('markdown=', sum(c['cell_type']=='markdown' for c in nb['cells'])); print('code=', sum(c['cell_type']=='code' for c in nb['cells']))"
```

Expected: a balanced mix of markdown and code cells, not a code-only notebook.

- [ ] **Step 3: Review the final diff**

Run:

```bash
git diff -- tensornetwork_introduction.ipynb
```

Expected: the diff shows the old minimal notebook replaced by the three-family tutorial.

- [ ] **Step 4: Commit**

```bash
git add tensornetwork_introduction.ipynb
git commit -m "feat: rebuild TensorNetwork introduction notebook"
```
