# TensorNetwork Introduction Notebook Design

Date: 2026-03-11
Target file: `tensornetwork_introduction.ipynb`

## Objective

Replace the current minimal notebook with a stronger tutorial that showcases manual construction,
visualization, and contraction of more complex tensor networks using the `tensornetwork` API used in
this repository.

The notebook should remain instructional and runnable, but it no longer needs to center quantum gates
or amplitudes. The emphasis is on visually interesting tensor-network topologies that are built by
hand with `tn.Node` and `^`, rendered in both 2D and 3D with `tensor_network_viz`, and contracted in
a way that teaches practical TensorNetwork usage.

## Audience

- Readers of this repository who want a better intuition for nontrivial tensor-network structures.
- Users who already saw the mini-Qiskit notebook and now want more expressive examples of the
  underlying TensorNetwork machinery.

## Scope

The notebook will cover three network families:

1. A layered feed-forward style network.
2. A loopy lattice patch.
3. A hierarchical tree network.

Each family will:

- be assembled manually with `tn.Node` objects,
- use named axes and explicit `^` edge connections,
- be rendered in both 2D and 3D,
- include one contraction path that is short enough to follow,
- explain what makes the topology structurally interesting.

## Out of Scope

- Using project helpers such as `build_control_tensors` or `build_control_gate_layers`.
- Framing the examples as quantum gates unless that adds clarity.
- Full performance benchmarking of contraction strategies.
- Deep MPS compression content; that belongs in a separate notebook.

## Notebook Structure

### 1. Introduction

Brief tutorial framing in English:

- what the notebook is for,
- what the reader should already know,
- what they will learn.

### 2. Shared Setup

One shared code cell with:

- imports,
- deterministic random-number generator,
- a helper to normalize random tensors,
- a helper to collect connected nodes with `tn.reachable(...)`,
- a helper to render a connected component in 2D and 3D,
- a helper for concise contraction reporting.

The setup should keep tensor sizes small so the notebook stays fast.

### 3. Family A: Layered Feed-Forward Network

Design:

- several rank-1 input tensors,
- one or more intermediate rank-3 or rank-4 tensors,
- a final readout tensor.

Teaching goals:

- explicit wiring with `^`,
- understanding open versus internal legs,
- partial contractions with `tn.contract_between(...)`,
- how output edge order changes intermediate tensor shape.

Expected result:

- the section should end with a compact output tensor or scalar,
- the text should explain why layered acyclic networks are easy to contract in stages.

### 4. Family B: Loopy Lattice Patch

Design:

- a small PEPS-like patch such as `2x3` or `3x3`,
- local tensors with internal bonds between neighboring sites,
- a few open boundary legs left dangling.

Teaching goals:

- show a visibly richer topology than a chain,
- demonstrate why loops make the structure more interesting to inspect,
- contract to a reduced final object without making the example too large.

Expected result:

- the visualization should be the highlight of the section,
- the contraction story should be readable and not require advanced path optimization discussion.

### 5. Family C: Hierarchical Tree Network

Design:

- several leaf tensors,
- binary merger tensors,
- a root tensor producing a scalar or low-rank output.

Teaching goals:

- contrast tree topology with the loopy lattice,
- demonstrate staged bottom-up contraction,
- give the reader a second nontrivial contraction pattern that is not chain-like.

Expected result:

- a scalar or very small root tensor,
- short explanation of why trees are structurally simpler than loopy networks.

### 6. Closing Comparison

Final markdown section comparing the three families:

- layered: acyclic and sequential,
- lattice: loopy and visually dense,
- tree: hierarchical and reduction-oriented.

The close should point out that TensorNetwork is useful beyond quantum-circuit diagrams.

## Visualization Requirements

- Every family should be rendered in 2D and 3D.
- Use `tensor_network_viz.show_tensor_network(...)` with `engine="tensornetwork"`.
- Keep figures readable with explicit titles and stable layout settings.
- Prefer visually distinct topologies over domain-heavy semantics.

## Contraction Requirements

- At least one contraction per family must be demonstrated.
- Use direct TensorNetwork APIs such as `tn.contract_between(...)` or
  `tn.contractors.greedy(...)`.
- The code should print final tensor shapes and, when small enough, the resulting values.
- The contraction path should be simple enough for tutorial purposes.

## Content Style

- All markdown text in English.
- More explanation than the current notebook: the reader should understand not only what is being
  called, but why that topology is interesting.
- Keep code cells focused and runnable from top to bottom.

## Risks and Mitigations

- Risk: 3D rendering may be visually noisy.
  Mitigation: keep networks small, use explicit titles, and reuse stable layouts where possible.

- Risk: random tensors can produce unreadable large numbers.
  Mitigation: normalize tensors and keep dimensions small.

- Risk: lattice contractions may become awkward if the network is too large.
  Mitigation: use a small patch and choose a readable contraction path.

## Verification Plan

- Execute all notebook code cells sequentially in a headless environment.
- Confirm that each family:
  - renders in 2D and 3D without API errors,
  - contracts successfully,
  - produces a final tensor or scalar with the expected rank/shape.
- Check that the notebook remains concise enough to be read as a tutorial, not as a dump of code.
