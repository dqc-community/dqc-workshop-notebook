#set document(
  author: "Bosonic",
  date: datetime.today(),
  title: "Distributed Quantum Computing Workshop",
  description: "Overview of DQC workshop contents for circulation"
)

#show title: set align(center)
#show heading: set block(below: 1em)
#set par(leading: 0.75em, spacing: 2em)
#show link: set text(blue)

#title()

#box(
  inset: 12pt,
  fill: gray.lighten(90%),
  radius: 8pt,
)[This workshop provides an introduction to key concepts in distributed quantum computing to students with basic familiarity of core quantum computing concepts. Students write their own distributed quantum circuits and are invited to contribute to a rapidly growing field.]

== Presentation

We begin by motivating the problem – what is distributed quantum computing, and why would anybody care about it? After defining the approach and its advantages, as well as highlighting support from major industry players, we move to the application layer. We introduce the concept of communication qubits and illustrate a gate teleportation protocol, drawing an analogy to transpiling down to native gates on standard monolithic hardware. We work through an example (the tutorial described below) before introducing the concept of circuit partitioning as an area for further exploration. We conclude with a list of open questions and an invitation to join our DQC ecosystem and contribute to our open-source initiatives.

== Tutorial

This tutorial is an interactive, code-driven exploration of how quantum circuits scale on monolithic versus distributed quantum hardware, aimed at students comfortable with basic quantum computing (Qiskit). All code is included in an #link("https://github.com/dqc-community/dqc-workshop-notebook")[open-source repo], and students can run the notebook on their own machines using a single terminal command.

The tutorial centers on a simple but powerful test case: $n$-qubit GHZ states. Students first build GHZ circuits in Qiskit and verify that ideal simulations produce only the expected bitstrings in ${0^n, 1^n}$, using a simple fidelity proxy based on measurement counts. They then run the same logical circuits through two backends: a monolithic, heavy-hex superconducting processor (IBM’s `FakeSherbrooke` backend) and a distributed trapped‑ion architecture modeled by the Bosonic SDK, which connects many small ion-trap modules via photonic links.

On the monolithic side, students see how limited connectivity forces SWAP-based routing, inflating depth and two‑qubit gate counts as circuits grow. On the distributed side, they learn how a system of all-to-all connected modules incurs a constant communication overhead per cross‑module gate, so communication cost grows roughly in step with the logical circuit rather than with the physical chip size. Students transpile circuits for various qubit counts on both backends and plot circuit metrics to explore scaling behavior.

The second half of the tutorial translates these circuit-level differences into an execution-time model. Using a simple worst-case time‑to‑solution (TTS) estimator, we compare this model against Qiskit’s own timing estimates, explore how TTS grows with circuit size on both architectures, and then extrapolate to “utility‑scale” regimes by fitting linear models to observed gate counts and projecting out to thousands of qubits. The tutorial concludes with a sandbox section where students can repeat the experiment with their own circuit generator.
