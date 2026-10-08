# Micrograd — A Tiny Autograd Engine, Built From Scratch

Implemented a scalar-valued automatic differentiation engine from an empty file: no
ML libraries involved. This is the mechanism that every deep learning framework —
PyTorch, TensorFlow — is built on top of, just without the speed and GPU support.

Built while following Andrej Karpathy's micrograd lecture from
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), typing every line myself.

---

## Why I Built This

I wanted to understand what `.backward()` is actually doing before trusting it
in bigger projects. Building the autograd engine by hand — the computation graph,
the chain rule, the reverse traversal — removed the "magic" and replaced it with
something I can now debug and reason about.

---

## What Is Implemented

- A `Value` class wrapping a scalar, tracking its data, gradient, the operation
  that produced it, and its parent nodes in the computation graph
- Operator overloading (`+`, `*`, `**`, etc.) so normal Python math builds the graph
  automatically as expressions are written
- Per-operation local gradient functions (`_backward`) implementing the chain rule
- Topological sort of the computation graph, then reverse traversal to propagate
  gradients from the output back to every input (`.backward()`)
- A tiny multi-layer perceptron built entirely out of `Value` objects and trained
  with manual gradient descent, to prove the engine actually works end to end
- A check against PyTorch: the same tanh neuron built with `torch` tensors gives the
  same output (0.7071) and the same gradients (x1 -1.5, w1 1.0, x2 0.5, w2 0.0)

---

## How to Run

```bash
git clone https://github.com/its-Ravi-Singh/neural-networks-from-scratch.git
cd neural-networks-from-scratch/micrograd
jupyter notebook micrograd_from_scratch.ipynb
```

---

## Tech Stack

`Python` — no external ML libraries used in the engine itself. PyTorch is only used for the gradient check.

---

## What I Learned

Backpropagation is just the chain rule applied systematically to a graph, computed
in reverse order because each node needs its output's gradient before it can
compute its own. Gradients must be zeroed before every new `.backward()` call, or
they silently accumulate across training steps.

---

*Contact: ravisingh7421@gmail.com · [LinkedIn](https://linkedin.com/in/ravi-rajaram-singh-47551a206)*
