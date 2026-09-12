# Neural Networks & Deep Learning — From Scratch

**Why this repo exists:** most ML work today means calling `.fit()` or `.backward()` on
a library and trusting it. I wanted to know what's actually happening underneath before
I kept building on top of it — so this is a running collection of core ML/DL mechanics
implemented from zero: no shortcuts, no copied notebooks, each one typed and understood
line by line.

This isn't my main portfolio (see links at the bottom for that) — it's proof of the
fundamentals underneath it.

---

## What's in here

| Project | What it builds | Core concepts | Stack |
|---|---|---|---|
| [`micrograd/`](./micrograd) | A scalar autograd engine from an empty file | Computation graphs, reverse-mode autodiff, backpropagation | Python only |
| [`numpy-vs-pytorch-nn/`](./numpy-vs-pytorch-nn) | A feedforward NN, twice | Manual forward/backward pass vs. framework autograd | NumPy, PyTorch |

---

## The projects, briefly

**[micrograd](./micrograd)** — Built a scalar-valued automatic differentiation engine
from scratch: a `Value` class that tracks data, gradient, and the operations that
produced it, with operator overloading so ordinary Python math builds a computation
graph automatically. `.backward()` does a topological sort of that graph and walks it
in reverse, applying the chain rule at every node. This is the exact mechanism every
deep learning framework runs — just without the speed.

**[numpy-vs-pytorch-nn](./numpy-vs-pytorch-nn)** — A feedforward network for binary
classification, implemented twice: once in raw NumPy with a hand-derived backward
pass and manual SGD, then rebuilt in PyTorch with `nn.Module` to confirm both produce
the same result. Built to stop treating `.backward()` as a black box.

Each folder has its own README with full implementation details and what I learned.

---

## More of my work

- Portfolio: [its-ravi-singh.github.io](https://its-ravi-singh.github.io)
- GitHub: [@its-Ravi-Singh](https://github.com/its-Ravi-Singh)
- LinkedIn: [ravi-rajaram-singh](https://linkedin.com/in/ravi-rajaram-singh-47551a206)
