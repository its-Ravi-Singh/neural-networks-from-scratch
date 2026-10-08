# Neural Networks & Deep Learning — From Scratch

**Why this repo exists:** most ML work today means calling `.fit()` or `.backward()` on
a library and trusting it. I wanted to know what's actually happening underneath before
I kept building on top of it, so this is a running collection of core ML/DL mechanics
built from zero and typed out line by line.

This isn't my main portfolio (see links at the bottom for that). It's the fundamentals
underneath it.

---

## What's in here

| Project | What it builds | Core concepts | Stack | Status |
|---|---|---|---|---|
| [`micrograd/`](./micrograd) | A scalar autograd engine, plus a tiny MLP trained with it | Computation graphs, reverse-mode autodiff, backpropagation | Python | Done |
| [`numpy-vs-pytorch-nn/`](./numpy-vs-pytorch-nn) | An MLP in plain NumPy, and PyTorch practice | Manual forward and backward pass | NumPy, PyTorch | In progress |

Related repo: [char-level-language-model](https://github.com/its-Ravi-Singh/char-level-language-model) —
character-level language models, starting with a bigram model built by counting and as a
one-layer neural net.

---

## The projects, briefly

**[micrograd](./micrograd)** — Built a scalar-valued automatic differentiation engine
from scratch: a `Value` class that tracks data, gradient, and the operations that
produced it, with operator overloading so ordinary Python math builds a computation
graph automatically. `.backward()` does a topological sort of that graph and walks it
in reverse, applying the chain rule at every node. I built it while following Andrej
Karpathy's micrograd lecture and checked its gradients against PyTorch autograd on the
same expression.

**[numpy-vs-pytorch-nn](./numpy-vs-pytorch-nn)** — An MLP for MNIST in plain NumPy that
I started but haven't finished (its README lists the bugs left to fix), plus a PyTorch
practice notebook. Once the NumPy version trains, I'll compare its gradients with
PyTorch autograd.

Each folder has its own README with the details.

---

## More of my work

- Portfolio: [its-ravi-singh.github.io](https://its-ravi-singh.github.io)
- GitHub: [@its-Ravi-Singh](https://github.com/its-Ravi-Singh)
- LinkedIn: [ravi-rajaram-singh](https://linkedin.com/in/ravi-rajaram-singh-47551a206)
