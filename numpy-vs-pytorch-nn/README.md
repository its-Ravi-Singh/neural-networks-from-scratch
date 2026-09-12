# Neural Network from Scratch — NumPy to PyTorch

Implemented a feedforward neural network in plain NumPy first, then rebuilt it in PyTorch — to make sure I understood what `.backward()` is actually doing before relying on it.

---

## Why I Built This

I kept using PyTorch without fully understanding what was happening during backpropagation. So I stopped and implemented it manually. It turned out to be more understandable than expected, and now when training goes wrong I have a much clearer sense of where to look.

---

## What Is Implemented

**Phase 1 — NumPy only** (`1/1.1/Start.ipynb`)
- Forward pass: linear transformation + activation functions (ReLU, Sigmoid)
- Loss computation: Binary Cross-Entropy and MSE
- Backward pass: manual gradient computation using the chain rule
- Weight updates: vanilla SGD and mini-batch gradient descent
- No autograd, no torch — just NumPy math

**Phase 2 — PyTorch rebuild** (`Pytorch_NN.ipynb`)
- Same architecture implemented using `nn.Module`
- Validated that both versions produce comparable gradients and outputs on the same inputs

---

## Architecture

Input → Dense(ReLU) → Dense(ReLU) → Output(Sigmoid)

Tested on binary classification tasks (XOR and a real dataset).

---

## How to Run

Both parts are notebooks — no separate scripts to install or run from a terminal.

```bash
git clone https://github.com/its-Ravi-Singh/neural-networks-from-scratch.git
cd neural-networks-from-scratch/numpy-vs-pytorch-nn
```

Then open either notebook in Jupyter, Google Colab, or Kaggle (whichever you have set up):
- `1/1.1/Start.ipynb` — the NumPy-only implementation (Phase 1)
- `Pytorch_NN.ipynb` — the PyTorch rebuild (Phase 2)

Requirements: `numpy`, `matplotlib`, `torch` (install with `pip install numpy matplotlib torch` if running locally; Colab/Kaggle already have these).

---

## Tech Stack

`Python` · `NumPy` · `PyTorch` · `Matplotlib`

---

## What I Learned

The chain rule for a multilayer network becomes straightforward once you draw the computation graph on paper. PyTorch's autograd is doing exactly what I implemented manually — just faster and with GPU support.

---

*Deep Learning coursework — University at Buffalo, Fall 2025*
*Contact: raviraja@buffalo.edu · [LinkedIn](https://linkedin.com/in/ravi-rajaram-singh-47551a206)*
