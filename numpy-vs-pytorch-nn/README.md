# Neural Network from Scratch — NumPy and PyTorch (work in progress)

Two notebooks from when I started learning how neural networks train. The NumPy version is not finished yet, so this folder is a work in progress.

---

## What Is Here

**`1/1.1/Start.ipynb` — NumPy MLP for MNIST (unfinished)**
- Loads the Kaggle digit-recognizer data and keeps 1,000 images aside for testing
- Parameter init, ReLU, softmax, forward pass and one-hot labels
- A first try at the backward pass and the gradient descent update

It does not run end to end yet. Bugs I still need to fix:
- `W2` should be 10x10, not 10x784
- the output softmax is applied to `A1` instead of `Z2`
- `np.arrange` should be `np.arange`, and `DZ1`/`DZ2` should be `dZ1`/`dZ2`
- the bias gradients sum over the wrong axis
- `update_params` updates `B1` twice and never updates `B2`
- there is no training loop yet

**`Pytorch_NN.ipynb` — PyTorch practice (Deep Learning coursework)**
- Autograd basics on a small expression
- Linear regression on synthetic data with a train/val/test split
- An MLP classifier on a tabular crime-incidents dataset, compared with logistic regression, random forest and KNN (all between 72% and 74% test accuracy; the MLP almost never predicts the minority class)

---

## Next Steps

1. Fix the bugs above and train the NumPy MLP on MNIST.
2. Build the same network in PyTorch with the same starting weights.
3. Compare the NumPy gradients with PyTorch autograd on the same batch.

For a from-scratch autograd engine that works end to end, see [`../micrograd`](../micrograd).

---

## Tech Stack

`Python` · `NumPy` · `PyTorch` · `scikit-learn` · `Matplotlib`

---

*Contact: ravisingh7421@gmail.com · [LinkedIn](https://linkedin.com/in/ravi-rajaram-singh-47551a206)*
