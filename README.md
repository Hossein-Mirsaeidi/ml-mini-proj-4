# Machine Learning Mini-Project 4 — Regularization, Optimization, CNNs, LSTMs & Word Embeddings

A single notebook covering five deep-learning topics with TensorFlow/Keras:

1. **Deep feedforward networks & backpropagation** — forward/backward equations for a
   2→3→2→1 network, plus a NumPy verification of one hand-computed update step.
2. **Regularization** — an MLP regressor on a synthetic cubic function, comparing
   *early stopping* vs *dropout* (training time and test MSE).
3. **Optimization** — a 3-layer MLP on MNIST comparing *SGD (momentum)* vs *Adam*.
4. **CNN vs LSTM** — a CNN on CIFAR-10 (confusion matrix, ROC, misclassified images)
   and an LSTM on IMDB sentiment (accuracy, precision, recall, F1).
5. **CBOW word embeddings** — a Continuous Bag-of-Words model learning 2-D embeddings
   directly, and 10-D embeddings projected to 2-D with PCA.

## Files

- `Regularization_Optimization_CNN_LSTM_Embeddings.ipynb` — main notebook (code, outputs, markdown).
- `figures/` — every plot exported as a vector PDF.
- `data/cbow_corpus.txt` — the ~1000-word text used for the CBOW model (Part 5).
- `Doc/report.tex` / `Doc/report.pdf` — the typeset report.
- `video_link.txt` — link to the video presentation.

## Setup

```bash
python3 -m venv venv
# ROS injects PYTHONPATH into the shell; clear it so the venv stays isolated
env -u PYTHONPATH ./venv/bin/pip install -r requirements.txt
```

## Run

The MNIST/CIFAR-10/IMDB datasets download automatically to `~/.keras/datasets` on
first use. Launch Jupyter with `PYTHONPATH` cleared to avoid the system ROS packages:

```bash
env -u PYTHONPATH ./venv/bin/jupyter lab Regularization_Optimization_CNN_LSTM_Embeddings.ipynb
```

A CUDA-capable GPU is used automatically if present (developed on a GTX 1650, 4 GB).
