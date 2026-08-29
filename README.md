# Machine Learning Mini-Project 4 — Regularization, Optimization, CNNs, LSTMs & Word Embeddings

A single notebook covering five deep-learning topics with TensorFlow/Keras:

1. **Deep feedforward networks & backpropagation** — forward/backward equations for a
   2→3→2→1 network.
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
- `report.pdf` — handwritten report.
- `video_link.txt` — link to the video presentation.

## Setup

```bash
python3 -m venv venv
env -u PYTHONPATH ./venv/bin/pip install -r requirements.txt
source setup_gpu_env.sh      # enable GPU + clear the ROS PYTHONPATH leak
```

`setup_gpu_env.sh` adds the `nvidia-*-cu12` pip wheel libraries to `LD_LIBRARY_PATH`
(TensorFlow 2.21 does not always find them on its own) and unsets the `PYTHONPATH` that
ROS injects into the shell. The project `venv`'s `activate` script already does the same,
so `source venv/bin/activate` is enough once the venv exists.

## Run

The MNIST/CIFAR-10/IMDB datasets download automatically to `~/.keras/datasets` on first use.

```bash
source venv/bin/activate     # sets CUDA lib path + clears ROS PYTHONPATH
jupyter lab Regularization_Optimization_CNN_LSTM_Embeddings.ipynb
```

A CUDA-capable GPU is used automatically if present (developed on a GTX 1650, 4 GB).
Epoch counts follow the specification by default but can be overridden with environment
variables (e.g. `CNN_EPOCHS=5`) for quick trials.
