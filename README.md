# AdaHAN-for-Image-Classification

This is an unofficial implementation of the AdaHAN model found in the ["Learning Visual Question Answering by Bootstrapping Hard Attention"](https://arxiv.org/pdf/1808.00300.pdf) research paper, by Mateusz Malinowski, Carl Doersch, Adam Santoro, and Peter Battaglia of DeepMind, London.

Here we omit the LSTM embedding as I am only trying to use this for image classification.

The structure of the models are inspired by the following repo: https://github.com/gnouhp/PyTorch-AdaHAN.


## How to run

1. Get the miniplaces dataset using the provided `get_miniplaces.sh` script.
2. Train by running the `train_on_miniplaces_AdaHAN.ipynb` notebook.
3. View results by running the `test_on_miniplaces_AdaHAN.ipynb` notebook.
