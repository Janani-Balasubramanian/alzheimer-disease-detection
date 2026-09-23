<p align="center"><img src=".github/repository-banner.svg" alt="Brain Image Classification — An exploratory deep-learning experiment" width="100%" /></p>

<p align="center"><a href="https://github.com/Janani-Balasubramanian">GitHub profile</a> · <a href="https://github.com/Janani-Balasubramanian/portfolio">Portfolio</a> · <a href="https://github.com/Janani-Balasubramanian/alzheimer-disease-detection/issues">Issues</a></p>

# Brain Image Classification

An exploratory TensorFlow script for classifying brain-image inputs into the labels `EMCI`, `LMCI`, and `NC`. The source was exported from a Colab notebook and includes preprocessing, augmentation, model training, evaluation, and plots.

## Explore the implementation

[`untitled3.py`](untitled3.py) contains the complete experiment.

- Resizes images to 64 × 64 and normalizes pixel values.
- Repeats one image channel across a depth of 32 to construct an input volume.
- Defines a Conv3D classifier with pooling, batch normalization, and dropout.
- Computes classification reports, a confusion matrix, and weighted precision, recall, and F1.
- Saves the model as `brain_advanced_3d_model.h5`.

## Data and environment

The script imports TensorFlow, NumPy, scikit-learn, Matplotlib, and image-loading utilities. The dataset is not included. It expects this structure:

```text
BRAIN IMAGE DATA/
├── train/  # EMCI/, LMCI/, NC/
├── val/    # EMCI/, LMCI/, NC/
└── test/   # EMCI/, LMCI/, NC/
```

## Current limitations

This is research code, not a validated diagnostic application. The depth dimension is generated from repeated 2D slices, not reconstructed from a true 3D scan. The current sequence of valid Conv3D operations and pooling shrinks the depth too far for later convolution layers; revise and validate the architecture before training. No reproducible benchmark results, dataset, or trained weights are committed.
