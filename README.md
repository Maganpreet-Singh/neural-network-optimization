# Neural Network Optimization

A practical collection of TensorFlow/Keras notebooks for understanding and experimenting with neural-network optimization and training techniques.

## Topics

- **Optimizers** — AdaGrad, Adam, RMSProp, SGD with Momentum, and Nesterov Accelerated Gradient
- **Regularization** — L1/L2 regularization, Dropout, Batch Normalization, and Early Stopping
- **Preprocessing** — Feature Scaling
- **Weight Initialization** — basic initialization, Xavier/Glorot, and He initialization
- **Hyperparameter Optimization** — Keras Tuner
- **Training Techniques** — Exponentially Weighted Moving Average

## Repository Structure

```text
neural-network-optimization/
├── data/
│   ├── climate/
│   │   └── DailyDelhiClimate.csv
│   └── datasets/
│       ├── concentric_circles.csv
│       ├── diabetes.csv
│       └── ushape.csv
├── notebooks/
│   ├── initialization/
│   ├── optimization/
│   ├── optimizers/
│   ├── preprocessing/
│   └── regularization/
└── README.md
```

## Getting Started

Create a Python environment and install the libraries required by the notebooks, then open the notebooks with Jupyter or Google Colab.

Most notebooks are designed as self-contained learning experiments, so run them from top to bottom.

## Learning Path

1. Feature scaling
2. Weight initialization
3. SGD and momentum-based optimizers
4. AdaGrad, RMSProp, and Adam
5. Regularization and dropout
6. Batch normalization and early stopping
7. Hyperparameter tuning with Keras Tuner
8. Compare optimization strategies experimentally

## Datasets

Datasets used by the notebooks are kept under `data/` so notebooks remain separated from raw data.

## Tech Stack

Python · TensorFlow · Keras · NumPy · Pandas · Scikit-learn · Matplotlib · Jupyter
