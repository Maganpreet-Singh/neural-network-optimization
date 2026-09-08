# Neural Network Optimization 🧠

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

> A hands-on deep learning repository for understanding practical techniques that make neural networks train more reliably, generalize better, and avoid common training problems.

## 📌 Overview

Training a neural network is not only about choosing an architecture and calling `fit()`. Input preprocessing, regularization, and training control can have a major impact on convergence, overfitting, and model generalization.

This repository explores core **neural network optimization techniques** through executable Jupyter notebooks. The current focus is on:

- **Feature Scaling** — preparing numerical inputs so optimization behaves more consistently.
- **Dropout Layers** — using stochastic regularization to reduce overfitting and improve generalization.
- **Early Stopping** — stopping training when validation performance stops improving instead of blindly training for a fixed number of epochs.

The notebooks are designed as practical learning material: each topic demonstrates the idea in code and uses visual or quantitative evaluation where appropriate.

## 🎯 Goals

The main goals of this project are to:

1. Understand why preprocessing matters for neural-network training.
2. Study practical regularization strategies.
3. Learn how validation performance can guide the training process.
4. Compare model behavior before and after applying optimization techniques.
5. Build intuition that can be transferred to larger deep-learning projects.

## 📂 Repository Structure

```text
neural-network-optimization/
│
├── Dropout_Layers.ipynb
├── Early_Stopping.ipynb
├── Feature_Scaling.ipynb
└── README.md
```

## 🔬 Topics Covered

### 1. Feature Scaling

Feature scaling transforms numerical input variables into comparable ranges. Neural networks are generally easier to optimize when input features do not operate on dramatically different scales.

Common approaches include:

- Standardization
- Min-Max scaling
- Other normalization strategies depending on the problem

A typical standardization transformation is:

```text
z = (x - μ) / σ
```

where `x` is the original feature value, `μ` is the feature mean, and `σ` is the feature standard deviation.

**Why it matters:**

- Can improve optimization stability.
- Can help gradient-based methods converge more efficiently.
- Prevents large-scale features from dominating smaller-scale features.
- Makes numerical training behavior easier to analyze.

Notebook: [`Feature_Scaling.ipynb`](./Feature_Scaling.ipynb)

---

### 2. Dropout Layers

**Dropout** is a regularization technique that randomly deactivates a fraction of neurons during training.

For a dropout rate `p`, a subset of activations is temporarily removed during each training step. This discourages the model from relying too heavily on individual neurons and can reduce overfitting.

Conceptually:

```text
Dense Layer → Dropout → Dense Layer → Output
```

Typical Keras usage:

```python
from tensorflow.keras.layers import Dropout

model.add(Dropout(0.5))
```

The dropout probability should be treated as a hyperparameter rather than a universal constant. Too little dropout may not provide enough regularization, while too much dropout can make learning unnecessarily difficult.

Notebook: [`Dropout_Layers.ipynb`](./Dropout_Layers.ipynb)

---

### 3. Early Stopping

**Early stopping** is a training-control technique that monitors a validation metric and stops training once the model has stopped improving.

A common workflow is:

```text
Training begins
      ↓
Validation performance improves
      ↓
Continue training
      ↓
Validation performance stops improving
      ↓
Wait for configured patience
      ↓
Stop training
```

Example with Keras:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)
```

This can help reduce unnecessary epochs, limit overfitting, preserve the best validation weights, and make training less dependent on guessing an exact epoch count.

The early-stopping notebook uses a synthetic circular classification problem and evaluates the model using standard classification metrics and decision-boundary visualization.

Notebook: [`Early_Stopping.ipynb`](./Early_Stopping.ipynb)

## 🧪 Dataset Used in the Early Stopping Experiment

The early-stopping notebook creates a binary classification dataset with scikit-learn's `make_circles`:

```python
X, y = make_circles(
    n_samples=1000,
    noise=0.10,
    factor=0.50,
    random_state=1
)
```

The generated dataset contains:

- **1,000 samples**
- **2 input features**
- **2 target classes**
- **Balanced classes** in the generated experiment

The notebook visualizes the circular structure and uses TensorFlow/Keras to train the neural network. It also uses metrics such as accuracy, a classification report, and a confusion matrix, along with decision-boundary visualization.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **TensorFlow / Keras** | Neural-network modeling and training |
| **NumPy** | Numerical computation |
| **Pandas** | Data inspection and manipulation |
| **Matplotlib** | Data and training visualizations |
| **Scikit-learn** | Dataset generation, splitting, and evaluation |
| **MLxtend** | Decision-region visualization |
| **Jupyter / Google Colab** | Interactive notebook environment |

The notebooks are configured for a Python 3 environment. The early-stopping notebook records TensorFlow `2.20.0` in its executed output and is configured for a GPU/Google Colab environment.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Maganpreet-Singh/neural-network-optimization.git
cd neural-network-optimization
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

A minimal environment for the notebooks can be installed with:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow mlxtend jupyter
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Then open any notebook and run the cells from top to bottom.

### Google Colab

The notebooks can also be uploaded directly to **Google Colab**, which is especially useful for experiments requiring GPU acceleration.

## 📈 Recommended Learning Path

For a clean progression, study the notebooks in this order:

```text
01 → Feature Scaling
02 → Dropout Layers
03 → Early Stopping
```

Start with input preprocessing, then move into regularization, and finally learn how validation feedback can control the training process.

## 🧠 Key Concepts

### Optimization is more than the optimizer

In deep learning, the word "optimization" can refer to more than the choice of SGD, Adam, or another optimizer. Practical model optimization also includes:

- Input preprocessing
- Regularization
- Training-duration control
- Validation monitoring
- Hyperparameter selection
- Evaluation and error analysis

This repository focuses on three of those practical levers.

### Generalization vs. training performance

A model can achieve very low training loss while still performing poorly on unseen data. Techniques such as dropout and early stopping are useful because they focus attention on **generalization**, not simply memorizing the training set.

### Validation data matters

When using techniques such as early stopping, validation performance becomes a key signal for deciding whether additional training is actually helping.

## 📊 What to Look For in the Experiments

While working through the notebooks, pay attention to:

- Training loss vs. validation loss.
- Training accuracy vs. validation accuracy.
- Changes in convergence behavior.
- Signs of overfitting.
- How regularization affects model capacity.
- The point at which validation performance is best.
- How decision boundaries change as the model learns.

These comparisons are more valuable than a single final accuracy number because they explain **why** a technique helps or hurts.

## 🔁 Experiment Ideas

The repository is also a good starting point for extending the experiments.

### Hyperparameter experiments

Try different values for:

- Learning rate
- Batch size
- Number of hidden units
- Number of hidden layers
- Dropout rate
- Early-stopping patience

### Optimizer comparison

Compare training behavior using:

```text
Adam
SGD
RMSprop
```

and record how convergence and validation performance change.

### Combine techniques

Build controlled experiments such as:

```text
Baseline
   ↓
Feature Scaling
   ↓
Feature Scaling + Dropout
   ↓
Feature Scaling + Dropout + Early Stopping
```

This makes it easier to isolate the contribution of each technique.

### Add learning-rate scheduling

A natural extension is to experiment with callbacks such as learning-rate reduction when validation loss plateaus.

### Track experiments systematically

For a stronger ML workflow, log:

- Hyperparameters
- Training/validation metrics
- Best epoch
- Final evaluation metrics
- Model size
- Training time

This turns a notebook collection into a more reproducible experimentation pipeline.

## ✅ Best Practices Demonstrated

- Use reproducible random seeds when possible.
- Keep training and validation data conceptually separate.
- Evaluate generalization rather than training performance alone.
- Visualize model behavior where possible.
- Treat regularization parameters as tunable hyperparameters.
- Restore the best validation weights when appropriate.

## ⚠️ Important Notes

This is an educational and experimental repository. The notebooks demonstrate concepts rather than providing a production-ready training framework.

Dependency versions can affect results, especially with deep-learning libraries. For reproducible experiments, pin exact package versions in a `requirements.txt` or environment file.

The included notebooks may contain previously executed outputs, so displayed metrics depend on the notebook state and execution environment. Re-running the notebooks is recommended when reproducing experiments.

## 🌱 Future Roadmap

Potential additions to this repository include:

- [ ] Learning-rate scheduling
- [ ] Batch normalization
- [ ] Weight regularization (L1/L2)
- [ ] Optimizer comparisons
- [ ] Learning-rate experiments
- [ ] Hyperparameter tuning
- [ ] Batch-size experiments
- [ ] Model checkpointing
- [ ] TensorBoard experiment tracking
- [ ] Reproducible `requirements.txt`
- [ ] A reusable Python training pipeline
- [ ] Benchmark tables comparing optimization techniques

## 🤝 Contributing

Contributions are welcome.

A good contribution should:

1. Focus on a clearly defined optimization or training technique.
2. Include a clean, reproducible notebook or implementation.
3. Explain the intuition behind the technique.
4. Include meaningful evaluation or visualization.
5. Keep the experiment easy to reproduce.

For larger changes, open an issue first so the direction can be discussed before implementation.

## 📚 Learning Outcomes

After completing the notebooks in this repository, you should have a stronger practical understanding of:

- Why feature scaling affects neural-network training.
- How dropout acts as a regularizer.
- Why validation performance is important when deciding when to stop training.
- How training curves can reveal overfitting.
- How to structure controlled experiments for model improvement.

## 👤 Author

**Maganpreet Singh**

GitHub: [@Maganpreet-Singh](https://github.com/Maganpreet-Singh)

## ⭐ Support the Project

If this repository helped you understand neural-network optimization, consider giving it a ⭐ on GitHub and using the notebooks as a base for your own experiments.

---

### Repository

[Maganpreet-Singh/neural-network-optimization](https://github.com/Maganpreet-Singh/neural-network-optimization)

> **Learn the concept → run the experiment → inspect the behavior → optimize with evidence.**
