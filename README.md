# BSTabDiff: Block-Subunit Diffusion Priors for High-Dimensional Tabular Data Generation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Task](https://img.shields.io/badge/Task-HDLSS%20Tabular%20Synthesis-orange)
![Model](https://img.shields.io/badge/Model-BSTabDiff-blueviolet)
![Architecture](https://img.shields.io/badge/Architecture-Block--Subunit%20Generator-informational)
![Latent Prior](https://img.shields.io/badge/Latent%20Prior-Diffusion%20%2F%20Flow-purple)
![Data Regime](https://img.shields.io/badge/Regime-n%20%3C%3C%20m-critical)
[![Workshop](https://img.shields.io/badge/Workshop-ICLR%202026%20DeLTa-blue)](https://iclr.cc/virtual/2026/workshop/10000780)
[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-red)](https://openreview.net/forum?id=RKNDy0KhGT)
[![Workshop Page](https://img.shields.io/badge/DeLTa%202026-Website-1f6feb)](https://delta-workshop.github.io/DeLTa2026/)
![Status](https://img.shields.io/badge/Status-Camera%20Ready-brightgreen)

<p align="center">
  <img src="./BSTabDiffArchi.png" alt="BSTabDiff Architecture" width="900">
</p>

BSTabDiff is a block-subunit generative framework for **High-Dimensional Low-Sample-Size (HDLSS) tabular data synthesis**. Rather than learning dependence directly in the original high-dimensional feature space, it partitions the feature space into **M latent blocks, where M ≪ m**, models global structure through a compact diffusion/flow prior over block latents, and decodes back to the full table using copula-based dependence, flexible feature-wise marginals, and explicit missingness modeling. This design makes BSTabDiff especially well suited for omics-style and other HDLSS settings, where direct high-dimensional density learning is often unstable. Across multiple HDLSS benchmarks, BSTabDiff generates more realistic and stable synthetic data than several widely used tabular generators, while often approaching downstream performance obtained from real data.

## Citation

Al Zadid Sultan Bin Habib, Md Younus Ahamed, Prashnna Kumar Gyawali, Gianfranco Doretto, and Donald A. Adjeroh.  
**“BSTabDiff: Block-Subunit Diffusion Priors for High-Dimensional Tabular Data Generation.”**  
In *ICLR 2026 2nd Workshop on Deep Generative Models in Machine Learning: Theory, Principle and Efficacy (DeLTa)*, 2026.


BibTeX:
```bibtex
@inproceedings{habib2026bstabdiff,
  title     = {BSTabDiff: Block-Subunit Diffusion Priors for High-Dimensional Tabular Data Generation},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Gyawali, Prashnna Kumar and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {ICLR 2026 2nd Workshop on Deep Generative Models in Machine Learning: Theory, Principle and Efficacy (DeLTa)},
  year      = {2026}
}
```

## Files and Repository Structure

### Python package: `bstabdiff/`

This folder contains the core BSTabDiff implementation:

- `__init__.py` - Package initializer and high-level API exports.
- `block_subunit_gen.py` - Main BSTabDiff implementation, including feature schema, empirical marginals, block-subunit emissions, diffusion/flow priors, training, and synthetic sampling utilities.

### Notebooks

- **`Dummy Example Usage.ipynb`**  
  Contains simple toy examples showing how to install/import the `bstabdiff` package, fit BSTabDiff on a dummy HDLSS dataset, and sample synthetic data.

- **`BSTabDiff_Colon.ipynb`**  
  Contains the Colon dataset experiments from the paper. The downstream classifiers include Logistic Regression, TabPFN-2.5 (currently applicable only when the number of features is within its supported range, so Colon is eligible), TANDEM (NeurIPS 2025), and CatBoost. This notebook also includes the paper’s ablation studies and related fidelity analysis.

- **`BSTabDiff_GLI.ipynb`**  
  Contains the GLI-85 experiments using Logistic Regression as the downstream classifier, along with selected fidelity analysis.

- **`BSTabDiff_Lung.ipynb`**  
  Contains the Lung dataset experiments using Logistic Regression as the downstream classifier, along with selected fidelity analysis.

- **`BSTabDiff_PIP_Install_Check.ipynb`**
    Demonstration of BSTabDiff in a Google Colab notebook using pip installation with some toy examples.

### Other top-level files

- **`requirements.txt`** - Python dependencies required to run the BSTabDiff package and notebooks.
- **`BSTabDiffArchi.png`** - High-level architecture diagram of the BSTabDiff framework.
- **`LICENSE`** - MIT license for this repository.
- **`README.md`** - Project overview, installation, usage instructions, and citation information.
- **`.gitignore`** - Standard Git ignore rules for Python and Jupyter projects.
- **`pyproject.toml`** - Build system and packaging metadata for installation.
- **`setup.cfg`** - Package configuration and installation metadata.

### Tested Environment

- Python 3.10.13
- torch 2.9.1+cu128
- numpy 2.2.6
- pandas 2.3.3
- scikit-learn 1.7.2
- catboost 1.2.8
- tabpfn 6.3.1

## Installation

You can install **BSTabDiff** in several ways depending on your workflow.

---

### Option 1: Clone the Repository (Recommended for Development)

```bash
git clone https://github.com/zadid6pretam/BSTabDiff.git
cd BSTabDiff
pip install -r requirements.txt
pip install -e .
```

### Option 2: Install Directly from GitHub (No Cloning Needed)

```bash
pip install "git+https://github.com/zadid6pretam/BSTabDiff.git"
```

### Option 3: Use a Virtual Environment

```bash
python -m venv bstabdiff-env
source bstabdiff-env/bin/activate  # On Windows: bstabdiff-env\Scripts\activate

git clone https://github.com/zadid6pretam/BSTabDiff.git
cd BSTabDiff
pip install -r requirements.txt
pip install -e .
```

### Option 4: Local Install Without Editable Mode

```bash
git clone https://github.com/zadid6pretam/BSTabDiff.git
cd BSTabDiff
pip install -r requirements.txt
pip install .
```

### Option 5: Install from PyPI (Planned)

```bash
pip install bstabdiff
```

## Example Usage
Below is a minimal example showing how to fit BSTabDiff on a dummy HDLSS dataset and generate synthetic samples.

```bash
import numpy as np
from bstabdiff import FeatureSpec, fit_block_subunit_generator

# Dummy HDLSS data
np.random.seed(42)
n, m = 80, 2000
X = np.random.randn(n, m).astype(np.float32)
y = np.random.randint(0, 2, size=n)
X[np.random.rand(n, m) < 0.1] = np.nan

# Feature schema
feature_specs = [FeatureSpec(name=f"f{j}", kind="continuous") for j in range(m)]

# Fit BSTabDiff
gen, train_info = fit_block_subunit_generator(
    X=X,
    feature_specs=feature_specs,
    y=y,
    M=20,
    blocks=None,
    permute_features=False,
    prior_type="diffusion",
    device="cpu",
    seed=42,
    prior_epochs=300,
    prior_batch=64,
    prior_lr=1e-3,
    verbose_every=100,
    save_dir=None,
    save_name="bstabdiff_demo",
    save_best=True,
    use_ema=True,
    ema_decay=0.999,
    return_train_info=True,
)

# Sample synthetic data
X_syn, R_syn, y_syn = gen.sample(n=50)

print("X_syn shape:", X_syn.shape)
print("R_syn shape:", R_syn.shape)
print("y_syn shape:", y_syn.shape if y_syn is not None else None)
print("Best training info:", train_info)
```

**For fuller experiments, ablations, and fidelity studies, see:**
- Dummy Example Usage.ipynb
- BSTabDiff_Colon.ipynb
- BSTabDiff_GLI.ipynb
- BSTabDiff_Lung.ipynb

## Our Previous Related Work on Tabular Deep Learning

BSTabDiff is part of our broader line of work on tabular deep learning and high-dimensional tabular modeling.

### TabSeq

Our earlier work on sequential modeling for tabular data:

- **TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering**  
  GitHub: https://github.com/zadid6pretam/TabSeq  
  Springer (ICPR 2024 proceedings): https://link.springer.com/chapter/10.1007/978-3-031-78128-5_27

```bibtex
@inproceedings{habib2024tabseq,
  title={TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering},
  author={Habib, Al Zadid Sultan Bin and Wang, Kesheng and Hartley, Mary-Anne and Doretto, Gianfranco and A. Adjeroh, Donald},
  booktitle={International Conference on Pattern Recognition},
  pages={418--434},
  year={2024},
  organization={Springer}
}
```

- If you are interested in sequential ordering for tabular data, deep sequential backbones, and early feature-ordering-based tabular modeling, please also refer to the **TabSeq** repository and paper.

### DynaTab

Our more recent work on learned feature ordering for high-dimensional tabular data:

- **DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data**  
  GitHub: https://github.com/zadid6pretam/DynaTab

```bibtex
@inproceedings{habib2026dynatab,
  title     = {{DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data}},
  author    = {Habib, Al Zadid Sultan Bin and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {Proceedings of the AAAI 2026 First International Workshop on Neuro for AI \& AI for Neuro: Towards Multi-Modal Natural Intelligence (NeuroAI)},
  year      = {2026},
  series    = {PMLR}
}
```

- If you are interested in learned feature ordering, neural rewiring for high-dimensional tabular data, and sequential backbone design for HDLSS settings, please also refer to the DynaTab repository and paper.
- DynaTab has completed camera-ready submission, and the public proceedings version is expected to appear online later.

## Contact

For any questions, issues, or suggestions related to this repository, please feel free to contact us or open an issue on GitHub.
