# SpatialTuring

[![Release](https://img.shields.io/github/v/release/Tako-liu/SpatialTuring?include_prereleases&color=success)](https://github.com/Tako-liu/SpatialTuring/releases)
[![PyTorch](https://img.shields.io/badge/PyTorch-Accelerated-red?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)]()

> **Unmasking the hidden reaction-diffusion logic in massive spatial transcriptomics data.**

**SpatialTuring** is a high-performance bioinformatics framework designed to discover **Turing patterns** (reaction-diffusion systems) in sub-cellular resolution spatial transcriptomics data (Xenium, Visium HD, Stereo-seq).

Unlike traditional tools that rely on morphological clustering or simple autocorrelation, SpatialTuring explicitly models the **Gierer-Meinhardt mechanism** (Short-range Activation vs. Long-range Inhibition), enabling researchers to identify potential morphogen pairs driving tissue organogenesis.

---

## ⚡ Key Features

* **🚀 GPU-Accelerated Engineering**: Built on `torch.nn.Conv2d`, turning hours of CPU computation into seconds of GPU inference. Capable of processing 1M+ cells in real-time.
* **🧬 Mechanism-First Discovery**: Moves beyond "clustering" to identify the underlying *generative rules* of tissue patterning.
* **📐 Scale-Invariant Logic**: Uses a topological DoG (Difference of Gaussians) filter to detect patterns regardless of tissue deformation or rotation.
* **🔌 Ecosystem Compatible**: Seamlessly integrates with [Scanpy](https://scanpy.readthedocs.io/) and `AnnData`.

---

## 📥 Installation

You can install the development version directly from GitHub:

```bash
pip install git+[https://github.com/Tako-liu/SpatialTuring.git](https://github.com/Tako-liu/SpatialTuring.git)

```

**Requirements:**

* Python >= 3.8
* PyTorch >= 1.10 (CUDA recommended for acceleration)
* Scanpy, NumPy, Pandas, SciPy

---

## 🏎️ Quick Start

```python
import scanpy as sc
from spatialturing import TuringPatternHunter

# 1. Load your spatial data (e.g., Xenium output)
adata = sc.read_h5ad("path/to/embryo.h5ad")

# 2. Initialize the Hunter (Auto-detects GPU)
# bin_size: converts physical units (e.g., microns) to analysis grid pixels
hunter = TuringPatternHunter(adata, bin_size=20)

# 3. Step I: Screen for Turing-like Candidates
# Scans the entire transcriptome for reaction-diffusion wave-numbers
candidates_u, candidates_v = hunter.screen_geometry(
    sigma_inner=2,   # Short-range activation scale
    sigma_outer=5,   # Long-range inhibition scale
    top_n=50
)

# 4. View Results
print(candidates_u[['gene', 'peak_score']].head())

```

---

## 🖼️ Gallery

### Visualizing the Hidden Logic

SpatialTuring applies a computational lens to separate biological logic from sequencing noise.

* **Left (Raw Input):** Gene expression with dropout and technical noise.
* **Right (Processed):** The extracted Turing field using the DoG filter, revealing the underlying periodic structure.

---

## 🔬 Theoretical Foundation

SpatialTuring is grounded in the mathematical theory of morphogenesis proposed by **Alan Turing (1952)** and refined by **Gierer & Meinhardt (1972)**.

We define a "Turing Candidate" as a gene  whose spatial expression field  satisfies the band-pass energy condition:

$$ E_{Turing} = || \mathcal{K}*{\sigma*{short}} * X_g - \mathcal{K}*{\sigma*{long}} * X_g ||_2^2 $$

Where  represents the Gaussian diffusion kernel. A high energy score indicates that the gene forms a stable pattern driven by local self-enhancement and long-range lateral inhibition.

---

## 🤝 Contributing

Contributions are welcome! If you are interested in improving the algorithm or adding new features (e.g., Graph Neural Networks support), please submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 Citation

If you use SpatialTuring in your research, please cite:

> **[Your Name]**, et al. (2025). *SpatialTuring: Unmasking reaction-diffusion logic in spatial transcriptomics.* GitHub Repository.

---

**License**: MIT
**Maintainer**: [Tako-liu](https://www.google.com/search?q=https://github.com/Tako-liu)
