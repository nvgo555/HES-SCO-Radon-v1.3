# HES-SCO Radon v1.3: Geometric Intelligence Architecture

**Status:** v1.3.3 (Stable) | **Architecture:** Post-Euclidean | **Dim:** D=64

## 🔬 Overview
Radon v1.3 is a **Geometric Artificial Intelligence** that replaces standard Deep Learning "Curve Fitting" with **Manifold Mapping**. Unlike Euclidean networks that require massive parameters ($D=1024+$) to memorize outliers, Radon uses Hyperbolic and Spherical topology to compress logic into a constrained latent space ($D=64$).

**The Core Thesis:** Intelligence is not a function of parameter count, but of geometric alignment.

## 📊 Engineering Benchmark (Radon vs. Goliath MLP)
We pitted Radon ($D=64$) against a massive Euclidean MLP ($D=1024$) on a Hierarchical Entailment task with **500% Out-of-Distribution (OOD) Noise**.

| Metric | Goliath (Standard MLP) | Radon v1.3 (Geometric) | Advantage |
| :--- | :--- | :--- | :--- |
| **Parameters** | 1,117,185 | **8,292** | **134x Compression** |
| **Memory Footprint** | 8.52 MB | **0.06 MB** | Smaller than a JPEG |
| **OOD Error (MSE)** | 2.7093 | **0.2860** | **92.5% More Robust** |
| **Inference Type** | Passive Matrix Mul | Active Riemannian Search | "System 2 Thinking" |

## 🧠 Architecture
1.  **The Manifold Cast:** Splits input $x$ into $\mathbb{R}^{32}$ (Intensity), $\mathbb{H}^{16}$ (Hierarchy), and $\mathbb{S}^{16}$ (Phase).
2.  **System 2 Engine ("The Ghost"):** An inference-time optimization loop that performs **Riemannian Gradient Descent** on the latent state to minimize "Epiplexity" (Geometric Stress) before answering.
3.  **Poincaré Projection:** A topological adapter that maps unbounded Lorentz coordinates to the bounded Poincaré ball, guaranteeing stability against infinite noise.

## 🚀 Quick Start
### Prerequisites
```bash
pip install -r requirements.txt