**Official PyTorch implementation of the paper:**  
**_"CR-DARTS: Channel Redistribution-based Differentiable Architecture Search"_**  
Submitted to *[Target Conference/Journal]*, 2025.

![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-Coming%20Soon-blue)

---

## 🚀 Overview

**CR-DARTS** automates neural architecture design but suffers from computational overhead and an architecture optimization gap between search and evaluation networks.
We propose CR-DARTS, a multi-stage search framework with an adaptive channel redistribution strategy. The approach:
- Compresses shared input features among candidate operations.
- Restores network dimensions via channel-wise feature concatenation.
- Progressively eliminates underperforming operations.
- Redistributes channels to relevant operations for improved feature extraction.

Key results:
- Up to 1.5× faster inference in image classification.
- A competitive fire segmentation network outperforming state-of-the-art methods while preserving computational efficiency.

Although developed primarily for fire segmentation, **FireSegUNet** generalizes well to other real-time segmentation tasks across domains like environmental monitoring, IoT, TinyML, and mobile vision applications.
---

## 🛠️ Features

- ✅ PyTorch implementation with modular codebase
- ✅ Pretrained models on FLAME, BoWFire, and Fire datasets
- ✅ Detailed training and evaluation scripts
- ✅ Deployment scripts for embedded platforms (A100, Cortex A76, Adreno 630/640)
- ✅ Easy-to-customize for other segmentation tasks

---

## 📖 Paper

> Ali Hassan, Mårten Sjöström, Karen Egiazarian, Tingting Zhang, Johan Johansson, Stefan Schulte  
> _FireSegUNet: Exploring Computationally Efficient Real-Time Fire Segmentation Network for UAVs_, 2025.

📄 [Paper Link (Coming Soon)](#)

---

## 📦 Code Release Timeline

| Stage | Status |
|:-----|:------|
| Code Cleaning | ✅ Completed |
| Paper Submission | ✅ Accepted |
| Repository Public Release | ⏳ Coming Soon upon publication |

---

## 📂 Project Structure

```bash
🔹 firesegunet/
│   🔹 models/           # FireSegUNet Architecture Components
│   🔹 datasets/         # Data loading utilities (FLAME, BoWFire, Fire)
│   🔹 train.py          # Training script
│   🔹 eval.py           # Evaluation script
│   🔹 deploy/           # Deployment scripts for edge devices
│   🔹 utils/            # Helper functions (metrics, logging, etc.)
🔹 configs/              # YAML config files for experiments
🔹 requirements.txt      # Python dependencies
🔹 README.md             # You are here 🚀
```

---

## 🔥 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/firesegunet.git
cd firesegunet

# Install dependencies
pip install -r requirements.txt

# Train FireSegUNet on FLAME dataset
python train.py --config configs/firesegunet_flame.yaml

# Evaluate pretrained model
python eval.py --weights checkpoints/firesegunet_best.pth
```

---

## 📊 Benchmarks

| Dataset | Mean IoU (%) | F1 Score (%) | Parameters (M) | Memory Reduction |
|:--------|:-------------|:-------------|:---------------|:-----------------|
| FLAME [5] | 85.25 | 92.04 | 1.28M | -81% |

Full benchmarking details available in [paper](#) 📄.

---

## 🤝 Acknowledgments

This work was supported by the European Joint Doctoral Programme on Plenoptic Imaging (PLENOPTIMA) and the EU Interreg Aurora project IMMERSE.  
We thank NAISS Sweden for computational resources.

---

## 📬 Citation

If you use CR-DARTS or our code, please cite:

```bibtex
@article{hassan2025firesegunet,
  title={CR-DARTS: CR-DARTS: Channel Redistribution-based Differentiable Architecture Search},
  author={Hassan, Ali and Sj{"o}str{"o}m, M{å}rten and Egiazarian, Karen and Zhang, Tingting},
  journal={[Target Journal/Conference]},
  year={2025}
}
```

---

## 🔥 Stay tuned for the official release!
