# CR-DARTS - Channel Redistribution-based Differentiable Architecture Search 

![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-Coming%20Soon-blue)

---

**Official PyTorch implementation of the paper:**  
**_"CR-DARTS: Channel Redistribution-based Differentiable Architecture Search"_**  
Accepted for Publication in *IEEE Access, 2025*.

---

## 🚀 Overview

**CR-DARTS** addresses the architecture-optimization gap between search and evaluation networks in the state-of-the-art DARTS framework.
We propose CR-DARTS, a multi-stage search framework with an adaptive channel redistribution strategy. The approach:
- Compresses shared input features among candidate operations.
- Restores network dimensions via channel-wise feature concatenation.
- Progressively eliminates underperforming operations.
- Redistributes channels to surviving candidate operations for improved feature extraction.
We have evaluated it on two computer vision applications:
- Image Classification
  - CIFAR-10 & CIFAR-100 dataset - https://docs.pytorch.org/vision/0.9/datasets.html#cifar
  - Face Blurred ImageNet 2017 dataset - https://www.c3se.chalmers.se/documentation/software/machine_learning/datasets/?h=dataset#imagenet
- Fire Segmentation
  - Fire Luminosity Airborne-based Machine learning Evaluation (FLAME) - [https://dx.doi.org/10.21227/qad6-r683](https://ieee-dataport.org//open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs))
  - Fire Dataset - https://github.com/hayatkhan8660-maker/Fire_Seg_Dataset?tab=readme-ov-file 

Key results:
- Reduce computational resources requirement by up to 4.3× while addressing the architecture optimization gap.
- Discovered architecture achieves up to 25.3% reduction in computational complexity and 50.6× faster inference in image classification.
- A competitive fire segmentation network outperforms state-of-the-art methods while preserving computational efficiency.

---
## 📂 Project Structure

```bash
🔹 CR-DARTS/
│   🔹 outputs/           # FireSegUNet Architecture Components
│         🔹 crdarts_cifar10_weights.pt    # Weights of the trained CR-DARTS (last epoch) architecture on CIFAR-10 dataset
│         🔹 crdarts_cifar100_weights.pt   # Weights of the trained CR-DARTS (last epoch) architecture on CIFAR-100 dataset
│   🔹 requirements.txt  # Python dependencies
│   🔹 train_search_cifar.py, train_cifar.py, test_cifar.py      # Related to CIFAR Dataset
│   🔹 train_search_imagenet.py, train_imagenet.py               # Related to ImageNet Dataset

```

## Start

```bash
# Clone the repository
git clone https://github.com/Realistic3D-MIUN/CR-DARTS.git
cd CR-DARTS

# Install dependencies
pip install -r requirements.txt

# Search on CIFAR dataset
python train_search_cifar.py

# Train CR-DARTS on CIFAR dataset
python train_cifar.py

# Evaluate pretrained model
python test_cifar.py
```

---

## 🤝 Acknowledgments

This work was supported by the European Joint Doctoral Programme on Plenoptic Imaging (PLENOPTIMA) and the EU Interreg Aurora project IMMERSE.  
We thank NAISS Sweden for computational resources.

---

## 📬 Citation

If you use CR-DARTS or our code, please cite:

```bibtex
@article{hassan2025crdarts,
  title={CR-DARTS: Channel Redistribution-based Differentiable Architecture Search},
  author={Hassan, Ali and Zhang, Tingting and Egiazarian, Karen and Sj{"o}str{"o}m, M{å}rten},
  journal={IEEE Access},
  year={2025}
}
```

---

