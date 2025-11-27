# CR-DARTS - Channel Redistribution-based Differentiable Architecture Search 

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0) ![image](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![image](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

---

**Official Author implementation of the paper:**  
**_"CR-DARTS: Channel Redistribution-based Differentiable Architecture Search"_**  
Published in *IEEE Access, 26 September 2025*. [Paper Link](https://ieeexplore.ieee.org/document/11269759)

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
  - CIFAR-10 & CIFAR-100 dataset - [Dataset Link](https://docs.pytorch.org/vision/0.9/datasets.html#cifar)
  - Face Blurred ImageNet dataset - [Dataset Link](https://www.c3se.chalmers.se/documentation/software/machine_learning/datasets/?h=dataset#imagenet)
- Fire Segmentation
  - Fire Luminosity Airborne-based Machine learning Evaluation (FLAME) - [Dataset Link](https://dx.doi.org/10.21227/qad6-r683)
  - Fire Dataset - [Dataset Link](https://github.com/hayatkhan8660-maker/Fire_Seg_Dataset?tab=readme-ov-file)

Key results:
- Reduce computational resources requirement by up to 4.3× while addressing the architecture optimization gap.
- Discovered architecture achieves up to 25.3% reduction in computational complexity and 50.6× faster inference in image classification.
- A competitive fire segmentation network outperforms state-of-the-art methods while preserving computational efficiency.

---
## 📂 Project Structure

```bash
🔹 CR-DARTS/
│   🔹 outputs/           # This directory contains the output of the proposed CR-DARTS algorithms
│         🔹 crdarts_cifar10_weights.pt    # Weights of the trained CR-DARTS (last epoch) architecture on CIFAR-10 dataset
│         🔹 crdarts_cifar100_weights.pt   # Weights of the trained CR-DARTS (last epoch) architecture on CIFAR-100 dataset
│   🔹 requirements.txt  # Python dependencies
│   🔹 train_search_cifar.py, train_cifar.py, test_cifar.py      # Related to CIFAR Dataset
│   🔹 train_search_imagenet.py, train_imagenet.py               # Related to ImageNet Dataset

```

## Using the Source Code
This section provides instructions on how to use the provided source code, such as searching for a new evaluation architecture, training the discovered architecture, and evaluating the fully trained discovered architecture with pretrained model weights.
### Setup the repository
```bash
# Clone the repository
git clone https://github.com/Realistic3D-MIUN/CR-DARTS.git
cd CR-DARTS

# Install dependencies
pip install -r requirements.txt
```

### Evaluating the proposed CR-DARTS final architecture
To evaluate the proposed architecture on the CIFAR dataset by loading the weights of the trained architecture, just set the number of classes on "Line # 39" of the "test_cifar.py" (i.e, CIFAR_CLASSES = 100) and then run the following prompt:
```bash
# Evaluate pretrained model
python test_cifar.py
```
Please note if you change the architecture, do not forget to update the "arch" parser argument, by adding the architecture genotype in the "genotypes.py" file.
Similarly, to evaluate the proposed architecture on the ImageNet dataset, you need access to the [ALVIS supercomputer](https://www.c3se.chalmers.se/about/Alvis/). The dataset is also hosted on the same cluster and is available upon signing an agreement.

For the extended evaluation of the fire segmentation task, please refer to the "fire_segmentation" directory, which contains further instructions for setting up and evaluating our proposed architecture.

If interested, you can also consider our recently published paper "REDARTS" in IEEE Transactions 2025. To compare the results with the REDARTS paper, use the "REDARTS" variable, available in "genotypes.py" file.

---

### Searching for the evaluation network using the proposed CR-DARTS framework

To search for the evaluation architecture, select the file you want to use based on the dataset.
For example, if you want to search on the CIFAR dataset, use the file "train_cifar.py".
Moreover, if you want to search on the CIFAR-100 dataset, set the argument "cifar100" to False in the "train_cifar.py" file.
After this, run the following command:
```bash
# Search on CIFAR dataset
python train_search_cifar.py
```

Once an evaluation architecture is identified (at the end of the log file), copy the architecture's Genotype, add a variable name, and paste it at the end of genotypes.py. Following this, set the argument "arch" name to the assigned variable name and run the complete convergence training using:
```bash
# Train CR-DARTS on CIFAR dataset
python train_cifar.py
```
---


## 🤝 Acknowledgments

This work was supported by the European Joint Doctoral Programme on Plenoptic Imaging (PLENOPTIMA), EU Interreg Aurora project IMMERSE, and MIUN internal funding. We thank NAISS Sweden for computational resources.

---

## 📬 Citation

If you use CR-DARTS or our code, please cite:

```bibtex
@article{hassan2025crdarts,
  title={CR-DARTS: Channel Redistribution-based Differentiable Architecture Search},
  author={Hassan, Ali and Zhang, Tingting and Egiazarian, Karen and Sjöström, Mårten},
  journal={IEEE Access},
  pages={1-17},
  year={2025}
}

@article{hassan2025redarts,
  author={Hassan, Ali and Sjöström, Mårten and Zhang, Tingting and Egiazarian, Karen},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence}, 
  title={REDARTS: Regressive Differentiable Neural Architecture Search for Exploring Optimal Light Field Disparity Estimation Network}, 
  year={2025},
  pages={1-12},
  doi={10.1109/TETCI.2025.3592281}}
```

This code is modified and heavily borrowed from [P-DARTS](https://github.com/chenxin061/pdarts)
The code they provided is greatly appreciated.

```bibtex
@inproceedings{chen2019progressive,
  title={Progressive differentiable architecture search: Bridging the depth gap between search and evaluation},
  author={Chen, Xin and Xie, Lingxi and Wu, Jun and Tian, Qi},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={1294--1303},
  year={2019}
}
```

