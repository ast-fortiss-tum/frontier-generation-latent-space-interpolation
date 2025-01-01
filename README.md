# Automated-Boundary-Testing-for-DL-Systems
Codebase for Giorgi Merabishvili's internship project on Neural Model-Based Test Generation for Deep Learning Systems.

## Installation

```bash
conda env create -f environment.yml
conda activate mimicry
```
MNIST:
- Gradual truncation was added to increase seed utilization.
  
FashionMNIST:
- Two feature-different images are generated after switching the feature layer, followed by pixel-wise interpolation.
- An additional search for exact 0.5 confidence is included.
- Background masking is applied to interpolate only with objects.

SVHN:
- Linear weight interpolation is performed directly in the latent space.
- A frontier pair is identified by fully changing the feature layer. Once found, linear weight interpolation is performed to approach the decision boundary.

CIFAR-10:
- Linear weight interpolation is performed directly in the latent space.
- A frontier pair is identified by fully changing the feature layer. Once found, linear weight interpolation is performed to approach the decision boundary.

## Getting started

1. Download pre-trained networks `*.pkl` files from [Huggingface](https://huggingface.co/awafa/cSG2) and put them under `./checkpoints/checkpoints`.

2. Select one of the following datasets and go to the corresponding directory:
  - MNIST: `./mnist` 
  - FashionMNIST: `./f-mnist` 
  - SVHN: `./svhn` 
  - CIFAR-10: `./CIFAR-10` 

3. Adjust the default config in `./<DATASET>/config.py`.
   - MNIST: Change paths in config.py
   - FashionMNIST: Change paths in config.py and in Model1_fmnist.py
   - SVHN: Change paths in config.py and gs_svhn.py
   - CIFAR-10: Change paths in config.py
   
5. Run the file `./<DATASET>/search.py` to generate frontier pairs.



## Acknowledgement

This code is developed based on [StyleGAN3](https://github.com/NVlabs/stylegan3) and [PTI](https://github.com/tianhaoxie/DragGAN_PTI/tree/27a9821085ce4d9b788aaf4bbb52b9b982b25bcd?tab=readme-ov-file)
