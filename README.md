# Pay attention to the activations a modular attention mechanism for fine grained image recognition
**Attend and Rectify: a gated attention mechanism for fine-grained recovery**

[arxiv](https://arxiv.org/abs/1907.13075)

Contents:
```
./logs
    best logs for the tables reported in the paper
./scripts
    sh scripts to reproduce experiments
./modules
    contains the attention module proposed in the paper.
./models
    contains the different networks used on cifar.
```

Usage:
1. git clone this repository && cd this repository
2. ./scripts/your_script.sh (edit to set dataset paths, etc)

Requirements:
1. The code requires pytorch >= 0.4


## Cite
This work has been published at IEEE Transactions of Multimedia as an extension of a former paper presented at ECCV:

```
@ARTICLE{8762109, 
author={P. {Rodriguez Lopez} and D. {Velazquez Dorta} and G. {Cucurull Preixens} and J. M. {Gonfaus Sitjes} and F. X. {Roca Marva} and J. {Gonzalez}}, 
journal={IEEE Transactions on Multimedia}, 
title={Pay attention to the activations: a modular attention mechanism for fine-grained image recognition}, 
year={2019}, 
volume={}, 
number={}, 
pages={1-1}, 
keywords={Computer architecture;Computational modeling;Image recognition;Task analysis;Proposals;Logic gates;Clutter;Image Retrieval Deep Learning Convolutional Neural Networks Attention-based Learning}, 
doi={10.1109/TMM.2019.2928494}, 
ISSN={1520-9210}, 
month={},}
```

```
@inproceedings{rodriguez2018attend,
  title={Attend and rectify: a gated attention mechanism for fine-grained recovery},
  author={Rodr{\'\i}guez, Pau and Gonfaus, Josep M and Cucurull, Guillem and XavierRoca, F and Gonzalez, Jordi},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={349--364},
  year={2018}
}
```
