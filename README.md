# DHAG-DTA

This repository contains code for replicating results from the associated paper:

Cheng Wang, Yang Liu, Kun Cao, Xiaoyan Liu, Shitao Song, Gaurav Sharma and Maozu
Guo, 
"DHAG-DTA: Dynamic Hierarchical Affinity Graph Model for Drug-Target Binding Affinity Prediction," in IEEE Transactions on Computational Biology and Bioinformatics, vol. 22, no. 2, pp. 697-709, March-April 2025, doi: 10.1109/TCBBIO.2025.3531938.

<p align="justify">
<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10847908">[Paper]</a>
<a href="https://ieeexplore.ieee.org/ielx8/10723156/10953288/10847908/supp1-3531938.pdf?arnumber=10847908">[Supplementary]</a>
<a href="https://github.com/ChengWang-hit/DHAG-DTA">[GitHub Site]</a>
<a href="https://codeocean.com/capsule/6526340/tree">[CodeOcean Site]</a>
</p>

## Dependencies

The program requires Python 3.6.13 and the following main packages:

* cudatoolkit v11.3.1
* networkx v2.5.1
* pytorch v1.10.2
* rdkit v2020.09.1.0
* lifelines v0.26.4
* torch-cluster v1.5.9
* torch-geometric v2.0.3
* torch-scatter v2.0.9
* torch-sparse v0.6.12
* torch-spline-conv v1.2.1

GPU is highly recommended.

## Run the Program using Code Ocean

We recommend using the Code Ocean version of this program, which can be run using Code Ocean's built-in interface: https://codeocean.com/capsule/6526340/tree (DOI: 10.24433/CO.8337276.v1)

See "/environment/Dockerfile" for installation details.

checkpoints and data can be downloaded from Code Ocean and are organized as follows:

`checkpoints/davis_S1.pkl`

`checkpoints/davis_S2.pkl`

`checkpoints/davis_S3.pkl`

`checkpoints/davis_S4.pkl`

`checkpoints/kiba_S1.pkl`

`checkpoints/kiba_S2.pkl`

`checkpoints/kiba_S3.pkl`

`checkpoints/kiba_S4.pkl`

`data/davis`

`data/kiba`

## Citation
If you use the code, please cite:
```BibTex
@ARTICLE{10847908,
  author={Wang, Cheng and Liu, Yang and Song, Shitao and Cao, Kun and Liu, Xiaoyan and Sharma, Gaurav and Guo, Maozu},
  journal={IEEE Transactions on Computational Biology and Bioinformatics}, 
  title={DHAG-DTA: Dynamic Hierarchical Affinity Graph Model for Drug-Target Binding Affinity Prediction}, 
  year={2025},
  volume={22},
  number={2},
  pages={697-709},
  keywords={Drugs;Predictive models;Graph neural networks;Artificial neural networks;Biological system modeling;Proteins;Atoms;Computational modeling;Bioinformatics;Entropy;Drug-target binding affinity prediction;drug discovery;dynamic hierarchical affinity graph;graph neural networks;transductive learning},
  doi={10.1109/TCBBIO.2025.3531938}}
```
