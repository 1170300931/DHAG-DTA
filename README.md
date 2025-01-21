# DHAG-DTA

This repository contains code for replicating results from the associated paper:
Cheng Wang, Yang Liu, Shitao Song, Kun Cao, Xiaoyan Liu, Gaurav Sharma, Maozu Guo, "DHAG-DTA: dynamic hierarchical affinity graph model for  drug-target binding affinity prediction" (DOI: [https://doi.org/10.1109/TCBBIO.2025.3531938](https://doi.org/10.1109/TCBBIO.2025.3531938))

The results shows in "results/*", including results in the Table 2, 3, 4 and 5.

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
