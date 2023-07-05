
# Automatic assessment for melodic pattern imitations

This repository contains code for automatic assessment for melodic pattern imitations. Data used in the machine experiments is available here: [**MAST Rhythm Data Set**](https://zenodo.org/record/8007358) Please refer to the paper below for a detailed description of the tools, experiments and experiment results.

```latex
@article{BozkurtBaysal2023,
  title={Automatic assessment of student vocal melodic repetition performances},
  author={Bozkurt, Baris and Baysal, Ozan},
  journal={submitted for review},
  volume={},
  number={},
  pages={},
  year={2023},
  publisher={}
}
```
and if you use this code please cite it as above. 

## Content

The repo contains the following folders:
*   'data' folder contains a copy of the annotation files as well as the melodic pattern midi files. 
*   'scripts/data_processing' folder contains all preprocessing and feature extraction code that prepares tabular data for machine learning experiments. 
*   'scripts/ML_experiments' folder contains code for machine learning experiments 

To run all steps from scratch run runFromScratch.py from its folder

# Acknowledgment

This study is supported by TUBITAK with grant number 121E198 as a part of the Scientific and Technological Research Projects Funding Program (1001).

