# 物理学実験II 生物物理学 Physics Experiment II, Biophysics

## プラスミドマップ Plasmid map

[pET28a-msfGFP-MinD-rev (in benchling)](https://benchling.com/s/seq-XDvIopQValuSYi5UeJv0?m=slm-kZW53Hcc6JggocCcs4z5)

[pET29-MinE-mCherry-His (in benchling)](https://benchling.com/s/seq-RITthGqZkVKIZYxX0kvM?m=slm-rHuFEeKrvNf43DbWSKE8)

[ColabFold (AlphaFold2)](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb)

## 2領域モデル Two-regions model

Simulation code for the two-regions model in the textbook.

<img src="https://github.com/dbkk/min_simplified_model/blob/main/plots/two_regions_equations.png" width="50%">

Simply run
```
/code/min_tworegions.py
```

Plot will be in `/plots`.

<img src="https://github.com/dbkk/min_simplified_model/blob/main/plots/minD_minE_sim.png" width="50%">

## 細胞膜モデル Cell membrane model

Implementation of 2D version of model described in [Bonny et al. Plos Comp Biol (2013)](https://doi.org/10.1371/journal.pcbi.1003347)

First, run
```
code/min_tworegions.py
```
This will make `.pkl` files in `/results`.

Then run
```
code/min_spatial_visualize_mem.py
```
This will make a `.gif` file in `/plots`.

![Example gif](https://github.com/dbkk/min_simplified_model/blob/main/plots/membrane_example.gif)


## プロットの作り方(基本編) How to make plots (the very basics)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dbkk/min_simplified_model/blob/main/tutorial_plots/plots_tutorial.ipynb)
