<!--
SPDX-FileCopyrightText: 2024 Matteo Lai <matteo.lai3@unibo.it>

SPDX-License-Identifier: NPOSL-3.0
-->

[<img src="https://img.shields.io/badge/Colab_tutorial_from_files-red.svg?logo=jupyter&logoColor=white">](https://colab.research.google.com/drive/1A70ENEZPpB9FT4mMU5ZiP0gFTkw--ysf?usp=drive_link)
[<img src="https://img.shields.io/badge/Colab_tutorial_from_pretrained_generator-blue.svg?logo=jupyter&logoColor=white">](https://colab.research.google.com/drive/1n-gcNzYoly9ZBUzczJu1Wr_gnyzs7L4I?usp=sharing)

# Configs

The Synthetic_Images_Metrics_Toolkit repository is designed to evaluate the quality of synthetic images. It provides flexible configurations to assess images obtained from two different sources:

1. 🗂️**From files** – Load and analyze pre-existing synthetic images stored in files or directories.

2. 🔄**From a pretrained model** – Generate synthetic images on-the-fly using a pretrained generative model.

## 1. 🗂️ Evaluating synthetic images from files

If synthetic images are saved in files or folders, you only need to define a function to load them.

📝 **Example configurations**: Find sample configuration files for different data types [here](Tutorials/from_files)

📖 **Colab tutorial**: Follow 
[this Google Colab tutorial](https://colab.research.google.com/drive/1A70ENEZPpB9FT4mMU5ZiP0gFTkw--ysf?usp=drive_link) to learn how to create a configuration file that loads both real and synthetic images from files or folders and computes the metrics of interest.

## 2. 🔄Evaluating synthetic images from a pre-trained generator

If you prefer to generate synthetic images dynamically using a pretrained generator, you need to define functions to:
- Load the pretrained generator
- Generate new samples

📝 **Example configurations**: Sample configuration files for various pretrained generators are available [here](Tutorials/from_pre-trained_model):
- [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) $-$ a state-of-the-art generative adversarial network;
- [PACGAN](https://github.com/MatteoLai/PACGAN) $-$ a custom generative adversarial network;
- [Mediffusion](https://github.com/BardiaKh/Mediffusion) $-$ a diffusion model.

📖 **Colab tutorial**: Learn how to set up a configuration file for loading pretrained models and computing image quality metrics in 
[this Google Colab tutorial](https://colab.research.google.com/drive/1A70ENEZPpB9FT4mMU5ZiP0gFTkw--ysf?usp=drive_link).