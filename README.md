# crossmodality-dl

This repository contains the source code of the method presented in the paper "An immunofluorescence-guided segmentation
model in H&E images is enabled by tissue artifact correction by CycleGAN"
(https://doi.org/10.1016/j.modpat.2024.100591). The source code (except for code modified or copied from other sources,
see further below) is licensed under the Apache License, Version 2.0. If you find this work useful in your research,
please consider citing the paper referenced above.

The code is organized as follows:

- Notebooks "prepare\_labeled\_dataset.ipynb", "prepare\_independent\_dataset.ipynb", and "prepare\_staintrans.ipynb"
  are used to preprocess the two data sets considered in this study for later model training and evaluation
- Scripts starting with "run" are mainly used to drive the different experiments, and include model training and
  evaluation

- Folder "datasets" contains code to interact with the two data sets
- Folder "multiscale" contains the (modified) implementation of the multi-scale segmentation model proposed by Schmitz
  et al. (2021)
- Folder "staintrans" contains all implementation related to stain transfer, including the baselines by Shaban et al.
  (2019) and De Bel et al. (2021), and the own developed method

- Folder "singlescale" contains single-scale segmentation models used by Bulten et al. (2019) and Schmitz et al. (2021).
  They can be trained using notebooks "singlescale\_bulten2019.ipynb" and "singlescale\_schmitz2021.ipynb",
  respectively. The single-scale models were used as baseline models to explore the merit of using multi-scale
  segmentation models for epithelium segmentation. This exploration was excluded from submission as it was not
  immediately relevant in the context of tissue artifact correction. The model used by Bulten et al. (2019) is also used
  as feature extractor for our developed stain transfer method
- Folder "stainnorm" contains a stain color normalization network proposed by Tellez et al. (2019). Stain color
  normalization was explored as simpler alternative to stain transfer, but it was excluded from submission due to its
  low performance in this particular study

The other files contain mostly utilities to conduct the training and evaluation, e.g. the employed sampling and tiling
mechanisms and data (pre)processing routines. Note: File "constants.py" contained internal paths and file names which
were removed for submission.

The required dependencies are listed in "torch-linux-exact.yml" (only tested on Linux) and can be used with mamba
(recommended) or conda.

Code modified or copied from other sources is declared as such using inline comments, including references to respective
licenses if available. All relevant licenses are included in the LICENSES folder of this repository.

The clean-fid dependency had to be included in source form, as it had to be monkey patched for compatibility with our
project specifics. Copies of its license are located at cleanfid/LICENSE and LICENSES/CLEANFID_LICENSE.
