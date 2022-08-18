# Deep Diffusion Models for Seismic Processing

PyTorch implementation of **Deep Diffusion Models for Seismic Processing**.

<p align="left"><img width="50%" src="images/demultiple_img.PNG" /></p>

## Overview
Seismic data processing involves techniques to deal with undesired effects that occur during acquisition and pre-processing.These effects mainly comprise coherent artefacts such as multiples, non-coherent signals such as electrical noise, and loss of signal information at the receivers that leads to incomplete traces. In this work, we introduce diffusion models for three seismic applications: demultiple, denoising and interpolation.

<p align="left"><img width="60%" src="images/demultiple_img2.PNG" /></p>
<p align="left"><img width="60%" src="images/denoising_img.PNG" /></p>
<p align="left"><img width="60%" src="images/inter_img.PNG" /></p>

## Code
The repository contains the following files:
 <ul>
  <li> run.py -> This file contains the specifications</li>
  <li> diffusion.py -> This file contains the diffusion fucntions</li>
  <li> unet.py -> This file contains the diffusion model (architecture)</li>
  <li> visualization.ipnyb -> This Jupyter notebook can run inference and displys the results</li>
  <li> dataset -> This folder contains a few multiple-infested gathers and their multiple-free conuterparts</li>
  
</ul> 

To use the code, we only need to run the. 

Note that you might need to installl thrid party libraries.

## Acknowledgement
We acknowledge the code from [lucidrains](https://github.com/lucidrains/denoising-diffusion-pytorch) & [Janspiry](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)
