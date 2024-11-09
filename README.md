<h1 align="center">
Learning Aerodynamics for the Control of Flying Humanoid Robots
</h1>


<div align="center">

_A. Paolino, G. Nava, F. Di Natale, F. Bergonti, P. Reddy Vanteddu, D. Grassi, L. Riccobene, A. Zanotti, R. Tognaccini, G. Iaccarino, D. Pucci_

</div>

<p align="center">

https://github.com/user-attachments/assets/acdfc5f7-8452-44e8-9078-ed70e781a90d

</p>

<div align="center">
  Nature Communications Engineering
</div>

<div align="center">
  <a href="#abstract"><b>Abstract</b></a> |
  <a href="#installation"><b>Installation</b></a> |
  <a href="#usage"><b>Usage</b></a>
</div>

## Abstract

Robots with multi-modal locomotion are an active research field due to their versatility in diverse environments. In this context, additional actuation can provide humanoid robots with aerial capabilities. Flying humanoid robots face challenges in modeling and control, particularly with aerodynamic forces. This paper addresses these challenges from a technological and scientific standpoint. The technological contribution includes the mechanical design of iRonCub-Mk1, a jet-powered humanoid robot, optimized for jet engine integration, and hardware modifications for wind tunnel experiments on humanoid robots for precise aerodynamic forces and surface pressure measurements. The scientific contribution offers a comprehensive approach to model and control aerodynamic forces using classical and learning techniques. Computational Fluid Dynamics (CFD) simulations calculate aerodynamic forces, validated through wind tunnel experiments on iRonCub-Mk1. An automated CFD framework expands the aerodynamic dataset, enabling the training of a Deep Neural Network and a linear regression model. These models are integrated into a simulator for designing aerodynamic-aware controllers, validated through flight simulations and balancing experiments on the iRonCub-Mk1 physical prototype.

## Installation

1. Clone the repository:
  ```bash
  git clone https://github.com/ami-iit/paper_paolino_2024_nature_learning-aerodynamics-control.git
  ```

2. Follow the instructions to install https://github.com/ami-iit/ironcub-mk1-software using `conda` environments.

3. Install additional Python dependencies by running:
  ```bash
  conda install -c conda-forge matplotlib scipy scikit-learn optuna
  ```

4. Install PyTorch in the `conda` environment following the instruction at https://pytorch.org/get-started/locally/ for `pip`


## Usage

Please follow the instructions in the README of each directory to use the code present there.


### Maintainer

This repository is maintained by:

| | |
|:---:|:---:|
| [<img src="https://github.com/antonellopaolino.png" width="40">](https://github.com/antonellopaolino) | [@antonellopaolino](https://github.com/antonellopaolino) |

<p align="left">
   <a href="https://github.com/ami-iit/paper_paolino_2024_nature_learning-aerodynamics-control/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ami-iit/paper_paolino_2024_nature_learning-aerodynamics-control" alt="Size" class="center"/></a>
</p>
