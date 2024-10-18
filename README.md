# SpiceCiM

## A SPICE-based Design and Optimization Framework for eNVM-based Analog In-memory Computing

**Cite our work**:
S. M. Mojahidul Ahsan, M. Sakib Shahriar, Mrittika Chowdhury, Tanvir Hossain, Md. Sakib Hasan, and Tamzidul Hoque. 2024. *Accurate, Yet Scalable: A SPICE-based Design and Optimization Framework for eNVM-based Analog In-memory Computing*. In IEEE/ACM International Conference on Computer-Aided Design (ICCAD '24), October 27–31, 2024, New York, NY, USA. ACM, New York, NY, USA, 9 pages. [https://doi.org/10.1145/3676536.3676827](https://doi.org/10.1145/3676536.3676827)


## Table of Contents

1. [Overview](#overview)
2. [Background](#background)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Features](#features)
6. [License](#license)
7. [Contact](#contact)

## Overview

This project introduces **SpiceCiM**, a SPICE-based design and optimization framework tailored for analog compute-in-memory (ACIM) architectures using emerging non-volatile memory (eNVM) technologies. ACIM architectures are essential for accelerating deep neural network (DNN) models, especially in edge AI applications where power, performance, and area efficiency are paramount. The framework provides a solution to challenges in simulating large-scale ACIM architectures, which are often sensitive to process, voltage, and temperature variations and analog noise.

### Key Features of the Framework:

1. **Automated SPICE Netlist Generation**: It generates SPICE netlists for DNN models, simplifying hardware simulation.
2. **DNN Weights Mapping**: Accurately maps DNN weights to eNVM conductance values, ensuring faithful hardware-level inference simulations.
3. **Optimized Simulation Runtime**: By introducing a seed crossbar array methodology, it reduces simulation time while maintaining accuracy, achieving up to 35x speedup over traditional methods.
4. **Comprehensive Integration**: It allows system-level simulations through integration with architectural simulators like GEM5, enabling comprehensive analysis of latency, power, and inference accuracy.

This framework pushes the boundary of analog in-memory computing research by enabling accurate, large-scale SPICE simulations for DNN inference, making it a powerful tool for researchers and designers developing the next generation of energy-efficient AI hardware.

## Background

## Installation

Step-by-step guide on how to install and run **SpiceCiM** locally:
```bash
# Clone the repository
git clone https://github.com/ASTHA-Lab/SpiceCiM.git

# Change the directory
cd SpiceCiM

# Install dependencies (if applicable)
npm install
```
## Usage
### Designing Neural Network:

### Design Generation:

### Setting-up Simulation Environment:



## Features

## License

## Contact
