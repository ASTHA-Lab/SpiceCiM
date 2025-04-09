# SpiceCiM

## A SPICE-based Design and Optimization Framework for eNVM-based Analog In-memory Computing

**Cite our work**:
@inproceedings{10.1145/3676536.3676827,
author = {Ahsan, S M Mojahidul and Shahriar, Muhammad Sakib and Chowdhury, Mrittika and Hossain, Tanvir and Hasan, Md Sakib and Hoque, Tamzidul},
title = {Accurate, Yet Scalable: A SPICE-based Design and Optimization Framework for eNVM based Analog In-memory Computing},
year = {2025},
isbn = {9798400710773},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3676536.3676827},
doi = {10.1145/3676536.3676827},
abstract = {This paper introduces a scalable SPICE-based tool infrastructure designed to optimize analog compute-in-memory (ACIM) architectures utilizing emerging non-volatile resistive memory (eNVM) technologies. The inherent efficiency of analog eNVM crossbar arrays in performing matrix-vector multiplications significantly enhances the power, performance, and area efficiency of edge AI devices and other applications. Our framework addresses the challenges of accurately simulating ACIM architectures, which are highly susceptible to variations in process, voltage, temperature, and analog noise. The framework uses SPICE for accurate analog and mixed-signal circuit simulation. It automates the generation of SPICE-level ACIM designs for deep neural networks. Additionally, it speeds up the simulation runtime by up to 35\texttimes{} for large DNN models while maintaining the same SPICE-level accuracy. Moreover, it ensures simulation convergence for large netlists, facilitating SPICE simulation of large-scale eNVM crossbars that were previously impractical. We demonstrated that our framework is capable of simulating inference using netlists for MLPs with over 800,000 parameters trained on the MNIST dataset within acceptable runtime, where contemporary SPICE simulators do not even converge. We validated the simulation results in terms of inference accuracy, which shows less than a 3.8\% accuracy drop compared to software-based inference results. Lastly, we demonstrated the integration of our framework with an architectural simulator, facilitating comprehensive system-level simulation.},
booktitle = {Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design},
articleno = {191},
numpages = {9},
keywords = {analog in-memory-computing, eNVM, SPICE, design automation},
location = {Newark Liberty International Airport Marriott, New York, NY, USA},
series = {ICCAD '24}
}


## 🚀 Current Version Updates (vX.X.X)

- ✅ **SLP Simulation Support**  
  The current version includes support for **SLP simulation**.

- 🐞 **MLP Simulation - Known Issues**  
  MLP simulation may contain bugs. We're actively working on resolving these in upcoming patches.



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

## Limitations:
1. Current version only supports Cadence Spectre. Support for Hspice and Opensource Ngspice are coming soon!
2. Current version only supports MLP and SLP network. Support for CNN is in progress!
3. Current verison only supports SkyWater 130nm RRAM model. Support for MRAM is coming soon!

# Run the the tool:
./run.sh


## Installation

Requirements
Python 3.x
Cadence SPECTRE simulator

Step-by-step guide on how to install and run **SpiceCiM** locally:
```bash
# Clone the repository
git clone https://github.com/ASTHA-Lab/SpiceCiM.git

# Change the directory
cd SpiceCiM

# Make run file executable:
chmod +x run.sh

# Change the config.ini file if necessary
# Make sure to add a valid transistor_model_path value (SPICE model comes with PDK supported by Spectre)
# Make sure to source appropriate Simulator License!  Current version only supports Cadence Spectre. Support for Hspice and Opensource Ngspice are coming soon!

# Run the the tool:
./run.sh

```

## Usage
Repository Structure
run.sh: Bash wrapper to run the SpiceCim.py after installing all the dependencies.
SpiceCim.py: Core script for creating the neural network, training it, generating the design, mapping weights, running simulations, and performing inference.
data_collect.py: Extracts waveform data and tabulates it.
checkInfer.py: Validates the inference results.
Getting Started
Running Instructions
Edit Configuration
Open config.ini and adjust the settings according to your requirements.

Load Cadence SPECTRE and Navigate to the Script Directory
Make sure Cadence SPECTRE is loaded and navigate to the directory containing the scripts.

Run spicecim.py
Execute this script to:

Create and train the neural network model.
Retrieve the trained weight matrix and hardware requirements.
Generate design specifications.
Map weights to the model.
Execute simulation and inference
```bash
python SpiceCim.py
```
Or just simply execute:
```bash
./run.sh
```

This script collects data from the waveform output and organizes it into a tabular format.
```bash
python data_collect.py
```
Run this script to check and validate the inference results.
```bash
python checkInfer.py
```


## License
MIT License

Copyright (c) 2025 ASTHA-Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
## Contact
Email: ahsan@ku.edu
