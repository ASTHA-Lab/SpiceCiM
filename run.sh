#!/bin/bash

# Update package list
sudo apt update -y

# Install necessary packages
sudo apt install -y python3 python3-pip

# Install required Python libraries
pip3 install --upgrade pip
pip3 install numpy tensorflow scipy pillow pandas scikit-image configparser

# Run the raptor script
python3 SpiceCim.py
