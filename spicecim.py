import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from scipy.ndimage import zoom
from PIL import Image
import pandas as pd
import math
import os
import shutil
import subprocess
import shlex
import csv
import re
from supports.rapsody import *
import sys
import configparser



# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Architecture Parameters
synaptic_array_size = tuple(map(int, config['ARCHITECTURE']['synaptic_array_size'].split(',')))
pe_size = tuple(map(int, config['ARCHITECTURE']['pe_size'].split(',')))
tile_size = tuple(map(int, config['ARCHITECTURE']['tile_size'].split(',')))

# Hardware Constants
Tox = float(config['HARDWARE']['Tox'])
Tref = float(config['HARDWARE']['Tref'])
Ik1 = float(config['HARDWARE']['Ik1'])
Vref = float(config['HARDWARE']['Vref'])
alpha = Ik1 / Vref

# Paths
golden_input_path = config['PATHS']['golden_input_path']
base_dir = config['PATHS']['base_dir']

# Hardware Parameters
Tmax = float(config['HARDWARE']['Tmax'])
Tmin = float(config['HARDWARE']['Tmin'])
Gmax, Gmin = conductance_calc(alpha, Tox, Tref, Tmax, Tmin)

# Wire Parasitics
res_val = float(config['HARDWARE']['res_val'])
cap_val = float(config['HARDWARE']['cap_val'])

# Simulation Parameters
PW = float(config['SIMULATION']['PW'])
Trise = float(config['SIMULATION']['Trise'])
Tfall = float(config['SIMULATION']['Tfall'])

# Inference Parameters
image_path = config['INFERENCE']['image_path']
size = tuple(map(int, config['INFERENCE']['size'].split(',')))
vmax = float(config['INFERENCE']['vmax'])

# Ensure distribution directory is set up
def setup_distribution_directory(base_dir):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
setup_distribution_directory(base_dir)

def main():
    #alpha = Ik1/Vref
    num_epoch = int(config['TRAINING']['num_epoch'])
    inputlen_sqrt = int(config['TRAINING']['inputlen_sqrt'])
    outputlen = int(config['TRAINING']['outputlen'])
    num_hidden_layer = int(config['TRAINING']['num_hidden_layer'])
    qbit = int(config['TRAINING']['qbit'])
    
    model, layer_weights, biases, hardware_reqs = create_and_train_ann_model(qbit, num_epoch, inputlen_sqrt, outputlen, num_hidden_layer, 16)
    binary_image, inf_image = load_and_process_to_bin_img(image_path, size, vmax, model)
    
    total_rows, sim_time = generate_pwl_sources('img_to_voltage_data.csv', PW, Trise, Tfall, synaptic_array_size[0])
    
    stdcell_file_path = config['PATHS']['stdcell_file_path']
    top_netlist = generate_subarray(synaptic_array_size[0], synaptic_array_size[1], stdcell_file_path)
    with open("top_netlist.scs", 'w') as file:
        file.write(top_netlist)
    
    for layer_idx, weights in layer_weights.items():
        distribute_weights({f"{layer_idx}": weights}, {f"{layer_idx}": hardware_reqs[f"{layer_idx}"]})
    
    baseline_file_path = config['PATHS']['baseline_file_path']
    
    base_dir = "weights_distribution"
    
    for layer_idx in sorted(layer_weights.keys(), key=lambda x: int(x.replace('Layer', ''))):
        layer_dir = os.path.join("weights_distribution", layer_idx)
        
        for root, dirs, files in os.walk(layer_dir, topdown=True):
            dirs.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
            files.sort(key=lambda x: int(re.findall(r'\d+', x.split('_')[1])[0]))
            
            for file in files:
                if file.endswith(".npy"):
                    full_path = os.path.join(root, file)
                    quantized_weights = np.load(full_path)
                    Tpos, Tneg = convert_to_conductance(quantized_weights, Gmax, Gmin, Tmin, qbit)
                    weight_mapping(Tpos, Tneg, baseline_file_path, res_val, cap_val)
                    infile_noext = os.path.splitext(full_path)[0]
                    
                    siminput_path_pos = os.path.join(os.getcwd(), "input_pos.scs")
                    siminput_path_neg = os.path.join(os.getcwd(), "input_neg.scs")
                    
                    if int(layer_idx.replace('Layer', '')) == 1:
                        run_string = setup_simulation_args(siminput_path_pos, infile_noext, '_pos')
                        map_primary_input(siminput_path_pos, siminput_path_neg, full_path, sim_time)
                        run_sim(run_string, infile_noext)
                    else:
                        run_string = setup_simulation_args(siminput_path_pos, infile_noext, '_pos')
                        map_secondary_input(siminput_path_pos, siminput_path_neg, full_path, sim_time)
                        run_sim(run_string, infile_noext)

if __name__ == "__main__":
    main()

