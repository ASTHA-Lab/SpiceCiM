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
import glob
import subprocess
import shlex
import csv
import re
from utils.spcimcore import *
import sys
import configparser
from utils.data_collect import *
from utils.dp import *

#os.makedirs('tmp')

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

# --- after config.read('config.ini') ---
binarize_images = config['INFERENCE'].getboolean('Binarize_image', fallback=False)


start_time_tick = 0
increment = 3e-9
#output_csv_path = "./tmp/output.csv"

# Ensure distribution directory is set up
def setup_distribution_directory(base_dir):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        #shutil.rmtree('tmp')
        glob.glob(os.path.join("./", "*.ahdlSimDB"))
        glob.glob(os.path.join("./", "*.srf"))
        glob.glob(os.path.join("./", "*.ic"))
        glob.glob(os.path.join("./", "*.fc"))
    os.makedirs(base_dir)
setup_distribution_directory(base_dir)
setup_distribution_directory('tmp')

def apply_hardware_activation(voltage_csv_in, voltage_csv_out,
                              activation="relu", vth=0.0, vmax=1.0):
    df = pd.read_csv(voltage_csv_in)
    # 1) Identify numeric columns (everything except any text columns)
    num_cols = df.select_dtypes(include=[np.number]).columns

    # 2) Cast them to float (in case they were strings of numbers):
    df[num_cols] = df[num_cols].astype(float)

    # 3) Apply your activation only to the numeric subset:
    if activation.lower() == "relu":
        df[num_cols] = df[num_cols].clip(lower=vth, upper=vmax)
    elif activation.lower() == "none":
        df[num_cols] = df[num_cols].clip(lower=0.0, upper=vmax)
    else:
        raise ValueError(f"Unknown activation {activation!r}")

    # 4) Write everything — numeric and non‑numeric — back out
    df.to_csv(voltage_csv_out, index=False)

def main():
    #alpha = Ik1/Vref
    num_epoch = int(config['TRAINING']['num_epoch'])
    inputlen_sqrt = int(config['TRAINING']['inputlen_sqrt'])
    outputlen = int(config['TRAINING']['outputlen'])
    #hidden_layer_sizes = list(config['TRAINING']['hidden_layers'])
    raw = config['TRAINING']['hidden_layers']
    qbit = int(config['TRAINING']['qbit'])
    try:
        hidden_layers = [int(size.strip()) for size in raw.split(',')]
        per_layer_ncount = hidden_layers + [outputlen]
    except:
        hidden_layers = None
        per_layer_ncount = [outputlen]
        #qbit = None
        
    
    
    #per_layer_ncount = hidden_layers.append(outputlen)
    
    

    #print(per_layer_ncount)
    
    model, layer_weights, biases, hardware_reqs, X_train = create_and_train_ann_model(qbit, num_epoch, inputlen_sqrt, outputlen, hidden_layers)
    binary_image, inf_image = load_and_process_images(image_path, size, vmax, model, binarize_images)
    print("Debug point 1")
    
    total_rows, sim_time = generate_pwl_sources('./tmp/img_to_voltage_data.csv', PW, Trise, Tfall, synaptic_array_size[0])
    
    print(total_rows, sim_time)
    
    print(layer_weights)
    
    stdcell_file_path = config['PATHS']['stdcell_file_path']
    top_netlist = generate_subarray(synaptic_array_size[0], synaptic_array_size[1], stdcell_file_path)
    with open("./tmp/top_netlist.scs", 'w') as file:
        file.write(top_netlist)
    
    print("Debug point 3")
    
    for layer_idx, weights in layer_weights.items():
        distribute_weights({f"{layer_idx}": weights}, {f"{layer_idx}": hardware_reqs[f"{layer_idx}"]})
    
    baseline_file_path = config['PATHS']['envm_model_path']
    

    compute_mapping_ranges(layer_weights, biases, X_train, file_out="./tmp/mapping_ranges.json")
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
                    #if qbit:
                    Tpos, Tneg = convert_to_conductance(quantized_weights, Gmax, Gmin, Tmin, qbit)
                    print(f'with QBIT: {Tpos, Tneg}')
                    #else:
                        #Tpos, Tneg = convert_to_conductance_continuous(quantized_weights, Gmax, Gmin, Tmin)
                        #print(f'without QBIT: {Tpos, Tneg}')
                    weight_mapping(Tpos, Tneg, baseline_file_path, res_val, cap_val)
                    infile_noext = os.path.splitext(full_path)[0]
                    
                    siminput_path_pos = os.path.join(os.getcwd(), "./tmp/input_pos.scs")
                    siminput_path_neg = os.path.join(os.getcwd(), "./tmp/input_neg.scs")
                    
                        
                    if int(layer_idx.replace('Layer', '')) == 1:
                        print(f"Layer-{layer_idx.replace('Layer', '')} Simulation: Positive")
                        run_string=setup_simulation_args(siminput_path_pos, infile_noext, '_pos')
                        map_primary_input(siminput_path_pos, siminput_path_neg, full_path, sim_time)
                        run_sim(run_string, infile_noext)

                        print(f"Layer-{layer_idx.replace('Layer', '')} Simulation: Negative")
                        run_string=setup_simulation_args(siminput_path_neg, infile_noext, '_neg')
                        map_primary_input(siminput_path_pos, siminput_path_neg, full_path, sim_time)
                        run_sim(run_string, infile_noext)
                        #collect_data(infile_noext, cyc_time, total_rows)
                    else:
                        print(f"Layer-{layer_idx.replace('Layer', '')} Simulation: Positive")
                        run_string=setup_simulation_args(siminput_path_pos, infile_noext, '_pos')
                        map_secondary_input(siminput_path_pos, siminput_path_neg, full_path, sim_time)
                        run_sim(run_string, infile_noext)

                        print(f"Layer-{layer_idx.replace('Layer', '')} Simulation: Negative")
                        run_string=setup_simulation_args(siminput_path_neg, infile_noext, '_neg')
                        map_secondary_input(siminput_path_pos, siminput_path_neg, full_path, sim_time)
                        run_sim(run_string, infile_noext)
                        
        ncolumn = int(f"{layer_idx.replace('Layer', '')}")-1
        print(per_layer_ncount[ncolumn])
        print(f'Debug: The current Layer ID running is: {layer_idx} and layer path is: {layer_dir} and ncolumn is: {ncolumn}')
        
        #process_files_in_folder(layer_dir, start_time_tick, increment, f'./tmp/output_layer_{layer_idx}.csv', synaptic_array_size[1])
        #voltage_data = process_crossbar_csv_to_image_csv(f'./tmp/output_layer_{layer_idx}.csv', f'./tmp/_voltage_{layer_idx}.csv',vmax)
        #generate_pwl_sources_ll(f'./tmp/_voltage_{layer_idx}.csv', PW, Trise, Tfall, synaptic_array_size[0])
        
        # … inside your for layer_idx in sorted(layer_weights) loop …

        # 1) collect raw layer outputs (as voltages in [0, vmax])
        process_files_in_folder(
            layer_dir,
            start_time_tick,
            increment,
            f'./tmp/output_layer_{layer_idx}.csv',
            synaptic_array_size[1]
        )
        _ = process_crossbar_csv_to_image_csv(
            f'./tmp/output_layer_{layer_idx}.csv',
            f'./tmp/_voltage_{layer_idx}.csv',
            layer_idx,
            vmax
        )
        
        # after generate_pwl_sources_ll(...) and SPICE run for layer1:
        # we get "./tmp/_voltage_Layer1.csv"
        apply_hardware_activation(
            f'./tmp/_voltage_{layer_idx}.csv',
            f'./tmp/_voltage_{layer_idx}_activated.csv',
            activation="relu",
            vth=0.0,
            vmax=vmax
        )
        # then feed "./tmp/_voltage_Layer1_activated.csv" into your next
        # process_crossbar_csv_to_image_csv / current_to_voltage_fixed call.


        #import pandas as pd
        # …inside your for layer_idx in sorted(layer_weights) loop…

        # 1) compute the bias‐voltages in [0, vmax]
        b_vec       = biases[layer_idx]               # e.g. length=64 for hidden, 10 for output
        bias_list   = [[float(b)] for b in b_vec]     # wrap each bias in a list
        b_voltages  = bias_to_voltage_signed(bias_list, layer_idx, vmax) #current_to_voltage_fixed(b_current , layer_idx, vmax)
        # flatten to a simple list of floats
        b_voltages  = [bv[0] for bv in b_voltages]

        # 2) load the layer’s voltage CSV
        df_v = pd.read_csv(f'./tmp/_voltage_{layer_idx}_activated.csv')

        # 3) add each neuron’s bias‐voltage, clipping at vmax
        for j, vb in enumerate(b_voltages):
            col = str(j)
            if col in df_v.columns:
                df_v[col] = (df_v[col] + vb).clip(upper=vmax)

        # 4) write it back out
        df_v.to_csv(f'./tmp/_voltage_{layer_idx}_activated.csv', index=False)

        # …now generate the PWL sources as before…
        generate_pwl_sources_ll(
            f'./tmp/_voltage_{layer_idx}_activated.csv',
            PW, Trise, Tfall,
            synaptic_array_size[0]
        )


        
    compare_predictions(f'./tmp/_voltage_{layer_idx}.csv', './tmp/tensorOut.csv', './tmp/Infer_out.csv')
        

if __name__ == "__main__":
    main()
    
