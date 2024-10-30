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
from rapsody import *
import sys
import configparser

inputConfigFile="config.ini"

config = configparser.ConfigParser()
config.read(inputConfigFile)

# Architecture Parameters
synaptic_array_size = tuple(map(int, config['Architecture_Parameters']['synaptic_array_size'].split(',')))
pe_size = tuple(map(int, config['Architecture_Parameters']['pe_size'].split(',')))
tile_size = tuple(map(int, config['Architecture_Parameters']['tile_size'].split(',')))

pwd=os.getcwd()

# Hardware Constants
Tox= float(config['Hardware_Constants']['Tox'].strip())
Tref= float(config['Hardware_Constants']['Tref'].strip())
Ik1= float(config['Hardware_Constants']['Ik1'].strip())
Vref= float(config['Hardware_Constants']['Vref'].strip())
alpha = Ik1/Vref

golden_input_path=pwd+"/supports/_golden_input.scs"


def setup_distribution_directory(base_dir):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

# Call this before you start processing any layers
base_dir = "weights_distribution"
setup_distribution_directory(base_dir)


def distribute_weights(layer_weights, hardware_requirements):
    rs, cs = synaptic_array_size  # Synaptic array size: rows and columns

    for layer_name, weights in layer_weights.items():
        m, n = weights.shape  # dimensions of the weight matrix from the model
        hardware_req = hardware_requirements[layer_name]
        
        num_arrays_vertically = (m + rs - 1) // rs
        num_arrays_horizontally = (n + cs - 1) // cs
        
        arrays_per_pe = pe_size[0] * pe_size[1]  # rpe * cpe
        pes_per_tile = tile_size[0] * tile_size[1]  # rt * ct
        total_pes = (num_arrays_vertically * num_arrays_horizontally + arrays_per_pe - 1) // arrays_per_pe  # Total PEs needed
        print(total_pes)
        total_tiles = (total_pes + pes_per_tile - 1) // pes_per_tile  # Total Tiles needed
        print(total_tiles)
        # Iterate over each Tile
        for tile_idx in range(total_tiles):
            # Iterate over each PE in the current Tile
            for pe_in_tile_idx in range(min(pes_per_tile, total_pes - tile_idx * pes_per_tile)):
                pe_idx = tile_idx * pes_per_tile + pe_in_tile_idx  # Global PE index

                start_array_idx = pe_idx * arrays_per_pe
                end_array_idx = start_array_idx + arrays_per_pe
                
                # Iterate over each Synaptic Array in the PE
                for array_idx in range(start_array_idx, min(end_array_idx, num_arrays_vertically * num_arrays_horizontally)):
                    sa_row = array_idx // num_arrays_horizontally
                    sa_col = array_idx % num_arrays_horizontally
                    start_row = sa_row * rs
                    start_col = sa_col * cs
                    end_row = min(start_row + rs, m)
                    end_col = min(start_col + cs, n)

                    synaptic_array = weights[start_row:end_row, start_col:end_col]
                    if synaptic_array.shape[0] < rs or synaptic_array.shape[1] < cs:
                        padded_array = np.full((rs, cs), 0)
                        padded_array[:synaptic_array.shape[0], :synaptic_array.shape[1]] = synaptic_array
                    else:
                        padded_array = synaptic_array

                    #print(tile_idx)
                    dir_path = os.path.join(base_dir, layer_name, f"Tile{tile_idx}", f"PE{pe_in_tile_idx}")
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)

                    array_filename = f"Array_{sa_row}_{sa_col}.npy"
                    np.save(os.path.join(dir_path, array_filename), padded_array)




def resize_images(images, new_size):
    resized_images = np.array([zoom(image, (new_size[0] / image.shape[0], new_size[1] / image.shape[1]), order=1) for image in images])
    return resized_images

def quantize_weights_to_1bit(weights):
    quantized_weights = np.where(weights > 0, 1, -1)
    return quantized_weights

def quantize_weights_to_int(weights, bits):
    """
    Quantize weights to integers using specified number of bits.
    Args:
        weights (numpy.ndarray): The original weight matrix.
        bits (int): Number of bits for quantization (1 to 8).
    Returns:
        numpy.ndarray: Quantized weight matrix with integer values.
    """
    # Calculate the number of quantization levels
    levels = 2 ** bits - 1
    
    # Normalize weights to [0, 1] and scale to [0, levels]
    min_w = np.min(weights)
    max_w = np.max(weights)
    scaled_weights = (weights - min_w) / (max_w - min_w) * levels

    # Round weights to nearest integer and map to symmetric range around zero
    quantized_weights = np.round(scaled_weights) - (levels // 2)

    # Convert to integer type
    return quantized_weights.astype(int)

def compute_hardware_requirements(weights_shape, sa_size, pe_size, tile_size):
    m, n = weights_shape  # dimensions of the weight matrix
    rs, cs = sa_size
    rpe, cpe = pe_size
    rt, ct = tile_size

    # Calculate height and width for synaptic arrays, then compute the total
    h_sa = math.ceil(m / rs)
    w_sa = math.ceil(n / cs)
    num_synaptic_arrays = h_sa * w_sa

    # Calculate height and width for PEs, then compute the total
    h_pe = math.ceil(h_sa / rpe)
    w_pe = math.ceil(w_sa / cpe)
    num_pes = h_pe * w_pe

    # Calculate height and width for tiles, then compute the total
    h_tile = math.ceil(h_pe / rt)
    w_tile = math.ceil(w_pe / ct)
    num_tiles = h_tile * w_tile
    
    return {
        'synaptic_arrays': {'total': num_synaptic_arrays, 'height': h_sa, 'width': w_sa},
        'pes': {'total': num_pes, 'height': h_pe, 'width': w_pe},
        'tiles': {'total': num_tiles, 'height': h_tile, 'width': w_tile}
    }

def create_and_train_ann_model(pqbit, nepoch, xsize, ntensor_out, num_hlayer, *hlayer_sizes):
    # Load and preprocess the dataset
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train_resized = resize_images(x_train, (xsize, xsize))
    x_train_resized = x_train_resized.reshape(x_train_resized.shape[0], xsize, xsize).astype('float32') / 255
    
    # Define the model structure
    hidden_layers = [Dense(size, activation='relu') for size in hlayer_sizes] if num_hlayer > 0 else []
    model = Sequential([Flatten(input_shape=(xsize, xsize)), *hidden_layers, Dense(ntensor_out, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train_resized, y_train, epochs=nepoch, verbose=1)
    
    # Extract and quantize weights, compute hardware requirements
    layer_weights = {}
    hardware_requirements = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
            weights = layer.get_weights()[0]
            quantized_weights = quantize_weights_to_int(weights,pqbit)
            layer_weights[f"Layer{i}"] = quantized_weights
            hardware_requirements[f"Layer{i}"] = compute_hardware_requirements(weights.shape, synaptic_array_size, pe_size, tile_size)
    
    return model, layer_weights, hardware_requirements


    

# Existing functions and class definitions are assumed to be included here
#Weight Mapping

def convert_to_conductance(quantized_weights, Gmax, Gmin, Tmin, qbit):
    delG = (Gmax - Gmin) / (2**qbit - 1)
    
    # Sampled device-to-device variations
    #Tox_variation = np.random.normal(Tox_mean, Tox_std)
    #Tref_variation = np.random.normal(Tref_mean, Tref_std)
    #alpha_variation = np.random.normal(alpha_mean, alpha_std)
    
    T_del = Tox + ((Tox - Tref) * np.log(((delG * np.abs(quantized_weights)) + Gmin) / alpha))

    #G_pos = np.where(quantized_weights >= 0, delG * np.abs(quantized_weights) + Gmin, Gmin)
    #G_neg = np.where(quantized_weights < 0, delG * np.abs(quantized_weights) + Gmin, Gmin)
    
    T_pos = np.where(quantized_weights >= 0, T_del, Tmin)
    T_neg = np.where(quantized_weights < 0, T_del, Tmin)
    
    
    
    return T_pos, T_neg


def weight_mapping(T_pos, T_neg, baseline_file_path):
    # Initialize the content for the device_model_pos.scs file
    device_model_pos_content = f"ahdl_include \"{baseline_file_path}\" \n"
    # Initialize the content for the device_model_neg.scs file
    device_model_neg_content = f"ahdl_include \"{baseline_file_path}\" \n"
    
    # Iterate over the T_pos and T_neg matrices
    for row_index, (row_pos, row_neg) in enumerate(zip(T_pos, T_neg)):
        for col_index, (T_pos_val, T_neg_val) in enumerate(zip(row_pos, row_neg)):
            device_model_content_pos = ""
            device_model_content_neg = ""
            
            # Device name
            device_name = f"rram_cell_1T1R_180N_STD_R{row_index}C{col_index}"
            
            # Add content for T_pos
            device_model_content_pos += f"// Cell name: {device_name}\n"
            device_model_content_pos += f"// View name: schematic\n"
            device_model_content_pos += f"subckt {device_name} BL SL WL\n"
            device_model_content_pos += "    NM0 (net1 WL SL 0) nmos1 w=(2u) l=180n as=1.2p ad=1.2p ps=5.2u pd=5.2u \\\n"
            device_model_content_pos += "        m=(1)*(1)\n"
            device_model_content_pos += f"    I0 (BL net1) sky130_fd_pr_reram__reram_cell area_ox=1.024e-13 \\\n"
            device_model_content_pos += f"        Tox=5e-09 Tfilament_max=4.9e-09 Tfilament_min=3.3e-09 \\\n"
            device_model_content_pos += f"        Tfilament_0=3.3e-09 Eact_generation=1.501 Eact_recombination=1.5 \\\n"
            device_model_content_pos += f"        I_k1=6.14e-05 Tfilament_ref=4.7249e-09 V_ref=0.43 velocity_k1=150 \\\n"
            device_model_content_pos += f"        gamma_k0=16.5 gamma_k1=-1.25 Temperature_0=300 \\\n"
            device_model_content_pos += f"        C_thermal=3.1825e-16 tau_thermal=2.3e-10 t_step=1e-09 initial_state={T_pos_val:.4e}\n"
            device_model_content_pos += f"ends {device_name}\n"
            device_model_content_pos += "// End of subcircuit definition.\n\n"

            device_model_pos_content += device_model_content_pos
            
            # Add content for T_neg
            device_model_content_neg += f"// Cell name: {device_name}\n"
            device_model_content_neg += f"// View name: schematic\n"
            device_model_content_neg += f"subckt {device_name} BL SL WL\n"
            device_model_content_neg += "    NM0 (net1 WL SL 0) nmos1 w=(2u) l=180n as=1.2p ad=1.2p ps=5.2u pd=5.2u \\\n"
            device_model_content_neg += "        m=(1)*(1)\n"
            device_model_content_neg += f"    I0 (BL net1) sky130_fd_pr_reram__reram_cell area_ox=1.024e-13 \\\n"
            device_model_content_neg += f"        Tox=5e-09 Tfilament_max=4.9e-09 Tfilament_min=3.3e-09 \\\n"
            device_model_content_neg += f"        Tfilament_0=3.3e-09 Eact_generation=1.501 Eact_recombination=1.5 \\\n"
            device_model_content_neg += f"        I_k1=6.14e-05 Tfilament_ref=4.7249e-09 V_ref=0.43 velocity_k1=150 \\\n"
            device_model_content_neg += f"        gamma_k0=16.5 gamma_k1=-1.25 Temperature_0=300 \\\n"
            device_model_content_neg += f"        C_thermal=3.1825e-16 tau_thermal=2.3e-10 t_step=1e-09 initial_state={T_neg_val:.4e}\n"
            device_model_content_neg += f"ends {device_name}\n"
            device_model_content_neg += "// End of subcircuit definition.\n\n"

            device_model_neg_content += device_model_content_neg

    # Write the device_model_pos_content to the device_model_pos.scs file
    with open("device_model_pos.scs", 'w') as device_model_pos_file:
        device_model_pos_file.write(device_model_pos_content)
        
    # Write the device_model_neg_content to the device_model_neg.scs file
    with open("device_model_neg.scs", 'w') as device_model_neg_file:
        device_model_neg_file.write(device_model_neg_content)




def run_sim(run_string, infile_noext):
    run_command = shlex.split(run_string)
    print(f"Running simulation for {infile_noext.split('/')[-1]}")
    #Simulation Run Call
    subprocess.call(run_command)  # Running the simulation
    print("Done.")

    output_file = "./" + infile_noext + "/psf/tran.tran.tran"
    header = True
    all_data = []

    # Prepare a dictionary to store IPRB values indexed by column number
    iprb_values = {f'IPRB{col}:in': [] for col in range(synaptic_array_size[1])}

    '''with open(output_file, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if "VALUE" in line:
                header = False  # Start reading values after this line
                continue
            
            if not header:
                line1 = line.replace('(', '')
                line2 = line1.replace(')', '')
                line_val = shlex.split(line2)
                
                for col in range(synaptic_array_size[1]):
                    if line_val[0].lower() == f"iprb{col}:in":
                        iprb_values[f'IPRB{col}:in'].append(float(line_val[1]))

    # Aggregate data for all columns for this run
    all_data.append(iprb_values)
    fields = sorted(all_data[0].keys()) 

    # Write results to a CSV file
    csv_file = f"{infile_noext}_results.csv"
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        
        # Finding the maximum length of lists to handle varying lengths
        max_length = max(len(lst) for lst in all_data[0].values())

        # Writing data
        for i in range(max_length):
            row = {field: all_data[0][field][i] if i < len(all_data[0][field]) else '' for field in fields}
            writer.writerow(row)'''

    #return all_data

def conductance_calc(Tmax, Tmin):
    Gmax = alpha * np.exp(-((Tox - Tmax) / (Tox - Tref)))
    Gmin = alpha * np.exp(-((Tox - Tmin) / (Tox - Tref)))
    return Gmax, Gmin
    
def load_and_process_test_image(image_path, size, vmax):
    # Create a dictionary to store voltage data for each image
    voltage_data = {}

    # Iterate over every file in the directory
    for file_name in os.listdir(image_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Check for common image file extensions
            file_path = os.path.join(image_path, file_name)
            # Open the image
            with Image.open(file_path) as img:
                # Convert to grayscale
                img_gray = img.convert('L')
                # Resize the image
                img_resized = img_gray.resize(size)
                # Normalize the image
                #image_normalized = image_resized.astype('float32') / 255
                # Reshape to match the model input shape
                #image_reshaped = image_normalized.reshape(1, size[0], size[1])
                # Flatten the image data
                img_flattened = np.array(img_resized).flatten()
                # Linearly map the pixel values to the range 0 to vmax
                voltage_values = np.round((img_flattened / 255) * vmax, 2)
                # Store in the dictionary
                voltage_data[file_name] = voltage_values

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(voltage_data, orient='index')
    # Save the DataFrame to a CSV file
    df.to_csv('img_to_voltage_data.csv', header=True, index_label='image_name')

    return 'Processing complete and data saved to CSV.'

def map_primary_input(input_path_pos, input_path_neg, full_path):
    # Extract row and column from the filename
    filename = os.path.basename(full_path)
    print(filename)
    match = re.search(r"Array_(\d+)_(\d+).npy", filename)
    if match:
        row_number = int(match.group(1))
        # column_number = int(match.group(2)) # Currently unused

    # Calculate the start and end lines for source fetching
    start_line = row_number * synaptic_array_size[0]
    end_line = start_line + synaptic_array_size[0]

    # Read the necessary source lines from _pwl_sources.scs
    with open('_pwl_sources.scs', 'r') as file:
        lines = file.readlines()
    selected_sources = "".join(lines[start_line:end_line])
    
    print(selected_sources, "\n\n")

    # Read the golden input template and replace the placeholder with the selected sources
    with open(golden_input_path, 'r') as file:
        content = file.read()
    content = content.replace('<input_source>', selected_sources)
    content = content.replace('<dev_mod>', 'device_model_pos.scs')

    # Write the modified content to the new input.scs file
    with open(input_path_pos, 'w') as file:
        file.write(content)
        
        
    content = content.replace('device_model_pos.scs', 'device_model_neg.scs')
    with open(input_path_neg, 'w') as file:
        file.write(content)

    print("Input sources mapped for simulation.")

def map_secondary_input(input_path_pos, input_path_neg, full_path):
    # Extract row and column from the filename
    filename = os.path.basename(full_path)
    print(filename)
    match = re.search(r"Array_(\d+)_(\d+).npy", filename)
    if match:
        row_number = int(match.group(1))
        # column_number = int(match.group(2)) # Currently unused

    # Calculate the start and end lines for source fetching
    start_line = row_number * synaptic_array_size[0]
    end_line = start_line + synaptic_array_size[0]

    # Read the necessary source lines from _pwl_sources.scs
    with open('_pwl_sources_HL.scs', 'r') as file:
        lines = file.readlines()
    selected_sources = "".join(lines[start_line:end_line])
    
    print(selected_sources, "\n\n")

    # Read the golden input template and replace the placeholder with the selected sources
    with open(golden_input_path, 'r') as file:
        content = file.read()
    content = content.replace('<input_source>', selected_sources)
    content = content.replace('<dev_mod>', 'device_model_pos.scs')

    # Write the modified content to the new input.scs file
    with open(input_path_pos, 'w') as file:
        file.write(content)
        
        
    content = content.replace('device_model_pos.scs', 'device_model_neg.scs')
    with open(input_path_neg, 'w') as file:
        file.write(content)

    print("Input sources mapped for simulation.")

'''def generate_pwl_sources(csv_path, pulse_width, trise, tfall):
    # Load CSV file
    df = pd.read_csv(csv_path, index_col=0)  # Assuming the first column is image names and used as index

    # Initialize the content to be written to the file
    content = ""

    # Process each column (each voltage source)
    for col in df.columns:
        time_start = 0
        voltage_start = 0
        waveform_data = []

        # Append each point in the waveform
        for voltage in df[col]:
            waveform_data.extend([
                (time_start, voltage_start),
                (time_start + pulse_width, voltage_start),
                (time_start + pulse_width + trise, voltage),
                (time_start + 2 * pulse_width, voltage),
                (time_start + 2 * pulse_width + tfall, voltage_start)
            ])
            time_start += 3 * pulse_width + trise + tfall

        # Generate the source string
        waveform_str = " ".join(f"{time} {voltage}" for time, voltage in waveform_data)
        content += f"VI{col} (IN{col} 0) vsource wave=[{waveform_str}] pwl_period={time_start} type=pwl\n"

    # Write to file
    with open('_pwl_sources.scs', 'w') as file:
        file.write(content)

    return "Netlist with PWL sources generated and saved."
'''

def generate_pwl_sources(csv_path, pulse_width, trise, tfall, scol):
    # Load CSV file
    df = pd.read_csv(csv_path, index_col=0)  # Assuming the first column is image names and used as index

    # Initialize the content to be written to the file
    content = ""
    node_counter = 0  # Initialize counter for node numbering

    # Process each column (each voltage source)
    for col in df.columns:
        time_start = 0
        voltage_start = 0
        waveform_data = []

        # Append each point in the waveform
        for voltage in df[col]:
            waveform_data.extend([
                (time_start, voltage_start),
                (time_start + pulse_width, voltage_start),
                (time_start + pulse_width + trise, voltage),
                (time_start + 2 * pulse_width, voltage),
                (time_start + 2 * pulse_width + tfall, voltage_start)
            ])
            time_start += 3 * pulse_width + trise + tfall

        # Generate the source string using modulus operation for node counter
        waveform_str = " ".join(f"{time} {voltage}" for time, voltage in waveform_data)
        content += f"VI{col} (IN{node_counter % (scol)} 0) vsource wave=[{waveform_str}] pwl_period={time_start} type=pwl\n"
        node_counter += 1  # Increment the node counter for the next source

    # Write to file
    with open('_pwl_sources.scs', 'w') as file:
        file.write(content)
    total_rows = df.shape[0]
    return total_rows, time_start

# Uncomment the function call in final version after writing and reviewing the code in PCI

def setup_simulation_args(siminput_path, infile_noext, suffix)
    spectre_args = ["spectre -64",
                    siminput_path,
                    "+escchars",
                    "=log ./" + infile_noext + f"{suffix}/psf/spectre.out",
                    "-format psfascii",
                    "-raw ./" + infile_noext + f"{suffix}/psf",
                    "+lqtimeout 900",
                    "-maxw 5",
                    "-maxn 5",
                    "+logstatus"]
                    
                    
    run_string = " ".join(spectre_args, cyc_time)
    return run_string
                    



def main():
    # Load and preprocess dataset, train model
    # Parameters
    num_epoch = int(config["Parameters"]["num_epoch"])
    inputlen_sqrt = int(config["Parameters"]["inputlen_sqrt"])
    outputlen = int(config["Parameters"]["outputlen"])
    num_hidden_layer = int(config["Parameters"]["num_hidden_layer"])
    hidden_layer_size=list(map(int, config['Parameters']['hidden_layer_size'].split(',')))
    qbit=int(config["Parameters"]["qbit"])
    
    #Hardware Parameters
    Tmax=float(config["Hardware_Parameters"]["Tmax"])
    Tmin=float(config["Hardware_Parameters"]["Tmin"])
    Gmax, Gmin = conductance_calc(Tmax, Tmin)
    #print(Gmax, Gmin)
    stdcell_file_path=pwd+'/supports/standard_cell.netlist'
    print(stdcell_file_path)
    
    #Inference Parameters
    image_path=fr"{config["Inference_Parameters"]["image_path"].strip()}"
    size=(inputlen_sqrt,inputlen_sqrt)
    vmax=float(config["Inference_Parameters"]["vmax"])
    
    PW= float(config["Inference_Parameters"]["PW"])
    Trise = float(config["Inference_Parameters"]["Trise"])
    Tfall = float(config["Inference_Parameters"]["Tfall"])
    
    cyc_time = PW+Trise+Tfall
    
    #input_path = "/home/s550a945/proj/simulation_framework/mlp_baseline_v0.5/input.scs"
    
    
    
    # Create and train the model
    model, layer_weights, hardware_reqs = create_and_train_ann_model(qbit, num_epoch, inputlen_sqrt, outputlen, num_hidden_layer, *hidden_layer_size)
    #print(layer_weights)
    print(hardware_reqs)
    
    
    #Prerparing Inference Material
    load_and_process_test_image(image_path, size, vmax) #Generates img_to_voltage_data.csv
    total_rows, sim_time = generate_pwl_sources('img_to_voltage_data.csv', PW, Trise, Tfall, synaptic_array_size[0]) # file, PW, trise, tfall


    #Generating subarray
    top_netlist=generate_subarray(synaptic_array_size[0], synaptic_array_size[1], stdcell_file_path)
    with open(f"top_netlist.scs", 'w') as file:
        file.write(top_netlist)
    
    # Now distribute and save weights according to hardware configuration
    for layer_idx, weights in layer_weights.items():
        #print(layer_idx)
        distribute_weights({f"{layer_idx}": weights}, {f"{layer_idx}": hardware_reqs[f"{layer_idx}"]})
    
    # Path to the baseline file needed for the hardware model
    baseline_file_path =pwd+"/supports/sky130_fd_pr_reram__reram_cell.v"
    print(baseline_file_path)

    # Iterate over the directories created by distribute_weights
    base_dir = "weights_distribution"
    

    # Process each layer in sequence
    for layer_idx in sorted(layer_weights.keys(), key=lambda x: int(x.replace('Layer', ''))):
        layer_dir = os.path.join("weights_distribution", layer_idx)
        
        for root, dirs, files in os.walk(layer_dir, topdown=True):
            dirs.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # Sort by numerical part if directory names have numbers
            files.sort(key=lambda x: int(re.findall(r'\d+', x.split('_')[1])[0]))  # Assuming filenames like array_row_col.npy

            for file in files:
                if file.endswith(".npy"):
                    full_path = os.path.join(root, file)
                    #Load the quantized weights from NPY array
                    quantized_weights = np.load(full_path)

                    #Get the Tfillament+ and Tfillamen- for the corresponding quantized weights
                    Tpos, Tneg = convert_to_conductance(quantized_weights, Gmax, Gmin, Tmin, qbit)

                    #Map the weight to model
                    weight_mapping(Tpos, Tneg, baseline_file_path)
                    infile_noext = os.path.splitext(full_path)[0]
                    
                    
                    siminput_path_pos = os.path.join(pwd, "input_pos.scs")
                    
                    siminput_path_neg = os.path.join(pwd, "input_neg.scs")
                    

                    
                    if int(layer_idx.replace('Layer', '')) == 1:
                        print(f"Layer-{layer_idx.replace('Layer', '')} Simulation: Positive")
                        run_string=setup_simulation_args(siminput_path_pos, infile_noext, '_pos')
                        map_primary_input(siminput_path_pos, siminput_path_neg, full_path)
                        #sim_data=run_sim(run_string, infile_noext)

                        print(f"Layer-{layer_idx.replace('Layer', '')} Simulation: Negative")
                        run_string=setup_simulation_args(siminput_path_pos, infile_noext, '_neg')
                        map_primary_input(siminput_path_pos, siminput_path_neg, full_path)
                        #run_sim(run_string, infile_noext)
                        #collect_data(infile_noext, cyc_time, total_rows)
                    else:
                        print(f"Layer-{layer_idx.replace('Layer', '')} Simulation: Positive")
                        run_string=setup_simulation_args(siminput_path_pos, infile_noext, '_pos')
                        map_secondary_input(siminput_path_pos, siminput_path_neg, full_path)
                        sim_data=run_sim(run_string, infile_noext)

                        print(f"Layer-{layer_idx.replace('Layer', '')} Simulation: Negative")
                        run_string=setup_simulation_args(siminput_path_pos, infile_noext, '_neg')
                        map_secondary_input(siminput_path_pos, siminput_path_neg, full_path)
                        run_sim(run_string, infile_noext)
                        #collect_data(infile_noext, cyc_time, total_rows)


                        #data_bit.py is going to clean up the current data and generate output.csv
                        
                        # Run sim and collect Result in temporary CSV
                        # Process and add the data, convert to voltage, Relu
                        # Prepare Next Layer Input
                        # Perform similar operation
                        # Get output through ADC and infer
                        


                    # Prepare simulation environment
                    #infile = os.path.join(root, "device_model.scs")
                    
                    #print(infile_noext)


                    

                    

                    #sim_data=run_sim(run_string, infile_noext)
                    
                    

                    #print(sim_data)
                    
if __name__ == "__main__":
    main()


