import math
import os
import shutil
import subprocess
import shlex
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
from skimage.filters import threshold_otsu
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
transistor_model =config['PATHS']['transistor_model_path']

# Wire Parasitics
res_val = float(config['HARDWARE']['res_val'])
cap_val = float(config['HARDWARE']['cap_val'])

# Simulation Parameters
process_corner = config['SIMULATION']['process_corner']
vdd_val = config['SIMULATION']['VDD']
vss_val = config['SIMULATION']['VSS']
temp_val = config['SIMULATION']['Temperature']
PW = float(config['SIMULATION']['PW'])
Trise = float(config['SIMULATION']['Trise'])
Tfall = float(config['SIMULATION']['Tfall'])

# Inference Parameters
image_path = config['INFERENCE']['image_path']
size = tuple(map(int, config['INFERENCE']['size'].split(',')))
vmax = float(config['INFERENCE']['vmax'])

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
        #print(total_pes)
        total_tiles = (total_pes + pes_per_tile - 1) // pes_per_tile  # Total Tiles needed
        #print(total_tiles)
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
            weights,biases = layer.get_weights()
            quantized_weights = quantize_weights_to_int(weights,pqbit)
            quantized_biases= quantize_weights_to_int(biases,pqbit)
            layer_weights[f"Layer{i}"] = quantized_weights
            hardware_requirements[f"Layer{i}"] = compute_hardware_requirements(weights.shape, synaptic_array_size, pe_size, tile_size)
    
    return model, layer_weights, quantized_biases, hardware_requirements


    

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


def weight_mapping(T_pos, T_neg, baseline_file_path, res_val, cap_val):
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
            device_model_content_pos += "    NM0 (net1 WL net2 0) nmos1 w=(2u) l=180n as=1.2p ad=1.2p ps=5.2u pd=5.2u \\\n"
            device_model_content_pos += "        m=(1)*(1)\n"
            device_model_content_pos += f"   R0 (net2 SL) resistor r={res_val} \nC1 (SL 0) capacitor c={cap_val/2} \nC0 (net2 0) capacitor c={cap_val/2}\n"
            device_model_content_pos += f"   I0 (BL net1) sky130_fd_pr_reram__reram_cell area_ox=1.024e-13 \\\n"
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
    with open("./tmp/device_model_pos.scs", 'w') as device_model_pos_file:
        device_model_pos_file.write(device_model_pos_content)
        
    # Write the device_model_neg_content to the device_model_neg.scs file
    with open("./tmp/device_model_neg.scs", 'w') as device_model_neg_file:
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
    

def conductance_calc(alpha, Tox, Tref, Tmax, Tmin):
    Gmax = alpha * np.exp(-((Tox - Tmax) / (Tox - Tref)))
    Gmin = alpha * np.exp(-((Tox - Tmin) / (Tox - Tref)))
    return Gmax, Gmin
    
def natural_sort_key(s):
    """ Sort strings containing numbers correctly by extracting parts of the string that are numeric. """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_and_process_to_bin_img(image_path, size, vmax,model):
    # Create dictionaries to store binary voltage data and binary images
    voltage_data = {}
    binary_images = {}
    
    file_names = sorted(os.listdir(image_path), key=natural_sort_key)
    tensorOut=open("./tmp/tensorOut.csv", "w")

    # Iterate over every file in the directory
    for file_name in file_names:
        if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Check for common image file extensions
            file_path = os.path.join(image_path, file_name)
            # Open the image
            with Image.open(file_path) as img:
                # Convert to grayscale
                img_gray = img.convert('L')
                # Resize the image
                img_resized = img_gray.resize(size)
                # Normalize for inference
                img_normalized = np.array(img_resized).astype('float32') / 255.0
                # Flatten the image data for thresholding
                img_flattened = img_normalized.flatten()

                # Apply Otsu's thresholding
                threshold_value = threshold_otsu(img_flattened)
                binary_image = np.where(img_flattened > threshold_value, 1, 0)
                
                # Generate binary voltage values
                voltage_values = binary_image * vmax
                
                # Store the binary voltages in the dictionary
                voltage_data[file_name] = voltage_values
                # Store binary images for returning
                binary_images[file_name] = binary_image.reshape(size)

                # Prepare image for model inference
                inference_ready_image = img_normalized.reshape(1, size[0], size[1])
                # Run inference
                prediction = model.predict(inference_ready_image)
                predicted_class = np.argmax(prediction, axis=1)
                print(f"Prediction for {file_name}: {predicted_class}")
                print(f"Prediction for {file_name},{predicted_class}", file=tensorOut)

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(voltage_data, orient='index')
    # Save the DataFrame to a CSV file
    df.to_csv('./tmp/img_to_voltage_data.csv', header=True, index_label='image_name')

    return binary_images, 'Processing complete and binary data saved to CSV.'




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
    df.to_csv('./tmp/img_to_voltage_data.csv', header=True, index_label='image_name')

    return 'Processing complete and data saved to CSV.'

def pixel_to_binary(binary_image, vread=1):
    # Initialize the netlist string
    netlist = ""
    
    # Iterate over the binary image data to create voltage sources for each input
    for i in range(min(num_inputs, len(binary_image))):
        voltage_value = binary_image[i] * vread  # 0V for '0', VDD for '1'
        voltage_data[file_name] = voltage_values
        #netlist += f"V{i} (IN{i} 0) vsource dc={voltage_value} type=dc\n"


def map_primary_input(input_path_pos, input_path_neg, full_path, sim_time):
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
    with open('./tmp/_pwl_sources.scs', 'r') as file:
        lines = file.readlines()
    selected_sources = "".join(lines[start_line:end_line])
    
    csnames = " ".join([f"IPRB{i}:in" for i in range(synaptic_array_size[1])])
    
    #print(selected_sources, "\n\n")

    # Read the golden input template and replace the placeholder with the selected sources
    with open(golden_input_path, 'r') as file:
        content = file.read()
    content = content.replace('<sim_time>', str(sim_time))
    content = content.replace('<input_source>', selected_sources)
    content = content.replace('<dev_mod>', './tmp/device_model_pos.scs')
    content = content.replace('<transistor_mod>', transistor_model)
    content = content.replace('<pcorner>', process_corner)
    content = content.replace('<vdd>', vdd_val)
    content = content.replace('<vss>', vss_val)
    content = content.replace('<temp>', temp_val)
    content = content.replace('<csnames>', csnames)
    
    # Write the modified content to the new input.scs file
    with open(input_path_pos, 'w') as file:
        file.write(content)
        
        
    content = content.replace('./tmp/device_model_pos.scs', './tmp/device_model_neg.scs')
    with open(input_path_neg, 'w') as file:
        file.write(content)

    print("Input sources mapped for simulation.")

def map_secondary_input(input_path_pos, input_path_neg, full_path, sim_time):
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
    with open('_PWL_source_HL.scs', 'r') as file:
        lines = file.readlines()
    selected_sources = "".join(lines[start_line:end_line])
    
    #print(selected_sources, "\n\n")
    
    csnames = " ".join([f"IPRB{i}:in" for i in range(synaptic_array_size[1])])

    # Read the golden input template and replace the placeholder with the selected sources
    with open(golden_input_path, 'r') as file:
        content = file.read()
    content = content.replace('<sim_time>', str(sim_time))
    content = content.replace('<input_source>', selected_sources)
    content = content.replace('<dev_mod>', './tmp/device_model_pos.scs')
    content = content.replace('<transistor_mod>', transistor_model)
    content = content.replace('<pcorner>', process_corner)
    content = content.replace('<vdd>', vdd_val)
    content = content.replace('<vss>', vss_val)
    content = content.replace('<temp>', temp_val)
    content = content.replace('<csnames>', csnames)
    

    # Write the modified content to the new input.scs file
    with open(input_path_pos, 'w') as file:
        file.write(content)
        
        
    content = content.replace('./tmp/device_model_pos.scs', './tmp/device_model_neg.scs')
    with open(input_path_neg, 'w') as file:
        file.write(content)

    print("Input sources mapped for simulation.")


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
    with open('./tmp/_pwl_sources.scs', 'w') as file:
        file.write(content)
    total_rows = df.shape[0]
    return total_rows, time_start

# Uncomment the function call in final version after writing and reviewing the code in PCI

def setup_simulation_args(siminput_path, infile_noext, suffix):
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
                    
                    
    run_string = " ".join(spectre_args)
    return run_string
                    


def generate_sl_decoder_netlist(num_address_pins, standard_cells):
    num_io = 2 ** num_address_pins
    Sub_ckt_name=f"BL_SL_DECODER_{num_address_pins}x{num_io}"
    Pins=" ".join([f"A{i}" for i in range(num_address_pins)]) + " VDD VSS " + " ".join([f"Y{i}" for i in range(num_io)]) + " " + " ".join([f"IO{i}" for i in range(num_io)]) + " VDDP PGMEN"
    # Start of the netlist
    netlist = f"// Library name: Custom_Library\n// Cell name: BL_SL_DECODER_{num_address_pins}x{num_io}\n// View name: schematic\nsubckt BL_SL_DECODER_{num_address_pins}x{num_io} " + " ".join([f"A{i}" for i in range(num_address_pins)]) + " VDD VSS " + " ".join([f"Y{i}" for i in range(num_io)]) + " " + " ".join([f"IO{i}" for i in range(num_io)]) + " VDDP PGMEN\n"

    # Generating Inverted Inputs using INVX1 cell
    inv_pins = standard_cells['INVX1']
    for i in range(num_address_pins):
        netlist += f"I_inv{i} ({' '.join(['A' + str(i) if pin == 'A' else 'X' + str(i) if pin == 'Y' else pin for pin in inv_pins])}) INVX1\n"

    # Helper function to generate AND gates
    def generate_and_gates(input_list, output_net, netlist, standard_cells):
        if len(input_list) == 1:
            return input_list[0], netlist, output_net
        else:
            new_output_net = f"net{output_net}"
            and_pins = standard_cells['AND2_X1']
            formatted_pins = ' '.join([input_list[0] if pin == and_pins[0] else input_list[1] if pin == and_pins[1] else new_output_net if pin == 'Y' else pin for pin in and_pins])
            netlist += f"I_and{output_net} ({formatted_pins}) AND2_X1\n"
            output_net += 1
            return generate_and_gates([new_output_net] + input_list[2:], output_net, netlist, standard_cells)

    
      # Generating AND Gates Logic for all combinations
    and_output_nets = []
    output_net_counter = 0
    for i in range(2**num_address_pins):
        # Reverse the binary string to switch MSB and LSB
        binary_str = format(i, f"0{num_address_pins}b")[::-1]
        and_inputs = [f"{'X' if binary_str[j] == '0' else 'A'}{j}" for j in range(num_address_pins)]
        
        final_output, netlist, output_net_counter = generate_and_gates(and_inputs, output_net_counter, netlist, standard_cells)
        and_output_nets.append(final_output)
    
    #and_output_nets = []
    #output_net_counter = 0
    #for i in range(2**num_address_pins):
        #binary_str = format(i, f"0{num_address_pins}b")
        #and_inputs = [f"{'X' if binary_str[j] == '0' else 'A'}{j}" for j in range(num_address_pins)]
        
        #final_output, netlist, output_net_counter = generate_and_gates(and_inputs, output_net_counter, netlist, standard_cells)
        #and_output_nets.append(final_output)
        
    # Connecting outputs of AND gates to BL_DRIVER block
    bl_driver_pins = standard_cells['BL_DRIVER']
    for i, and_output in enumerate(and_output_nets):
        formatted_pins = ' '.join(['Y' + str(i) if pin == 'OUT' else and_output if pin == 'RDEN' else 'PGMEN' if pin == 'PGMEN' else 'VDDP' if pin == 'VD_PGM' else f'IO{i}' if pin == 'VD_READ' else pin for pin in bl_driver_pins])
        netlist += f"I_bl_driver{i} ({formatted_pins}) BL_DRIVER\n"

    # End of the netlist
    netlist += f"ends BLDECODER_{num_address_pins}x{num_io}\n// End of subcircuit definition.\n"

    return netlist, Sub_ckt_name, Pins


# Example usage
#standard_cells = parse_standard_cells('/home/s550a945/proj/simulation_framework/standard_cell.netlist')
#decoder_netlist = generate_bl_decoder_netlist(4, standard_cells)
#print(decoder_netlist)


def generate_bl_wl_decoder_netlist(num_inputs, standard_cells):
    num_io = 2 ** num_inputs
    Sub_ckt_name=f"BL_WL_DECODER_{num_inputs}x{num_io}"
    Pins = (f"AE " + " ".join([f"A{i}" for i in range(num_inputs)]) + " VDD VSS " + " ".join([f"Y{i}" for i in range(num_io)]) + " " + " ".join([f"IO{i}" for i in range(num_io)]) + " VDDP PGMEN " + " ".join([f"T{i}" for i in range(num_io)]))
    # Start of the netlist
    netlist = f"// Library name: Custom_Library\n// Cell name: BL_WL_DECODER_{num_inputs}x{num_io}\n// View name: schematic\nsubckt BL_WL_DECODER_{num_inputs}x{num_io} AE " + " ".join([f"A{i}" for i in range(num_inputs)]) + " VDD VSS " + " ".join([f"Y{i}" for i in range(num_io)]) + " " + " ".join([f"IO{i}" for i in range(num_io)]) + " VDDP PGMEN " + " ".join([f"T{i}" for i in range(num_io)]) + "\n"

    # Generating Inverted Inputs using INVX1 cell
    inv_pins = standard_cells['INVX1']
    for i in range(num_inputs):
        netlist += f"I_inv{i} ({' '.join(['A' + str(i) if pin == 'A' else 'X' + str(i) if pin == 'Y' else pin for pin in inv_pins])}) INVX1\n"

    # Helper function to generate AND gates
    def generate_and_gates(input_list, output_net, netlist, standard_cells):
        if len(input_list) == 1:
            return input_list[0], netlist, output_net
        else:
            new_output_net = f"net{output_net}"
            and_pins = standard_cells['AND2_X1']
            formatted_pins = ' '.join([input_list[0] if pin == and_pins[0] else input_list[1] if pin == and_pins[1] else new_output_net if pin == 'Y' else pin for pin in and_pins])
            netlist += f"I_and{output_net} ({formatted_pins}) AND2_X1\n"
            output_net += 1
            return generate_and_gates([new_output_net] + input_list[2:], output_net, netlist, standard_cells)

    # Generating AND Gates Logic for all combinations
    and_output_nets = []
    output_net_counter = 0
    for i in range(num_io):
        binary_str = format(i, f"0{num_inputs}b")[::-1]
        and_inputs = [f"{'X' if binary_str[j] == '0' else 'A'}{j}" for j in range(num_inputs)]
        
        final_output, netlist, output_net_counter = generate_and_gates(and_inputs, output_net_counter, netlist, standard_cells)
        and_output_nets.append(final_output)

    # Connecting outputs of AND gates to BL_DRIVER block and OR gates
    bl_driver_pins = standard_cells['BL_DRIVER']
    or_pins = standard_cells['OR2_X1']
    for i, and_output in enumerate(and_output_nets):
        formatted_pins_bl_driver = ' '.join(['Y' + str(i) if pin == 'OUT' else and_output if pin == 'RDEN' else 'PGMEN' if pin == 'PGMEN' else 'VDDP' if pin == 'VD_PGM' else f'IO{i}' if pin == 'VD_READ' else pin for pin in bl_driver_pins])
        netlist += f"I_bl_driver{i} ({formatted_pins_bl_driver}) BL_DRIVER\n"
        formatted_pins_or = ' '.join([and_output if pin == or_pins[0] else 'AE' if pin == or_pins[1] else 'T' + str(i) if pin == 'Y' else pin for pin in or_pins])
        netlist += f"I_or{i} ({formatted_pins_or}) OR2_X1\n"

    # End of the netlist
    netlist += f"ends BL_WL_DECODER_{num_inputs}x{num_io}\n// End of subcircuit definition.\n"

    return netlist, Sub_ckt_name, Pins

def generate_crossbar_netlist(size_bwl, size_sl):
    subcircuit_name = f"rram_array_{size_bwl}x{size_sl}"
    bl_pins = [f"BL{i}" for i in range(size_bwl)]
    sl_pins = [f"SL{i}" for i in range(size_sl)]
    wl_pins = [f"WL{i}" for i in range(size_bwl)]

    netlist = f"// Library name: Custom Library\n"
    netlist += f"// Cell name: {subcircuit_name}\n"
    netlist += f"// View name: schematic\n"
    netlist += f"subckt {subcircuit_name} {' '.join(bl_pins + sl_pins + wl_pins)}\n"

    for i in range(size_sl):
        for j in range(size_bwl):
            instance_name = f"I{i * size_bwl + j}"
            bl_pin = bl_pins[j]
            sl_pin = sl_pins[i]
            wl_pin = wl_pins[j]
            # Modify the cell_subckt_name to include row and column indices
            cell_subckt_name = f"rram_cell_1T1R_180N_STD_R{j}C{i}"
            netlist += f"    {instance_name} ({bl_pin} {sl_pin} {wl_pin}) {cell_subckt_name}\n"

    netlist += f"ends {subcircuit_name}\n"
    netlist += "// End of subcircuit definition."

    return netlist, subcircuit_name, wl_pins, bl_pins, sl_pins


def parse_standard_cells(file_path):
    # Dictionary to store the pin sequences of standard cells
    standard_cells = {}
    current_cell = None
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('subckt'):
                parts = line.split()
                current_cell = parts[1]
                standard_cells[current_cell] = parts[2:]  # Store the pin sequence
            elif line.startswith('ends') and current_cell:
                current_cell = None
    #print(standard_cells)
    return standard_cells


def generate_subarray(input_length, num_tensor_out, stdcell_file_path):
    #column size
    bl_wl_row_size = input_length  # Or any other size (Size of input shape) # Need to change to automatic sizing of bl_wl_row
    sl_col_size =num_tensor_out # Row size (Size of Output shape)
    subarray_capacity = bl_wl_row_size * sl_col_size
   
    # Calculate the number of inputs for the decoders
    bl_wl_input_size = int(math.ceil(math.log2(bl_wl_row_size)))
    sl_input_size = int(math.ceil(math.log2(sl_col_size)))
    
    # Read and copy the standard cell netlist
    with open(stdcell_file_path, 'r') as file:
        standard_cell_netlist = file.read()

    # Generate the crossbar array netlist
    crossbar_netlist, xSC_name, xWL_pins, xBL_pins, xSL_pins = generate_crossbar_netlist(bl_wl_row_size, sl_col_size)
    
    #print(xSC_name, xWL_pins, xBL_pins, xSL_pins)

    # Generate the decoders netlist
    standard_cells = parse_standard_cells(stdcell_file_path)
    
    
    #wl_decoder_netlist = generate_decoder_netlist(sl_input_size, standard_cells)
    bl_decoder_netlist, bwdecSC_name, bwdec_pins = generate_bl_wl_decoder_netlist(bl_wl_input_size, standard_cells)
    #print(bwdecSC_name, bwdec_pins)
    sl_decoder_netlist, bsdecSC_name, bsdec_pins = generate_sl_decoder_netlist(sl_input_size, standard_cells)
    #print(bsdecSC_name, bsdec_pins)

    # Combine all netlists
    subarray_netlist = standard_cell_netlist + "\n\n" + crossbar_netlist + "\n\n"
    subarray_netlist += bl_decoder_netlist + "\n\n" + sl_decoder_netlist + "\n\n"
    
    
    #extra_pins_count = (sl_input_size**2) - sl_col_size
    VSS_map = sl_col_size

     # Top level subcircuit calls
    subarray_netlist += f"I1 ({' '.join(xBL_pins)} {' '.join(['net' + str(i*2 + 1) + 'i' for i in range(len(xSL_pins))])} {' '.join(xWL_pins)}) {xSC_name}\n"
    subarray_netlist += f"I2 (AE {' '.join(['A' + str(i) for i in range(bl_wl_input_size)])} VDD VSS {' '.join(['BL' + str(i) for i in range(2**bl_wl_input_size)])} {' '.join(['IN' + str(i) for i in range(2**bl_wl_input_size)])} VDDP PGMEN {' '.join(['WL' + str(i) for i in range(2**bl_wl_input_size)])}) {bwdecSC_name}\n"
    subarray_netlist += f"I3 ({' '.join(['SA' + str(i) for i in range(sl_input_size)])} VDD VSS {' '.join(['net' + str(i*2) + 'i' for i in range(sl_col_size)])} {' '.join(['VSS' for _ in range(VSS_map)])} VDDP_SL SLP) {bsdecSC_name}\n"
    
    # Adding iprobes
    for i in range(sl_col_size):
        subarray_netlist += f"IPRB{i} (net{i*2 + 1}i net{i*2}i) iprobe\n"
    
    return subarray_netlist
