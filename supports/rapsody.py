import math
import os
import shutil
import subprocess
import shlex




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

