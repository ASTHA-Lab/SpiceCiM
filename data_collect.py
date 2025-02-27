import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def read_and_process_file(file_path):
    """Reads a text file and processes it into a DataFrame."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    processed_data = []
    header_passed = False
    for line in lines:
        if "VALUE" in line:
            header_passed = True
            continue
        if header_passed:
            line = line.replace('(', '').replace(')', '').strip()
            parts = line.split()
            if len(parts) >= 2:
                processed_data.append(parts[:2])

    df = pd.DataFrame(processed_data, columns=['Param', 'Value'])
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    return df.reset_index(drop=True)

def find_indices(df, start_time, end_time):
    """Finds start and end indices for the specified time range."""
    time_data = df[df['Param'] == '"time"']
    start_index = time_data[time_data['Value'] >= start_time].index[0]
    end_index = time_data[time_data['Value'] >= end_time].index[0]
    return start_index, end_index

def find_stable_number(values):
    """Finds the most stable number in a list of values."""
    if not values:
        return None
    return max(set(values), key=values.count)

def find_max_number(values):
    """Finds the maximum number in a list of values."""
    return max(values, default=None)

def extract_values_between(df, start_index, end_index, param):
    """Extracts values between two indices for a specific parameter."""
    return df[(df['Param'] == f'"{param}"') & (df.index >= start_index) & (df.index <= end_index)]['Value']

def stable_value_finder(df, start_index, end_index, target_net):
    """Finds the stable value for a given parameter within a range."""
    if start_index is not None and end_index is not None:
        values_df = extract_values_between(df, start_index, end_index, target_net)
        if not values_df.empty:
            values_list = values_df.tolist()
            stable_number = find_max_number(values_list)
            return stable_number
    return None

def time_index_finder(df, start_time, increment):
    """Finds and prints start and end indices for specified start time and increment."""
    time_data = df[df['Param'] == '"time"']
    start_indices = []
    end_indices = []

    while True:
        start_row = time_data[time_data['Value'] >= start_time]
        if start_row.empty:
            break
        start_index = start_row.index[0]
        start_time_value = start_row.iloc[0]['Value']

        start_indices.append(start_index)
        end_time = start_time + increment

        end_row = time_data[time_data['Value'] >= end_time]
        if end_row.empty:
            end_index = time_data.index[-1]
            end_indices.append(end_index)
            break

        end_index = end_row.index[0]
        end_indices.append(end_index)

        start_time = end_time

    return start_indices, end_indices

def process_folder(folder_path, start_time_tick, increment):
    """Processes a single folder and returns results."""
    target_path = os.path.join(folder_path, 'psf', 'tran.tran.tran')
    if not os.path.exists(target_path):
        return None

    print(f"Processing file: {target_path}")
    df = read_and_process_file(target_path)
    start_indices, end_indices = time_index_finder(df, start_time_tick, increment)

    if not start_indices or not end_indices:
        return None

    result_row = {'Folder': os.path.basename(folder_path)}
    iprb_labels = [f'IPRB{col}:in' for col in range(16)]

    for label in iprb_labels:
        result_row[label] = []
        for start_index, end_index in zip(start_indices, end_indices):
            stable_number = stable_value_finder(df, start_index, end_index, label)
            result_row[label].append(stable_number)

    return result_row

def process_files_in_folder(base_path, start_time_tick, increment, output_csv_path):
    """Processes all tran.tran.tran files in specified subdirectories and creates a CSV file."""
    folders = [os.path.join(root, dir_name) for root, dirs, _ in os.walk(base_path) for dir_name in dirs]
    data = []

    # Use parallel processing to speed up folder processing
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_folder, folder, start_time_tick, increment) for folder in folders]
        for future in as_completed(futures):
            result = future.result()
            if result:
                data.append(result)

    df_output = pd.DataFrame(data)
    df_output.to_csv(output_csv_path, index=False)
    print(f"Output CSV file saved at: {output_csv_path}")

# Example usage
base_path = r"weights_distribution"
start_time_tick = 0
increment = 3e-9
output_csv_path = "./tmp/output.csv"

process_files_in_folder(base_path, start_time_tick, increment, output_csv_path)
