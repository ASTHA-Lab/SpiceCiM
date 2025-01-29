import pandas as pd
import time
import os

# def read_and_process_file(file_path):
#     """Reads a text file and processes it into a DataFrame."""
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#
#     processed_data = []
#     header_passed = False
#     for line in lines:
#         if "VALUE" in line:
#             header_passed = True
#             continue
#         if header_passed:
#             line = line.replace('(', '').replace(')', '').strip()
#             parts = line.split()
#             if len(parts) >= 2:
#                 processed_data.append(parts[:2])
#     return pd.DataFrame(processed_data, columns=['Param', 'Value'])

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
    df = df.reset_index()  # This adds the 'index' column explicitly
    return df




def find_indices(df, start_time, end_time):
    """Finds start and end indices for the specified time range."""
    # Filter rows where 'Param' is "time"
    time_data = df[df['Param'] == '"time"']
    time_data_csv = time_data.reset_index()
    # Save the DataFrame to CSV, including the new index column
    time_data_csv.to_csv('time_data.csv', index=False)
    print(time_data_csv)
    # Find the index of the row matching start_time
    start_index = time_data.index[time_data['Value'] == start_time].tolist()
    # If start_index has values, get the first element
    if start_index:
        start_index = start_index[0]
    # Check if start_index is empty and find the nearest upper value if needed
    if not start_index:
        start_index = time_data.index[time_data['Value'] > start_time].min()

    end_index = time_data.index[time_data['Value'] == end_time].tolist()
    if end_index:
        end_index = end_index[0]
    if not end_index:
        # Find the index of the row with the closest value greater than end_time
        end_index = time_data.index[time_data['Value'] > end_time].min()

    # start_index = start_index[0]
    # end_index = end_index[0]
    # Print the results
    print("Start index:", start_index)  # Expected to show an exact match or the closest higher value
    print("End index:", end_index)  # Expected to show the closest higher value if exact match is not found

    # start_index = df[(df['Param'] == '"time"') & (df['Value'] == start_time)].index
    # end_index = df[(df['Param'] == '"time"') & (df['Value'] == end_time)].index
    # if start_index.empty or end_index.empty:
    #     return None, None
    return start_index, end_index

def find_stable_number(values):

    """Finds the most stable number in a list of values."""
    current_number, current_count, max_count, stable_number = None, 0, 0, None
    for value in values:
        if value == current_number:
            current_count += 1
        else:
            if current_count > max_count:
                max_count = current_count
                stable_number = current_number
            current_number, current_count = value, 1
    if current_count > max_count:
        stable_number = current_number
    return stable_number

def find_max_number(values):
    """Finds the maximum number in a list of values."""
    return max(values, default=None)


def extract_values_between(df, start_index, end_index, param):
    """Extracts values between two indices for a specific parameter."""
    indices = df[(df['Param'] == f'"{param}"') & (df.index > start_index) & (df.index < end_index)].index
    return df.loc[indices, 'Value'].reset_index(drop=True)

def index_finder(file_path, start_time_tick, end_time_tick):
    df = read_and_process_file(file_path)
    start_index, end_index = find_indices(df, start_time_tick, end_time_tick)
    return df, start_index, end_index

def stable_value_finder(df, start_index, end_index, target_net):
    print("I am in stable_value_finder")
    print("Start index: ", start_index)
    print("End index: ", end_index)
    print("Target net: ", target_net)
    if start_index is not None and end_index is not None:
        values_df = extract_values_between(df, start_index, end_index, target_net)
        if not values_df.empty:
            values_list = values_df.tolist()
            #stable_number = find_stable_number(values_list)
            stable_number = find_max_number(values_list)
            print(f"The stable number is: {stable_number}")
            return stable_number
        else:
            print("No values found for the specified parameter in the given time range.")
    else:
        print("Time range indices not found, please check the data.")


def time_index_finder(df, start_time, increment):
    """Finds and prints start and end indices for specified start time and increment, and returns them."""
    time_data = df[df['Param'] == '"time"']
    start_indices = []
    end_indices = []
    start_times = []
    end_times = []

    while True:
        if time_data['Value'].ge(start_time).any():
            start_row = time_data[time_data['Value'] >= start_time].iloc[0]
            start_index = start_row.name  # Use the name attribute to get the row index
            start_time_value = start_row['Value']
        else:
            break

        start_indices.append(start_index)
        start_times.append(start_time_value)
        end_time = start_time + increment

        if time_data['Value'].ge(end_time).any():
            end_row = time_data[time_data['Value'] >= end_time].iloc[0]
            end_index = end_row.name  # Use the name attribute to get the row index
            end_time_value = end_row['Value']
        else:
            end_index = time_data.iloc[-1].name
            end_time_value = time_data.iloc[-1]['Value']
            end_indices.append(end_index)
            end_times.append(end_time_value)
            break

        end_indices.append(end_index)
        end_times.append(end_time_value)
        print(f"Start Index: {start_index}, Start Time: {start_time_value}")
        print(f"End Index: {end_index}, End Time: {end_time_value}")
        start_time = end_time_value
    # Remove the last elements from start_indices and end_indices before returning
    if start_indices:
        start_indices = start_indices[:-1]
    if end_indices:
        end_indices = end_indices[:-1]

    return start_indices, end_indices


# def time_index_finder(df, start_time, increment):
#     """Finds and prints start and end indices for specified start time and increment, and returns them."""
#
#     # Filter out the rows where 'Param' is "time"
#     time_data = df[df['Param'] == '"time"']
#
#     # Initialize variables
#     start_indices = []
#     end_indices = []
#     start_times = []
#     end_times = []
#
#     # Loop to find the start and end indices based on start time and increment
#     while True:
#         # Find the row for the start time
#         if time_data['Value'].ge(start_time).any():
#             start_row = time_data[time_data['Value'] >= start_time].iloc[0]
#             start_index = start_row['index']
#             start_time_value = start_row['Value']
#         else:
#             break  # Exit if no start_time match found
#
#         start_indices.append(start_index)
#         start_times.append(start_time_value)
#
#         # Calculate the end time
#         end_time = start_time + increment
#
#         # Find the row for the end time
#         if time_data['Value'].ge(end_time).any():
#             end_row = time_data[time_data['Value'] >= end_time].iloc[0]
#             end_index = end_row['index']
#             end_time_value = end_row['Value']
#         else:
#             end_index = time_data.iloc[-1]['index']
#             end_time_value = time_data.iloc[-1]['Value']
#             end_indices.append(end_index)
#             end_times.append(end_time_value)
#             break  # Exit loop if no end_time match found
#
#         end_indices.append(end_index)
#         end_times.append(end_time_value)
#
#         # Print the results for this iteration
#         print(f"Start Index: {start_index}, Start Time: {start_time_value}")
#         print(f"End Index: {end_index}, End Time: {end_time_value}")
#
#         # Update start_time for the next iteration
#         start_time = end_time_value
#
#     # Return start and end indices
#     return start_indices, end_indices

def process_files_in_folder(base_path, start_time_tick, increment, output_csv_path):
    """Processes all tran.tran.tran files in specified subdirectories and creates a CSV file."""
    iprb_labels = [f'IPRB{col}:in' for col in range(16)]

    data = []  # List to store results

    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            target_path = os.path.join(folder_path, 'psf', 'tran.tran.tran')

            if os.path.exists(target_path):
                print(f"Processing file: {target_path}")
                df = read_and_process_file(target_path)
                print("#####################################################")
                start_indices, end_indices = time_index_finder(df, start_time_tick, increment)  # Now expecting lists
                print("Printing the start and end indices from time_index_finder")
                print("start_indices:", start_indices)
                print("end_indices:", end_indices)
                print("#####################################################")
                if start_indices and end_indices:
                    result_row = {'Folder': os.path.basename(folder_path)}  # Initialize result row here
                    for label in iprb_labels:
                        result_row[label] = []  # Initialize a list to store results for each label

                    for i, (start_index, end_index) in enumerate(zip(start_indices, end_indices)):
                        for label in iprb_labels:

                            stable_number = stable_value_finder(df, start_index, end_index, label)
                            result_row[label].append(stable_number)  # Append results to the respective label's list

                    data.append(result_row)

    # Create DataFrame from the collected data
    df_output = pd.DataFrame(data)

    # Save DataFrame to CSV
    df_output.to_csv(output_csv_path, index=False)
    print(f"Output CSV file saved at: {output_csv_path}")


# Example usage
base_path = r"weights_distribution"
start_time_tick = 0.5e-9
increment = 3.2e-9
output_csv_path = "output.csv"

process_files_in_folder(base_path, start_time_tick, increment, output_csv_path)
