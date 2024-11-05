import json
import os
import glob
import json
import random
# base_directory = "./data/dataset"


def list_up_arc_data_collections(provider_path = "./settings/data_provider_list.json"):
    with open(provider_path, 'r' ) as file:
        file = json.load(file)
    return file

def get_arc_data_collection(provider_name):
    provider_setting_list = list_up_arc_data_collections()
    provider_setting_path = provider_setting_list[provider_name]
    with open(provider_setting_path, 'r' ) as file:
        file = json.load(file)
    return file


def arc_data_collection_figure(setting_directory='settings/datapathType2', base_directory='data/dataset'):
    # Dictionary to store JSON file paths and the dataset names
    json_info = {}
    base_name = '_data_file.json'
    # Use glob to find all JSON files under 'data/datasets'
    json_files = glob.glob(os.path.join(base_directory, "*", "**", "*.json"), recursive=True)
    # json_files = glob.glob(os.path.join(base_directory, "**", "*.json"), recursive=True)
    file_list = {}
    for json_file in json_files:
        # Extract the relative path starting from 'data/datasets'
        relative_path = os.path.relpath(json_file, base_directory)
    
        # Split the relative path to get the dataset name
        path_parts = relative_path.split(os.sep)
        dataset_name = path_parts[0]  # This should be the dataset name under 'data/datasets'
        # Store the file path and its dataset name
        json_info[ dataset_name ] = {}

    # display(file_list,json_info)
    # Process each JSON file found
    for json_file in json_files:
        # Extract the relative path starting from 'data/datasets'
        relative_path = os.path.relpath(json_file, base_directory)

        # Split the relative path to get the dataset name
        path_parts = relative_path.split(os.sep)
        dataset_name = path_parts[0]
        datafolder_name = path_parts[-2]  # This should be the dataset name under 'data/datasets'
        taskId = path_parts[-1][:-5]
        # print(path_parts, dataset_name)
        # Store the file path and its dataset name
        if not datafolder_name in json_info[ dataset_name ]:
            json_info[ dataset_name ][ datafolder_name ] = {}
        json_info[ dataset_name ][ datafolder_name ][taskId] = json_file

    with open(setting_directory+'/arc_collections.json', 'w') as datafile:
        json.dump(json_info, datafile, indent=4)


def split_dict(data, train_ratio=0.4, eval_ratio=0.4, test_ratio=0.2, seed=42):
    # Check if the ratios sum to 1
    if train_ratio + eval_ratio + test_ratio != 1:
        raise ValueError("The sum of train_ratio, eval_ratio, and test_ratio must be 1.")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle the keys
    keys = list(data.keys())
    random.shuffle(keys)
    
    # Calculate the split indices
    total = len(keys)
    train_end = int(total * train_ratio)
    eval_end = train_end + int(total * eval_ratio)
    
    # Split keys
    train_keys = keys[:train_end]
    eval_keys = keys[train_end:eval_end]
    test_keys = keys[eval_end:]
    
    # Create split dictionaries
    train_data = {k: data[k] for k in train_keys}
    eval_data = {k: data[k] for k in eval_keys}
    test_data = {k: data[k] for k in test_keys}
    
    return train_data, eval_data, test_data