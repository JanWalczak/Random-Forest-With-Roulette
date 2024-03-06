__author__  = "Jan Walczak, Patryk Jankowicz"

import pandas as pd
import configparser
import os


def load_data(file_name):
    """Function loads data from a csv file
    """
    original = pd.read_csv(f'Data/{file_name}.csv')
    subset_list =  divide_data(original)
    return subset_list


def divide_data(original_data):
    grouped = original_data.groupby(original_data.iloc[:, -1]) #Grouping data based by last column

    training_sizes = {}
    for group_name, group in grouped:
        training_sizes[group_name] = int(len(group) / 5)  

    subset_list = []
    for _ in range(4):  
        subset = pd.DataFrame()
        for group_name, training_size in zip(grouped.groups, training_sizes.values()):
            grouped = original_data.groupby(original_data.iloc[:, -1])
            group = grouped.get_group(group_name)
            
            sampled_group = group.sample(training_size, replace=False)
            subset = pd.concat([subset, sampled_group])
            
            non_training_indexes = original_data.index.difference(subset.index)
            original_data = original_data.loc[non_training_indexes]

        subset_list.append(subset)

    subset_list.append(original_data)
    
    return subset_list


def validate_data(subset_list):
    """Function giving ability to validate split of training and test data
    """
    total = pd.concat(subset_list, ignore_index=True)
    for subset in subset_list:
        print(f'Size: {len(subset)/len(total)}%')
        subset_value_counts = subset.iloc[:,-1].value_counts()
        total_value_counts = total.iloc[:,-1].value_counts()
        for (value, subset_value_count, total_value_count) in zip(subset_value_counts.index, subset_value_counts, total_value_counts):
            print(f"Value: {value}: {subset_value_count/total_value_count}%")
    return total.iloc[:, -1].unique()
 

def load_base_settings():
    """Function load base(default) settings used for each test from a config file

    """
    base_settings = {}
    config = configparser.ConfigParser()
    config.read('conf.ini')
    base_settings['max_depth'] = config.getint('Base_settings','base_max_depth')
    base_settings['min_number_of_values'] = config.getint('Base_settings','base_min_number_of_values')
    base_settings['threshold_percentage'] = float(config.get('Base_settings','base_threshold_percentage'))
    base_settings['attributes_percentage'] = float(config.get('Base_settings','base_attributes_percentage'))
    base_settings['number_of_trees'] = config.getint('Base_settings','base_number_of_trees')
    base_settings['mode'] = config.getint('Base_settings','base_mode')
    return base_settings
    
    
def load_test_settings():
    """Function loads settings for all tests from config file 

    Returns:
        _type_: _description_
    """
    test_settings = {}
    config = configparser.ConfigParser()
    config.read('conf.ini')
    test_settings['max_depth_range'] = [int(max_depth) for max_depth in config.get('Test_settings','max_depth_range').split(',')]
    test_settings['min_number_of_values_range'] = [int(min_values) for min_values in config.get('Test_settings','min_number_of_values_range').split(',')]
    test_settings['threshold_percentage_range'] = [float(treshold) for treshold in config.get('Test_settings','threshold_percentage_range').split(',')]
    test_settings['attributes_percentage_range'] = [float(attributes) for attributes in config.get('Test_settings','attributes_percentage_range').split(',')]
    test_settings['number_of_trees_range'] = [int(number_of_trees) for number_of_trees in config.get('Test_settings','number_of_trees_range').split(',')]
    test_settings['mode_range'] = [int(mod_range) for mod_range in config.get('Test_settings','mode_range').split(',')]
    return test_settings
    
    
def load_system_settings(): 
    """Function loads system settings (like dataset file name, number of processes to use etc.) from the config file

    """
    config = configparser.ConfigParser()
    config.read('conf.ini')
    file_name = config.get('System_settings','file_name')
    attribute_types = [int(attribute_type) for attribute_type in config.get('System_settings','attribute_types').split(',')]
    number_of_samples = config.getint('System_settings','number_of_samples')
    number_of_processes = config.getint('System_settings','number_of_processes')
    return file_name, attribute_types, number_of_samples, number_of_processes
    
    
def create_log_file(file_path,classes):
    """Function creates a log file
    """
    f = open(file_path, "w")
    f.write('number_of_trees,max_depth,min_number_of_values,threshold_percentage,attributes_percentage,mode')
    for i in classes:
        for j in classes:
            f.write(f',{i}:{j}')
    f.write('\n')
    

def create_test_files_folder(folder_path,folder_name):
    """Function creates an empty folder for specified test

    """
    existing_folders = []

    for folder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder)) and folder.startswith(folder_name+'_'):
            existing_folders.append(folder)
            
    if not existing_folders:
        new_number = 1
    else:
        highest_number = 0
        for folder in existing_folders:
            try:
                number = int(folder[len(folder_name+'_'):])
                if number > highest_number:
                    highest_number = number
            except ValueError:
                pass
        new_number = highest_number + 1

    new_folder_name = f'{folder_path}{folder_name+'_'}{new_number}'
    os.makedirs(new_folder_name)
    return new_folder_name