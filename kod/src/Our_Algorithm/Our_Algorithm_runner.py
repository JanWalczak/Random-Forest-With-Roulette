__author__  = "Jan Walczak, Patryk Jankowicz"

import sys
sys.path.append('src')
from Our_Algorithm.Forest.Forest import Forest
import Utility as utl
import csv
import pandas as pd

def fit_data(dataframe_subset_list,classes,samples,base_settings,tested_value,value_range,attribute_types,file_name,folder_path,number_of_processes):
    test_base_settings = base_settings.copy()
    logs_file_name = f'{folder_path}\\{tested_value}.csv'
    utl.create_log_file(logs_file_name,classes)
    for x in value_range:
        test_base_settings[tested_value] = x
        for i in range(5):
            K = dataframe_subset_list[i]
            T = pd.concat([df for j, df in enumerate(dataframe_subset_list) if j != i], ignore_index=True)
            
            for _ in range(samples):
                forest = Forest(
                    T=T,
                    K=K,
                    N=test_base_settings['number_of_trees'], 
                    attribute_types=attribute_types,
                    max_depth=test_base_settings['max_depth'], 
                    min_number_of_values=test_base_settings['min_number_of_values'],
                    threshold=test_base_settings['threshold_percentage'],
                    attributes=test_base_settings['attributes_percentage'],
                    mode=test_base_settings['mode'],
                    processes=number_of_processes)
                
                confusion_matrix, run_time = forest.create_random_forest()
                
                print(f'File:{file_name},Num Trees:{test_base_settings['number_of_trees']},Max depth:{test_base_settings['max_depth']},Min values:{test_base_settings['min_number_of_values']},Threshold:{test_base_settings['threshold_percentage']*100}%,Atributes:{test_base_settings['attributes_percentage']*100}%,t:{run_time}s,mode:{test_base_settings['mode']}')
                
                with open(logs_file_name, mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    confusion_matrix_list =[]
                    for k in confusion_matrix.keys():
                        for n in confusion_matrix[k].keys():
                            confusion_matrix_list.append(confusion_matrix[k][n])
                    csv_writer.writerow([test_base_settings['number_of_trees'],test_base_settings['max_depth'],test_base_settings['min_number_of_values'],test_base_settings['threshold_percentage'],test_base_settings['attributes_percentage'],test_base_settings['mode']]+confusion_matrix_list)


def main():
    file_name, attribute_types, number_of_samples, number_of_processes = utl.load_system_settings()
    base_settings = utl.load_base_settings()
    test_settings = utl.load_test_settings()
    dataframe_subset_list = utl.load_data(file_name)
    classes = utl.validate_data(dataframe_subset_list)

    folder_path = utl.create_test_files_folder('src\\Our_Algorithm\\Logs\\',file_name)        
        
    fit_data(dataframe_subset_list,classes,number_of_samples,base_settings,'max_depth',test_settings['max_depth_range'],attribute_types,file_name,folder_path,number_of_processes)
    fit_data(dataframe_subset_list,classes,number_of_samples,base_settings,'min_number_of_values',test_settings['min_number_of_values_range'],attribute_types,file_name,folder_path,number_of_processes)
    fit_data(dataframe_subset_list,classes,number_of_samples,base_settings,'threshold_percentage',test_settings['threshold_percentage_range'],attribute_types,file_name,folder_path,number_of_processes)
    fit_data(dataframe_subset_list,classes,number_of_samples,base_settings,'attributes_percentage',test_settings['attributes_percentage_range'],attribute_types,file_name,folder_path,number_of_processes)
    fit_data(dataframe_subset_list,classes,number_of_samples,base_settings,'number_of_trees',test_settings['number_of_trees_range'],attribute_types,file_name,folder_path,number_of_processes)
    fit_data(dataframe_subset_list,classes,number_of_samples,base_settings,'mode',test_settings['mode_range'],attribute_types,file_name,folder_path,number_of_processes)
                
                            
if __name__ == '__main__':
    main()