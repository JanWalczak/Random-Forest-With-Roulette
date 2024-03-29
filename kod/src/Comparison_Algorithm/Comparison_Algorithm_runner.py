__author__  = "Jan Walczak, Patryk Jankowicz"

import sys
sys.path.append('src')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import Utility as utl
import numpy as np
import csv
import time
import pandas as pd

def fit_data(dataframe_subset_list, classes,samples,base_settings,tested_value,value_range,attribute_types,file_name,folder_path,number_of_processes):
    test_base_settings = base_settings.copy()
    logs_file_name = f'{folder_path}\\{tested_value}.csv'
    utl.create_log_file(logs_file_name,classes)
    for x in value_range:
        test_base_settings[tested_value] = x
        for _ in range(samples):
            for i in range(5):
                K = dataframe_subset_list[i]
                T = pd.concat([df for j, df in enumerate(dataframe_subset_list) if j != i], ignore_index=True)
                
                start_time = time.time()
                
                forest = RandomForestClassifier(n_estimators=test_base_settings['number_of_trees'],
                                                criterion='gini',
                                                max_depth=test_base_settings['max_depth'],
                                                min_samples_split=test_base_settings['min_number_of_values'],
                                                max_features=test_base_settings['attributes_percentage'], 
                                                n_jobs=number_of_processes)
                
                forest.fit(T.iloc[:, :-1],T.iloc[:, -1])
                predictions = forest.predict(K.iloc[:, :-1])
                conf_matrix = confusion_matrix(K.iloc[:, -1], predictions)
                print(f'File:{file_name},Num Trees:{test_base_settings['number_of_trees']},Max depth:{test_base_settings['max_depth']},Min values:{test_base_settings['min_number_of_values']},Threshold:{test_base_settings['threshold_percentage']*100}%,Atributes:{test_base_settings['attributes_percentage']*100}%,t:{time.time()-start_time}s,mode:{test_base_settings['mode']}')
                    
                with open(logs_file_name, mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    confusion_matrix_list = []
                    rows, columns = np.shape(conf_matrix)
                    for i in range(rows):
                        for j in range(columns):
                            confusion_matrix_list.append(conf_matrix[i][j])
                    csv_writer.writerow([test_base_settings['number_of_trees'],test_base_settings['max_depth'],test_base_settings['min_number_of_values'],test_base_settings['threshold_percentage'],test_base_settings['attributes_percentage'],test_base_settings['mode']]+confusion_matrix_list)


def main():
    file_name, attribute_types, number_of_samples, number_of_processes = utl.load_system_settings()
    base_settings = utl.load_base_settings()
    test_settings = utl.load_test_settings()
    dataframe_subset_list = utl.load_data(file_name)
    classes = utl.validate_data(dataframe_subset_list)
    
    folder_path = utl.create_test_files_folder('src\\Comparison_Algorithm\\Logs\\',file_name)
    
    fit_data(dataframe_subset_list,classes,number_of_samples,base_settings,'max_depth',test_settings['max_depth_range'],attribute_types,file_name,folder_path,number_of_processes)
    fit_data(dataframe_subset_list,classes,number_of_samples,base_settings,'min_number_of_values',test_settings['min_number_of_values_range'],attribute_types,file_name,folder_path,number_of_processes)
    fit_data(dataframe_subset_list,classes,number_of_samples,base_settings,'threshold_percentage',test_settings['threshold_percentage_range'],attribute_types,file_name,folder_path,number_of_processes)
    fit_data(dataframe_subset_list,classes,number_of_samples,base_settings,'attributes_percentage',test_settings['attributes_percentage_range'],attribute_types,file_name,folder_path,number_of_processes)
    fit_data(dataframe_subset_list,classes,number_of_samples,base_settings,'number_of_trees',test_settings['number_of_trees_range'],attribute_types,file_name,folder_path,number_of_processes)
    fit_data(dataframe_subset_list,classes,number_of_samples,base_settings,'mode',test_settings['mode_range'],attribute_types,file_name,folder_path,number_of_processes)
                        
                            
                            
if __name__ == '__main__':
    main()