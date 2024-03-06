__author__  = "Jan Walczak, Patryk Jankowicz"
import os
import sys
sys.path.append('src')
import Utility as utl
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import textalloc as ta
from collections import defaultdict
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import utils
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def calculate_average(group):
        return round(group.mean(), 15)


def calculate_averages(df,number_of_samples):
    return df.groupby(df.index // number_of_samples).apply(calculate_average)


def get_data_as_matrix(row, n):
    """Function converts passed data and returns it as a matrix - an array of arrays

    """
    values_list = row.iloc[6:].tolist()
    list_of_lists = []
    for i in range(n):
        sub_list = []
        for j in range(i*n,(i+1)*n):
            sub_list.append(values_list[j])
        list_of_lists.append(sub_list)
    data = np.array(list_of_lists)
    return data


def get_measures(TP,TN,FP,FN,classes_measures,class_name):
    """Function calculates all chosen quality indicators 
    """
    if TP+TN+FP+FN != 0:
        classes_measures[class_name]['ACC'] = round((TP+TN)/(TP+TN+FP+FN),4)
    else: 
        classes_measures[class_name]['ACC'] = 0
        
    if TP+FN != 0:
        classes_measures[class_name]['TPR'] = round((TP)/(TP+FN),4)
    else:
        classes_measures[class_name]['TPR'] = 0
        
    if TP+FP != 0:
        classes_measures[class_name]['PPV'] = round((TP)/(TP+FP),4) 
    else:
        classes_measures[class_name]['PPV'] = 0
    
    if FP+TN != 0:
        classes_measures[class_name]['TNR'] = round((TN)/(FP+TN),4)  
    else: 
        classes_measures[class_name]['TNR'] = 0
    
    if TN+FP != 0:
        classes_measures[class_name]['FAR'] = round((FP)/(TN+FP),4)  
    else: 
        classes_measures[class_name]['FAR'] = 0
        
    if classes_measures[class_name]['PPV']+classes_measures[class_name]['TPR'] != 0:
        classes_measures[class_name]['F1'] = round(2*(classes_measures[class_name]['PPV']*classes_measures[class_name]['TPR'])/(classes_measures[class_name]['PPV']+classes_measures[class_name]['TPR']),4)
    else: 
        classes_measures[class_name]['F1'] = 0
        
    if (TP+FP)!=0 and (TP+FN)!=0 and (TN+FP)!=0 and (TN+FN)!=0:
        classes_measures[class_name]['MCC'] = round((TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5),4)
    else: 
        classes_measures[class_name]['MCC'] = 0
    classes_measures[class_name]['FPR'] = round(1 - classes_measures[class_name]['TNR'],4)


def calculate_class_measures(classes,data,n, additional_dict, name):
    """Function summarizes TP, TN, FP, FN values and calls function 'get_measures' to calculate quality indicators for them

    """
    classes_measures = {}
    for c in range(n):
        classes_measures[classes[c]] = defaultdict(int)
    classes_measures['overall'] = defaultdict(int)   
    for c in range(n):
        for i in range(n):
            for j in range(n):
                if i!=c and j!=c:
                    classes_measures[classes[c]]['TN'] += data[i][j]
                elif i!=c and j==c:
                    classes_measures[classes[c]]['FP'] += data[i][j]
                elif i==c and j!=c:
                    classes_measures[classes[c]]['FN'] += data[i][j]
                elif i==c==j:
                    classes_measures[classes[c]]['TP'] += data[i][j]
                    
        classes_measures['overall']['TN'] += classes_measures[classes[c]]['TN']
        classes_measures['overall']['FP'] += classes_measures[classes[c]]['FP']
        classes_measures['overall']['FN'] += classes_measures[classes[c]]['FN']
        classes_measures['overall']['TP'] += classes_measures[classes[c]]['TP']
        
        get_measures(classes_measures[classes[c]]['TP'],classes_measures[classes[c]]['TN'],classes_measures[classes[c]]['FP'],classes_measures[classes[c]]['FN'],classes_measures,classes[c])
    
     
    get_measures(classes_measures['overall']['TP'],classes_measures['overall']['TN'],classes_measures['overall']['FP'],classes_measures['overall']['FN'],classes_measures,'overall')
    additional_dict[name] = classes_measures['overall']
    
    return classes_measures


def create_table(classes_measures, path): 
    """Function converts passed data and transforms it into a table that is nextly saved to a file
    """
    table_data = classes_measures.copy()

    labels = list(table_data.keys())
    col_labels = list(table_data[labels[0]].keys())
    data = [[f'{table_data[c][l]:.4f}' for l in col_labels] for c in labels]

    _, ax = plt.subplots(figsize=(1, 1))
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=col_labels, rowLabels=labels, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5) 
    table.auto_set_column_width(col=list(range(len(col_labels))))
    for i in range(4,len(col_labels)):
        column_list = []
        for j in range(0, len(labels)):
            column_list.append(data[j][i])
        maximum = max(column_list)
        for index, value in enumerate(column_list):
            if value == maximum:
                table[index+1,i].set_text_props(fontweight='bold')
    for i in range(0,len(col_labels)):
        for j in range(0, len(labels)):
            table[j+1,i].get_text().set_text(str(data[j][i]).replace('.', ',')) 
    plt.savefig(path,bbox_inches='tight')
    plt.close()


def create_heatmap(algorithm, data_matrix, classes,_current_test_, _current_file, current_path_final, current_test):
    """Function creates heatmap for passed data and saves it to a file
    """
    row_sums = data_matrix.sum(axis=1, keepdims=True)
    fractions = data_matrix / row_sums
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(fractions, annot=True, fmt=".2%", cmap="Reds", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if algorithm:
        plt.title(f'Macierz pomyłek (sklearn):{'_'.join(_current_file.split('_')[:-1])} | {current_test} : {_current_test_}')
    else:
        plt.title(f'Macierz pomyłek (nasz alg): {'_'.join(_current_file.split('_')[:-1])} | {current_test} : { _current_test_}')
    
    plt.savefig(f'{current_path_final}/heatmap.png')
    plt.close()


def calculate_classes(dataframe):
    """Function returns list of classes read from log files

    """
    classes = []
    for column_name in list(dataframe.columns[6:].values):
        names = column_name.split(':')
        if (names[0]==names[1]):
            classes.append(names[1])
    return classes   


def generate_tabels_and_heatmaps(algorithm, number_of_samples, param_dict):
    """Fundamental funtion in quality tool that generates tables and heatmaps for both our and comparison algorithms (depending on 'algorithm' variable value)

    """
    if algorithm:
        path = "Comparison_Algorithm"     
    else:
        path = "Our_Algorithm"
    dir = f'src/{path}/Logs/'
    for _all_tests_dir_ in os.listdir(dir):
        if _all_tests_dir_ == ".gitkeep":
            continue
        param_dict[_all_tests_dir_] = {}
        for _current_test_ in os.listdir(f'{dir}/{_all_tests_dir_}'): # Acquiring data from generated logs (.csv files)
            df = pd.read_csv(os.path.join(dir, _all_tests_dir_, _current_test_))
            classes = calculate_classes(df)
            path_for_files = f'src/Quality_tool/Measures/'
            averages = calculate_averages(df,number_of_samples)

            os.makedirs(os.path.join(path_for_files, _all_tests_dir_), exist_ok=True)
            os.makedirs(os.path.join(path_for_files, _all_tests_dir_, _current_test_[:-4]), exist_ok=True)

            path_for_averages = os.path.join(path_for_files,_all_tests_dir_, _current_test_[:-4], f"{path}_averages.csv")
            utl.create_log_file(path_for_averages,classes)  # Saving calculated averages to file
            averages.to_csv(path_for_averages, sep=',', index=False, encoding='utf-8') 

            n = len(classes)    
            test_table_dict = {}
            for _, row_in_test in averages.iterrows():
                current_value = row_in_test[str(_current_test_[:-4])]
                data_matrix = get_data_as_matrix(row_in_test,n)
                classes_measures = calculate_class_measures(classes,data_matrix,n, test_table_dict, str(current_value))
                current_path_final = os.path.join(path_for_files,_all_tests_dir_, _current_test_[:-4], str(current_value))
                current_path_final = os.path.join(path_for_files,_all_tests_dir_, _current_test_[:-4], str(current_value), path)
                os.makedirs(current_path_final, exist_ok=True)
                create_table(classes_measures, os.path.join(path_for_files,_all_tests_dir_, _current_test_[:-4], str(current_value), path, "classes_measures_table.png"))
                create_heatmap(algorithm, data_matrix, classes,current_value, _all_tests_dir_, current_path_final, _current_test_[:-4])
            path_for_test_table_dict = os.path.join(path_for_files,_all_tests_dir_, _current_test_[:-4], f"{path}_test_table.png")
            create_table(test_table_dict,path_for_test_table_dict)
            param_dict[_all_tests_dir_][_current_test_[:-4]]=test_table_dict            


def generate_graphs(our_algorithm_dict,comparison_algorithm_dict):
    """Functions generates 2 graphs: ROC and ACC value depending on tested parameter value for both algorithms

    """
    if our_algorithm_dict and comparison_algorithm_dict:
        for test_name in our_algorithm_dict.keys():
            if not test_name in comparison_algorithm_dict.keys():
                print("DATA ERROR")
                return
            for test_parameter in our_algorithm_dict[test_name].keys():
                if not test_parameter in comparison_algorithm_dict[test_name].keys():
                    print("DATA ERROR")
                    return
                tested_values = []
                our_acc = []
                our_tpr = []
                our_fpr = []
                comparison_acc = []
                comparison_tpr = []
                comparison_fpr = []
                for value_of_parameter in comparison_algorithm_dict[test_name][test_parameter].keys():
                    if not value_of_parameter in comparison_algorithm_dict[test_name][test_parameter].keys():
                        print("DATA ERROR")
                        return
                    tested_values.append(value_of_parameter)
                    our_acc.append(our_algorithm_dict[test_name][test_parameter][value_of_parameter]['ACC'])
                    comparison_acc.append(comparison_algorithm_dict[test_name][test_parameter][value_of_parameter]['ACC'])
                    our_tpr.append(our_algorithm_dict[test_name][test_parameter][value_of_parameter]['TPR'])
                    comparison_tpr.append(comparison_algorithm_dict[test_name][test_parameter][value_of_parameter]['TPR'])
                    our_fpr.append(our_algorithm_dict[test_name][test_parameter][value_of_parameter]['FPR'])
                    comparison_fpr.append(comparison_algorithm_dict[test_name][test_parameter][value_of_parameter]['FPR'])
                    
                    
                plt.scatter(our_fpr, our_tpr, label='Nasz algorytm', marker='o', color='blue')
                for i, label in enumerate(tested_values):
                    plt.annotate(label, (our_fpr[i], our_tpr[i]), textcoords="offset points", xytext=(0,10), ha='center')

                plt.scatter(comparison_fpr, comparison_tpr, label='Algorytm porównawczy', marker='s', color='red')
                for i, label in enumerate(tested_values):
                    plt.annotate(label, (comparison_fpr[i], comparison_tpr[i]), textcoords="offset points", xytext=(0,10), ha='center')

                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.title(f'Wykres ROC dla zmiany {test_parameter}')
                plt.legend()
                plt.savefig(f'src/Quality_tool/Measures/{test_name}/{test_parameter}/ROC.png')
                plt.close()
        
                plt.plot(tested_values, our_acc, label='Nasz algorytm')
                plt.plot(tested_values, comparison_acc, label='Algorytm porównawczy (sklearn)')
                plt.xlabel('Testowane wartości dla: '+ test_parameter)
                plt.ylabel('Acc')
                plt.title(f'Wykres porównawczy algorytmów dla parametru: {test_parameter}')
                plt.legend()
                plt.savefig(f'src/Quality_tool/Measures/{test_name}/{test_parameter}/graph.png')
                plt.close()
                
    elif our_algorithm_dict:
        print('Give comparison algorithm')
    elif comparison_algorithm_dict:
        print('Give our algorithm')
    else:
        print("ERROR NO DATA")
        

def get_adjusted_dimensions(img,image_width,image_height):
    """Functions adjusts an image's size
    """
    scale_width = image_width / img.getSize()[0]
    scale_height = image_height / img.getSize()[1]
    scale_factor = min(scale_width, scale_height)

    adjusted_width = img.getSize()[0] * scale_factor
    adjusted_height = img.getSize()[1] * scale_factor
    return adjusted_width, adjusted_height


def pdf_draw_number_line(c,x,y,i,j,k,font_size,text):
    """Function generates chapter's numbers

    """
    font_type = "Arial-bold"
    c.setFont(font_type, font_size)
    c.drawString(x, y, f'{i}.{j}.1   ')
    text_width = c.stringWidth(f'{i}.{j}.{k}   ', font_type, font_size)
    font_type = "Arial"
    c.setFont(font_type, font_size)
    c.drawString(x + text_width, y, text)


def create_pdf_with_images_and_text(c, images, text_list,i,j):
    """Function adds to currently created pdf file created tables, graphs and heatmaps. 

    """
    pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-bold', 'arialbd.ttf'))
    width, height = letter
    font_size = 12

    img_path = images.pop(0)
    img = utils.ImageReader(img_path)
    adjusted_width, adjusted_height = get_adjusted_dimensions(img, width/2, height/3-25)

    y_position = height - font_size - 8
    x_position = 5
    pdf_draw_number_line(c,x_position,y_position,i,j,1,font_size,text_list[0])

    y_position = y_position - adjusted_height - 3
    x_position = (width/2 - adjusted_width) / 2
    c.drawImage(img_path, x_position, y_position, width=adjusted_width, height=adjusted_height)

    img_path = images.pop(0)
    img = utils.ImageReader(img_path)

    x_position = width/2 + (width/2 - adjusted_width) / 2
    c.drawImage(img_path, x_position, y_position, width=adjusted_width, height=adjusted_height)

    y_position -= font_size
    x_position = 5
    pdf_draw_number_line(c,x_position,y_position,i,j,2,font_size,text_list[1])
    
    img_path = images.pop(0)
    img = utils.ImageReader(img_path)
    adjusted_width, adjusted_height = get_adjusted_dimensions(img, width, height/3-25)

    y_position = y_position - adjusted_height - 8
    x_position = (width - adjusted_width) / 2
    c.drawImage(img_path, x_position, y_position, width=adjusted_width, height=adjusted_height)

    img_path = images.pop(0)
    img = utils.ImageReader(img_path)

    y_position -=  font_size 
    x_position = 5
    pdf_draw_number_line(c,x_position,y_position,i,j,3,font_size,text_list[2])
    
    y_position = y_position - adjusted_height - 8
    x_position = (width - adjusted_width) / 2
    c.drawImage(img_path, x_position, y_position, width=adjusted_width, height=adjusted_height)


def generate_pdf():
    """Function generates final pdf with all of the expermiment's results 
    """
    path = 'src/Quality_tool/Measures'
    for test_name in os.listdir(path):
        if not os.path.isdir(os.path.join(path, test_name)):
            continue
        test_path = os.path.join(path, test_name)
        c = canvas.Canvas(f"{test_path}/final_raport.pdf", pagesize=letter)
        i = 1
        for test_parameter_name in os.listdir(test_path):
            if not os.path.isdir(os.path.join(test_path,test_parameter_name)):
                continue
            test_parameter_name_path = os.path.join(test_path,test_parameter_name)
            j = 1
            picture_list = [f'{test_parameter_name_path}/graph.png',
                            f'{test_parameter_name_path}/ROC.png',
                            f'{test_parameter_name_path}/Our_Algorithm_test_table.png',
                            f'{test_parameter_name_path}/Comparison_Algorithm_test_table.png']

            string_list = [f'Wykres precyzji oraz wykres ROC dla zmiany parametru: {test_parameter_name}',
                           f'Tablica naszego algorytmu dla zmiany parametru: {test_parameter_name}',
                           f'Tablica porównawczego algorytmu dla zmiany parametru: {test_parameter_name}']
                           
            create_pdf_with_images_and_text(c, picture_list, string_list,i,j)
            c.showPage()
            j+=1
            for test_parameter_value in os.listdir(test_parameter_name_path):
                if not os.path.isdir(os.path.join(test_parameter_name_path,test_parameter_value)):
                    continue
                test_parameter_value_path = os.path.join(test_parameter_name_path,test_parameter_value)
                picture_list = [f'{test_parameter_value_path}/Our_Algorithm/heatmap.png',
                                f'{test_parameter_value_path}/Comparison_Algorithm/heatmap.png',
                                f'{test_parameter_value_path}/Our_Algorithm/classes_measures_table.png',
                                f'{test_parameter_value_path}/Comparison_Algorithm/classes_measures_table.png']

                string_list = [f'Porównanie confusion matrix dla {test_parameter_name} = {test_parameter_value}',
                            f'Tablica naszego algorytmu dla {test_parameter_name} = {test_parameter_value}',
                            f'Tablica porównawczego algorytmu dla {test_parameter_name} = {test_parameter_value}']
                            
                create_pdf_with_images_and_text(c, picture_list, string_list,i,j)
                c.showPage()
                j+=1
            i+=1

        c.save()


def main():
    np.set_printoptions(suppress=True)
    0,1,2 
    pick = 2
    numberofsamples = 15
    param_dict_our_algorithm = {}
    param_dict_comparison_algorithm = {}
    
    if pick == 0:
        generate_tabels_and_heatmaps(0, numberofsamples, param_dict_our_algorithm)
    elif pick == 1:
        generate_tabels_and_heatmaps(1, numberofsamples, param_dict_comparison_algorithm)
    elif pick ==2:
        generate_tabels_and_heatmaps(0, numberofsamples, param_dict_our_algorithm)
        generate_tabels_and_heatmaps(1, numberofsamples, param_dict_comparison_algorithm)
        generate_graphs(param_dict_our_algorithm,param_dict_comparison_algorithm)
        generate_pdf()
    else:
        print("Error")


if __name__ == '__main__':
    main()