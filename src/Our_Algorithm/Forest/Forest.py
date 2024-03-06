__author__  = "Jan Walczak, Patryk Jankowicz"
import sys
sys.path.append('src')
from Our_Algorithm.Tree.Tree import Tree
from collections import defaultdict
from multiprocessing import Pool
import time
import random


class Forest:
    def __init__(self,T,N,K,attribute_types,max_depth,min_number_of_values,threshold,attributes,mode,processes):
        self.training_data = T
        self.number_of_trees = N
        self.data_to_classify = K
        self.attribute_types = attribute_types
        self.max_depth = max_depth
        self.min_number_of_values = min_number_of_values
        self.threshold = threshold
        self.attributes = attributes
        self.mode = mode
        self.processes = processes # Number of proccesses (used to optimize runtime by multithreading)
        
        
    def generate_tree(self,training_data, attribute_types, min_number_of_values, max_depth, threshold, mode):
        """ Function creates a Tree object 

        Args:
            training_data (dataframe): dataframe with passed training data set
            attribute_types (array): array with attribute types (district '0' or continous '1')

        Returns:
            Tree: created tree object 
        """
        n = len(self.training_data)
        num_column = len(self.training_data.columns) - 1
        attribute_list = list(range(0, num_column))
        number_of_attributes = int(self.attributes*num_column)
        if number_of_attributes == 0:
            number_of_attributes = 1
        chosen_attributes = random.sample(attribute_list, number_of_attributes)
        selected_rows = training_data.sample(n = n, replace = True)
        tree = Tree(selected_rows,chosen_attributes,attribute_types,min_number_of_values,max_depth,threshold,mode) 
        tree.create_decision_tree()
        return tree

        
    def create_random_forest(self):
        """Function generates a forest with applied multithreading. It also counts taken time.

        Returns:
            dictionary: confusion matrix for generated forest and calculated time
        """
        start_time = time.time()
        list_of_models = []

        with Pool(processes=self.processes) as pool:
            result = pool.starmap(self.generate_tree, [(self.training_data, self.attribute_types,
                                                         self.min_number_of_values, self.max_depth, self.threshold, self.mode)
                                                        for _ in range(self.number_of_trees)])
        
        list_of_models = result
        
        classes_list = self.training_data.iloc[:, -1].unique()
        confusion_matrix = {}
        for i in classes_list:
            confusion_matrix[i] = {}
            for j in classes_list:
                confusion_matrix[i][j] = 0
                
        for k in self.data_to_classify.values: # Classyfing passed data (K subset)
            classes = defaultdict(int)
            data_to_classify = k[:-1]
            for t in list_of_models:
                classes[t.classify_data(data_to_classify)] += 1
            best_class = max(classes, key=lambda k: classes[k]) #Choosing final, best class (answer) by majority vote
            confusion_matrix[k[-1]][best_class] += 1
        return confusion_matrix, time.time()-start_time