__author__  = "Jan Walczak, Patryk Jankowicz"
import sys
sys.path.append('src')
from Our_Algorithm.Tree.Tree_node import Tree_node
import pandas as pd
import random


class Tree:
    def  __init__(self,T,X,parameter_types,min_number_of_values,max_depth,threshold,mode):
        self.dataframe = T # Training data 
        self.attributes = X  # Atribiutes (columns' numbers)
        self.param_types = parameter_types # Type of each column (district '0' or continous '1')
        self.min_number_of_values = min_number_of_values
        self.max_depth = max_depth # Maximum value of tree's levels number
        self.threshold = threshold # Number of conditions to be removed before roulette
        self.mode = mode # Way of treating cotinous values: 0 - average of 2 adjacent values, 1 - average of all values, 2 - median of all values, 3 - spliting data to bins (equal in terms of the length) 


    def create_decision_tree(self):
        """Function responsible for initiating a process of tree creation
        """
        self.root = self.create_decision_tree_recursive(None,self.dataframe,0)

    def create_decision_tree_recursive(self, parent_node,T,depth):
        """Function that recursively generates a tree

        Args:
            parent_node (Tree_node): precedent node in the tree
            T (dataframe): split subset of data
            depth (int): current depth of the tree

        Returns:
            _type_: Tree_node
        """
        node = Tree_node()
        node.parent = parent_node
        node.value = self.choose_best_class(T,parent_node) # Setting value for current node 
        

        # Checking first conditions - if subset of data is to small or current depth of tree exceed the limit 
        if len(T) < self.min_number_of_values or depth > self.max_depth:
            return node
        elif T.iloc[:,-1].nunique() == 1:
            return node


        # Calculating value of gini index and removing worst results 
        calculated_gini = self.calculate_gini_index(T)
        if not calculated_gini:
            return node
        calculated_gini = self.clear_worst_results(calculated_gini,self.threshold)
        

        # Roulette - choosing randomly final condition from the rest (with calculated weights)
        condition = random.choices(list(calculated_gini.keys()), weights=list(calculated_gini.values()), k=1)
        node.condition = condition
        
        # Splitting data into the subsets, accordingly to condition in the node
        if self.param_types[condition[0][0]]:
            df1 = T[T.iloc[:,condition[0][0]]<condition[0][1]]
            df2 = T[T.iloc[:,condition[0][0]]>=condition[0][1]]
        else:
            df1 = T[T.iloc[:,condition[0][0]]==condition[0][1]]
            df2 = T[T.iloc[:,condition[0][0]]!=condition[0][1]]      
        

        # If any of the subsets is empty, end the creation process
        if (len(df1)==0 or len(df2)==0):
            node.condition = None
            return node
        
        # Setting left and right nodes with split data; calling function recursively
        node.left = self.create_decision_tree_recursive(node,df1,depth+1)
        node.right = self.create_decision_tree_recursive(node,df2,depth+1)
        return node
    
    
    def classify_data(self,data_to_classify):
        """funtion initiates process of classifying data 

        Args:
            data_to_classify (array): current row to be classified

        Returns:
            calling recursive function
        """
        return self.classify_data_recursive(self.root,data_to_classify)
    
    
    def classify_data_recursive(self,node,data_to_classify):
        """_summary_

        Args:
            node (Tree_node): currently check node
            data_to_classify (array): current row to be classified
        """
        if node.condition == None: #If node is a leaf, return current node's value
            return node.value
        if self.compare_condition_and_data(node.condition,data_to_classify): #If condition is fullfilled "go" left otherwise "go" right in tree
            return self.classify_data_recursive(node.left,data_to_classify)
        else:
            return self.classify_data_recursive(node.right,data_to_classify)   
    
    
    def choose_best_class(self,dataframe,parent_node):
        """Function sets the parent node's value (if possible) for current node, otherwise choosing it at random; used to choose final class in case size of leafs' subsets are equal

        Args:
            dataframe (dataframe): data
            parent_node (Tree_node): current node's parent node

        Returns:
            _type_: best class for the node
        """
        class_counts = dataframe.iloc[:, -1].value_counts()
        max_count = class_counts.max()
        if (class_counts == max_count).sum() > 1:
            if parent_node:
                best_class = parent_node.value
            else:
                best_class = random.choice(class_counts.index.tolist())
        else:
            best_class = class_counts.idxmax()
        return best_class
    
    
    def mode_of_data_preparation(self,T,param):
        """Function prepares continous values for calculating gini index, depending on the choosen mode (way of treating the continuous values)
        0 - average of 2 adjacent values, 
        1 - average of all values,
        2 - median of all values, 
        3 - spliting data to bins (equal in terms of the length) 
        Args:
            T (dataframe): Data
            param (string): passed attribute

        Returns:
            _type_: calculated values based on choosen method
        """
        values = []
        if self.mode == 0:
            sorted_list = sorted(T.iloc[:,param].unique())
            values = [(x + y) / 2 for x, y in zip(sorted_list, sorted_list[1:])]
        elif self.mode == 1:
            values.append(T.iloc[:,param].mean())
        elif self.mode == 2:
            values.append(T.iloc[:,param].median())
        elif self.mode == 3:
            _, values = pd.qcut(T.iloc[:,param],q=int(len(T)**0.5), retbins=True, duplicates='drop')
        return values
    
    
    def gini_index(self,dataframe):
        """Function calculates Gini index for entire subset of data 

        Args:
            dataframe (dataframe): subset of data in current node

        Returns:
            value of calculated gini 
        """
        l = len(dataframe)
        gini = 1.0
        class_counts = dataframe.iloc[:, -1].value_counts()
        for value in class_counts:
            proportion = value / l
            gini -= proportion * proportion
        return gini


    def total_gini_index(self,dataframe,parameter_type,parameter_value):
        """ Function calculates change of Gini index for specified attribute

        Args:
            dataframe (_type_): subset of data
            parameter_type (string): current attribute
            parameter_value : current attribute's gini index value  

        Returns:
            Gini index value
        """
        param_values = dataframe.iloc[:, parameter_type]
        condition = param_values < parameter_value if self.param_types[parameter_type] else param_values == parameter_value

        df1 = dataframe[condition]
        df2 = dataframe[~condition]

        len_df1, len_df2 = len(df1), len(df2)
        total_len = len_df1 + len_df2

        gini_1 = self.gini_index(df1)
        gini_2 = self.gini_index(df2)

        total_gini = (gini_1 * len_df1 + gini_2 * len_df2) / total_len
        return total_gini
        
    # choosing conditions
    def calculate_gini_index(self,T):
        """ Function calculates final value of Gini index for each attribute

        Args:
            T (dataframe): Passed subset of data

        Returns:
            dictionary: dictionary where key is a tuple with attribute and value that is unique attribute values or calulcated ones (depending on its type),
                        dicitionary value is equal to calculated probability for a condition
        """
        base_index = self.gini_index(T)
        calculated_gini = {}
        for param in self.attributes: 
            if self.param_types[param]:   
                values = self.mode_of_data_preparation(T,param)
            else:
                values = T.iloc[:,param].unique()
                
            for value in values:
                gini = base_index-self.total_gini_index(T,param,value)
                if gini > 0:
                    calculated_gini[(param, value)]=gini
                    
        return calculated_gini
                    

    def clear_worst_results(self,data,percent):
        """Function removes conditions that don't match the set threshold

        Args:
            data (dictionary): dictionary with calculated probabilities
            percent (int): set threshold

        Returns:
            dictionary: passed dictionary with removed worst conditions
        """
        n = int(len(data) * percent)
        if n == len(data):
            n = n - 1
        keys_to_delete = sorted(data, key=data.get)[:n]
        for key in keys_to_delete:
            del data[key]

        return data


    def compare_condition_and_data(self,condition,data):
        """Function checks (depending on attribute type) if it fullfills condition 

        Args:
            condition: current node's condition
            data_to_classify (array): current row to be classified

        Returns:
            boolean: whether the condition is fullfilled 
        """
        if self.param_types[condition[0][0]]:
            if data[condition[0][0]] < condition[0][1]:
                return True
            else:
                return False 
        else:
            if data[condition[0][0]] == condition[0][1]:
                return True
            else:
                return False 