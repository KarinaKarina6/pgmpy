import os
from itertools import combinations
from pgmpy.base import DAG
import networkx as nx
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch
from pgmpy.estimators import K2Score, BicScore, CompositeScore
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from sklearn.preprocessing import LabelEncoder
from LE import CustomLabelEncoder
from write_txt import Write
from comparison import Comparison

# Чтобы это работало, нужно отключить Step 1, Step 2, Step 3 в файле HillClimbingSearch.py 


def get_init_edges(file, number):
    path = 'C:/Users/KShakhyan/OneDrive - ITMO UNIVERSITY/Projects/GB/GOLEM/examples/results/CHC/results'
    # path = 'results/classical'    
    file_name = 'classical_HC_' + file + '_run_' + str(number)
    textfile = open(path + '/' + file_name + '.txt', "r")
    rows = textfile.read().split("\n")[:-1]
    structure = eval(rows[0].split(' = ')[1])
    return structure


uci_datasets = ['Balance', 'Block', 'Breast_cancer', 'Breast_tissue',
 'CPU', 'Ecoli', 'Glass', 'Iris', 'Liver', 'Parkinsons',
 'QSAR Aquatic', 'QSAR fish toxicity', 'Sonar', 'Vehicle', 
 'Vowel', 'Wdbc', 'Wine', 'Wpbc', 'Yeast']
bnlearn_datasets = ['asia', 'cancer', 'earthquake', 'sachs', 'survey', 'healthcare', 'sangiovese', 
                    'alarm', 'barley', 'child', 'insurance', 'mildew', 'water', 'ecoli70', 'magic-niab', 'mehra-complete', 'hailfinder']
datasets = bnlearn_datasets + uci_datasets # uci_datasets + bnlearn_datasets # 'asia', 'cancer', 'earthquake', 'healthcare', 'sachs', 'sangiovese', 'Sonar' 1 
random_seed = [87, 60, 37, 99, 42, 92, 48, 91, 86, 33]
len_train = 0
for file in datasets:
    if file == 'Block':
        continue

    for number in range(0, 10):

        vars_of_interest_comp = {}
        vars_of_interest = {}

        if file in bnlearn_datasets:
            df = pd.read_csv('data/csv/' + file + ".csv") 
            df.drop(['Unnamed: 0'], axis=1, inplace=True) 
            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)
        else:
            path = 'data/' + file + '/' + file 
            df_train = pd.read_csv(path + '_train_fold_' + str(number) + ".csv") 
            df_test = pd.read_csv(path + '_test_fold_' + str(number) + ".csv") 

            df = pd.concat([df_train, df_test])
            if 'class' in df.columns:
                df.drop(['class'], axis=1, inplace=True)
            df.drop(['Unnamed: 0'], axis=1, inplace=True) 
            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True) 
            len_train = len(df_train)

        str_columns = [col for col in df.columns.tolist() if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])]

        init_edges = get_init_edges(file, number)

        if str_columns:
            custom_encoder = CustomLabelEncoder()
            df_encoded = custom_encoder.fit_transform(df[str_columns])
            df[str_columns] = df_encoded

        df.attrs = {'len_train': len_train,
                    'str_columns': str_columns,
                    'random_seed': random_seed[number]}
        # Learn the model structure using Hill-Climb Search
        # scoring_method = BicScore(data=df)
        scoring_method = CompositeScore(data=df)

        est = HillClimbSearch(data=df)
        estimated_model = est.estimate(
            scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4),
            fixed_edges=init_edges
        )
        # get_f1_score(estimated_model, model)
        structure_comp = list(estimated_model.edges())
        print(structure_comp) # структура
        for node in estimated_model.nodes:
            model = estimated_model.nodes[node]['ML_model']
            print(f'{node} -> {model}')
        models_comp = {node:estimated_model.nodes[node]['ML_model'] for node in estimated_model.nodes}
        likelihood_comp = scoring_method.score(estimated_model)


        vars_of_interest_comp['Structure'] = structure_comp
        vars_of_interest_comp['Likelihood'] = likelihood_comp
        vars_of_interest_comp['Models'] = models_comp

        if file in bnlearn_datasets:

            with open('data/txt/'+(file)+'.txt') as f:
                lines = f.readlines()
            true_net = []
            for l in lines:
                e0 = l.split()[0]
                e1 = l.split()[1].split('\n')[0]
                true_net.append((e0, e1))    
            
            comparison = Comparison()
            f1 = comparison.F1(structure_comp, true_net)
            SHD = comparison.precision_recall(structure_comp, true_net)['SHD']
            vars_of_interest_comp['f1'] = f1
            vars_of_interest_comp['SHD'] = SHD
        
        write = Write()
        write.write_txt(vars_of_interest_comp, path = os.path.join('results', 'bnlearn'), file_name = 'catboost_model_HC_' + file + '_run_' + str(number) + '.txt')




