
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDRegressor
)
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
import json
from random import choice
import numpy as np

class ML_models():
    def __init__(self, data):
        self.operations_by_types = {
        'xgbreg': 'XGBRegressor',
        'adareg': 'AdaBoostRegressor',
        'gbr': 'GradientBoostingRegressor',
        'dtreg': 'DecisionTreeRegressor',
        'treg': 'ExtraTreesRegressor',
        'rfr': 'RandomForestRegressor',
        'linear': 'LinearRegression',
        'ridge': 'Ridge',
        'lasso': 'Lasso',
        'svr': 'LinearSVR',
        'sgdr': 'SGDRegressor',
        'lgbmreg': 'LGBMRegressor',
        'catboostreg': 'CatBoostRegressor',

        # 'xgboost': 'XGBClassifier',
        'logit': 'LogisticRegression',
        'bernb': 'BernoulliNB',
        'multinb': 'MultinomialNB',
        'dt': 'DecisionTreeClassifier',
        'rf': 'RandomForestClassifier',
        'mlp': 'MLPClassifier',
        'lgbm': 'LGBMClassifier',
        'catboost': 'CatBoostClassifier',

        'kmeans': 'KMeans',
    }
        
        self.dict_models = {
        'XGBRegressor': XGBRegressor,
        'AdaBoostRegressor': AdaBoostRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'ExtraTreesRegressor': ExtraTreesRegressor,
        'RandomForestRegressor': RandomForestRegressor,
        'LinearRegression': LinearRegression,
        'Ridge': Ridge,
        'Lasso': Lasso,
        'LinearSVR': LinearSVR,
        'SGDRegressor': SGDRegressor,
        'LGBMRegressor': LGBMRegressor,
        'CatBoostRegressor': CatBoostRegressor,

        # 'XGBClassifier': XGBClassifier,
        'LogisticRegression': LogisticRegression,
        'BernoulliNB': BernoulliNB,
        'MultinomialNB': MultinomialNB,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'MLPClassifier': MLPClassifier,
        'LGBMClassifier': LGBMClassifier,
        'CatBoostClassifier': CatBoostClassifier,

        'KMeans': KMeans,
    }
        self.data = data

    def get_model_by_children_type(self, node):
        candidates = []
        if node.content['type'] == 'cont':
            type_model = 'regr'
        else:
            type_model = 'class'
        forbidden_tags = ['non-default', 'expensive']
        with open('examples/composite_bn/model_repository.json', 'r') as f:
            models_json = json.load(f)
            models = models_json['operations']
            for model, value in models.items():
                if model not in ['knnreg', 'knn', 'qda', 'svr', 'xgboost'] and list(set(value['tags']).intersection(forbidden_tags)) == [] and type_model in value['meta']:
                    candidates.append(model)   
        return self.operations_by_types[choice(candidates)]
    

    def get_all_models_by_children_type(self, node):
        candidates = []
        # if node.content['type'] == 'cont':
        # if(self.data[node].dtype == np.float64 or self.data[node].dtype == np.int64):
        if node not in self.data.attrs['str_columns']:
            type_model = 'regr'
        else:
            type_model = 'class'
        forbidden_tags = ['non-default', 'expensive']
        with open('model_repository.json', 'r') as f:
            models_json = json.load(f)
            models = models_json['operations']
            for model, value in models.items():
                if model not in ['knnreg', 'knn', 'qda', 'svr', 'xgboost'] and list(set(value['tags']).intersection(forbidden_tags)) == [] and type_model in value['meta']:
                    candidates.append(model)   

        # Для экспериментов с catboost
        # if node not in self.data.attrs['str_columns']:
        #     candidates = ['catboostreg']
        # else:
        #     candidates = ['catboost']      

        models = list(map(self.operations_by_types.get, candidates))
        return models