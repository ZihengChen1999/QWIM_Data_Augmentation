__author__ = "Ziheng Chen"
__version__ = "1"
__status__ = "Prototype"  # Status should typically be one of "Prototype", "Development", or "Production".

import random
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

class Evaluation_model:

    def __init__(self, train_true_data, wrong_data_number=10):
        self.train_true_data=train_true_data
        self.data_length=self.train_true_data.shape[0]
        self.data_number=self.train_true_data.shape[1]
        self.train_wrong_data = self.generate_wrong_data(wrong_data_number)
        self.clf = KNeighborsClassifier()

        self.train_true_data=self.train_true_data.pct_change().fillna(0)

    def get_features(self, data):
        features_dic={}
        features_dic["mean"]=data.mean().values
        features_dic["std"]=data.std().values
        features_dic["skw"]=data.skew().values
        features_dic["kur"]=data.kurtosis().values
        for i in range(0,100,5):
            features_dic[str(i)+" percentile"]=data.quantile(i/100).values
        for j in [1,2,5,10,20]:
            features_dic["acf"+str(j)]=[data[i].autocorr(lag=j) for i in data.columns]

        features=np.array([list(i) for i in features_dic.values()])
        return features.T


    def generate_wrong_data(self,number):
        wrong_data=np.array([[random.uniform(-0.5, 0.5) for i in range(self.data_length)] for j in range(number)])
        return pd.DataFrame(wrong_data.T)

    def train(self):
        wrong_data_features=self.get_features(self.train_wrong_data)
        true_data_features=self.get_features(self.train_true_data)

        X=np.append(true_data_features, wrong_data_features,axis=0)
        y=np.array([1]*len(true_data_features)+[0]*len(wrong_data_features))
        self.clf.fit(X,y)

    def predict(self,test_data):
        test_features=self.get_features(test_data.pct_change().fillna(0))
        return self.clf.predict_proba(test_features)



