from typing import Dict
from sklearn.linear_model import Perceptron


class Perceptron_():

    def __init__(self , args:Dict):
        self.model = Perceptron()
        self.trained = False
        self.num_epoch = args.get('num_epoch',5)

    def fit(X,y):
        


