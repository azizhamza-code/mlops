import string
from mzops.data import etl
from mzops.model import encoder_, extract
from mzops.utils import get_unique_classes
import numpy as np

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

import joblib

from mzops.config import d

def print_evaluation_scores(y_val, predicted,title:str):
    print(title)
    print(accuracy_score(y_val, predicted))
    print(f1_score(y_val, predicted, average='weighted'))
    print(average_precision_score(y_val, predicted))


# TODO: test load and saving encoder-extractor



def train():

    save_it = True

    x_train, y_train, x_val, y_val, x_test = etl(dev_mode = d['sub_data'])

    encoder = encoder_(classes=get_unique_classes(y_train))
    encoder.fit(y_train)
    y_train, y_val = encoder.encode(y_train), encoder.encode(y_val)
    if save_it:
        encoder.save(target_dir=d['artifactdir'])

    vectorizer = extract()
    vectorizer.fit(x_train)
    x_train, x_val , x_test = vectorizer.encode(x_train), vectorizer.encode(x_val) , vectorizer.encode(x_test)
    if save_it:
        vectorizer.save(target_dir=d['artifactdir'])


    model = OneVsRestClassifier(LogisticRegression())
    model.fit(x_train , y_train)

    y_train_pred = model.predict(x_train)
    y_val_predict = model.predict(x_val)

    if save_it:
        joblib.dump(model, d['model_artifact'])

    print_evaluation_scores(y_train , y_train_pred , "train evalution")
    print_evaluation_scores(y_val , y_val_predict , "val evalution")




