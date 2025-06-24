import os
import math

import pytest
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from ml.random_data import random_data
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)


# TODO: implement the first test. Change the function name and input as needed
def test_data_split():
    '''
    Test train_test_split to determine if train and test data are split approx 80/20.
    '''

    data = random_data()
    train, test = train_test_split(data, test_size=0.20)

    assert type(train) == pd.core.frame.DataFrame
    assert type(test) == pd.core.frame.DataFrame
    
    assert train.shape[0] == int(data.shape[0] * .80)
    assert test.shape[0] == math.ceil(data.shape[0] * .20)
    

# TODO: implement the second test. Change the function name and input as needed
def test_process_data():
    '''
    Test process_data function to determine if correct data types are returned.
    '''

    data = random_data()
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    
    cat_features = [
        'category_1',
        'category_2',
        'category_3',
        'category_4',
        'category_5',
        'category_6'
    ]
    X_train, y_train, encoder, lb = process_data(
    X=train, categorical_features=cat_features, label="target", training=True
    )
    
    assert type(X_train) == np.ndarray
    assert X_train.shape[0] == train.shape[0]

    assert type(y_train) == np.ndarray
    assert y_train.shape[0] == train.shape[0]

    assert type(encoder) == OneHotEncoder
    assert type(lb) == LabelBinarizer


# TODO: implement the second test. Change the function name and input as needed
def test_train_model():
    '''
    Test train_model function to determine if a Random Forrest Classifier is returned.
    '''

    data = random_data()
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    
    cat_features = [
        'category_1',
        'category_2',
        'category_3',
        'category_4',
        'category_5',
        'category_6'
    ]
    
    X_train, y_train, encoder, lb = process_data(
    X=train, categorical_features=cat_features, label="target", training=True
    )

    model = train_model(X_train, y_train)
    
    assert type(model) == RandomForestClassifier
    assert isinstance(model, BaseEstimator)
    assert isinstance(model, ClassifierMixin)


# TODO: implement the fourth test. Change the function name and input as needed
def test_inference():
    '''
    Test inference function to determine if the number of predictions is equel to the
    size of the imput DataFrame.
    '''

    data = random_data()
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    
    cat_features = [
        'category_1',
        'category_2',
        'category_3',
        'category_4',
        'category_5',
        'category_6'
    ]
    
    X_train, y_train, encoder, lb = process_data(
    X=train, categorical_features=cat_features, label="target", 
    training=True)

    X_test, y_test, _, _ = process_data(
    X=test, categorical_features=cat_features, label="target",
    training=False, encoder=encoder, lb=lb)

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    
    assert X_test.shape[0] == 8191
    assert preds.shape[0] == 8191
    assert X_test.shape[0] == preds.shape[0]





# End of Page
