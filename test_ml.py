
import pytest

import math
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from ml.random_data import data_samples
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    performance_on_categorical_slice,
    train_model,
)


# Implement the first test.
def test_data_split():
    '''
    Test train_test_split to determine if train and test data are split approx 80/20.
    '''

    # Get Sample Data
    data = data_samples('random_data')

    # Split Sample Data
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    
    assert type(train) == pd.core.frame.DataFrame
    assert type(test) == pd.core.frame.DataFrame
    
    assert train.shape[0] == int(data.shape[0] * .80)
    assert test.shape[0] == math.ceil(data.shape[0] * .20)
    

# Implement the second test.
def test_process_data():
    
    '''
    Test process_data function to determine if correct data types are returned.
    '''

    # Get Data Samples
    train, cat_features = data_samples('process_training_data')

    # Process Sample Data
    X_train, y_train, encoder, lb = process_data(
    X=train, categorical_features=cat_features, label="target", training=True
    )
    
    assert type(X_train) == np.ndarray
    assert X_train.shape[0] == train.shape[0]

    assert type(y_train) == np.ndarray
    assert y_train.shape[0] == train.shape[0]

    assert type(encoder) == OneHotEncoder
    assert type(lb) == LabelBinarizer


# Implement the third test.
def test_train_model():
    '''
    Test train_model function to determine if a Random Forrest Classifier is returned.
    '''

    # Get Sample Data
    X_train, y_train = data_samples('model_data')

    # Create ML Model
    model = train_model(X_train, y_train)
    
    assert type(model) == RandomForestClassifier
    assert isinstance(model, BaseEstimator)
    assert isinstance(model, ClassifierMixin)


# Implement the fourth test.
def test_inference():
    '''
    Test inference function to determine if the number of predictions is equel to the
    size of the imput DataFrame.
    '''

    # Get Sample Data
    X_test, model = data_samples('inference_data')

    # Get Predictions
    preds = inference(model, X_test)
    
    assert X_test.shape[0] == 8191
    assert preds.shape[0] == 8191
    assert X_test.shape[0] == preds.shape[0]


# Implement the fifth test.
def test_compute_model_metrics():
    '''
    Test compute model metrics function to determine if reasonable metrics are returned for random data.
    '''
    # Get Sampale Data
    y_test, preds = data_samples('metrics_data')

    # Compute Metrics
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    
    assert round(precision, 1) == 0.7
    assert round(recall, 1) == 0.9
    assert round(fbeta, 1) == 0.8


# Implement the sixth test.
def test_performance_on_categorical_slice():
    '''
    Test performance on categorical slice function to determine if reasonable metrics are returned for random data.
    '''

    # Get Sampale Data
    test, cat_features, encoder, lb, model = data_samples('slice_data')

    # Compute Slice Metrics
    precision, recall, fbeta = performance_on_categorical_slice(
            data = test, 
            column_name = 'category_1', 
            slice_value = 'Alpha',
            categorical_features = cat_features, 
            label = 'target',
            encoder = encoder, 
            lb = lb, 
            model = model
        )
    
    assert round(precision, 1) == 0.7
    assert round(recall, 1) == 0.9
    assert round(fbeta, 1) == 0.8





# End of Page
