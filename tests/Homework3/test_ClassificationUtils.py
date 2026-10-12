import pytest
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from Homework3.ClassificationUtils import ClassificationUtils

def test_prepare_target():
    df_train = pd.DataFrame({'col1': [1, 2], 'target': [0, 1]})
    df_val = pd.DataFrame({'col1': [3, 4], 'target': [1, 0]})
    df_test = pd.DataFrame({'col1': [5, 6], 'target': [0, 1]})
    
    y_train, y_val, y_test = ClassificationUtils.prepare_target(df_train, df_val, df_test, 'target')
    
    assert np.array_equal(y_train, np.array([0, 1]))
    assert np.array_equal(y_val, np.array([1, 0]))
    assert np.array_equal(y_test, np.array([0, 1]))

def test_prepare_data():
    df_train = pd.DataFrame({'col1': [1, 2], 'target': [0, 1]})
    df_val = pd.DataFrame({'col1': [3, 4], 'target': [1, 0]})
    df_test = pd.DataFrame({'col1': [5, 6], 'target': [0, 1]})
    
    df_train_new, df_val_new, df_test_new = ClassificationUtils.prepare_data(df_train, df_val, df_test, 'target')
    
    assert 'target' not in df_train_new.columns
    assert 'target' not in df_val_new.columns
    assert 'target' not in df_test_new.columns
    assert len(df_train_new) == 2
    assert len(df_val_new) == 2
    assert len(df_test_new) == 2
    # Verify original dataframes are also modified (since they are passed by reference)
    assert 'target' not in df_train.columns
    assert 'target' not in df_val.columns
    assert 'target' not in df_test.columns

def test_calculate_mi():
    df = pd.DataFrame({
        'cat': [1, 1, 0, 0],
    })
    target = np.array([1, 1, 0, 0])
    # Mutual information should be high
    mi = ClassificationUtils.calculate_mi(df, 'cat', target)
    assert isinstance(mi, float)
    assert mi >= 0

def test_ohe_data():
    df = pd.DataFrame({'feature': ['a', 'b', 'a']})
    dv = DictVectorizer(sparse=False)
    
    # Test fit_transform
    X_train = ClassificationUtils.ohe_data(df, dv, True)
    assert X_train.shape == (3, 2)
    
    # Test transform
    df_test = pd.DataFrame({'feature': ['b', 'a']})
    X_test = ClassificationUtils.ohe_data(df_test, dv, False)
    assert X_test.shape == (2, 2)
    # Check that it's one-hot encoded correctly
    # For 'a' and 'b', it should have 1 in one column and 0 in the other.
    assert np.all(X_test.sum(axis=1) == 1)

def test_train_model():
    X_train = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
    y_train = np.array([1, 1, 0, 0])
    X_val = np.array([[1, 0], [0, 1]])
    y_val = np.array([1, 0])
    c = 1.0
    
    accuracy = ClassificationUtils.train_model(X_train, y_train, X_val, y_val, c)
    
    assert 0 <= accuracy <= 1
    # In this case, X_val[0] is [1, 0] which is same as X_train[0], so it should predict 1.
    # X_val[1] is [0, 1] which is same as X_train[3], so it should predict 0.
    # So accuracy should be 1.0
    assert accuracy == 1.0

def test_calculate_accuracy():
    df_train = pd.DataFrame({'feature': ['a', 'b', 'a', 'b'], 'target': [1, 0, 1, 0]})
    df_val = pd.DataFrame({'feature': ['a', 'b'], 'target': [1, 0]})
    y_train = df_train['target'].values
    y_val = df_val['target'].values
    c = 1.0
    
    accuracy = ClassificationUtils.calculate_accuracy(df_train, df_val, y_train, y_val, c)
    
    assert 0 <= accuracy <= 1
    # With these data, it should be trivially 1.0
    assert accuracy == 1.0
