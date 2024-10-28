import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier

def define_training_validation(train: pd.DataFrame, 
                                valid: pd.DataFrame, 
                                predictors: list, 
                                target: str) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Prepares the training and validation data.

    Parameters:
    - train: The training DataFrame.
    - valid: The validation DataFrame.
    - predictors: A list of column names to be used as features.
    - target: The target column name.

    Returns:
    - A tuple containing:
        - train_X: Training feature data (as a DataFrame).
        - train_Y: Training labels (as a NumPy array).
        - valid_X: Validation feature data (as a DataFrame).
        - valid_Y: Validation labels (as a NumPy array).
    """
    
    train_X = train[predictors]
    train_Y = train[target].values
    valid_X = valid[predictors]
    valid_Y = valid[target].values
    
    return train_X, train_Y, valid_X, valid_Y

def train_and_predict_rf(train_X: pd.DataFrame, 
                         train_Y: np.ndarray, 
                         valid_X: pd.DataFrame, 
                         random_state: int = 42, 
                         n_estimators: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trains a RandomForestClassifier on the training data and returns predictions
    for both training and validation data.

    Parameters:
    - train_X: Training features (pandas DataFrame)
    - train_Y: Training labels (NumPy array or pandas Series)
    - valid_X: Validation features (pandas DataFrame)
    - random_state: Random state for reproducibility (default: 42)
    - n_estimators: Number of trees in the forest (default: 100)

    Returns:
    - preds_tr: Predictions for the training data (NumPy array)
    - preds: Predictions for the validation data (NumPy array)
    """
    
    # Initialize the RandomForestClassifier
    clf = RandomForestClassifier(n_jobs=-1, 
                                 random_state=random_state,
                                 criterion="gini",
                                 n_estimators=n_estimators,
                                 verbose=False)
    
    # Fit the classifier with the training data
    clf.fit(train_X, train_Y)
    
    # Predict the training data (for checking training error)
    preds_tr = clf.predict(train_X)
    
    # Predict the validation data
    preds = clf.predict(valid_X)
    
    return preds_tr, preds