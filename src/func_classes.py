"""
This file does not contain any tests
Additional time would be needed to 
write a proper test file
"""
# Loading the libraries
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, fbeta_score
from sklearn.svm import SVC
from sklearn.utils import resample
from pathlib import Path
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import optuna
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, List
from lightgbm import LGBMClassifier

class Utils():
    def __init__(self):
        pass
    def load_scaler(self):
        """
        Loads the standard scaler from the models directory.
        Returns:
            scaler: The standard scaler object.
        """
        # Load the scaler
        cwd = Path.cwd()
        root = cwd.parent
        scaler_path = root / "models" / "Standardscaler.pkl"
        scaler = joblib.load(scaler_path)
        return scaler

    def create_pipeline(
        self,
        model: object,
        scaler: bool = False,
        feature_selector: Optional[object] = None
    ):
        """
        Creates a pipeline for the model.
        Args:
            model: The model object.
            scaler: Whether to use a scaler.
            feature_selector: The feature selector object.
        Returns:
            pipeline: The pipeline object.
        """
        # Initialize the steps for the pipeline
        steps = []
        # Add the scaler if needed
        # Load the scaler
        scaler = self.load_scaler()
        steps.append(('scaler', scaler))
        # Add the feature selector
        if feature_selector is not None:
            steps.append(('feature_selector', feature_selector))
        else:
            steps.append(('feature_selector', 'passthrough'))
        # Add the model to the pipeline
        steps.append(('model', model))
        pipeline = Pipeline(steps)
        return pipeline

    def train_model_and_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        pipeline: Pipeline,
        root_path: str = '',
        filename: str = '',
        save: bool = False,
        default_path: bool = True
    ):
        """
        Trains the model, stores it and makes predictions.
        Args:
            X_train: The training data.
            y_train: The training labels.
            x_test: The test data.
            pipeline: The pipeline object.
            root_path: Assignment-1 folder
            filename: Name of the output file or custom path after root if default_path=False
            save: Whether to save the model
            default_path: whether to save in models (True) or another folder (False)
        """
        # Fit the pipeline to the training data
        pipeline.fit(X_train, y_train)
        # Save the pipeline, default path is the models folder
        if save:
            if default_path:
                model_path = root_path / 'models' / filename
                joblib.dump(pipeline, model_path)
            # Save the pipeline to the specified path, this is for the winner model
            else:
                model_path = os.path.join(root_path, filename)
                joblib.dump(pipeline, model_path)
        # Make predictions
        y_pred = pipeline.predict(x_test)
        return y_pred, pipeline

class rnCV():
    def __init__(self):
        pass

    def kfold_loop(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        pipeline: Pipeline,
        n_splits: int = 3,
        n_repeats: int = 5,
        n_outer: int = 10,
        estimators: List[object] = [LogisticRegression(), RandomForestClassifier(), LGBMClassifier(), SVC(), LinearDiscriminantAnalysis()],
        hyperparameters: List[dict] = [{'C': [1, 10], 'class_weight': ['balanced']}, {'max_depth': [2, 4, 6, 8, 10], 'n_estimators': [100, 200, 300, 400, 500]}, {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [2, 4, 6, 8, 10]}, {'C': [1, 10], 'class_weight': ['balanced']}, {'C': [1, 10, 100, 1000], 'class_weight': ['balanced']}],
    ):
        pass

def main():
    pass

if __name__=="__main__":
    # Call the main function
    main()