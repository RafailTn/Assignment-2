"""
This file does not contain any tests
Additional time would be needed to 
write a proper test file
"""
# Loading the libraries
from atom import ATOMClassifier
from atom.feature_engineering import FeatureSelector
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, matthews_corrcoef, average_precision_score, f1_score, make_scorer
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, List, Dict
class Utils():
    def __init__(self):
        pass
    def load_scaler_and_imputer(self):
        """
        Loads the standard scaler from the models directory.
        Returns:
            scaler: The standard scaler object.
        """
        # Load the scaler
        cwd = Path.cwd()
        root = cwd.parent
        scaler_path = root / "models" / "RobustScaler.pkl"
        imputer_path = root / "models" / "KNNImputer.joblib"
        scaler = joblib.load(scaler_path)
        imputer = joblib.load(imputer_path)
        return scaler, imputer

    def create_pipeline(
        self,
        model: object,
        scaler: bool = False,
        feature_selector: Optional[object] = None,
        imputer: Optional[object] = None,
    ):
        """
        Creates a pipeline for the model.
        Args:
            model: The model object.
            scaler: Whether to use a scaler.
            feature_selector: The feature selector object,
            imputer: The imputer object.
        Returns:
            pipeline: The pipeline object.
        """
        # Initialize the steps for the pipeline
        steps = []
        # Add the scaler and imputer if needed
        if scaler:
            scaler, imputer = self.load_scaler_and_imputer()
            steps.append(('imputer', imputer))
            steps.append(('scaler', scaler))
        else:
            steps.append(('imputer', 'passthrough'))
            steps.append(('scaler', 'passthrough'))
        # Add the feature selector
        if feature_selector is not None:
            steps.append(('feature_selector', feature_selector))
        else:
            steps.append(('feature_selector', 'passthrough'))
        # Add the model to the pipeline
        steps.append(('model', model))
        pipeline = Pipeline(steps)
        return pipeline
    
    def calculate_statistics(self, scores: List[float]):
        """
        Calculates the mean, standard deviation, and median of a list of scores.
        Args:
            scores (List[float]): List of score values.
        Returns:
            Tuple of (mean, std, median)
        """
        mean = np.mean(scores)
        std = np.std(scores)
        median = np.median(scores)
        return mean, std, median

    def create_boxplot(
        self,
        df: pd.DataFrame,
        model_name: str,
        metrics: List[str]
    ):
        """
        Creates boxplots for selected metrics of a specific model.
        Args:
            df (pd.DataFrame): DataFrame with results (must contain 'model' and metric columns).
            model_name (str): Name of the model to filter and plot.
            metrics (List[str]): List of metric names to visualize.
        """
        model_df = df[df["model"] == model_name]
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
        if n == 1:
            axes = [axes]
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = model_df[metric]
            mean, std, median = self.calculate_statistics(values)
            sns.boxplot(y=values, color="skyblue", ax=ax)
            ax.axhline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.3f}')
            ax.axhline(median, color='blue', linestyle='--', linewidth=2, label=f'Median: {median:.3f}')
            ax.axhline(mean - std, color='purple', linestyle='-.', linewidth=2, label=f'Mean - 1 Std: {mean - std:.3f}')
            ax.axhline(mean + std, color='purple', linestyle='-.', linewidth=2, label=f'Mean + 1 Std: {mean + std:.3f}')
            ax.set_title(f"{metric.upper()} - {model_name}")
            ax.set_ylabel(metric.upper())
            ax.set_xlabel("")
            ax.legend()
        plt.tight_layout()
        plt.show()

class RNcvAtom:
    '''
    Class to run repeated n-fold cross-validation for a list of models.
    Args:
        X (pd.DataFrame): DataFrame with features.
        y (pd.Series): Series with target variable.
        models (List[object]): List of models to evaluate.
        param_spaces (Dict[object, List[object]]): Dictionary with model names as keys and lists of parameter values as values.
        n_repeats (int): Number of times to repeat the cross-validation.
        n_splits (int): Number of splits for the cross-validation.
        n_trials (int): Number of trials for the optimization.
        inner_cv (int): Number of folds for the inner cross-validation.
        seed (int): Random seed for reproducibility.
    '''
    def __init__(
            self, 
            X: pd.DataFrame,
            y: pd.Series,
            models: List[str],
            param_spaces: Dict[str, List[object]] = None,
            n_repeats: int = 10, 
            n_splits: int = 5, 
            n_trials: int = 50, 
            inner_cv: object|int = StratifiedKFold(n_splits=3),
            fs: str = 'pca',
            seed=42
            ):
        self.X = X
        self.y = y
        self.models = models
        self.param_spaces = param_spaces
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.inner_cv = inner_cv
        self.fs = fs
        self.seed = seed
        self.results = []

    def run(self):
        """
        Runs the repeated n-fold cross-validation.
        """
        f2_weighted = make_scorer(fbeta_score, beta=2, average="weighted", zero_division=0)

        rkf = RepeatedStratifiedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.seed
        )
        
        for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(self.X, self.y)):     
            print(f"Processing fold: {fold_idx}")
            
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            # Perform feature selection
            pca = PCA(n_components=10)
            X_train_transformed = pca.fit_transform(X_train)
            X_test_transformed = pca.transform(X_test)
                        
            for model in self.models:
                # Create a new ATOM instance for the model training
                model_atom = ATOMClassifier(X_train_transformed, y_train, random_state=self.seed + fold_idx, verbose=2)
                # Train the model on the transformed training data
                model_atom.run(
                    models=[model],
                    metric=f2_weighted,
                    n_trials=self.n_trials,
                    ht_params={'cv': self.inner_cv}
                )
                best_model = model_atom.winner
                
                # Make predictions on the transformed test data
                y_pred = best_model.predict(X_test_transformed)
                
                self.results.append({
                    "model": model,
                    "fold": fold_idx,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f2": fbeta_score(y_test, y_pred, beta=2, average="weighted", zero_division=0),
                    "mcc": matthews_corrcoef(y_test, y_pred),
                })

    def get_results(self):
        return pd.DataFrame(self.results)
        
def main():
    pass

if __name__=="__main__":
    # Call the main function
    main()