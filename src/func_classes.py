"""
This file does not contain any tests
Additional time would be needed to 
write a proper test file
"""
# Loading the libraries
from atom import ATOMClassifier
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, List
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
    
    # Function to create a boxplot of the scores
    def create_boxplot(
        self,
        scores_list: List[List[float]],
        model_name: str,
        metrics: List[str],
        means: List[float],
        stds: List[float],
        medians: List[float]
    ):
        """
        Creates a boxplot of the scores.
        Args:
            scores_list: The list of scores.
            model_name: The name of the model.
            metrics: The metrics to plot.
        """
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
        # If there's only one metric, ensure axes is iterable.
        if n == 1:
            axes = [axes]
        for i, metric in enumerate(metrics):
            ax = axes[i]
            sns.boxplot(data=scores_list[i], color="skyblue", ax=ax)
            ax.axhline(means[i], color='red', linestyle='dashed', linewidth=2, label=f'Mean: {means[i]:.3f}')
            ax.axhline(medians[i], color='blue', linestyle='dashed', linewidth=2, label=f'Median: {medians[i]:.3f}')
            ax.axhline(means[i] - stds[i], color='purple', linestyle='dashdot', linewidth=2, label=f'Mean - 1 Std: {(means[i]- stds[i]):.3f}')
            ax.axhline(means[i] + stds[i], color='purple', linestyle='dashdot', linewidth=2, label=f'Mean + 1 Std: {(means[i]+ stds[i]):.3f}')
            ax.set_xlabel("Samples")
            ax.set_ylabel(f"{metric} Score")
            ax.set_title(f"{metric} for {model_name}")
            ax.legend()
        plt.tight_layout()
        plt.show()

    def calculate_statistics(scores):
        """
        Calculates the mean, standard deviation, and median of a list of scores.
        Args:
            scores: The list of scores.
        """
        mean = np.mean(scores)
        std = np.std(scores)
        median = np.median(scores)
        return mean, std, median

class rnCV():
    def __init__(self):
        pass

    def kfold_loop(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        models: List[object],
        metric: str,
        metrics: List[str],
        n_splits: int = 3,
        outer_cv: int = 5,
        n_repeats: int = 10,
        verbose: bool = False,
    ):
        atom = ATOMClassifier(x, y)
        tnrs = []
        fnrs = []
        tprs = []
        fprs = []
        aucs = []
        mccs = []

        for r in range(n_repeats):
            atom.run(models, metric=metrics, n_trials=50, cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=outer_cv, random_state=r))
            fprs.append(atom.results['FPR'])
            tprs.append(atom.results['TPR'])
            tnrs.append(atom.results['TNR'])
            fnrs.append(atom.results['FNR'])
            aucs.append(atom.results['AUC'])
            mccs.append(atom.results['MCC'])
        
        return fprs, tprs, tnrs, fnrs, aucs, mccs
        
def main():
    pass

if __name__=="__main__":
    # Call the main function
    main()