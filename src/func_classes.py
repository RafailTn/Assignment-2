"""
This file does not contain any tests
Additional time would be needed to 
write a proper test file
"""
# Loading the libraries
from atom import ATOMClassifier
from atom.feature_engineering import FeatureSelector
import ast
from sklearn.decomposition import PCA
from sklearn.utils import resample
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, fbeta_score, matthews_corrcoef, f1_score, make_scorer
from collections import Counter
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Any, Optional, List, Dict
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
    
    def get_best_from_folds(self, results: pd.DataFrame):
        """
        Gets the best model from the results of the inner cross-validation by counting the best model across folds.
        Args:
            results (pd.DataFrame): DataFrame with results of the cross-validation.
        """
        results_df = pd.read_csv(results, header=0)
        model_name = results_df["model"].values
        # Rank models by frequency
        ranked_models = Counter(model_name)
        ranked_models = ranked_models.most_common()
        print(f"Best model based on frequency: {ranked_models[0][0]}")
        print(f"Models ranked by frequency: {ranked_models}")

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

    def process_optuna_results(self, optuna_res):
        # Convert 'hyper_params' string to actual dict
        optuna_res["hyper_params"] = optuna_res["hyper_params"].apply(ast.literal_eval)
        # Expand the hyperparameters into separate columns
        hyper_df = optuna_res["hyper_params"].apply(pd.Series)
        # Merge with original metrics (optional)
        full_df = pd.concat([optuna_res.drop(columns=["hyper_params"]), hyper_df], axis=1)
        return full_df
    
    def plot_optuna_results(self, full_df):
        for col in full_df.columns:
            print(f"\n Analyzing: {col}")
            # If it's numeric, plot histogram
            # I had to use chatgpt for the next conditional statement that checks the dtype of each column
            if pd.api.types.is_numeric_dtype(full_df[col]):
                plt.figure(figsize=(6, 4))
                sns.histplot(full_df[col], bins=10, kde=True)
                plt.title(f"Distribution of {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.show()
            else:
                print(full_df[col].value_counts())

    def make_winner_pipeline(
            self,
            imputer: object,
            scaler: object,
            model: object,
            x: pd.DataFrame,
            y: pd.Series,
        ):
        """
        Creates a pipeline for the provided model.
        Args:
            model (object): List of objects to be used as the final estimator.
        """
        pipeline = Pipeline([
            ('imputer', imputer),
            ('scaler', scaler),
            ('fs', PCA(n_components=10)),
            ('model', model)
        ])
        pipeline.fit(x,y)
        joblib.dump(pipeline, Path.cwd().parent / "models" / "best_model_pipeline.pkl")
    
    def pipeline_predict(self, df_path: str):
        """
        Predicts the target variable using the pipeline.
        Args:
            df_path (str): Path to the DataFrame with features.
        Returns:
            pd.Series: Series with predicted target variable.
        """
        df = pd.read_csv(df_path)
        x = df.drop(columns=["diagnosis", 'id'])
        y = df["diagnosis"]
        pipeline = joblib.load(Path.cwd().parent / "models" / "best_model_pipeline.pkl")
        y_pred = pipeline.predict(x)
        return y_pred

class RNcvAtom:
    '''
    Class to run repeated n-fold cross-validation for a list of models.
    Args:
        X (pd.DataFrame): DataFrame with features (must contain a 'diagnosis' column).
        y (pd.Series): Series with target variable (must be a binary classification problem).
        models (List[str]): List of models to evaluate.
        param_spaces (Dict[str, List[object]]): Dictionary with model names as keys and lists of parameter values as values.
            If None, the default parameter spaces will be used.
        n_repeats (int): Number of times to repeat the cross-validation (default: 10).
        n_splits (int): Number of splits for the cross-validation (default: 5).
        fs (bool): Whether to perform feature selection (default: False).
        fs_method (object): Feature selection method to use (default: FeatureSelector(strategy="pca", n_features=10)).
        n_trials (int): Number of trials for the optimization (default: 50).
        inner_cv (int): Number of folds for the inner cross-validation (default: StratifiedKFold(n_splits=3)).
        metric (object): Metric to use for evaluation (default: make_scorer(fbeta_score, beta=2, average="weighted", zero_division=0)).
        seed (int): Random seed for reproducibility (default: 42).
    '''
    def __init__(
            self, 
            X: pd.DataFrame,
            y: pd.Series,
            models: List[str]|List[object],
            param_spaces: Dict[str, List[object]] = None,
            n_repeats: int = 10, 
            n_splits: int = 5,
            fs: bool = False,
            fs_method: object = FeatureSelector(strategy="pca", n_features=10),
            n_trials: int = 50, 
            inner_cv: object|int = StratifiedKFold(n_splits=3),
            metric: object = make_scorer(fbeta_score, beta=2, average="weighted", zero_division=0),
            seed=42
            ):
        self.X = X
        self.y = y
        self.models = models
        self.param_spaces = param_spaces
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.fs = fs
        # If feature selection is enabled, set the feature selection method
        if self.fs:
            self.fs_method = fs_method
        self.n_trials = n_trials
        self.inner_cv = inner_cv
        self.seed = seed
        self.metric = metric
        self.results = []
        self.results_baseline_per_fold = []
        self.best_model_results = []
        self.results_bootstrap = []

    def baseline_run(
            self,
            model_inst: List[object] = None
            ):
        """
        Runs the repeated n-fold cross-validation.
        """
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
            if self.fs:
                X_train_transformed = self.fs_method.fit_transform(X_train, y_train)
                X_test_transformed = self.fs_method.transform(X_test)
            else:
                X_train_transformed = X_train
                X_test_transformed = X_test
            # Create a new ATOM instance for the model training
            model_atom = ATOMClassifier(X_train_transformed, y_train, random_state=self.seed + fold_idx, verbose=2)
            if not model_inst:
                # Train the model on the transformed training data
                model_atom.run(
                    models=self.models,
                    metric=self.metric,
                    ht_params={'cv': self.inner_cv},
                )
                for model_name in model_atom.models:
                    model = model_atom[model_name]
                    # Make predictions on the transformed test data, for every model
                    y_pred = model.predict(X_test_transformed)
                    self.results.append({
                        "model": model_name,
                        "fold": fold_idx,
                        "accuracy": balanced_accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                        "f2": fbeta_score(y_test, y_pred, beta=2, average="weighted", zero_division=0),
                        "mcc": matthews_corrcoef(y_test, y_pred),
                    })
                # Winner per fold and its test performance (outer_cv metrics)
                winner = model_atom.winner
                self.results_baseline_per_fold.append({
                    "model": winner,
                    "fold": fold_idx,
                    "Metric": model_atom.metric
                })
            else:
                # Train the model on the transformed training data
                model_atom.run(
                    models=model_inst,
                    metric=self.metric,
                    ht_params={'cv': self.inner_cv},
                )
                model_obj = model_atom.winner
                # Make predictions on the transformed test data, for every model
                y_pred = model_obj.predict(X_test_transformed)
                self.results.append({
                    "model": model_obj.name,
                    "fold": fold_idx,
                    "accuracy": balanced_accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f2": fbeta_score(y_test, y_pred, beta=2, average="weighted", zero_division=0),
                    "mcc": matthews_corrcoef(y_test, y_pred),
                })

    def fine_tune(
        self,
        model: List[str],
        metric: object = make_scorer(fbeta_score, beta=2, average="weighted", zero_division=0),
        ):
        """
        Fine-tunes the provided model(s) on the provided data.
        Args:
            model (List[str]): List of models to fine-tune.
            metric (object): Metric to use for evaluation.
        """
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
            X_train_transformed = self.fs_method.fit_transform(X_train, y_train)
            X_test_transformed = self.fs_method.transform(X_test)
            # Create a new ATOM instance for the model training
            model_atom = ATOMClassifier(X_train_transformed, y_train, random_state=self.seed + fold_idx, verbose=2)
            model_atom.run(
            models=model,
            metric=metric,
            n_trials=self.n_trials,
            ht_params={'cv': self.inner_cv}
            )
            # Get the best model
            best_model = model_atom.winner
            winner_name = best_model.name
            self.best_model_results.append({
                "model": winner_name,
                "hyper_params": model_atom[winner_name].estimator.get_params(),
                "fold": fold_idx,
                "accuracy": balanced_accuracy_score(y_test, best_model.predict(X_test_transformed)),
                "precision": precision_score(y_test, best_model.predict(X_test_transformed), average="weighted", zero_division=0),
                "recall": recall_score(y_test, best_model.predict(X_test_transformed), average="weighted", zero_division=0),
                "f1": f1_score(y_test, best_model.predict(X_test_transformed), average="weighted", zero_division=0),
                "f2": fbeta_score(y_test, best_model.predict(X_test_transformed), beta=2, average="weighted", zero_division=0),
                "mcc": matthews_corrcoef(y_test, best_model.predict(X_test_transformed))
            })

    def bootstrap(
        self, 
        train_set: pd.DataFrame, 
        eval_set: pd.DataFrame, 
        n_samples: int = 1000,
        model_inst: List[object] = None,
        ):
        """
        Performs bootstrapping on the provided data.
        Args:
            train_set (pd.DataFrame): DataFrame with training data.
            eval_set (pd.DataFrame): DataFrame with evaluation data.
            n_samples (int): Number of samples to generate.
            model object: Model to evaluate.
        """
        x = train_set.drop(columns=["diagnosis"])
        y = train_set["diagnosis"]
        x_val = eval_set.drop(columns=["diagnosis"])
        y_val = eval_set["diagnosis"]
        # Perform feature selection
        if self.fs:
            X_train_transformed = self.fs_method.fit_transform(x,y)
            X_test_transformed = self.fs_method.transform(x_val)
        else:
            X_train_transformed = x
            X_test_transformed = x_val
        # Create a new ATOM instance for the model training
        model_atom = ATOMClassifier(X_train_transformed, y, random_state=self.seed, verbose=2)
        # Check if a specific model is provided
        if not model_inst:
            model_atom.run(models=self.models, metric=self.metric, ht_params={'cv': self.inner_cv})
            for i in range(n_samples):
                # Resample the data
                x_boot, y_boot = resample(X_test_transformed, y_val, random_state=self.seed + i)
                # Iterate over the models
                for model_name in model_atom.models:
                    model = model_atom[model_name]
                    # Make predictions on the transformed test data
                    y_pred = model.predict(x_boot)
                    self.results_bootstrap.append({
                        "model": model_name,
                        "accuracy": balanced_accuracy_score(y_boot, y_pred),
                        "precision": precision_score(y_boot, y_pred, average="weighted", zero_division=0),
                        "recall": recall_score(y_boot, y_pred, average="weighted", zero_division=0),
                        "f1": f1_score(y_boot, y_pred, average="weighted", zero_division=0),
                        "f2": fbeta_score(y_boot, y_pred, beta=2, average="weighted", zero_division=0),
                        "mcc": matthews_corrcoef(y_boot, y_pred),
                    })

        else:
            model_atom.run(models=model_inst, metric=self.metric, ht_params={'cv': self.inner_cv})
            for i in range(n_samples):
                # Resample the data
                x_boot, y_boot = resample(X_test_transformed, y_val, random_state=self.seed + i)
                # Iterate over the models
                # Make predictions on the transformed test data
                trained_model = model_atom.winner
                y_pred = trained_model.predict(x_boot)
                self.results_bootstrap.append({
                    "model": trained_model.name,
                    "accuracy": balanced_accuracy_score(y_boot, y_pred),
                    "precision": precision_score(y_boot, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y_boot, y_pred, average="weighted", zero_division=0),
                    "f1": f1_score(y_boot, y_pred, average="weighted", zero_division=0),
                    "f2": fbeta_score(y_boot, y_pred, beta=2, average="weighted", zero_division=0),
                    "mcc": matthews_corrcoef(y_boot, y_pred),
                })
        
    def get_best_from_inner_cv(self):
        return pd.DataFrame(self.results_baseline_per_fold)

    def get_baseline_results(self):
        return pd.DataFrame(self.results)
    
    def get_best_model_results(self):
        return pd.DataFrame(self.best_model_results)
    
    def get_bootstrap_results(self):
        return pd.DataFrame(self.results_bootstrap)
            
def main():
    pass

if __name__=="__main__":
    # Call the main function
    main()