import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_classif, RFE, chi2, f_classif, mutual_info_classif, f_classif
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sklearn

from scipy.stats import ttest_ind
import scipy

from xgboost import XGBClassifier
import xgboost


def get_feature_importance(X_train, y_train, features, random_state=42):
    """
    Trains a Random Forest model on the given training data and returns the features
    sorted by their importance to the model.

    Parameters:
    - X_train: Training data features (e.g., a pandas DataFrame or NumPy array).
    - y_train: Training data labels (e.g., a pandas Series or NumPy array).
    - features: List or array of feature names (same order as X_train columns).
    - random_state: An integer to control the random state of the Random Forest model.

    Returns:
    - A list of feature names sorted by their importance in descending order.
    """

    # Initialize the RandomForestClassifier with the provided random state for reproducibility
    forest = RandomForestClassifier(random_state=random_state)

    # Fit the model to the training data
    forest.fit(X_train, y_train)

    # Get the feature importances (how important each feature is in making predictions)
    importances = forest.feature_importances_

    # Get the indices of the features, sorted in descending order of importance
    indices = np.argsort(importances)[::-1]

    # Return the features, sorted by their importance
    return np.array(features)[indices]



def get_top_features(X_train, y_train, k=10, score_func=f_classif):
    """
    Selects the top k features from the training data using the SelectKBest method.

    Parameters:
    - X_train: Training data features (e.g., a pandas DataFrame or NumPy array).
    - y_train: Training data labels (e.g., a pandas Series or NumPy array).
    - k: Number of top features to select (default is 10).
    - score_func: Function used to compute the scores for the features (default is f_classif).

    Returns:
    - indices: Array of indices of the selected top k features.
    """

    # Initialize SelectKBest with the specified scoring function and number of top features to select
    select_kbest = SelectKBest(score_func=score_func, k=k)

    # Fit SelectKBest to the training data
    select_kbest.fit(X_train, y_train)

    # Get the indices of the selected top k features
    selected_indices = select_kbest.get_support(indices=True)

    # Get the names of the selected features
    selected_features = X_train.columns[selected_indices].tolist()
        
    return selected_features


def get_feature_importances_xgb(X_train, y_train, features, random_state=42):
    """Trains an XGBoost classifier on the given training data and returns the features
    sorted by their importance.

    Parameters:
        - X_train: Training data features (e.g., a pandas DataFrame or NumPy array)
        - y_train: Training data labels (e.g., a pandas Series or NumPy array)
        - FEATURES: List or array of feature names (same order as X_train columns)
        - random_state: Integer seed for the random number generator (default is 42)

    Returns:
        - sorted_features: A list of feature names sorted by their importance in descending order.
    """

    # Initialize the XGBoost classifier with the specified random state for reproducibility
    xgb = XGBClassifier(random_state=random_state)

    # Fit the XGBoost model to the training data
    xgb.fit(X_train, y_train)

    # Get feature importances from the trained model
    xgb_importances = xgb.feature_importances_

    # Get the indices of the features sorted by importance in descending order
    indices_xgb = np.argsort(xgb_importances)[::-1]

    # Return the features sorted by their importance
    sorted_features = np.array(features)[indices_xgb].tolist()

    return sorted_features



# Create a function for performing the Chi2 test
def execute_chi2_test(X_categorical, y, categorical_features):
    """Performs the Chi-Square test of independence for categorical features and prints
    the results.

    Parameters:
        - X_categorical: Feature matrix with categorical data (e.g., a pandas DataFrame).
        - y: Labels corresponding to the features (e.g., a pandas Series or NumPy array).
        - categorical_features: List or array of feature names (same order as X_categorical columns).

    Returns:
        - chi2_results: DataFrame containing the Chi-Square scores and p-values for each feature.
    """

    # Perform the Chi-Square test
    chi2_scores, p_values = chi2(X_categorical, y)

    # Create a DataFrame with the results
    chi2_results = pd.DataFrame({
        "Feature": categorical_features,
        "Chi2 Score": chi2_scores,
        "p-value": p_values
    })

    return chi2_results



# Create a function for perfroming the Anova Test
def execute_anova_test(X, y, continuous_features):
    """Performs the ANOVA F-test for continuous features and prints the results

    Parameters:
        - X: Feature matrix (e.g., a pandas DataFrame).
        - y: Labels corresponding to the features (e.g., a pandas Series or NumPy array).
        - continuous_features: List or array of feature names (same order as X columns).

    Returns:
        - anova_results: DataFrame containing the F-scores and p-values for each continuous feature.
    """

    # Select the continuous features from the feature matrix
    X_continuous = X[continuous_features]

    # Perform the ANOVA F-test
    f_scores, p_values = f_classif(X_continuous, y)

    # Create a DataFrame with the results
    anova_results = pd.DataFrame({
        "Feature": continuous_features,
        "F-Score": f_scores,
        "p-value": p_values
    })

    return anova_results


# Create a function for performing the T-Test
def execute_t_test(X, y, continuous_features):
    """Performs T-tests for continuous features between two groups and prints the results.

    Parameters:
        - X: Feature matrix (e.g., a pandas DataFrame).
        - y: Labels corresponding to the features (e.g., a pandas Series or NumPy array).
        - continuous_features: List or array of continuous feature names (same order as X columns).

    Returns:
        - t_test_results_df: DataFrame containing the T-statistics and p-values for each continuous feature.
    """

    # Dictionary to store T-test results
    t_test_results = {}

    # Perform T-test for each continuous feature
    for feature in continuous_features:
        group1 = X[y == 0][feature]
        group2 = X[y == 1][feature]
        t_stat, p_value = ttest_ind(group1, group2)
        t_test_results[feature] = {"T-Statistic": t_stat, "p-value": p_value}

    # Create a DataFrame with the results
    t_test_results_df = pd.DataFrame(t_test_results).T

    return t_test_results_df 


# Create a function for mutual sf_scores
def execute_mutual_score(X, y, features):
    """Computes Mutual Information scores for each feature and prints the results.

    Parameters:
        - X: Feature matrix (e.g., a pandas DataFrame).
        - y: Labels corresponding to the features (e.g., a pandas Series or NumPy array).
        - features: List or array of feature names (same order as X columns).

    Returns:
        - mutual_info_results: DataFrame containing the Mutual Information scores for each feature.
    """

    # Compute Mutual Information scores
    mutual_score_results = mutual_info_classif(X, y)

    # Create a DataFrame with the results
    mutual_score_results = pd.DataFrame({
        "feature": features,
        "mutual_Information": mutual_score_results
    })

    return mutual_score_results



def standardize_data(data):
    """Standardizes the input data using StandardScaler

    Parameters:
        - data: DataFrame or array-like structure containing the data to be standardized.

    Returns:
        - standardized_data: Standardized version of the input data.
    """
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit and transform the data
    standardized_data = scaler.fit_transform(data)
    
    return standardized_data