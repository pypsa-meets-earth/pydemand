import numpy as np
import pandas as pd
import logging

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae

seed = 1
np.random.seed(seed)
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

def get_sklearn_train_val(X, y): 
    """
    Standard script to obtain train and validation performances of data X in
    predicting y. For performance and off-the-shelf performance, we use
    gradient boosting regression
    
    Args:
        X(np.ndarray or pd.DataFrame): features
        y(np.ndarray or pd.DataFrame): target 
    
    Returns:
        float: train r2 score
        float: train mean abs error
        float: val r2 score
        float: val mean abs error
    """ 

    if not isinstance(X, np.ndarray):
        X = X.to_numpy()
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()

    if len(X.shape) == 1:
        X = np.atleast_2d(X).T

    if len(y.shape) == 1:
        y = np.atleast_2d(y).T
        
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    scaled_X = X_scaler.fit_transform(X)
    scaled_y = y_scaler.fit_transform(y)
    
    n_folds = 5

    results = pd.DataFrame(np.zeros((n_folds, 4)), 
        columns=[
            "train_r2",
            "train_mae",
            "val_r2",
            "val_mae",
        ])
    
    kfold = KFold(n_splits=5, shuffle=True)

    for i, (train_index, val_index) in enumerate(kfold.split(scaled_X, scaled_y)):

        train_X, train_y = scaled_X[train_index], scaled_y[train_index]
        val_X, val_y = scaled_X[val_index], scaled_y[val_index]

        model = GradientBoostingRegressor()
        model = model.fit(train_X, train_y.flatten())

        pred_train = np.atleast_2d(model.predict(train_X)).T
        pred_val = np.atleast_2d(model.predict(val_X)).T

        results.at[i, "train_r2"] = r2_score(train_y, pred_train)
        results.at[i, "val_r2"] = r2_score(val_y, pred_val)
        results.at[i, "train_mae"] = mae(
            y_scaler.inverse_transform(train_y).flatten(),
            y_scaler.inverse_transform(pred_train).flatten())
        results.at[i, "val_mae"] = mae(
            y_scaler.inverse_transform(val_y).flatten(),
            y_scaler.inverse_transform(pred_val).flatten())
    
    return results.mean(axis=0)


def get_feature_importances(X, y):
    """
    Runs the following procedure to how features contribute
    to the overall prediction of y:
    
    1. Runs regression on individual features
    2. In a greedy-forward scheme add features to an feauture-free dataset
    3. During this, observes the changes to r2_score and MAE
    
    The procedure only contains methods that are computationally slim
    The method is inspired by Table I from 
    https://ojs.aaai.org/index.php/AAAI/article/view/7806
    
    Args:
        X(pd.DataFrame): predictive data
        y(pd.Series): target
    """ 

    features = X.columns
    metrices = ["train_r2", "train_mae", "val_r2", "val_mae"]
    individual_performances = pd.DataFrame(index=features, columns=metrices)
    individual_performances["abs_pearson_corr"] = X.corrwith(y).abs()
   
    print("Testing individual feature importances.") 
    for feature in features:
        individual_performances.loc[feature, metrices] = get_sklearn_train_val(
            X[[feature]],
            y
        )
    
    individual_performances.sort_values(by="val_mae", 
                                        ascending=True, 
                                        inplace=True)
    print(individual_performances)

    sorted_index = individual_performances.index
    accum_performances = pd.DataFrame(index=sorted_index, columns=metrices)
    
    print("Testing accumulated features starting with most predictive.") 
    for i, feature in enumerate(individual_performances.index):

        features = individual_performances.index[:i+1]
        curr_perf = get_sklearn_train_val(X[features], y)
        accum_performances.loc[feature] = curr_perf
    
    print(accum_performances)

    return individual_performances, accum_performances


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    X, y = data["data"], pd.Series(data["target"])
    X = pd.DataFrame(X, columns=data["feature_names"])

    indiv_perf, accum_perf = get_feature_importances(X, y)