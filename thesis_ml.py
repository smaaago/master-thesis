import numpy as np

from typing import Union, Callable
from sklearn.model_selection import (
    GridSearchCV, 
    StratifiedKFold
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


def run_algo(
        X_train: np.ndarray, 
        X_val: np.ndarray, 
        X_test: np.ndarray, 
        Y_train: np.ndarray,
        algo_name: str, 
        params: dict[str, list], 
        cv: Callable=StratifiedKFold(n_splits=5),
        cv_metric: str='f1'
    ) -> Union[tuple[np.ndarray, np.ndarray], 
               tuple[np.ndarray, np.ndarray, np.ndarray]]:
    '''
    Description:
        Runs specified Machine Learning classifier with cross-validation

    Args:
        X_train, X_val, X_test (np.ndarray): train, val, test array of features 
        Y_train (np.ndarray): train array of target values
        algo_name (str): name of the ML classifier
        params (dict[str, list]): dictionary of arrays of hyperparameteres to iterate through[]
        cv (Callable): split on cross validation. Default value = StratifiedKFold(n_splits=10)
        cv_metric (str): statistical metric used in cross-validation. Default value = 'f1'

    Returns:[]
        np.ndarray | tuple(np.ndarray, np.ndarray): array of OOS predicted probabilities or 
        tuple of it and feature importances from CatBoost and RandomForest if those are specified
    '''
    
    clfs = {
        'bayes': GaussianNB(),
        
        'logit': LogisticRegression(solver='saga', class_weight='balanced', 
                                    max_iter=500, n_jobs=-1),
        
        'svm': SVC(probability=True, class_weight='balanced'),
        
        'cat': CatBoostClassifier(auto_class_weights='Balanced', verbose=0),
        
        'rf': RandomForestClassifier(class_weight='balanced', n_jobs=-1)
    }

    if algo_name not in clfs.keys():
        raise ValueError(
            f'''Input correct 'algo_name'! The allowed names: {tuple(clfs.keys())}'''
        )

    grid_search = GridSearchCV(estimator=clfs[algo_name], param_grid=params, 
                               cv=cv, scoring=cv_metric, 
                               n_jobs=-1, pre_dispatch='2*n_jobs')
    grid_search.fit(X_train, Y_train)
    model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"'{algo_name}' best params: {best_params}")
    
    val_probas = model.predict_proba(X_val)[:, 1]
    test_probas = model.predict_proba(X_test)[:, 1]

    if algo_name in ['cat', 'rf']:
        fi = model.feature_importances_
        return val_probas, test_probas, fi
    
    return val_probas, test_probas