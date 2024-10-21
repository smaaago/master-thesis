import os 
import random

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from hmeasure import h_score

from thesis_ml import run_algo
from thesis_combos import ComboMethods


data = pd.read_excel('thesis_data.xlsx')
to_describe = data[[
    'closed',
    'decentralized',
    'wire_transfer',
    'credit_card',
    'lifetime',
    'coins_traded',
    'pairs_traded',
    'public_team',
    'cer_score', 
    'pen_test',
    'proof_of_funds', 
    'bug_bounty',
    'mozilla_score',
    'hacked',
    'volume_mln'
]].copy()
# .to_latex('desc_stats.tex')

Y = data['closed'].copy()
X = data[[
    'decentralized',
    'wire_transfer',
    'credit_card',
    'lifetime',
    'coins_traded',
    'pairs_traded',
    'public_team',
    'cer_score', 
    'pen_test',
    'proof_of_funds', 
    'bug_bounty',
    'mozilla_score',
    'hacked',
    'volume_mln'
]].copy()

scaler = StandardScaler()
X = scaler.fit_transform(X) 

# set seeds manually
SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

test_size1 = 0.35
test_size2 = 0.35
params_grid = {
    'bayes': {},
    
    'logit': {
        'penalty': ['l1', 'l2'], 
        'C': [0.01, 0.1, 1]
    },
    
    'svm': {
        'C': [0.01, 0.1, 1],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    },
    
    'cat': {
        'iterations': [50, 100, 150],
        'min_child_samples': [5, 10],
        'depth': [2, 3, 5],
        'learning_rate': [0.01, 0.05, 0.1]
    },
        
    'rf': {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 6, 9],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [5, 10]
    }
}

np.seterr(divide = 'ignore') 

with (open('preds_val.txt', 'a') as f1, 
      open('preds_test.txt', 'a') as f2, 
      open('feature_imps.txt', 'a') as f3):

    for _ in tqdm(range(1000), desc='Simulations'):
        fis = {}
        preds_val = {}
        preds_test = {}

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y.values, 
                                                            stratify=Y, 
                                                            test_size=test_size1)
        X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, 
                                                        stratify=Y_test, 
                                                        test_size=test_size2)

        for algo_name, params in tqdm(params_grid.items(), desc='ML algos', leave=False):
            if algo_name in ['cat', 'rf']:
                val_probas, test_probas, fi = run_algo(X_train=X_train, X_val=X_val, 
                                                       X_test=X_test, Y_train=Y_train, 
                                                       algo_name=algo_name, params=params)
                fis[algo_name] = fi
            else:
                val_probas, test_probas = run_algo(X_train=X_train, X_val=X_val, 
                                                   X_test=X_test, Y_train=Y_train,
                                                   algo_name=algo_name, params=params)

            preds_val[algo_name] = val_probas
            preds_test[algo_name] = test_probas

        fis = pd.DataFrame(fis)
        preds_val = pd.DataFrame(preds_val)
        preds_test = pd.DataFrame(preds_test)

        ComboClass = ComboMethods(y_score=preds_val.values, y_true=Y_val)
        combo_preds_val = {}
        combo_preds_test = {}

        for glp_spec in ComboClass.glp_funcs().keys():
            glp = ComboClass.glp_fit(name=glp_spec)
            glp_preds = ComboClass.glp_funcs()[glp_spec](preds_test.values, glp['params'])

            combo_preds_val[glp_spec] = glp['scores']
            combo_preds_test[glp_spec] = glp_preds

        # some weird error in calculations, which I have to handle somehow
        for bmethod_name in ['blp', 'bmc2', 'bmc3']:
            try:
                bfit, bfunc = map(
                    lambda x: getattr(ComboClass, f'{bmethod_name}_{x}'), ['fit', 'func']
                )
                bfit = bfit() if bmethod_name == 'blp' else bfit(M=int(bmethod_name[-1]))
                bmethod_preds = (
                    bfunc(preds_test.values, bfit['params']) 
                    if bmethod_name == 'blp' 
                    else bfunc(preds_test.values, bfit['params'], M=int(bmethod_name[-1]))
                )
                combo_preds_val[bmethod_name] = bfit['scores']
                combo_preds_test[bmethod_name] = bmethod_preds
            except ValueError:
                print(f'same shit error in {bmethod_name}')
                fill_vals = np.full(len(combo_preds_val['simple']), np.nan)
                combo_preds_val[bmethod_name] = fill_vals
                combo_preds_test[bmethod_name] = fill_vals

        combo_preds_val = pd.DataFrame(combo_preds_val)
        combo_preds_test = pd.DataFrame(combo_preds_test)

        preds_val = pd.concat([preds_val, combo_preds_val], axis=1)
        preds_test = pd.concat([preds_test, combo_preds_test], axis=1) 

        preds_val['Y_true'] = Y_val
        preds_test['Y_true'] = Y_test

        np.savetxt(f1, preds_val.values, fmt='%.4f')
        f1.write('\n')

        np.savetxt(f2, preds_test.values, fmt='%.4f')
        f2.write('\n')

        np.savetxt(f3, fis.values, fmt='%.4f')
        f3.write('\n')