import numpy as np
import pandas as pd
import statistics
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

warnings.simplefilter(action='ignore', category=FutureWarning)


def predict(xtrain, ytrain, xtest):
    out_of_fold = np.array([0.0] * xtrain.shape[0])
    predicted_val = np.array([0.0] * xtest.shape[0])
    scores = []
    temp = {}

    for i in xtrain.columns:
        temp[i] = []

    for train_group, test_group in KFold(n_splits=5, shuffle=True, random_state=11).split(xtrain):
        xtrainkf, xtestkf, ytrainkf, ytestkf = xtrain.iloc[train_group], xtrain.iloc[test_group], ytrain.iloc[train_group], ytrain.iloc[test_group]

        params = {
            'bagging_seed': 11,
            'boosting_type': 'gbdt',
            'colsample_bytree': 0.2,
            'learning_rate': 0.01,
            'max_depth': -1,
            'metric': 'mae',
            'min_child_samples': 79,
            'num_leaves': 128,
            'objective': 'gamma',
            'reg_alpha': 0.1302650970728192,
            'reg_lambda': 0.3603427518866501,
            'subsample': 0.8126672064208567,
            'subsample_freq': 5,
            'verbosity': -1
        }

        feature_model = lgb.LGBMRegressor(**params, n_estimators=50000, n_jobs=-1)
        feature_model.fit(xtrainkf, ytrainkf, eval_set=[(xtrainkf, ytrainkf), (xtestkf, ytestkf)], eval_metric='mae', verbose=10000, early_stopping_rounds=200)
        ykf_predicted = feature_model.predict(xtestkf)
        predicted_val += feature_model.predict(xtest, num_iteration=feature_model.best_iteration_)

        out_of_fold[test_group] = ykf_predicted.reshape(-1, )
        scores.append(mean_absolute_error(ytestkf, ykf_predicted))

        # feature importance scores
        ctr = 0
        for key, val in temp.items():
            temp[key].append(feature_model.feature_importances_[ctr])
            ctr += 1

    fi = {}
    for key, val in temp.items():
        fi[key] = statistics.mean(val)

    feature_importances = pd.DataFrame(list(sorted(fi.items(), key=lambda v: v[1], reverse=True)), columns=['features', 'importance_score'])

    plt.figure(figsize=(16, 12))
    sns.barplot(x='importance_score', y='features', data=feature_importances.head(50))
    plt.title('LightGBM Feature Importances')
    plt.show()

    print(f'CV mean score: {np.mean(scores):.4f}, std: {np.std(scores):1.4f}.')

    return predicted_val / 5, out_of_fold
